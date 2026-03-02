"""
train.py

Entrena un modelo Ridge para predecir ventas mensuales.
Carga datos preparados (monthly.pkl y base.pkl), crea features (lags, month, avg_price),
evalúa RMSE en el último bloque y guarda el modelo en artifacts/model.joblib.
"""


import argparse
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

from src.common.logging_utils import setup_logger

CLIP_MIN = 0
CLIP_MAX = 20

FEATURE_COLUMNS = [
    "shop_id",
    "item_id",
    "item_category_id",
    "month",
    "lag1_cnt",
    "lag12_cnt",
    "avg_price",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Training step: train Ridge model")
    parser.add_argument("--prep-dir", type=str, default="data/prep")
    parser.add_argument("--monthly-file", type=str, default="monthly.pkl")
    parser.add_argument("--base-file", type=str, default="base.pkl")
    parser.add_argument("--output-path", type=str, default="artifacts/model.joblib")
    parser.add_argument("--alpha", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    logger = setup_logger("train")
    start_time = time.time()
    logger.info("Iniciando entrenamiento...")

    # src/training/train.py -> parents[2] = raíz del repo
    project_root = Path(__file__).resolve().parents[2]
    args = parse_args()

    prep_dir = project_root / args.prep_dir
    monthly_path = prep_dir / args.monthly_file
    base_path = prep_dir / args.base_file
    model_path = project_root / args.output_path

    if not monthly_path.exists():
        logger.error("No se encontró: %s", monthly_path)
        raise FileNotFoundError(monthly_path)

    if not base_path.exists():
        logger.error("No se encontró: %s", base_path)
        raise FileNotFoundError(base_path)

    monthly = pd.read_pickle(monthly_path)
    base = pd.read_pickle(base_path)

    logger.info("Cargado %s (rows=%d, cols=%d)", monthly_path.name, len(monthly), monthly.shape[1])
    logger.info("Cargado %s (rows=%d, cols=%d)", base_path.name, len(base), base.shape[1])

    last_block = int(monthly["date_block_num"].max())
    logger.info("Último date_block_num: %d", last_block)

    # Lag 1
    lag1 = monthly[["date_block_num", "shop_id", "item_id", "item_cnt_month"]].copy()
    lag1["date_block_num"] = lag1["date_block_num"] + 1
    lag1 = lag1.rename(columns={"item_cnt_month": "lag1_cnt"})
    base = base.merge(lag1, on=["date_block_num", "shop_id", "item_id"], how="left")
    base["lag1_cnt"] = base["lag1_cnt"].fillna(0)

    # Lag 12
    lag12 = monthly[["date_block_num", "shop_id", "item_id", "item_cnt_month"]].copy()
    lag12["date_block_num"] = lag12["date_block_num"] + 12
    lag12 = lag12.rename(columns={"item_cnt_month": "lag12_cnt"})
    base = base.merge(lag12, on=["date_block_num", "shop_id", "item_id"], how="left")
    base["lag12_cnt"] = base["lag12_cnt"].fillna(0)

    # Mes del año
    base["month"] = base["date_block_num"] % 12

    # Imputación avg_price
    item_avg_price = monthly.groupby("item_id")["avg_price"].mean()
    base["avg_price"] = base["avg_price"].fillna(base["item_id"].map(item_avg_price))
    base["avg_price"] = base["avg_price"].fillna(monthly["avg_price"].median())

    train_data = (
        base[base["date_block_num"] <= last_block]
        .dropna(subset=["item_cnt_month"])
        .copy()
    )
    logger.info("Train rows (con target): %d", len(train_data))

    features = train_data[FEATURE_COLUMNS].astype(float)
    target = train_data["item_cnt_month"].astype(float)

    is_train = train_data["date_block_num"] < last_block
    is_valid = train_data["date_block_num"] == last_block

    model = Ridge(alpha=float(args.alpha), random_state=0)
    model.fit(features[is_train], target[is_train])

    pred_valid = model.predict(features[is_valid])
    pred_valid = np.clip(pred_valid, CLIP_MIN, CLIP_MAX)

    rmse = float(np.sqrt(mean_squared_error(target[is_valid], pred_valid)))
    logger.info("Modelo entrenado - RMSE valid (último mes): %.6f", rmse)

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    logger.info("Modelo guardado: %s", model_path)

    logger.info("Fin entrenamiento. Tiempo: %.2fs", time.time() - start_time)


if __name__ == "__main__":
    main()