"""
preprocess.py

Script de preprocesamiento del dataset con SageMaker.
La conexión con el bucket se hace en sm-processing_byoc_sklearn.ipynb

Prepara los datos crudos:
carga CSVs desde /opt/ml/processing/input/,
agrega ventas a nivel mensual
y guarda outputs intermedios en /opt/ml/processing/output/ (monthly.pkl y base.pkl).
"""

import argparse
import time
from pathlib import Path

import pandas as pd

import logging
from datetime import datetime


def setup_logger(log_prefix: str) -> logging.Logger:
    """Configura y retorna un logger para el script indicado."""
    # src/common/logging_utils.py -> parents[2] = raíz del repo
    project_root = Path(__file__).resolve().parents[2]
    log_dir = project_root / "artifacts" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"{log_prefix}_{timestamp}.log"

    logger = logging.getLogger(log_prefix)
    logger.setLevel(logging.INFO)
    logger.propagate = False  # evita duplicados por el root logger

    # Si ya está configurado, no vuelvas a agregar handlers
    if logger.handlers:
        return logger

    fmt = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    # Archivo
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)

    # Consola
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(ch)

    logger.info("Logger inicializado. Archivo: %s", log_path)
    return logger


FILES = [
    "sales_train.csv",
    "test.csv",
    "items.csv",
    "item_categories.csv",
    "shops.csv",
    "sample_submission.csv",
]

REQUIRED_SALES_COLS = [
    "date_block_num",
    "shop_id",
    "item_id",
    "item_cnt_day",
    "item_price",
]
REQUIRED_TEST_COLS = ["shop_id", "item_id"]
REQUIRED_ITEMS_COLS = ["item_id", "item_category_id"]

CLIP_MIN = 0
CLIP_MAX = 20


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocessing: build monthly.pkl and base.pkl")
    parser.add_argument(
        "--raw-dir",
        type=str,
        default="/opt/ml/processing/input",
        help="Directorio de entrada con CSVs raw (relativo a la raíz del repo).",
    )
    parser.add_argument(
        "--prep-dir",
        type=str,
        default="/opt/ml/processing/output",
        help="Directorio de salida para pickles (relativo a la raíz del repo).",
    )
    return parser.parse_args()


def validate_columns(df: pd.DataFrame, required: list, name: str, logger=None):
    """
    Verifica si las columnas requeridas existen en el dataframe.
    Si se añade el logger como parámetro, este registra un error en caso de que haya columnas faltantes.
    """
    missing = [c for c in required if c not in df.columns]
    if missing:
        if logger:
            logger.error("%s sin columnas requeridas: %s", name, missing)
        raise ValueError(f"Missing columns in {name}: {missing}")


def main() -> None:
    logger = setup_logger("prep")
    start_time = time.time()
    logger.info("Iniciando preprocesamiento...")

    # src/preprocessing/prep.py -> parents[2] = raíz del repo
    project_root = Path(__file__).resolve().parents[0] #contrato sagemaker
    args = parse_args()

    raw_dir = project_root / args.raw_dir
    prep_dir = project_root / args.prep_dir

    # Cargar archivos raw
    tablas: dict[str, pd.DataFrame] = {}
    for filename in FILES:
        path = raw_dir / filename
        if not path.exists():
            logger.error("No se encontró: %s", path)
            raise FileNotFoundError(path)

        df = pd.read_csv(path)
        tablas[filename] = df
        logger.info("Cargado %s (rows=%d, cols=%d)", path.name, len(df), df.shape[1])

    ventas_diarias = tablas["sales_train.csv"].copy()
    test = tablas["test.csv"].copy()
    items = tablas["items.csv"].copy()

    # Validación de columnas mínimas
    validate_columns(ventas_diarias, REQUIRED_SALES_COLS, "sales_train.csv", logger)
    validate_columns(test, REQUIRED_TEST_COLS, "test.csv", logger)
    validate_columns(items, REQUIRED_ITEMS_COLS, "items.csv", logger)

    items = items[["item_id", "item_category_id"]].copy()

    # Agregación mensual
    monthly = ventas_diarias.groupby(
        ["date_block_num", "shop_id", "item_id"], as_index=False
    ).agg(item_cnt_month=("item_cnt_day", "sum"), avg_price=("item_price", "mean"))
    monthly["item_cnt_month"] = monthly["item_cnt_month"].clip(CLIP_MIN, CLIP_MAX)
    monthly = monthly.merge(items, on="item_id", how="left")

    last_block = int(monthly["date_block_num"].max())
    test_block = last_block + 1
    logger.info("Último date_block_num: %d | Bloque test: %d", last_block, test_block)

    test_base = test.copy()
    test_base["date_block_num"] = test_block
    test_enriched = test_base.merge(items, on="item_id", how="left")

    base = pd.concat(
        [
            monthly[
                [
                    "date_block_num",
                    "shop_id",
                    "item_id",
                    "item_cnt_month",
                    "avg_price",
                    "item_category_id",
                ]
            ],
            test_enriched[["date_block_num", "shop_id", "item_id", "item_category_id"]],
        ],
        ignore_index=True,
        sort=False,
    )

    prep_dir.mkdir(parents=True, exist_ok=True)
    monthly_path = prep_dir / "monthly.pkl"
    base_path = prep_dir / "base.pkl"

    monthly.to_pickle(monthly_path)
    base.to_pickle(base_path)

    logger.info("Guardado %s (rows=%d)", monthly_path.name, len(monthly))
    logger.info("Guardado %s (rows=%d)", base_path.name, len(base))
    logger.info("Fin prep. Tiempo: %.2fs", time.time() - start_time)


if __name__ == "__main__":
    main()
