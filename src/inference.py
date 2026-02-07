"""
inference.py

Carga el modelo entrenado y genera predicciones para el archivo de envío
submission.csv.
"""

import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from logging_utils import setup_logger

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = PROJECT_ROOT / "artifacts" / "model.joblib"
INFERENCE_DIR = PROJECT_ROOT / "data" / "inference"
PRED_DIR = PROJECT_ROOT / "data" / "predictions"
OUT_PATH = PRED_DIR / "submission.csv"

CLIP_MIN = 0
CLIP_MAX = 20


if __name__ == "__main__":
    logger = setup_logger("prediccion")
    start_time = time.time()

    logger.info("Iniciando predicción...")

    try:
        model = joblib.load(MODEL_PATH)
    except FileNotFoundError:
        logger.error("Modelo no encontrado: %s", MODEL_PATH.name)
        raise

    x_test_path = INFERENCE_DIR / "xtest.pkl"
    test_path = INFERENCE_DIR / "test.pkl"

    if not x_test_path.exists():
        logger.error("No se encontró: %s", x_test_path.name)
        raise FileNotFoundError(x_test_path)

    if not test_path.exists():
        logger.error("No se encontró: %s", test_path.name)
        raise FileNotFoundError(test_path)

    x_test = pd.read_pickle(x_test_path)
    test = pd.read_pickle(test_path)

    logger.info(
        "Cargado %s (rows=%d, cols=%d)", x_test_path.name, len(x_test), x_test.shape[1]
    )
    logger.info(
        "Cargado %s (rows=%d, cols=%d)", test_path.name, len(test), test.shape[1]
    )

    if "ID" not in test.columns:
        logger.error("La columna 'ID' no existe en %s", test_path.name)
        raise ValueError("Missing required column: ID")

    pred_test = model.predict(x_test)
    pred_test = np.clip(pred_test, CLIP_MIN, CLIP_MAX)

    submission = pd.DataFrame({"ID": test["ID"], "item_cnt_month": pred_test})

    PRED_DIR.mkdir(parents=True, exist_ok=True)
    submission.to_csv(OUT_PATH, index=False)

    logger.info("Predicciones guardadas: %s (rows=%d)", OUT_PATH.name, len(submission))
    logger.info("Fin predicción. Tiempo: %.2fs", time.time() - start_time)
