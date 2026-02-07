"""
logging_utils.py

Configura logging consistente (archivo + consola).
"""

import logging
from datetime import datetime
from pathlib import Path


def setup_logger(log_prefix: str) -> logging.Logger:
    """Configura y retorna un logger para el script indicado."""
    project_root = Path(__file__).resolve().parents[1]
    log_dir = project_root / "artifacts" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_dir / f"{log_prefix}_{timestamp}.log"),
            logging.StreamHandler(),
        ],
        force=True,
    )

    return logging.getLogger(log_prefix)
