import logging
import os
from pathlib import Path

logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)


class AppPath:
    ROOT_DIR = Path(".")
    DATA_DIR = ROOT_DIR / "data"
    # store raw data
    RAW_DATA_DIR = DATA_DIR / "raw_data"
    # store preprocessed training data
    TRAIN_DATA_DIR = DATA_DIR / "train_data"
    # store configs for deployments
    MODEL_CONFIG_DIR = DATA_DIR / "model_config"
    # store captured data
    CAPTURED_DATA_DIR = DATA_DIR / "captured_data"


AppPath.RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
AppPath.TRAIN_DATA_DIR.mkdir(parents=True, exist_ok=True)
AppPath.MODEL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
AppPath.CAPTURED_DATA_DIR.mkdir(parents=True, exist_ok=True)


class AppConfig:
    MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI")
    MLFLOW_MODEL_PREFIX = "model"
