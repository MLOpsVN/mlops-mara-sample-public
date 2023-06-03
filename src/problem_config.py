import json
from utils import AppPath


class ProblemConst:
    PHASE1 = "phase-1"
    PROB1 = "prob-1"


class ProblemConfig:
    # required inputs
    phase_id: str
    prob_id: str
    test_size: float
    random_state: int

    # for original data
    raw_data_path: str
    feature_config_path: str
    category_index_path: str
    train_data_path: str
    train_x_path: str
    train_y_path: str
    test_x_path: str
    test_y_path: str

    # ml-problem properties
    target_col: str
    numerical_cols: list
    categorical_cols: list
    ml_type: str

    # for data captured from API
    captured_data_dir: str
    processed_captured_data_dir: str
    # processed captured data
    captured_x_path: str
    uncertain_y_path: str


def load_feature_configs_dict(config_path: str) -> dict:
    with open(config_path) as f:
        features_config = json.load(f)
    return features_config


def create_prob_config(phase_id: str, prob_id: str) -> ProblemConfig:
    prob_config = ProblemConfig()
    prob_config.prob_id = prob_id
    prob_config.phase_id = phase_id
    prob_config.test_size = 0.2
    prob_config.random_state = 123

    # construct data paths for original data
    prob_config.raw_data_path = (
        AppPath.RAW_DATA_DIR / f"{phase_id}" / f"{prob_id}" / "raw_train.parquet"
    )
    prob_config.feature_config_path = (
        AppPath.RAW_DATA_DIR / f"{phase_id}" / f"{prob_id}" / "features_config.json"
    )
    prob_config.train_data_path = AppPath.TRAIN_DATA_DIR / f"{phase_id}" / f"{prob_id}"
    prob_config.train_data_path.mkdir(parents=True, exist_ok=True)

    prob_config.category_index_path = (
        prob_config.train_data_path / "category_index.pickle"
    )
    prob_config.train_x_path = prob_config.train_data_path / "train_x.parquet"
    prob_config.train_y_path = prob_config.train_data_path / "train_y.parquet"
    prob_config.test_x_path = prob_config.train_data_path / "test_x.parquet"
    prob_config.test_y_path = prob_config.train_data_path / "test_y.parquet"

    # get properties of ml-problem
    feature_configs = load_feature_configs_dict(prob_config.feature_config_path)
    prob_config.target_col = feature_configs.get("target_column")
    prob_config.categorical_cols = feature_configs.get("category_columns")
    prob_config.numerical_cols = feature_configs.get("numeric_columns")
    prob_config.ml_type = feature_configs.get("ml_type")

    # construct data paths for API-captured data
    prob_config.captured_data_dir = (
        AppPath.CAPTURED_DATA_DIR / f"{phase_id}" / f"{prob_id}"
    )
    prob_config.captured_data_dir.mkdir(parents=True, exist_ok=True)
    prob_config.processed_captured_data_dir = (
        prob_config.captured_data_dir / "processed"
    )
    prob_config.processed_captured_data_dir.mkdir(parents=True, exist_ok=True)
    prob_config.captured_x_path = (
        prob_config.processed_captured_data_dir / "captured_x.parquet"
    )
    prob_config.uncertain_y_path = (
        prob_config.processed_captured_data_dir / "uncertain_y.parquet"
    )

    return prob_config


def get_prob_config(phase_id: str, prob_id: str):
    prob_config = create_prob_config(phase_id, prob_id)
    return prob_config
