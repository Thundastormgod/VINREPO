"""Train PaddleOCR from a config-driven command."""

import sys
from pathlib import Path

import mlflow
from zenml import step
from zenml.logger import get_logger

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from vin_ocr.training.runner import run_training_from_config
from vin_ocr.utils.zenml import experiment_tracker_name

logger = get_logger(__name__)


@step(enable_cache=False, experiment_tracker=experiment_tracker_name())  # type: ignore[untyped-decorator]
def train_paddleocr_step(
    dataset_info: dict[str, str],
    config_path: str,
    output_dir: str,
    paddleocr_repo_revision: str = "",
    paddleocr_pretrained_model: str = "",
) -> dict[str, str | float]:
    """Train PaddleOCR with the provided configuration."""
    variables = {
        **dataset_info,
        "output_dir": output_dir,
        "paddleocr_repo_revision": paddleocr_repo_revision,
        "paddleocr_pretrained_model": paddleocr_pretrained_model,
    }

    mlflow.log_param("paddleocr_config", config_path)
    for key, value in variables.items():
        mlflow.log_param(f"paddleocr_{key}", value)

    logger.info("Starting PaddleOCR training with %s", config_path)
    result = run_training_from_config(config_path, variables)

    for key, value in result.items():
        if isinstance(value, float):
            mlflow.log_metric(f"paddleocr_{key}", value)

    return result
