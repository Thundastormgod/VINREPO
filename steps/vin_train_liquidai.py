"""Train LiquidAI OCR from a config-driven command."""

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
def train_liquidai_step(
    dataset_info: dict[str, str],
    config_path: str,
    output_dir: str,
    liquidai_model_revision: str = "",
) -> dict[str, str | float]:
    """Train LiquidAI OCR with the provided configuration."""
    variables = {
        **dataset_info,
        "output_dir": output_dir,
        "liquidai_model_revision": liquidai_model_revision,
    }

    mlflow.log_param("liquidai_config", config_path)
    for key, value in variables.items():
        mlflow.log_param(f"liquidai_{key}", value)

    logger.info("Starting LiquidAI training with %s", config_path)
    result = run_training_from_config(config_path, variables)

    for key, value in result.items():
        if isinstance(value, float):
            mlflow.log_metric(f"liquidai_{key}", value)

    return result
