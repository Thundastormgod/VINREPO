"""PyTorch-related MLflow logging helpers."""

import mlflow
import torch


def log_model_architecture(model: torch.nn.Module) -> None:
    """Log the string representation of a PyTorch model to MLflow.

    Args:
        model: Model to serialise via `str(model)`.
    """
    mlflow.log_text(str(model), "model_architecture.txt")
