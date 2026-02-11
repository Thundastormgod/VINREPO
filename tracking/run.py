"""Entry points for starting MLflow runs with project metadata."""

import mlflow

from tracking.logbook import get_logbook


def start_run() -> mlflow.ActiveRun:
    """Start a new MLflow run and attach the latest logbook entry.

    Returns:
        mlflow.ActiveRun: The active run context manager.
    """
    return mlflow.start_run(description=get_logbook())
