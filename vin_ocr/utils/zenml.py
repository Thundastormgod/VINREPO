"""ZenML helper utilities."""

from __future__ import annotations

from zenml.client import Client


def experiment_tracker_name() -> str | None:
    """Return the active experiment tracker name if configured."""
    try:
        tracker = Client().active_stack.experiment_tracker
        return tracker.name if tracker else None
    except Exception:
        return None
