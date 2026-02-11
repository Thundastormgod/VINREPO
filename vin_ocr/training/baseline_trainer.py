"""Baseline trainer shared by starter OCR training scripts."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from vin_ocr.training import placeholder

logger = logging.getLogger(__name__)


def run_baseline_training(
    model_name: str,
    train_labels: str,
    val_labels: str | None,
    test_labels: str | None,
    output_dir: str,
    metrics_file: str | None,
    model_file: str | None,
    max_samples: int,
) -> None:
    """Run the baseline VIN OCR trainer and write metrics/model outputs."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    train_entries = placeholder._read_labels_file(Path(train_labels), max_samples)
    model = placeholder._train_position_model([entry.label for entry in train_entries])
    default_prediction = placeholder._build_default_prediction(model)

    metrics: dict[str, float] = {}
    metrics.update(placeholder._evaluate_split("train", train_entries, default_prediction))

    if val_labels:
        val_entries = placeholder._read_labels_file(Path(val_labels), max_samples)
        metrics.update(placeholder._evaluate_split("val", val_entries, default_prediction))

    if test_labels:
        test_entries = placeholder._read_labels_file(Path(test_labels), max_samples)
        metrics.update(placeholder._evaluate_split("test", test_entries, default_prediction))

    model_payload = {
        "model_name": model_name,
        "default_prediction": default_prediction,
        "position_model": model,
        "train_samples": len(train_entries),
        "trainer": "baseline",
    }

    model_path = Path(model_file) if model_file else output_path / "baseline_model.json"
    model_path.write_text(json.dumps(model_payload, indent=2, sort_keys=True), encoding="utf-8")

    metrics_path = Path(metrics_file) if metrics_file else output_path / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")

    logger.info("Baseline training complete for %s", model_name)
    logger.info("Wrote model to %s", model_path)
    logger.info("Wrote metrics to %s", metrics_path)
