"""Baseline VIN OCR trainer.

Builds a simple position-wise character model from labels and evaluates
predictions using filename VINs when available, otherwise the trained model.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from vin_ocr.utils.metrics import compute_metrics
from vin_ocr.utils.vin_labels import extract_vin_from_filename, normalize_vin, postprocess_vin

logger = logging.getLogger(__name__)

DEFAULT_MAX_SAMPLES = 0


@dataclass(frozen=True)
class LabelEntry:
    image_path: Path
    label: str


def _read_labels_file(path: Path, max_samples: int) -> list[LabelEntry]:
    if not path.exists():
        raise FileNotFoundError(f"Label file not found: {path}")

    entries: list[LabelEntry] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) < 2:
            raise ValueError(f"Invalid label line at {path}:{line_number}")
        image_path = Path(parts[0])
        label = parts[1].strip().upper()
        if not label:
            continue
        entries.append(LabelEntry(image_path=image_path, label=label))
        if max_samples and len(entries) >= max_samples:
            break

    if not entries:
        raise ValueError(f"No labels found in {path}")

    return entries


def _train_position_model(labels: Iterable[str]) -> dict[str, object]:
    length_counts = Counter(len(label) for label in labels if label)
    if not length_counts:
        raise ValueError("No valid labels provided for training.")

    default_length = length_counts.most_common(1)[0][0]
    position_counts: list[Counter[str]] = [Counter() for _ in range(default_length)]
    global_counts: Counter[str] = Counter()

    for label in labels:
        normalized = label.strip().upper()
        if not normalized:
            continue
        global_counts.update(normalized)
        for idx, char in enumerate(normalized[:default_length]):
            position_counts[idx][char] += 1

    default_char = global_counts.most_common(1)[0][0] if global_counts else "0"
    positions: list[dict[str, object]] = []
    for idx, counter in enumerate(position_counts, start=1):
        if counter:
            char, count = counter.most_common(1)[0]
            total = sum(counter.values())
            positions.append(
                {
                    "position": idx,
                    "top_char": char,
                    "top_prob": count / total if total else 0.0,
                }
            )
        else:
            positions.append({"position": idx, "top_char": default_char, "top_prob": 0.0})

    return {
        "default_length": default_length,
        "default_char": default_char,
        "positions": positions,
    }


def _build_default_prediction(model: dict[str, object]) -> str:
    positions = model.get("positions", [])
    if not isinstance(positions, list) or not positions:
        raise ValueError("Model positions are missing or invalid.")
    return "".join(str(item.get("top_char", "")) for item in positions)


def _evaluate_split(
    name: str,
    entries: list[LabelEntry],
    default_prediction: str,
) -> dict[str, float]:
    predictions: list[str] = []
    labels: list[str] = []
    filename_hits = 0

    for entry in entries:
        vin_from_name = extract_vin_from_filename(entry.image_path)
        if vin_from_name:
            predictions.append(postprocess_vin(vin_from_name))
            filename_hits += 1
        else:
            predictions.append(postprocess_vin(default_prediction))
        labels.append(normalize_vin(entry.label))

    metrics = compute_metrics(predictions, labels)
    metrics["filename_hit_rate"] = filename_hits / len(entries) if entries else 0.0
    metrics["sample_count"] = float(len(entries))

    return {f"{name}_{key}": float(value) for key, value in metrics.items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Baseline VIN OCR trainer.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--train-labels", required=True)
    parser.add_argument("--val-labels")
    parser.add_argument("--test-labels")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--metrics-file")
    parser.add_argument("--model-file")
    parser.add_argument("--max-samples", type=int, default=DEFAULT_MAX_SAMPLES)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_entries = _read_labels_file(Path(args.train_labels), args.max_samples)
    model = _train_position_model([entry.label for entry in train_entries])
    default_prediction = _build_default_prediction(model)

    metrics: dict[str, float] = {}
    metrics.update(_evaluate_split("train", train_entries, default_prediction))

    if args.val_labels:
        val_entries = _read_labels_file(Path(args.val_labels), args.max_samples)
        metrics.update(_evaluate_split("val", val_entries, default_prediction))

    if args.test_labels:
        test_entries = _read_labels_file(Path(args.test_labels), args.max_samples)
        metrics.update(_evaluate_split("test", test_entries, default_prediction))

    model_payload = {
        "model_name": args.model,
        "default_prediction": default_prediction,
        "position_model": model,
        "train_samples": len(train_entries),
    }

    model_path = Path(args.model_file) if args.model_file else output_dir / "baseline_model.json"
    model_path.write_text(json.dumps(model_payload, indent=2, sort_keys=True), encoding="utf-8")

    metrics_path = Path(args.metrics_file) if args.metrics_file else output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")

    logger.info("Wrote model to %s", model_path)
    logger.info("Wrote metrics to %s", metrics_path)


if __name__ == "__main__":
    main()
