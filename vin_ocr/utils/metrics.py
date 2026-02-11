"""Evaluation metrics for VIN OCR."""

from __future__ import annotations

from collections import defaultdict


def _levenshtein_distance(a: str, b: str) -> int:
    """Compute Levenshtein distance between two strings."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)

    dp = list(range(len(b) + 1))
    for i, char_a in enumerate(a, start=1):
        prev = dp[0]
        dp[0] = i
        for j, char_b in enumerate(b, start=1):
            temp = dp[j]
            cost = 0 if char_a == char_b else 1
            dp[j] = min(
                dp[j] + 1,
                dp[j - 1] + 1,
                prev + cost,
            )
            prev = temp
    return dp[-1]


def compute_metrics(predictions: list[str], labels: list[str]) -> dict[str, float]:
    """Compute VIN-level metrics from predictions and labels."""
    if len(predictions) != len(labels):
        raise ValueError("Predictions and labels must have the same length.")

    total = len(labels)
    if total == 0:
        return {
            "exact_match": 0.0,
            "char_f1": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "cer": 0.0,
            "ned": 0.0,
        }

    exact_matches = 0
    total_correct = 0
    total_pred = 0
    total_true = 0
    total_edit = 0
    total_ned = 0.0
    position_hits: dict[int, int] = defaultdict(int)
    position_total: dict[int, int] = defaultdict(int)

    for pred, label in zip(predictions, labels, strict=True):
        pred = pred.strip().upper()
        label = label.strip().upper()

        if pred == label:
            exact_matches += 1

        total_pred += len(pred)
        total_true += len(label)

        correct = sum(1 for p, l in zip(pred, label, strict=False) if p == l)
        total_correct += correct

        edit = _levenshtein_distance(pred, label)
        total_edit += edit
        denom = max(len(pred), len(label)) or 1
        total_ned += edit / denom

        for idx, (p_char, l_char) in enumerate(zip(pred, label, strict=False), start=1):
            position_total[idx] += 1
            if p_char == l_char:
                position_hits[idx] += 1

    precision = total_correct / total_pred if total_pred else 0.0
    recall = total_correct / total_true if total_true else 0.0
    char_f1 = (2 * precision * recall / (precision + recall)) if precision + recall else 0.0
    cer = total_edit / total_true if total_true else 0.0
    ned = total_ned / total

    metrics = {
        "exact_match": exact_matches / total,
        "char_f1": char_f1,
        "precision": precision,
        "recall": recall,
        "cer": cer,
        "ned": ned,
    }

    for position, total_count in position_total.items():
        metrics[f"pos_{position:02d}_acc"] = (
            position_hits[position] / total_count if total_count else 0.0
        )

    return metrics
