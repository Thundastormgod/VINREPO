"""VIN label extraction helpers."""

from __future__ import annotations

import re
from pathlib import Path

VIN_REGEX = re.compile(r"[A-HJ-NPR-Z0-9]{17}")

VIN_TRANSLITERATION = {
    **{str(digit): digit for digit in range(10)},
    "A": 1,
    "B": 2,
    "C": 3,
    "D": 4,
    "E": 5,
    "F": 6,
    "G": 7,
    "H": 8,
    "J": 1,
    "K": 2,
    "L": 3,
    "M": 4,
    "N": 5,
    "P": 7,
    "R": 9,
    "S": 2,
    "T": 3,
    "U": 4,
    "V": 5,
    "W": 6,
    "X": 7,
    "Y": 8,
    "Z": 9,
}

VIN_WEIGHTS = (8, 7, 6, 5, 4, 3, 2, 10, 0, 9, 8, 7, 6, 5, 4, 3, 2)

CONFUSABLE_MAP = {
    "O": "0",
    "I": "1",
    "Q": "0",
    "S": "5",
    "Z": "2",
    "B": "8",
}


def extract_vin_from_filename(path: str | Path) -> str | None:
    """Extract a VIN from a filename, if present.

    Args:
        path: File path or filename.

    Returns:
        str | None: VIN string if found.
    """
    name = Path(path).name.upper()
    match = VIN_REGEX.search(name)
    if match:
        return match.group(0)
    return None


def normalize_vin(value: str) -> str:
    """Normalize VIN text by stripping whitespace and non-alphanumerics."""
    normalized = "".join(char for char in value.upper() if char.isalnum())
    return normalized


def _replace_confusables(value: str) -> str:
    return "".join(CONFUSABLE_MAP.get(char, char) for char in value)


def compute_check_digit(vin: str) -> str | None:
    """Compute VIN check digit using ISO 3779 rules."""
    if len(vin) != 17:
        return None

    total = 0
    for char, weight in zip(vin, VIN_WEIGHTS, strict=True):
        value = VIN_TRANSLITERATION.get(char)
        if value is None:
            return None
        total += value * weight

    remainder = total % 11
    return "X" if remainder == 10 else str(remainder)


def postprocess_vin(prediction: str) -> str:
    """Apply VIN postprocessing rules to reduce common OCR errors."""
    normalized = normalize_vin(prediction)
    if not normalized:
        return normalized

    corrected = _replace_confusables(normalized)
    if len(corrected) != 17:
        return corrected

    check_digit = compute_check_digit(corrected)
    if check_digit is None:
        return corrected

    return f"{corrected[:8]}{check_digit}{corrected[9:]}"
