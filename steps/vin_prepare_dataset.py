"""Prepare VIN OCR datasets with train/val/test splits."""

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable
from urllib.parse import urlparse

import yaml
from sklearn.model_selection import train_test_split
from zenml import step
from zenml.logger import get_logger

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from vin_ocr.utils.vin_labels import extract_vin_from_filename

logger = get_logger(__name__)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


try:
    import cv2  # type: ignore
    import numpy as np  # type: ignore
except ImportError:  # pragma: no cover - handled in runtime when preprocessing is invoked
    cv2 = None
    np = None

try:
    import fsspec  # type: ignore
except ImportError:  # pragma: no cover - handled in runtime when remote mode is invoked
    fsspec = None


@dataclass(frozen=True)
class PreprocessConfig:
    target_width: int = 1024
    min_height: int = 32
    max_height: int = 512
    maintain_aspect: bool = True
    clahe_clip_limit: float = 2.0
    clahe_tile_grid: tuple[int, int] = (8, 8)
    bilateral_diameter: int = 7
    bilateral_sigma_color: float = 50.0
    bilateral_sigma_space: float = 50.0
    morph_kernel: tuple[int, int] = (3, 3)
    gamma: float = 1.2
    unsharp_amount: float = 1.0
    unsharp_sigma: float = 1.0
    adaptive_thresh_block: int = 31
    adaptive_thresh_c: int = 10


def _require_cv2() -> None:
    if cv2 is None or np is None:
        raise ImportError(
            "OpenCV and numpy are required for VIN preprocessing. "
            "Install opencv-python and numpy in your environment."
        )


def _require_fsspec() -> None:
    if fsspec is None:
        raise ImportError(
            "fsspec is required for remote data mode. "
            "Install fsspec and s3fs in your environment."
        )


def _collect_images(raw_data_dir: Path) -> tuple[list[Path], list[str]]:
    images: list[Path] = []
    labels: list[str] = []

    for path in raw_data_dir.rglob("*"):
        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        vin = extract_vin_from_filename(path)
        if not vin:
            continue
        images.append(path)
        labels.append(vin)

    return images, labels


def _remove_zero_byte_images(raw_data_dir: Path) -> int:
    removed = 0
    for path in raw_data_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            if path.stat().st_size == 0:
                path.unlink(missing_ok=True)
                removed += 1
    return removed


def _can_stratify(labels: list[str]) -> bool:
    if len(labels) < 2:
        return False
    if len(set(labels)) == len(labels):
        return False
    counts: dict[str, int] = {}
    for label in labels:
        counts[label] = counts.get(label, 0) + 1
    return min(counts.values()) >= 2


def _split_with_optional_stratify(
    images: list[str | Path],
    labels: list[str],
    test_size: float,
    seed: int,
) -> tuple[list[str | Path], list[str | Path], list[str], list[str]]:
    stratify_labels = None
    try:
        return train_test_split(
            images,
            labels,
            test_size=test_size,
            random_state=seed,
            shuffle=True,
            stratify=stratify_labels,
        )
    except ValueError:
        logger.warning("Stratified split failed; falling back to unstratified split.")
        return train_test_split(
            images,
            labels,
            test_size=test_size,
            random_state=seed,
            shuffle=True,
            stratify=None,
        )


def _collect_images_remote(remote_uri: str) -> tuple[list[str], list[str]]:
    _require_fsspec()
    fs, fs_path = fsspec.core.url_to_fs(remote_uri)
    protocol = fs.protocol[0] if isinstance(fs.protocol, (list, tuple)) else fs.protocol
    paths = fs.find(fs_path)
    images: list[str] = []
    labels: list[str] = []

    for path in paths:
        if not any(path.lower().endswith(ext) for ext in IMAGE_EXTENSIONS):
            continue
        full_path = path if protocol == "file" else f"{protocol}://{path}"
        vin = extract_vin_from_filename(full_path)
        if not vin:
            continue
        images.append(full_path)
        labels.append(vin)

    return images, labels


def _write_labels(label_path: Path, entries: list[tuple[Path, str]]) -> None:
    label_path.parent.mkdir(parents=True, exist_ok=True)
    with label_path.open("w", encoding="utf-8") as handle:
        for image_path, label in entries:
            handle.write(f"{image_path.as_posix()}\t{label}\n")


def _read_label_entries(label_path: Path) -> list[tuple[Path, str]]:
    entries: list[tuple[Path, str]] = []
    for line_number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) < 2:
            raise ValueError(f"Invalid label line at {label_path}:{line_number}")
        entries.append((Path(parts[0]), parts[1].strip()))
    return entries


def _is_remote_path(path: str | Path) -> bool:
    if isinstance(path, Path):
        return False
    return bool(urlparse(path).scheme)


def _source_name(path: str | Path) -> str:
    if isinstance(path, Path):
        return path.name
    return path.rstrip("/").split("/")[-1]


def _load_image(path: str | Path) -> "np.ndarray":
    _require_cv2()
    if _is_remote_path(path):
        _require_fsspec()
        with fsspec.open(str(path), "rb") as handle:
            data = handle.read()
        image_array = np.frombuffer(data, dtype=np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    else:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Failed to read image: {path}")
    return image


def _to_gray(image: "np.ndarray") -> "np.ndarray":
    if len(image.shape) == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def _apply_clahe(gray: "np.ndarray", config: PreprocessConfig) -> "np.ndarray":
    clahe = cv2.createCLAHE(clipLimit=config.clahe_clip_limit, tileGridSize=config.clahe_tile_grid)
    return clahe.apply(gray)


def _denoise(gray: "np.ndarray", config: PreprocessConfig) -> "np.ndarray":
    return cv2.bilateralFilter(
        gray,
        d=config.bilateral_diameter,
        sigmaColor=config.bilateral_sigma_color,
        sigmaSpace=config.bilateral_sigma_space,
    )


def _morph_enhance(gray: "np.ndarray", config: PreprocessConfig) -> "np.ndarray":
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, config.morph_kernel)
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    enhanced = cv2.addWeighted(gray, 1.0, blackhat, 1.0, 0)
    return enhanced


def _gamma_correction(gray: "np.ndarray", config: PreprocessConfig) -> "np.ndarray":
    inv_gamma = 1.0 / max(config.gamma, 1e-6)
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype("uint8")
    return cv2.LUT(gray, table)


def _unsharp_mask(gray: "np.ndarray", config: PreprocessConfig) -> "np.ndarray":
    blur = cv2.GaussianBlur(gray, (0, 0), config.unsharp_sigma)
    return cv2.addWeighted(gray, 1.0 + config.unsharp_amount, blur, -config.unsharp_amount, 0)


def _deskew(gray: "np.ndarray") -> "np.ndarray":
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return gray
    largest = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(largest)
    angle = rect[-1]
    if angle < -45:
        angle += 90
    (h, w) = gray.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(gray, matrix, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def _adaptive_resize(gray: "np.ndarray", config: PreprocessConfig) -> "np.ndarray":
    height, width = gray.shape[:2]
    scale = config.target_width / max(width, 1)
    new_height = int(height * scale)
    if new_height < config.min_height:
        scale = config.min_height / max(height, 1)
    elif new_height > config.max_height:
        scale = config.max_height / max(height, 1)

    new_width = max(1, int(width * scale))
    new_height = max(1, int(height * scale))
    resized = cv2.resize(gray, (new_width, new_height), interpolation=cv2.INTER_CUBIC)

    if not config.maintain_aspect:
        return cv2.resize(resized, (config.target_width, new_height), interpolation=cv2.INTER_CUBIC)

    if new_width < config.target_width:
        pad = config.target_width - new_width
        left = pad // 2
        right = pad - left
        return cv2.copyMakeBorder(resized, 0, 0, left, right, cv2.BORDER_REPLICATE)
    if new_width > config.target_width:
        start = (new_width - config.target_width) // 2
        return resized[:, start : start + config.target_width]
    return resized


def _binarize(gray: "np.ndarray", config: PreprocessConfig) -> "np.ndarray":
    block_size = config.adaptive_thresh_block if config.adaptive_thresh_block % 2 == 1 else config.adaptive_thresh_block + 1
    return cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        block_size,
        config.adaptive_thresh_c,
    )


def _preprocess_default(image: "np.ndarray", config: PreprocessConfig) -> "np.ndarray":
    gray = _to_gray(image)
    gray = _apply_clahe(gray, config)
    gray = _denoise(gray, config)
    gray = _morph_enhance(gray, config)
    return _adaptive_resize(gray, config)


def _preprocess_extended(image: "np.ndarray", config: PreprocessConfig) -> "np.ndarray":
    gray = _to_gray(image)
    gray = _apply_clahe(gray, config)
    gray = _denoise(gray, config)
    gray = _morph_enhance(gray, config)
    gray = _gamma_correction(gray, config)
    gray = _deskew(gray)
    gray = _unsharp_mask(gray, config)
    gray = _adaptive_resize(gray, config)
    return _binarize(gray, config)


def _process_entries(
    entries: list[tuple[str | Path, str]],
    image_root: Path,
    processor: Callable[["np.ndarray", PreprocessConfig], "np.ndarray"],
    config: PreprocessConfig,
    use_symlinks: bool,
) -> list[tuple[Path, str]]:
    image_root.mkdir(parents=True, exist_ok=True)
    mapped_entries: list[tuple[Path, str]] = []

    for source, label in entries:
        destination = image_root / _source_name(source)
        if destination.exists():
            mapped_entries.append((destination, label))
            continue
        if use_symlinks and isinstance(source, Path):
            destination.symlink_to(source.resolve())
            mapped_entries.append((destination, label))
            continue

        try:
            image = _load_image(source)
        except ValueError as exc:
            logger.warning("Skipping unreadable image %s: %s", source, exc)
            continue
        processed = processor(image, config)
        cv2.imwrite(str(destination), processed)
        mapped_entries.append((destination, label))

    return mapped_entries


@step(enable_cache=False)  # type: ignore[untyped-decorator]
def prepare_vin_dataset_step(
    raw_data_dir: str,
    dataset_dir: str,
    data_mode: str = "local",
    remote_data_uri: str = "",
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42,
    use_symlinks: bool = False,
) -> dict[str, str]:
    """Create train/val/test splits for VIN OCR.

    Returns a dict with dataset paths and label files.
    """
    _validate_split_ratios(train_ratio, val_ratio)
    if data_mode not in {"local", "remote"}:
        raise ValueError("data_mode must be 'local' or 'remote'.")

    raw_path = Path(raw_data_dir)
    dataset_path = Path(dataset_dir)

    if data_mode == "local":
        removed = _remove_zero_byte_images(raw_path)
        if removed:
            logger.warning("Removed %s zero-byte images from %s", removed, raw_path)
        preformatted_train = raw_path / "train" / "images"
        preformatted_test = raw_path / "test" / "images"
        preformatted_labels = raw_path / "train" / "train_labels.txt"
        use_presplit = (
            preformatted_train.exists()
            and preformatted_test.exists()
            and preformatted_labels.exists()
        )

        if use_presplit:
            _ensure_presplit_labels(raw_path)
            logger.info("Detected pre-split dataset layout under %s", raw_path)
            train_entries = _read_label_entries(raw_path / "train" / "train_labels.txt")
            val_entries = _read_label_entries(raw_path / "val" / "val_labels.txt")
            test_entries = _read_label_entries(raw_path / "test" / "test_labels.txt")
        else:
            images, labels = _collect_images(raw_path)
            if not images:
                raise ValueError(f"No VIN images found under {raw_path}")

            train_images, temp_images, train_labels, temp_labels = _split_with_optional_stratify(
                images,
                labels,
                test_size=1 - train_ratio,
                seed=seed,
            )

            val_ratio_adjusted = val_ratio / max(1 - train_ratio, 1e-6)
            val_images, test_images, val_labels, test_labels = _split_with_optional_stratify(
                temp_images,
                temp_labels,
                test_size=1 - val_ratio_adjusted,
                seed=seed,
            )

            train_entries = list(zip(train_images, train_labels, strict=True))
            val_entries = list(zip(val_images, val_labels, strict=True))
            test_entries = list(zip(test_images, test_labels, strict=True))
    else:
        if not remote_data_uri:
            raise ValueError("remote_data_uri is required when data_mode='remote'.")
        images, labels = _collect_images_remote(remote_data_uri)
        if not images:
            raise ValueError(f"No VIN images found under remote path {remote_data_uri}")

        train_images, temp_images, train_labels, temp_labels = _split_with_optional_stratify(
            images,
            labels,
            test_size=1 - train_ratio,
            seed=seed,
        )

        val_ratio_adjusted = val_ratio / max(1 - train_ratio, 1e-6)
        val_images, test_images, val_labels, test_labels = _split_with_optional_stratify(
            temp_images,
            temp_labels,
            test_size=1 - val_ratio_adjusted,
            seed=seed,
        )

        train_entries = list(zip(train_images, train_labels, strict=True))
        val_entries = list(zip(val_images, val_labels, strict=True))
        test_entries = list(zip(test_images, test_labels, strict=True))

    dataset_path.mkdir(parents=True, exist_ok=True)
    base_root = dataset_path.parent
    default_root = base_root / "preprocessed_default"
    extended_root = base_root / "preprocessed_extended"

    config = PreprocessConfig()

    train_default = _process_entries(
        train_entries,
        default_root / "train" / "images",
        _preprocess_default,
        config,
        use_symlinks,
    )
    val_default = _process_entries(
        val_entries,
        default_root / "val" / "images",
        _preprocess_default,
        config,
        use_symlinks,
    )
    test_default = _process_entries(
        test_entries,
        default_root / "test" / "images",
        _preprocess_default,
        config,
        use_symlinks,
    )

    train_extended = _process_entries(
        train_entries,
        extended_root / "train" / "images",
        _preprocess_extended,
        config,
        use_symlinks,
    )
    val_extended = _process_entries(
        val_entries,
        extended_root / "val" / "images",
        _preprocess_extended,
        config,
        use_symlinks,
    )
    test_extended = _process_entries(
        test_entries,
        extended_root / "test" / "images",
        _preprocess_extended,
        config,
        use_symlinks,
    )

    _write_labels(default_root / "train" / "train_labels.txt", train_default)
    _write_labels(default_root / "val" / "val_labels.txt", val_default)
    _write_labels(default_root / "test" / "test_labels.txt", test_default)

    _write_labels(extended_root / "train" / "train_labels.txt", train_extended)
    _write_labels(extended_root / "val" / "val_labels.txt", val_extended)
    _write_labels(extended_root / "test" / "test_labels.txt", test_extended)

    split_meta = {
        "train_count": len(train_default),
        "val_count": len(val_default),
        "test_count": len(test_default),
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "preprocess_default": [
            "clahe",
            "bilateral_denoise",
            "morph_blackhat",
            "adaptive_resize",
        ],
        "preprocess_extended": [
            "clahe",
            "bilateral_denoise",
            "morph_blackhat",
            "gamma_correction",
            "deskew",
            "unsharp_mask",
            "adaptive_resize",
            "adaptive_binarize",
        ],
        "default_output_dir": default_root.as_posix(),
        "extended_output_dir": extended_root.as_posix(),
    }
    with (base_root / "preprocess_report.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(split_meta, handle)

    return {
        "dataset_dir": default_root.as_posix(),
        "train_labels": (default_root / "train" / "train_labels.txt").as_posix(),
        "val_labels": (default_root / "val" / "val_labels.txt").as_posix(),
        "test_labels": (default_root / "test" / "test_labels.txt").as_posix(),
        "preprocessed_default_dir": default_root.as_posix(),
        "preprocessed_extended_dir": extended_root.as_posix(),
        "train_labels_extended": (extended_root / "train" / "train_labels.txt").as_posix(),
        "val_labels_extended": (extended_root / "val" / "val_labels.txt").as_posix(),
        "test_labels_extended": (extended_root / "test" / "test_labels.txt").as_posix(),
        "preprocess_report": (base_root / "preprocess_report.yaml").as_posix(),
    }


def _validate_split_ratios(train_ratio: float, val_ratio: float) -> None:
    if not 0.0 < train_ratio < 1.0:
        raise ValueError("train_ratio must be between 0 and 1 (exclusive).")
    if not 0.0 <= val_ratio < 1.0:
        raise ValueError("val_ratio must be between 0 and 1 (exclusive of 1).")
    if train_ratio + val_ratio >= 1.0:
        raise ValueError("train_ratio + val_ratio must be < 1.0 to leave room for test split.")


def _ensure_presplit_labels(raw_path: Path) -> None:
    expected = [
        raw_path / "train" / "train_labels.txt",
        raw_path / "val" / "val_labels.txt",
        raw_path / "test" / "test_labels.txt",
    ]
    missing = [p for p in expected if not p.exists()]
    if missing:
        missing_list = ", ".join(p.as_posix() for p in missing)
        raise FileNotFoundError(f"Pre-split dataset detected but missing label files: {missing_list}")
