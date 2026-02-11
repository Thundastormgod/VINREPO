"""Data loading step for the ZenML pipeline."""

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from zenml import step
from zenml.logger import get_logger

logger = get_logger(__name__)


def _load_split(data_dir: str, split: str) -> tuple[np.ndarray, np.ndarray]:
    """Load a dataset split (train/test) from disk.

    Args:
        data_dir: Root directory containing dataset splits.
        split: Split name to load.

    Returns:
        tuple[np.ndarray, np.ndarray]: Feature and label arrays.
    """
    images = []
    labels = []
    label_map: dict[str, int] = {}

    split_path = Path(data_dir) / split
    if not split_path.exists():
        raise ValueError(f"Split directory not found: {split_path}")

    characters = sorted([d for d in split_path.iterdir() if d.is_dir()])

    for idx, character_dir in enumerate(characters):
        label_map[character_dir.name] = idx
        image_files = [f for f in character_dir.iterdir() if f.name.endswith((".png", ".jpg"))]
        for file in image_files:
            img = Image.open(file).convert("L")
            img = img.resize((28, 28), Image.LANCZOS)
            transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,), (0.5,)),
                ]
            )
            images.append(transform(img))
            labels.append(label_map[character_dir.name])

    if not images:
        raise ValueError(f"No images found under {split_path}")

    features = torch.stack(images).numpy()
    labels_array = torch.tensor(labels, dtype=torch.long).numpy()
    return features, labels_array


@step(enable_cache=False)  # type: ignore[untyped-decorator]
def load_data_step(
    data_dir: str = "./data",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load train and test splits; fallback to dummy data if needed.

    Args:
        data_dir: Root directory containing train/test splits.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: Train features, train labels,
            test features, and test labels.
    """
    try:
        train_features, train_labels = _load_split(data_dir, "train")
        test_features, test_labels = _load_split(data_dir, "test")
        logger.info(
            "Loaded %s train samples and %s test samples",
            len(train_labels),
            len(test_labels),
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Falling back to dummy data: %s", exc)
        train_features = torch.randn(1000, 1, 28, 28).numpy()
        train_labels = torch.randint(0, 10, (1000,)).numpy()
        test_features = torch.randn(200, 1, 28, 28).numpy()
        test_labels = torch.randint(0, 10, (200,)).numpy()

    return train_features, train_labels, test_features, test_labels
