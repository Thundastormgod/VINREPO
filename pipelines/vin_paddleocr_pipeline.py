"""ZenML pipeline for PaddleOCR training on VIN data."""

from __future__ import annotations

from typing import Any

from steps.vin_fetch_dagshub import fetch_vin_data_step
from steps.vin_prepare_dataset import prepare_vin_dataset_step
from steps.vin_train_paddleocr import train_paddleocr_step
from zenml import pipeline


@pipeline  # type: ignore[untyped-decorator]
def vin_paddleocr_pipeline(
    raw_data_dir: str,
    dataset_dir: str,
    data_mode: str = "local",
    remote_data_uri: str = "",
    fetch_from_dagshub: bool = False,
    dagshub_repo_url: str = "https://dagshub.com/Thundastormgod/jlr-vin-ocr",
    dagshub_repo_data_path: str = "data",
    dagshub_clone_dir: str = "vin_ocr_data/.dagshub_repo",
    dagshub_branch: str = "main",
    dagshub_clean: bool = False,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42,
    use_symlinks: bool = False,
    paddle_config: str = "vin_ocr/configs/paddleocr_finetune.yaml",
    paddle_output_dir: str = "vin_ocr_outputs/paddleocr",
    paddleocr_repo_revision: str = "",
    paddleocr_pretrained_model: str = "",
) -> dict[str, Any]:
    """Train PaddleOCR on VIN data."""
    if fetch_from_dagshub:
        raw_data_dir = fetch_vin_data_step(
            repo_url=dagshub_repo_url,
            repo_data_path=dagshub_repo_data_path,
            output_dir=raw_data_dir,
            clone_dir=dagshub_clone_dir,
            branch=dagshub_branch,
            clean=dagshub_clean,
        )

    dataset_info = prepare_vin_dataset_step(
        raw_data_dir=raw_data_dir,
        dataset_dir=dataset_dir,
        data_mode=data_mode,
        remote_data_uri=remote_data_uri,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        seed=seed,
        use_symlinks=use_symlinks,
    )

    result = train_paddleocr_step(
        dataset_info=dataset_info,
        config_path=paddle_config,
        output_dir=paddle_output_dir,
        paddleocr_repo_revision=paddleocr_repo_revision,
        paddleocr_pretrained_model=paddleocr_pretrained_model,
    )

    return {"dataset": dataset_info, "paddleocr": result}
