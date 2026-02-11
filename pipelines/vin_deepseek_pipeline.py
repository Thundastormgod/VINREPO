"""ZenML pipeline for DeepSeek OCR training on VIN data."""

from __future__ import annotations

from typing import Any

from steps.vin_fetch_dagshub import fetch_vin_data_step
from steps.vin_prepare_dataset import prepare_vin_dataset_step
from steps.vin_train_deepseek import train_deepseek_step
from zenml import pipeline


@pipeline  # type: ignore[untyped-decorator]
def vin_deepseek_pipeline(
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
    deepseek_config: str = "vin_ocr/configs/deepseek_finetune.yaml",
    deepseek_output_dir: str = "vin_ocr_outputs/deepseek",
    deepseek_model_revision: str = "",
) -> dict[str, Any]:
    """Train DeepSeek OCR on VIN data."""
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

    result = train_deepseek_step(
        dataset_info=dataset_info,
        config_path=deepseek_config,
        output_dir=deepseek_output_dir,
        deepseek_model_revision=deepseek_model_revision,
    )

    return {"dataset": dataset_info, "deepseek": result}
