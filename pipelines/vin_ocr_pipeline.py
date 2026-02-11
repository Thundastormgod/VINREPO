"""ZenML pipeline for VIN OCR training."""

from __future__ import annotations

from typing import Any

from steps.vin_fetch_dagshub import fetch_vin_data_step
from steps.vin_prepare_dataset import prepare_vin_dataset_step
from steps.vin_train_deepseek import train_deepseek_step
from steps.vin_train_liquidai import train_liquidai_step
from steps.vin_train_paddleocr import train_paddleocr_step
from zenml import pipeline


@pipeline  # type: ignore[untyped-decorator]
def vin_ocr_pipeline(
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
    train_paddle: bool = True,
    paddle_config: str = "vin_ocr/configs/paddleocr_finetune.yaml",
    paddle_output_dir: str = "vin_ocr_outputs/paddleocr",
    paddleocr_repo_revision: str = "",
    paddleocr_pretrained_model: str = "",
    train_deepseek: bool = True,
    deepseek_config: str = "vin_ocr/configs/deepseek_finetune.yaml",
    deepseek_output_dir: str = "vin_ocr_outputs/deepseek",
    deepseek_model_revision: str = "",
    train_liquidai: bool = True,
    liquidai_config: str = "vin_ocr/configs/liquidai_finetune.yaml",
    liquidai_output_dir: str = "vin_ocr_outputs/liquidai",
    liquidai_model_revision: str = "",
) -> dict[str, Any]:
    """Train PaddleOCR, DeepSeek, and LiquidAI models on VIN data."""
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

    results: dict[str, Any] = {"dataset": dataset_info}

    if train_paddle:
        results["paddleocr"] = train_paddleocr_step(
            dataset_info=dataset_info,
            config_path=paddle_config,
            output_dir=paddle_output_dir,
            paddleocr_repo_revision=paddleocr_repo_revision,
            paddleocr_pretrained_model=paddleocr_pretrained_model,
        )

    if train_deepseek:
        results["deepseek"] = train_deepseek_step(
            dataset_info=dataset_info,
            config_path=deepseek_config,
            output_dir=deepseek_output_dir,
            deepseek_model_revision=deepseek_model_revision,
        )

    if train_liquidai:
        results["liquidai"] = train_liquidai_step(
            dataset_info=dataset_info,
            config_path=liquidai_config,
            output_dir=liquidai_output_dir,
            liquidai_model_revision=liquidai_model_revision,
        )

    return results
