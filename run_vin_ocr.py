"""Entrypoint for VIN OCR training pipeline."""

from __future__ import annotations

import argparse

from pipelines.vin_ocr_pipeline import vin_ocr_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the VIN OCR training pipeline.")
    parser.add_argument("--raw-data-dir", default="vin_ocr_data/raw", help="Path to raw VIN data.")
    parser.add_argument(
        "--dataset-dir",
        default="vin_ocr_data/processed",
        help="Path to processed train/val/test data.",
    )
    parser.add_argument(
        "--data-mode",
        choices=["local", "remote"],
        default="local",
        help="Read data from local path or remote URI.",
    )
    parser.add_argument(
        "--remote-data-uri",
        default="",
        help="Remote URI for images (e.g., s3://bucket/path).",
    )
    parser.add_argument("--fetch-from-dagshub", action="store_true")
    parser.add_argument(
        "--dagshub-repo-url",
        default="https://dagshub.com/Thundastormgod/jlr-vin-ocr",
    )
    parser.add_argument("--dagshub-repo-data-path", default="data")
    parser.add_argument("--dagshub-clone-dir", default="vin_ocr_data/.dagshub_repo")
    parser.add_argument("--dagshub-branch", default="main")
    parser.add_argument("--dagshub-clean", action="store_true")
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use-symlinks", action="store_true")
    parser.add_argument("--no-train-paddle", dest="train_paddle", action="store_false")
    parser.add_argument("--no-train-deepseek", dest="train_deepseek", action="store_false")
    parser.add_argument("--no-train-liquidai", dest="train_liquidai", action="store_false")
    parser.add_argument(
        "--paddle-config",
        default="vin_ocr/configs/paddleocr_finetune.yaml",
    )
    parser.add_argument(
        "--deepseek-config",
        default="vin_ocr/configs/deepseek_finetune.yaml",
    )
    parser.add_argument(
        "--liquidai-config",
        default="vin_ocr/configs/liquidai_finetune.yaml",
    )
    parser.add_argument(
        "--paddle-output-dir",
        default="vin_ocr_outputs/paddleocr",
    )
    parser.add_argument(
        "--paddleocr-repo-revision",
        default="",
        help="Git revision (commit/tag/branch) for PaddleOCR repo.",
    )
    parser.add_argument(
        "--paddleocr-pretrained-model",
        default="",
        help="Override PaddleOCR pretrained weights path or URL.",
    )
    parser.add_argument(
        "--deepseek-output-dir",
        default="vin_ocr_outputs/deepseek",
    )
    parser.add_argument(
        "--deepseek-model-revision",
        default="",
        help="HF model revision (commit/tag/branch) for DeepSeek OCR.",
    )
    parser.add_argument(
        "--liquidai-output-dir",
        default="vin_ocr_outputs/liquidai",
    )
    parser.add_argument(
        "--liquidai-model-revision",
        default="",
        help="HF model revision (commit/tag/branch) for LiquidAI OCR.",
    )
    parser.set_defaults(train_paddle=True, train_deepseek=True, train_liquidai=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    vin_ocr_pipeline.with_options(
        settings={"experiment_tracker.mlflow": {"experiment_name": "VIN OCR Training"}}
    )(
        raw_data_dir=args.raw_data_dir,
        dataset_dir=args.dataset_dir,
        data_mode=args.data_mode,
        remote_data_uri=args.remote_data_uri,
        fetch_from_dagshub=args.fetch_from_dagshub,
        dagshub_repo_url=args.dagshub_repo_url,
        dagshub_repo_data_path=args.dagshub_repo_data_path,
        dagshub_clone_dir=args.dagshub_clone_dir,
        dagshub_branch=args.dagshub_branch,
        dagshub_clean=args.dagshub_clean,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
        use_symlinks=args.use_symlinks,
        train_paddle=args.train_paddle,
        paddle_config=args.paddle_config,
        paddle_output_dir=args.paddle_output_dir,
        paddleocr_repo_revision=args.paddleocr_repo_revision,
        paddleocr_pretrained_model=args.paddleocr_pretrained_model,
        train_deepseek=args.train_deepseek,
        deepseek_config=args.deepseek_config,
        deepseek_output_dir=args.deepseek_output_dir,
        deepseek_model_revision=args.deepseek_model_revision,
        train_liquidai=args.train_liquidai,
        liquidai_config=args.liquidai_config,
        liquidai_output_dir=args.liquidai_output_dir,
        liquidai_model_revision=args.liquidai_model_revision,
    )


if __name__ == "__main__":
    main()
