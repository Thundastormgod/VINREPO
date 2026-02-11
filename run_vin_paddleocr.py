"""Entrypoint for PaddleOCR VIN training pipeline."""

from __future__ import annotations

import argparse

from pipelines.vin_paddleocr_pipeline import vin_paddleocr_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the PaddleOCR VIN training pipeline.")
    parser.add_argument("--raw-data-dir", default="vin_ocr_data/raw", help="Path to raw VIN data.")
    parser.add_argument(
        "--dataset-dir",
        default="vin_ocr_data/processed",
        help="Path to processed train/val/test data.",
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
    parser.add_argument(
        "--paddle-config",
        default="vin_ocr/configs/paddleocr_finetune.yaml",
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    vin_paddleocr_pipeline.with_options(
        settings={"experiment_tracker.mlflow": {"experiment_name": "VIN OCR PaddleOCR"}}
    )(
        raw_data_dir=args.raw_data_dir,
        dataset_dir=args.dataset_dir,
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
        paddle_config=args.paddle_config,
        paddle_output_dir=args.paddle_output_dir,
        paddleocr_repo_revision=args.paddleocr_repo_revision,
        paddleocr_pretrained_model=args.paddleocr_pretrained_model,
    )


if __name__ == "__main__":
    main()
