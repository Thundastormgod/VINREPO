"""Entrypoint for DeepSeek VIN training pipeline."""

from __future__ import annotations

import argparse

from pipelines.vin_deepseek_pipeline import vin_deepseek_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the DeepSeek VIN training pipeline.")
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
        "--deepseek-config",
        default="vin_ocr/configs/deepseek_finetune.yaml",
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    vin_deepseek_pipeline.with_options(
        settings={"experiment_tracker.mlflow": {"experiment_name": "VIN OCR DeepSeek"}}
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
        deepseek_config=args.deepseek_config,
        deepseek_output_dir=args.deepseek_output_dir,
        deepseek_model_revision=args.deepseek_model_revision,
    )


if __name__ == "__main__":
    main()
