"""PaddleOCR fine-tuning wrapper."""

from __future__ import annotations

import argparse
import json
import logging
import subprocess  # nosec B404
from pathlib import Path

from vin_ocr.training import placeholder
from vin_ocr.utils.cli import run_command

DEFAULT_CONFIG_PATH = "configs/rec/rec_svtrnet.yml"


def _count_labels(label_path: str | None) -> int:
    if not label_path:
        return 0
    entries = placeholder._read_labels_file(Path(label_path), 0)
    return len(entries)


def _find_checkpoint(output_dir: Path) -> Path:
    candidates = sorted(output_dir.rglob("*.pdparams"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError(f"No PaddleOCR checkpoints found under {output_dir}")
    return candidates[0]


def _ensure_git_clean(repo_path: Path) -> None:
    result = subprocess.run(
        ["git", "-C", str(repo_path), "status", "--porcelain"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Failed to check git status in {repo_path}")
    if result.stdout.strip():
        raise RuntimeError(
            "PaddleOCR repo has uncommitted changes. Commit/stash before using --repo-revision."
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PaddleOCR fine-tuning.")
    parser.add_argument("--train-labels", required=True)
    parser.add_argument("--val-labels")
    parser.add_argument("--test-labels")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--metrics-file")
    parser.add_argument("--paddleocr-repo", default="third_party/PaddleOCR")
    parser.add_argument("--config-path", default=DEFAULT_CONFIG_PATH)
    parser.add_argument(
        "--repo-revision",
        default=None,
        help="Git revision (commit hash, tag, or branch) for the PaddleOCR repo.",
    )
    parser.add_argument(
        "--pretrained-model",
        default=None,
        help="Override PaddleOCR pretrained weights path or URL for reproducible runs.",
    )
    parser.add_argument("--checkpoint-path")
    parser.add_argument("--export-onnx", action="store_true")
    parser.add_argument("--onnx-output")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    repo_path = Path(args.paddleocr_repo)
    config_path = Path(args.config_path)
    if not repo_path.exists():
        raise FileNotFoundError(f"PaddleOCR repo not found at {repo_path}")
    
    # For absolute path verification, join with repo_path
    full_config_path = config_path if config_path.is_absolute() else repo_path / config_path
    if not full_config_path.exists():
        raise FileNotFoundError(f"PaddleOCR config not found at {full_config_path}")
    
    # Use the original config_path (relative to repo) for the command since cwd is repo_path
    config_path_for_command = args.config_path

    if args.repo_revision:
        _ensure_git_clean(repo_path)
        run_command(["git", "-C", str(repo_path), "checkout", args.repo_revision])

    # Detect if CUDA is available
    try:
        import paddle
        use_gpu = paddle.device.is_compiled_with_cuda() and paddle.device.cuda.device_count() > 0
    except Exception:
        use_gpu = False

    train_labels_path = Path(args.train_labels).resolve()
    # data_dir should be the main repo root since label files contain full relative paths
    # e.g., "vin_ocr_data/preprocessed_default/train/images/filename.jpg"
    # We need to resolve to absolute path since PaddleOCR runs from third_party/PaddleOCR
    workspace_root = Path.cwd().resolve()

    # Compute batch size based on available samples
    num_train_samples = _count_labels(args.train_labels)
    # PaddleOCR requires batch_size <= num_samples, so we clamp it
    batch_size = min(32, max(1, num_train_samples))
    if batch_size < 32:
        logging.info(
            "Adjusted batch size from 32 to %d to match %d training samples",
            batch_size,
            num_train_samples,
        )

    train_command = [
        "python",
        "tools/train.py",
        "-c",
        config_path_for_command,
        "-o",
        f"Global.use_gpu={str(use_gpu).lower()}",
        f"Global.save_model_dir={output_dir.resolve()}",
        f"Global.epoch_num=3",
        f"Train.dataset.data_dir={workspace_root}",
        f"Train.dataset.label_file_list=['{train_labels_path}']",
        f"Train.loader.batch_size_per_card={batch_size}",
    ]
    if args.pretrained_model:
        train_command.append(f"Global.pretrained_model={args.pretrained_model}")
    if args.val_labels:
        val_labels_path = Path(args.val_labels).resolve()
        num_val_samples = _count_labels(args.val_labels)
        eval_batch_size = min(32, max(1, num_val_samples))
        train_command.append(f"Eval.dataset.data_dir={workspace_root}")
        train_command.append(f"Eval.dataset.label_file_list=['{val_labels_path}']")
        train_command.append(f"Eval.loader.batch_size_per_card={eval_batch_size}")

    run_command(train_command, cwd=str(repo_path))

    metrics = {
        "train_samples": float(_count_labels(args.train_labels)),
        "val_samples": float(_count_labels(args.val_labels)),
        "test_samples": float(_count_labels(args.test_labels)),
    }

    metrics_path = Path(args.metrics_file) if args.metrics_file else output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")

    if args.export_onnx:
        checkpoint = Path(args.checkpoint_path).resolve() if args.checkpoint_path else _find_checkpoint(output_dir).resolve()
        # Remove .pdparams extension if present - PaddleOCR expects path without extension
        checkpoint_str = str(checkpoint)
        if checkpoint_str.endswith(".pdparams"):
            checkpoint_str = checkpoint_str[:-9]
        inference_dir = (output_dir / "inference").resolve()
        run_command(
            [
                "python",
                "tools/export_model.py",
                "-c",
                config_path_for_command,
                "-o",
                f"Global.pretrained_model={checkpoint_str}",
                f"Global.save_inference_dir={inference_dir}",
            ],
            cwd=str(repo_path),
        )

        # Detect model filename format - newer PaddlePaddle uses .json, older uses .pdmodel
        model_json = inference_dir / "inference.json"
        model_pdmodel = inference_dir / "inference.pdmodel"
        if model_json.exists():
            model_filename = "inference.json"
        elif model_pdmodel.exists():
            model_filename = "inference.pdmodel"
        else:
            logging.warning("No inference model found, skipping ONNX export")
            return

        onnx_output = Path(args.onnx_output).resolve() if args.onnx_output else (output_dir / "export.onnx").resolve()
        run_command(
            [
                "paddle2onnx",
                "--model_dir",
                str(inference_dir),
                "--model_filename",
                model_filename,
                "--params_filename",
                "inference.pdiparams",
                "--save_file",
                str(onnx_output),
            ],
            cwd=str(repo_path),
        )


if __name__ == "__main__":
    main()
