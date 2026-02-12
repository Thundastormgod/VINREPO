"""PaddleOCR fine-tuning wrapper."""

from __future__ import annotations

import argparse
import json
import logging
import subprocess  # nosec B404
import sys
import tarfile
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

from vin_ocr.training import placeholder
from vin_ocr.utils.cli import run_command
from vin_ocr.utils.metrics import compute_metrics
from vin_ocr.utils.vin_labels import normalize_vin, postprocess_vin

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


def _is_url(value: str) -> bool:
    parsed = urllib.parse.urlparse(value)
    return parsed.scheme in {"http", "https"}


def _download_file(url: str, dest: Path, retries: int = 5) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    logging.info("Downloading pretrained model from %s", url)

    attempt = 0
    while attempt < retries:
        attempt += 1
        tmp_path = dest.with_suffix(dest.suffix + ".part")
        try:
            existing_size = tmp_path.stat().st_size if tmp_path.exists() else 0
            headers = {}
            if existing_size > 0:
                headers["Range"] = f"bytes={existing_size}-"
            request = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(request) as response:  # nosec B310
                expected_size = response.headers.get("Content-Length")
                expected_size_int = int(expected_size) if expected_size else None
                mode = "ab" if existing_size > 0 else "wb"
                with open(tmp_path, mode) as f:
                    downloaded = existing_size
                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        f.write(chunk)
                        downloaded += len(chunk)
                if expected_size_int is not None and downloaded < expected_size_int:
                    raise IOError(
                        f"Download incomplete: got {downloaded} of {expected_size_int} bytes"
                    )
            tmp_path.replace(dest)
            if not dest.exists() or dest.stat().st_size == 0:
                raise IOError(f"Downloaded file is missing or empty: {dest}")
            return dest
        except Exception as exc:
            logging.warning("Download attempt %d failed: %s", attempt, exc)
            if tmp_path.exists():
                tmp_path.unlink(missing_ok=True)
            if attempt >= retries:
                raise

    return dest


def _remote_content_length(url: str) -> int | None:
    try:
        request = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(request) as response:  # nosec B310
            size = response.headers.get("Content-Length")
            return int(size) if size else None
    except Exception:
        return None


def _extract_archive(archive_path: Path, extract_dir: Path) -> None:
    extract_dir.mkdir(parents=True, exist_ok=True)
    if tarfile.is_tarfile(archive_path):
        with tarfile.open(archive_path) as tar:
            tar.extractall(path=extract_dir)  # nosec B202
        return
    if zipfile.is_zipfile(archive_path):
        with zipfile.ZipFile(archive_path) as zipf:
            zipf.extractall(path=extract_dir)
        return
    raise ValueError(f"Unsupported archive format: {archive_path}")


def _resolve_pretrained_model(pretrained_model: str, output_dir: Path) -> str:
    if not pretrained_model:
        return ""

    if not _is_url(pretrained_model):
        candidate = Path(pretrained_model)
        if not candidate.is_absolute():
            candidate = (Path.cwd() / candidate).resolve()
        return candidate.as_posix()

    filename = Path(urllib.parse.urlparse(pretrained_model).path).name or "pretrained_model"
    download_dir = output_dir / "pretrained"
    download_path = download_dir / filename
    expected_size = _remote_content_length(pretrained_model)
    if download_path.exists() and expected_size:
        local_size = download_path.stat().st_size
        if local_size != expected_size:
            logging.warning(
                "Pretrained file size mismatch (local %d vs remote %d). Re-downloading.",
                local_size,
                expected_size,
            )
            download_path.unlink(missing_ok=True)
    if not download_path.exists() or download_path.stat().st_size == 0:
        _download_file(pretrained_model, download_path)

    if tarfile.is_tarfile(download_path) or zipfile.is_zipfile(download_path):
        extract_dir = download_dir / "extracted"
        if not extract_dir.exists() or not any(extract_dir.rglob("*.pdparams")):
            _extract_archive(download_path, extract_dir)
        candidates = sorted(extract_dir.rglob("*.pdparams"))
        if not candidates:
            raise FileNotFoundError(f"No .pdparams found in extracted archive {download_path}")
        return candidates[0].resolve().as_posix()

    if not download_path.exists() or download_path.stat().st_size == 0:
        raise FileNotFoundError(f"Pretrained model download failed: {download_path}")
    return download_path.resolve().as_posix()


def _extract_rec_image_shape(config: dict) -> str:
    transforms = (
        config.get("Eval", {})
        .get("dataset", {})
        .get("transforms", [])
    )
    if isinstance(transforms, list):
        for item in transforms:
            if isinstance(item, dict) and "RecResizeImg" in item:
                resize_cfg = item.get("RecResizeImg", {})
                image_shape = resize_cfg.get("image_shape")
                if isinstance(image_shape, (list, tuple)) and len(image_shape) == 3:
                    return ", ".join(str(v) for v in image_shape)
    return "3, 32, 100"


def _export_inference_model(
    repo_path: Path,
    config_path_for_command: str,
    checkpoint: Path,
    inference_dir: Path,
) -> None:
    checkpoint_str = str(checkpoint)
    if checkpoint_str.endswith(".pdparams"):
        checkpoint_str = checkpoint_str[:-9]
    inference_dir.mkdir(parents=True, exist_ok=True)
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


def _build_text_recognizer(
    repo_path: Path,
    inference_dir: Path,
    config: dict,
    use_gpu: bool,
    rec_batch_num: int,
) -> "TextRecognizer":
    if str(repo_path) not in sys.path:
        sys.path.insert(0, str(repo_path))

    import tools.infer.utility as paddle_utility
    from tools.infer.predict_rec import TextRecognizer

    parser = paddle_utility.init_args()
    args = parser.parse_args([])

    global_cfg = config.get("Global", {}) if isinstance(config.get("Global"), dict) else {}
    arch_cfg = config.get("Architecture", {}) if isinstance(config.get("Architecture"), dict) else {}

    args.use_gpu = bool(use_gpu)
    args.rec_model_dir = inference_dir.as_posix()
    args.rec_algorithm = arch_cfg.get("algorithm", "CRNN")
    args.rec_char_dict_path = global_cfg.get("character_dict_path", "ppocr/utils/en_dict.txt")
    if isinstance(args.rec_char_dict_path, str):
        char_path = Path(args.rec_char_dict_path)
        if not char_path.is_absolute():
            args.rec_char_dict_path = str((repo_path / char_path).resolve())
    args.use_space_char = bool(global_cfg.get("use_space_char", False))
    args.max_text_length = int(global_cfg.get("max_text_length", 25))
    args.rec_image_shape = _extract_rec_image_shape(config)
    args.rec_batch_num = rec_batch_num
    args.use_onnx = False
    args.benchmark = False
    args.precision = "fp32"

    return TextRecognizer(args)


def _evaluate_split(
    name: str,
    label_path: str,
    text_recognizer: "TextRecognizer",
    workspace_root: Path,
    batch_size: int,
) -> dict[str, float]:
    import cv2

    entries = placeholder._read_labels_file(Path(label_path), 0)
    images = []
    labels = []
    skipped = 0

    for entry in entries:
        image_path = entry.image_path
        if not image_path.is_absolute():
            image_path = (workspace_root / image_path).resolve()
        img = cv2.imread(image_path.as_posix())
        if img is None:
            skipped += 1
            continue
        images.append(img)
        labels.append(normalize_vin(entry.label))

    if not images:
        return {
            f"{name}_sample_count": 0.0,
            f"{name}_skipped": float(skipped),
        }

    predictions: list[str] = []
    for start in range(0, len(images), batch_size):
        batch = images[start : start + batch_size]
        rec_res, _ = text_recognizer(batch)
        for item in rec_res:
            text = item[0] if isinstance(item, (list, tuple)) and item else ""
            predictions.append(postprocess_vin(str(text)))

    if len(predictions) != len(labels):
        logging.warning(
            "Prediction/label length mismatch in %s split: %d preds vs %d labels",
            name,
            len(predictions),
            len(labels),
        )
        return {
            f"{name}_sample_count": float(len(labels)),
            f"{name}_skipped": float(skipped),
        }

    metrics = compute_metrics(predictions, labels)
    metrics["sample_count"] = float(len(labels))
    metrics["skipped"] = float(skipped)
    return {f"{name}_{key}": float(value) for key, value in metrics.items()}


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
    resolved_pretrained = _resolve_pretrained_model(args.pretrained_model, output_dir)
    if resolved_pretrained:
        train_command.append(f"Global.pretrained_model={resolved_pretrained}")
    if args.val_labels:
        val_labels_path = Path(args.val_labels).resolve()
        num_val_samples = _count_labels(args.val_labels)
        eval_batch_size = min(32, max(1, num_val_samples))
        train_command.append(f"Eval.dataset.data_dir={workspace_root}")
        train_command.append(f"Eval.dataset.label_file_list=['{val_labels_path}']")
        train_command.append(f"Eval.loader.batch_size_per_card={eval_batch_size}")

    run_command(train_command, cwd=str(repo_path))

    metrics: dict[str, float] = {
        "train_samples": float(_count_labels(args.train_labels)),
        "val_samples": float(_count_labels(args.val_labels)),
        "test_samples": float(_count_labels(args.test_labels)),
    }

    config = json.loads(Path(full_config_path).read_text(encoding="utf-8")) if full_config_path.suffix == ".json" else None
    if config is None:
        import yaml

        config = yaml.safe_load(full_config_path.read_text(encoding="utf-8"))

    checkpoint = Path(args.checkpoint_path).resolve() if args.checkpoint_path else _find_checkpoint(output_dir).resolve()
    inference_dir = (output_dir / "inference").resolve()
    _export_inference_model(repo_path, config_path_for_command, checkpoint, inference_dir)

    rec_batch_num = max(
        1,
        min(
            32,
            int(
                _count_labels(args.val_labels)
                or _count_labels(args.test_labels)
                or _count_labels(args.train_labels)
                or 1
            ),
        ),
    )
    text_recognizer = _build_text_recognizer(
        repo_path=repo_path,
        inference_dir=inference_dir,
        config=config,
        use_gpu=use_gpu,
        rec_batch_num=rec_batch_num,
    )

    metrics.update(
        _evaluate_split(
            "train",
            args.train_labels,
            text_recognizer,
            workspace_root=workspace_root,
            batch_size=rec_batch_num,
        )
    )
    if args.val_labels:
        metrics.update(
            _evaluate_split(
                "val",
                args.val_labels,
                text_recognizer,
                workspace_root=workspace_root,
                batch_size=rec_batch_num,
            )
        )
    if args.test_labels:
        metrics.update(
            _evaluate_split(
                "test",
                args.test_labels,
                text_recognizer,
                workspace_root=workspace_root,
                batch_size=rec_batch_num,
            )
        )

    metrics_path = Path(args.metrics_file) if args.metrics_file else output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")

    if args.export_onnx:
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
