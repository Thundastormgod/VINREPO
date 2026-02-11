"""Fetch VIN dataset from a DagsHub repo using DVC."""

from __future__ import annotations

import argparse
import os
import random
import shutil
import subprocess  # nosec B404
from pathlib import Path


def _build_repo_url(repo_url: str) -> str:
    user = os.environ.get("DAGSHUB_USER")
    token = os.environ.get("DAGSHUB_TOKEN")
    if user and token and repo_url.startswith("https://"):
        return repo_url.replace("https://", f"https://{user}:{token}@", 1)
    return repo_url


def _run(command: list[str], cwd: Path | None = None) -> None:
    subprocess.run(command, cwd=cwd, check=True)  # nosec B603


def _run_dvc_get(repo_url: str, repo_data_path: str, output_dir: Path, branch: str) -> None:
    repo_url = _build_repo_url(repo_url)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    _run(
        [
            "dvc",
            "get",
            repo_url,
            repo_data_path,
            "-o",
            str(output_dir),
            "--rev",
            branch,
        ]
    )


def _run_dagshub_download(bucket: str, bucket_path: str, output_dir: Path) -> None:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    _run(["dagshub", "download", "--bucket", bucket, bucket_path, str(output_dir)])


def _copy_data(source: Path, dest: Path) -> None:
    if source.is_dir():
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(source, dest)
    else:
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, dest)


def _sample_images(
    source_dir: Path,
    output_dir: Path,
    sample_count: int,
    seed: int,
) -> None:
    image_paths = [p for p in source_dir.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    if not image_paths:
        raise SystemExit(f"No images found under {source_dir}")

    random.seed(seed)
    sample = image_paths if sample_count <= 0 or sample_count >= len(image_paths) else random.sample(image_paths, sample_count)

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for path in sample:
        shutil.copy2(path, output_dir / path.name)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch VIN data from DagsHub using DVC.")
    parser.add_argument(
        "--repo-url",
        default="https://dagshub.com/Thundastormgod/jlr-vin-ocr",
        help="DagsHub repo URL.",
    )
    parser.add_argument(
        "--branch",
        default="main",
        help="Repo branch to clone.",
    )
    parser.add_argument(
        "--repo-data-path",
        default="data",
        help="Path to data within the repo after dvc pull.",
    )
    parser.add_argument(
        "--output-dir",
        default="vin_ocr_data/raw",
        help="Destination directory for the dataset.",
    )
    parser.add_argument(
        "--clone-dir",
        default="vin_ocr_data/.dagshub_repo",
        help="Directory used to clone the DagsHub repo.",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove any existing clone directory before cloning.",
    )
    parser.add_argument(
        "--public",
        action="store_true",
        help="Use dvc get without cloning (useful for public repos).",
    )
    parser.add_argument(
        "--dagshub-bucket",
        default="",
        help="DagsHub bucket name for direct download (e.g., Thundastormgod/jlr-vin-ocr).",
    )
    parser.add_argument(
        "--dagshub-bucket-path",
        default="",
        help="Path inside the DagsHub bucket (e.g., data_fixed/train/images).",
    )
    parser.add_argument(
        "--dvc-track",
        action="store_true",
        help="Initialize DVC (if needed) and track the output directory.",
    )
    parser.add_argument(
        "--sample-count",
        type=int,
        default=0,
        help="Sample a fixed number of images into the output dir (0 = all).",
    )
    parser.add_argument(
        "--sample-from",
        default="",
        help="Subdirectory inside the data path to sample from (e.g., train/images).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    clone_dir = Path(args.clone_dir)
    output_dir = Path(args.output_dir)
    repo_data_path = Path(args.repo_data_path)
    repo_root = Path(__file__).resolve().parents[2]

    if args.clean and clone_dir.exists():
        shutil.rmtree(clone_dir)

    if args.dagshub_bucket and args.dagshub_bucket_path:
        _run_dagshub_download(args.dagshub_bucket, args.dagshub_bucket_path, output_dir)
    elif args.public:
        _run_dvc_get(args.repo_url, args.repo_data_path, output_dir, args.branch)
    else:
        if not clone_dir.exists():
            clone_dir.parent.mkdir(parents=True, exist_ok=True)
            repo_url = _build_repo_url(args.repo_url)
            try:
                _run(["git", "clone", "--depth", "1", "--branch", args.branch, repo_url, str(clone_dir)])
            except subprocess.CalledProcessError:
                _run_dvc_get(args.repo_url, args.repo_data_path, output_dir, args.branch)
                return

        if not (clone_dir / ".dvc").exists():
            _run_dvc_get(args.repo_url, args.repo_data_path, output_dir, args.branch)
            return

        try:
            _run(["dvc", "pull"], cwd=clone_dir)
        except subprocess.CalledProcessError:
            _run_dvc_get(args.repo_url, args.repo_data_path, output_dir, args.branch)
            return

        source_path = clone_dir / repo_data_path
        if not source_path.exists():
            raise SystemExit(f"Repo data path not found: {source_path}")

        _copy_data(source_path, output_dir)

    if args.sample_count:
        sample_root = output_dir
        if args.sample_from:
            sample_root = output_dir / args.sample_from
        _sample_images(sample_root, output_dir, args.sample_count, args.seed)

    if args.dvc_track:
        if not (repo_root / ".dvc").exists():
            _run(["dvc", "init", "--subdir"], cwd=repo_root)
        _run(["dvc", "add", str(output_dir)], cwd=repo_root)


if __name__ == "__main__":
    main()
