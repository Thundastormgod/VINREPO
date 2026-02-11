#!/usr/bin/env python3
"""Utility script to replay the state of an MLflow run in the local Git workspace.

Given an MLflow run name, the script will:
1. Locate the run (optionally within a specific experiment).
2. Read the Git commit hash stored with the run metadata.
3. Download the git diff artifact saved under ``git_info/tmp*.diff``.
4. Restore the current working tree to the recorded commit.
5. Apply the downloaded diff on top of that commit.

The current branch is preserved; only the working tree and index are updated.

Examples:
    python tracking/apply_run_state.py "awesome-run-id"
    python tracking/apply_run_state.py "awesome-run-id" --experiment-id "Demo Experiment ID"
    python tracking/apply_run_state.py "awesome-run-id" --tracking-uri http://localhost:5000
"""

import argparse
import os
import shutil
import subprocess  # nosec B404
import tempfile
from pathlib import Path

import mlflow
from mlflow.entities import ViewType
from mlflow.tracking import MlflowClient
from zenml.logger import get_logger

logger = get_logger(__name__)

GIT_BIN = shutil.which("git")
DVC_BIN = shutil.which("dvc")


def _require_executable(name: str, resolved_path: str | None) -> str:
    """Ensure a named executable is available on PATH.

    Args:
        name: Executable name to resolve.
        resolved_path: Resolved absolute path from shutil.which.

    Returns:
        Absolute path to the executable.
    """
    if not resolved_path:
        raise SystemExit(f"Error: '{name}' is not installed or not on PATH.")
    return resolved_path


def _git_bin() -> str:
    """Return the resolved git executable path."""
    return _require_executable("git", GIT_BIN)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for reproducing an MLflow run.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Replay an MLflow run's Git state.")
    parser.add_argument("run_id", help="MLflow run ID to replay.")
    parser.add_argument(
        "--experiment-id",
        help="MLflow experiment ID to search within.",
    )
    parser.add_argument(
        "--tracking-uri",
        help="Override MLflow tracking URI. Defaults to MLflow's configured URI.",
    )
    parser.add_argument(
        "--repo",
        default=".",
        help="Path to the Git repository root (defaults to current directory).",
    )
    parser.add_argument(
        "--diff-pattern",
        default="tmp*.diff",
        help="Glob pattern (under git_info/) used to locate the diff artifact.",
    )
    return parser.parse_args()


def generate_reproduce_run_command(run_id: str, experiment_id: str) -> str:
    """Return the CLI command that replays the state of an MLflow run.

    Args:
        run_id: Identifier of the MLflow run.
        experiment_id: Identifier of the experiment that owns the run.

    Returns:
        str: Shell command that reproduces the run's Git state.
    """
    return f"uv run python tracking/reproduce_run.py {run_id} --experiment-id {experiment_id}"


def ensure_git_repo(repo_path: Path) -> None:
    """Verify that `repo_path` points to a Git repository.

    Args:
        repo_path: Path expected to be a Git working tree.

    Raises:
        SystemExit: If the path is not a Git repository.
    """
    try:
        git_bin = _git_bin()
        subprocess.run(  # nosec B603
            [git_bin, "-C", str(repo_path), "rev-parse", "--is-inside-work-tree"],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except subprocess.CalledProcessError as err:
        raise SystemExit(f"Error: {repo_path} is not a Git repository.") from err


def ensure_clean_worktree(repo_path: Path) -> None:
    """Guard that the Git working tree has no uncommitted changes.

    Args:
        repo_path: Path to the Git repository.

    Raises:
        SystemExit: If the working tree is dirty.
    """
    git_bin = _git_bin()
    status = subprocess.run(  # nosec B603
        [git_bin, "-C", str(repo_path), "status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
    )
    if status.stdout.strip():
        raise SystemExit(
            "Error: Working tree has local changes. Commit or stash them before proceeding."
        )


def find_run(
    client: MlflowClient,
    run_id: str,
    experiment_id: str | None,
) -> mlflow.entities.Run:
    """Locate the MLflow run matching `run_id`.

    Args:
        client: MLflow client used to query runs.
        run_id: Identifier of the run to locate.
        experiment_id: Restrict the search to a specific experiment.

    Returns:
        mlflow.entities.Run: The matching MLflow run.

    Raises:
        SystemExit: If no matching run can be found.
    """
    if experiment_id:
        experiment_ids = [experiment_id]
    else:
        experiments = client.search_experiments(view_type=ViewType.ACTIVE_ONLY)
        experiment_ids = []
        for exp in experiments:
            experiment_ids.append(exp.experiment_id)

        if not experiment_ids:
            raise SystemExit("Error: No active experiments available to search.")

    filter_string = f"attributes.run_id = '{run_id}'"

    for exp_id in experiment_ids:
        runs = client.search_runs([exp_id], filter_string=filter_string, max_results=1)
        if runs:
            return runs[0]

    raise SystemExit(f"Error: No run with ID '{run_id}' found.")


def fetch_commit_hash(run: mlflow.entities.Run) -> str:
    """Extract the Git commit hash from a tracked MLflow run.

    Args:
        run: Run with Git metadata tags.

    Returns:
        str: Git commit hash associated with the run.

    Raises:
        SystemExit: If the run lacks a Git commit hash.
    """
    commit = (
        run.data.tags.get("mlflow.source.git.commit")
        or run.data.tags.get("git.commit")
        or run.data.params.get("git_commit")
    )
    if not commit:
        raise SystemExit("Error: Run does not contain a Git commit hash in its tags or params.")
    return str(commit)


def download_diff_artifact(
    client: MlflowClient,
    run_id: str,
    diff_pattern: str,
) -> Path | None:
    """Download the diff artifact for a run and return a local copy if present.

    Args:
        client: MLflow client used for artifact downloads.
        run_id: Run identifier that owns the artifact.
        diff_pattern: Glob pattern used to locate the diff file.

    Returns:
        Path | None: Local path to the diff file, or None if not available.
    """
    with tempfile.TemporaryDirectory(prefix="mlflow_git_info_") as tmp_dir:
        try:
            local_git_info = Path(client.download_artifacts(run_id, "git_info", tmp_dir))
        except Exception:
            logger.warning(
                "No git_info artifacts found for run %s. Skipping diff application.",
                run_id,
            )
            return None

        matches = sorted(local_git_info.glob(diff_pattern))
        if not matches:
            logger.warning(
                "No diff matching pattern '%s' found under git_info/. Skipping diff application.",
                diff_pattern,
            )
            return None

        # Copy the diff to a stable temp file outside the context manager scope.
        fd, path = tempfile.mkstemp(prefix="mlflow_diff_", suffix=".diff")
        os.close(fd)
        destination = Path(path)
        shutil.copy(matches[0], destination)
        return destination


def verify_commit_exists(repo_path: Path, commit_hash: str) -> None:
    """Ensure the Git commit hash exists in the local repository.

    Args:
        repo_path: Path to the Git repository.
        commit_hash: Commit hash to validate.

    Raises:
        SystemExit: If the commit is absent locally.
    """
    try:
        git_bin = _git_bin()
        subprocess.run(  # nosec B603
            [
                git_bin,
                "-C",
                str(repo_path),
                "cat-file",
                "-e",
                f"{commit_hash}^{{commit}}",
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except subprocess.CalledProcessError as err:
        raise SystemExit(
            f"Error: Commit '{commit_hash}' is not present in the local repository."
        ) from err


def restore_to_commit(repo_path: Path, commit_hash: str) -> None:
    """Restore the working tree to the specified commit without checkout.

    Args:
        repo_path: Path to the Git repository.
        commit_hash: Commit hash to restore.
    """
    git_bin = _git_bin()
    subprocess.run(  # nosec B603
        [
            git_bin,
            "-C",
            str(repo_path),
            "restore",
            "--source",
            commit_hash,
            "--worktree",
            ":/",
        ],
        check=True,
    )


def apply_diff(repo_path: Path, diff_path: Path) -> None:
    """Apply the diff artifact onto the current working tree.

    Args:
        repo_path: Path to the Git repository.
        diff_path: Path to the diff file.
    """
    git_bin = _git_bin()
    subprocess.run(  # nosec B603
        [git_bin, "-C", str(repo_path), "apply", "--whitespace=nowarn", str(diff_path)],
        check=True,
    )


def run_dvc_pull(repo_path: Path) -> None:
    """Run `dvc pull` from the repository path, warning on failure."""
    try:
        if not DVC_BIN:
            logger.warning("DVC is not installed or not on PATH; skipped dvc pull.")
            return
        subprocess.run(  # nosec B603
            [DVC_BIN, "pull"],
            check=True,
            cwd=repo_path,
        )
        logger.info("DVC pull completed.")
    except FileNotFoundError:
        logger.warning("DVC is not installed or not on PATH; skipped dvc pull.")
    except subprocess.CalledProcessError as exc:
        logger.warning("DVC pull failed: %s", exc)


def main() -> None:
    """Entry point for the CLI that restores an MLflow run's Git state."""
    args = parse_args()

    if args.tracking_uri:
        mlflow.set_tracking_uri(args.tracking_uri)

    repo_path = Path(args.repo).resolve()
    ensure_git_repo(repo_path)
    ensure_clean_worktree(repo_path)

    if args.tracking_uri:
        mlflow.set_tracking_uri(args.tracking_uri)

    client = MlflowClient()
    run = find_run(client, args.run_id, args.experiment_id)
    commit_hash = fetch_commit_hash(run)

    verify_commit_exists(repo_path, commit_hash)
    diff_path = download_diff_artifact(client, run.info.run_id, args.diff_pattern)

    logger.info(f"Restoring working tree to commit {commit_hash}...")
    restore_to_commit(repo_path, commit_hash)

    if diff_path:
        logger.info(f"Applying diff from {diff_path}...")
        try:
            apply_diff(repo_path, diff_path)
        finally:
            diff_path.unlink(missing_ok=True)
    else:
        logger.info("No diff artifact found; working tree restored to commit only.")

    run_dvc_pull(repo_path)

    logger.info("Done. The working tree now reflects the run's state.")


if __name__ == "__main__":
    main()
