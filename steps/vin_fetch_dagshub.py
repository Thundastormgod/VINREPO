"""Fetch VIN data from DagsHub using a helper script."""

import sys
from pathlib import Path

from zenml import step
from zenml.logger import get_logger

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from vin_ocr.tools.dagshub_fetch import main as dagshub_fetch_main

logger = get_logger(__name__)


@step(enable_cache=False)  # type: ignore[untyped-decorator]
def fetch_vin_data_step(
    repo_url: str,
    repo_data_path: str,
    output_dir: str,
    clone_dir: str,
    branch: str = "main",
    clean: bool = False,
) -> str:
    """Fetch VIN data from DagsHub and return the local path."""
    args = [
        "--repo-url",
        repo_url,
        "--repo-data-path",
        repo_data_path,
        "--output-dir",
        output_dir,
        "--clone-dir",
        clone_dir,
        "--branch",
        branch,
    ]
    if clean:
        args.append("--clean")

    logger.info("Fetching VIN data from %s", repo_url)
    dagshub_fetch_main(args)  # type: ignore[arg-type]
    return Path(output_dir).as_posix()
