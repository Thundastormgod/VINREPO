"""CLI helpers for running external training commands."""

from __future__ import annotations

import os
import shlex
import subprocess  # nosec B404
from typing import Iterable


def format_command(command: Iterable[str], variables: dict[str, str]) -> list[str]:
    """Format command tokens using the provided variables."""
    formatted = []
    for token in command:
        formatted.append(token.format_map(variables))
    return formatted


def resolve_command(command: str | Iterable[str]) -> list[str]:
    """Resolve a command string or token list to a list of args."""
    if isinstance(command, str):
        return shlex.split(command)
    return list(command)


def run_command(
    command: list[str],
    cwd: str | None = None,
    env: dict[str, str] | None = None,
) -> None:
    """Run a command and raise if it fails."""
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)

    subprocess.run(command, cwd=cwd, env=merged_env, check=True)  # nosec B603
