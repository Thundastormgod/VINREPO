"""Training runner utilities for external OCR models."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from vin_ocr.utils.cli import format_command, resolve_command, run_command

DEFAULT_DIR_KEYS = (
    "dataset_dir",
    "preprocessed_default_dir",
    "preprocessed_extended_dir",
)


def _ensure_dirs(variables: dict[str, str], output_dir: Path) -> None:
    for key in DEFAULT_DIR_KEYS:
        path_str = variables.get(key)
        if path_str:
            Path(path_str).mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)


def _read_metrics(metrics_file: str | None, output_dir: Path) -> dict[str, str | float]:
    metrics_path = Path(metrics_file) if metrics_file else output_dir / "metrics.json"
    if not metrics_path.exists():
        return {}
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    result: dict[str, str | float] = {"metrics_file": metrics_path.as_posix()}
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            result[key] = float(value)
    return result


def _run_hf_trainer(config: dict[str, object], variables: dict[str, str]) -> dict[str, str | float]:
    module = config.get("module")
    if not isinstance(module, str) or not module:
        raise ValueError("HF trainer config requires a 'module' string.")

    output_dir_str = str(config.get("output_dir") or variables.get("output_dir") or "").strip()
    if not output_dir_str:
        raise ValueError("HF trainer config requires 'output_dir'.")
    output_dir = Path(output_dir_str)

    _ensure_dirs(variables, output_dir)

    command = [
        "python",
        "-m",
        module,
        "--train-labels",
        variables["train_labels"],
        "--output-dir",
        output_dir.as_posix(),
    ]
    if variables.get("val_labels"):
        command.extend(["--val-labels", variables["val_labels"]])
    if variables.get("test_labels"):
        command.extend(["--test-labels", variables["test_labels"]])

    metrics_file = config.get("metrics_file")
    if isinstance(metrics_file, str) and metrics_file:
        command.extend(["--metrics-file", metrics_file])

    model_id = config.get("model_id")
    if isinstance(model_id, str) and model_id:
        command.extend(["--model-id", model_id])

    model_revision = config.get("model_revision")
    if isinstance(model_revision, str) and model_revision:
        command.extend(["--model-revision", model_revision])

    if config.get("trust_remote_code") is True:
        command.append("--trust-remote-code")
    if config.get("fp16") is True:
        command.append("--fp16")

    for key, flag in (
        ("batch_size", "--batch-size"),
        ("eval_batch_size", "--eval-batch-size"),
        ("num_epochs", "--num-epochs"),
        ("learning_rate", "--learning-rate"),
        ("max_train_samples", "--max-train-samples"),
        ("max_eval_samples", "--max-eval-samples"),
        ("max_new_tokens", "--max-new-tokens"),
        ("max_target_length", "--max-target-length"),
        ("seed", "--seed"),
    ):
        value = config.get(key)
        if value is not None:
            command.extend([flag, str(value)])

    prompt = config.get("prompt")
    if isinstance(prompt, str) and prompt:
        command.extend(["--prompt", prompt])

    if config.get("export_onnx") is True:
        command.append("--export-onnx")
        onnx_task = config.get("onnx_task")
        if isinstance(onnx_task, str) and onnx_task:
            command.extend(["--onnx-task", onnx_task])
        onnx_output = config.get("onnx_output")
        if isinstance(onnx_output, str) and onnx_output:
            command.extend(["--onnx-output", onnx_output])
            Path(onnx_output).parent.mkdir(parents=True, exist_ok=True)

    run_command(command)

    result: dict[str, str | float] = {"output_dir": output_dir.as_posix()}
    metrics_file_str = metrics_file if isinstance(metrics_file, str) else None
    result.update(_read_metrics(metrics_file_str, output_dir))
    return result


def _run_paddleocr_trainer(config: dict[str, object], variables: dict[str, str]) -> dict[str, str | float]:
    output_dir_str = str(config.get("output_dir") or variables.get("output_dir") or "").strip()
    if not output_dir_str:
        raise ValueError("PaddleOCR config requires 'output_dir'.")
    output_dir = Path(output_dir_str)

    _ensure_dirs(variables, output_dir)

    repo_path = config.get("repo_path")
    config_path = config.get("config_path")
    if not isinstance(repo_path, str) or not repo_path:
        raise ValueError("PaddleOCR config requires 'repo_path'.")
    if not isinstance(config_path, str) or not config_path:
        raise ValueError("PaddleOCR config requires 'config_path'.")

    command = [
        "python",
        "-m",
        "vin_ocr.training.finetune_paddleocr",
        "--train-labels",
        variables["train_labels"],
        "--output-dir",
        output_dir.as_posix(),
        "--paddleocr-repo",
        repo_path,
        "--config-path",
        config_path,
    ]
    if variables.get("val_labels"):
        command.extend(["--val-labels", variables["val_labels"]])
    if variables.get("test_labels"):
        command.extend(["--test-labels", variables["test_labels"]])

    metrics_file = config.get("metrics_file")
    if isinstance(metrics_file, str) and metrics_file:
        command.extend(["--metrics-file", metrics_file])

    repo_revision = config.get("repo_revision")
    if isinstance(repo_revision, str) and repo_revision:
        command.extend(["--repo-revision", repo_revision])

    pretrained_model = config.get("pretrained_model")
    if isinstance(pretrained_model, str) and pretrained_model:
        command.extend(["--pretrained-model", pretrained_model])

    checkpoint_path = config.get("checkpoint_path")
    if isinstance(checkpoint_path, str) and checkpoint_path:
        command.extend(["--checkpoint-path", checkpoint_path])

    if config.get("export_onnx") is True:
        command.append("--export-onnx")
        onnx_output = config.get("onnx_output")
        if isinstance(onnx_output, str) and onnx_output:
            command.extend(["--onnx-output", onnx_output])
            Path(onnx_output).parent.mkdir(parents=True, exist_ok=True)

    run_command(command)

    result: dict[str, str | float] = {"output_dir": output_dir.as_posix()}
    metrics_file_str = metrics_file if isinstance(metrics_file, str) else None
    result.update(_read_metrics(metrics_file_str, output_dir))
    return result


def run_training_from_config(
    config_path: str,
    variables: dict[str, str],
) -> dict[str, str | float]:
    """Run a training command defined in a YAML config file.

    The config should define:
      - command: list or string
      - output_dir: str (optional)
      - metrics_file: str (optional)
      - cwd: str (optional)
      - env: dict (optional)
    """
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError(f"Invalid config at {config_path}")

    command_raw = config.get("command")
    if command_raw:
        output_dir = config.get("output_dir", variables.get("output_dir"))
        metrics_file = config.get("metrics_file")
        cwd = config.get("cwd")
        env = config.get("env")

        command = resolve_command(command_raw)
        command = format_command(command, {**variables, "output_dir": str(output_dir or "")})

        run_command(command, cwd=cwd, env=env)

        result: dict[str, str | float] = {}
        if output_dir:
            result["output_dir"] = str(output_dir)

        if metrics_file:
            metrics_path = Path(metrics_file.format_map(variables))
            if metrics_path.exists():
                metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
                for key, value in metrics.items():
                    if isinstance(value, (int, float)):
                        result[key] = float(value)
                result["metrics_file"] = metrics_path.as_posix()

        return result

    trainer = config.get("trainer")
    if trainer == "paddleocr":
        return _run_paddleocr_trainer(config, variables)
    if trainer in {"deepseek", "liquidai", "hf"}:
        return _run_hf_trainer(config, variables)
    raise ValueError(f"Missing command or trainer in {config_path}")
