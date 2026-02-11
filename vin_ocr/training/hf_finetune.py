"""Hugging Face vision-to-text fine-tuning with ONNX export.

This module supports training on multiple hardware platforms:
- CUDA (NVIDIA GPU) - Recommended for training, supports flash_attention_2
- MPS (Apple Silicon) - Supports SDPA attention
- CPU - Slowest, uses eager attention

For cloud deployment, use CUDA with flash_attention_2 for best performance.
For local development, install the flash-attn-stub package for non-CUDA systems:
    pip install -e ./stubs/flash_attn_stub
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import yaml

from vin_ocr.training import placeholder
from vin_ocr.utils.metrics import compute_metrics
from vin_ocr.utils.vin_labels import normalize_vin, postprocess_vin

logger = logging.getLogger(__name__)

DEFAULT_BATCH_SIZE = 2
DEFAULT_EVAL_BATCH_SIZE = 2
DEFAULT_EPOCHS = 1
DEFAULT_LR = 5e-5
DEFAULT_MAX_TRAIN_SAMPLES = 0
DEFAULT_MAX_EVAL_SAMPLES = 200
DEFAULT_SEED = 42
DEFAULT_ONNX_TASK = "image-to-text"

# Model config directory
MODEL_CONFIGS_DIR = Path(__file__).parent.parent / "configs" / "models"


@dataclass
class HardwareInfo:
    """Detected hardware capabilities."""
    platform: str  # "cuda", "mps", "cpu"
    has_cuda: bool
    has_mps: bool
    cuda_version: str | None
    vram_gb: float | None
    ram_gb: float | None
    recommended_attn: str  # "flash_attention_2", "sdpa", "eager"


def detect_hardware() -> HardwareInfo:
    """Detect available hardware and recommend attention implementation."""
    import torch
    import platform as plat
    import psutil
    
    has_cuda = torch.cuda.is_available()
    has_mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    
    cuda_version = None
    vram_gb = None
    
    if has_cuda:
        cuda_version = torch.version.cuda
        try:
            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        except Exception:
            pass
    
    # Get system RAM
    try:
        ram_gb = psutil.virtual_memory().total / (1024**3)
    except Exception:
        ram_gb = None
    
    # Determine platform and recommended attention
    if has_cuda:
        platform = "cuda"
        # Check if flash_attn is available (real package, not stub)
        try:
            import flash_attn
            # Check if it's the real package or our stub
            if hasattr(flash_attn, 'FlashAttnNotAvailableError'):
                # It's our stub - use sdpa
                recommended_attn = "sdpa"
            else:
                recommended_attn = "flash_attention_2"
        except ImportError:
            recommended_attn = "sdpa"
    elif has_mps:
        platform = "mps"
        recommended_attn = "sdpa"
    else:
        platform = "cpu"
        recommended_attn = "eager"
    
    return HardwareInfo(
        platform=platform,
        has_cuda=has_cuda,
        has_mps=has_mps,
        cuda_version=cuda_version,
        vram_gb=vram_gb,
        ram_gb=ram_gb,
        recommended_attn=recommended_attn,
    )


@dataclass
class ModelConfig:
    """Configuration for a HuggingFace model."""
    model_id: str
    model_name: str
    hardware: dict[str, Any] = field(default_factory=dict)
    alternatives: list[dict[str, Any]] = field(default_factory=list)
    requirements: dict[str, str] = field(default_factory=dict)
    fallback: dict[str, Any] = field(default_factory=dict)
    training: dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def from_yaml(cls, path: Path) -> "ModelConfig":
        """Load model config from YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(
            model_id=data["model_id"],
            model_name=data.get("model_name", data["model_id"].split("/")[-1]),
            hardware=data.get("hardware", {}),
            alternatives=data.get("alternatives", []),
            requirements=data.get("requirements", {}),
            fallback=data.get("fallback", {}),
            training=data.get("training", {}),
        )
    
    @classmethod
    def from_model_id(cls, model_id: str) -> "ModelConfig":
        """Create a minimal config from just a model ID."""
        return cls(
            model_id=model_id,
            model_name=model_id.split("/")[-1],
        )
    
    def get_attn_implementation(self, hw: HardwareInfo) -> str:
        """Get the appropriate attention implementation for detected hardware."""
        # If explicitly set in training config
        attn_impl = self.training.get("attn_implementation", "auto")
        if attn_impl != "auto":
            return attn_impl
        
        # Check hardware profile for platform-specific setting
        supported = self.hardware.get("supported", [])
        for entry in supported:
            if entry.get("platform") == hw.platform:
                return entry.get("attn_implementation", hw.recommended_attn)
        
        # Default to hardware recommendation
        return hw.recommended_attn


def load_model_config(config_name: str | None = None, model_id: str | None = None) -> ModelConfig:
    """Load model config by name or create from model_id."""
    if config_name:
        config_path = MODEL_CONFIGS_DIR / f"{config_name}.yaml"
        if config_path.exists():
            return ModelConfig.from_yaml(config_path)
        raise FileNotFoundError(f"Model config not found: {config_path}")
    
    if model_id:
        # Try to find a config that matches this model_id
        if MODEL_CONFIGS_DIR.exists():
            for config_file in MODEL_CONFIGS_DIR.glob("*.yaml"):
                config = ModelConfig.from_yaml(config_file)
                if config.model_id == model_id:
                    return config
        # No config found, create minimal one
        return ModelConfig.from_model_id(model_id)
    
    raise ValueError("Either config_name or model_id must be provided")


def log_hardware_info(hw: HardwareInfo, config: ModelConfig) -> None:
    """Log detected hardware and configuration."""
    logger.info("=" * 60)
    logger.info("HARDWARE DETECTION")
    logger.info("=" * 60)
    logger.info(f"Platform: {hw.platform.upper()}")
    logger.info(f"CUDA available: {hw.has_cuda}" + (f" (v{hw.cuda_version})" if hw.cuda_version else ""))
    logger.info(f"MPS available: {hw.has_mps}")
    if hw.vram_gb:
        logger.info(f"GPU VRAM: {hw.vram_gb:.1f} GB")
    if hw.ram_gb:
        logger.info(f"System RAM: {hw.ram_gb:.1f} GB")
    logger.info(f"Recommended attention: {hw.recommended_attn}")
    logger.info(f"Using attention: {config.get_attn_implementation(hw)}")
    
    # Check if optimal platform
    optimal = config.hardware.get("optimal", "cuda")
    if hw.platform != optimal:
        logger.warning(
            f"Running on {hw.platform.upper()} but model is optimized for {optimal.upper()}. "
            f"For best performance, deploy to a {optimal.upper()} environment."
        )
    logger.info("=" * 60)


@dataclass(frozen=True)
class LabelEntry:
    image_path: Path
    label: str


def _require_deps() -> None:
    try:
        import datasets  # noqa: F401
        import torch  # noqa: F401
        import transformers  # noqa: F401
        from PIL import Image  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "Missing training dependencies. Install datasets, torch, transformers, pillow, and optimum."
        ) from exc


def _read_entries(label_path: str, max_samples: int) -> list[LabelEntry]:
    entries = placeholder._read_labels_file(Path(label_path), max_samples)
    return [LabelEntry(image_path=entry.image_path, label=entry.label) for entry in entries]


def _build_dataset(entries: list[LabelEntry]) -> list[dict[str, str]]:
    return [
        {
            "image_path": entry.image_path.as_posix(),
            "text": entry.label,
        }
        for entry in entries
    ]


def _load_processor(model_id: str, trust_remote_code: bool, revision: str | None):
    from transformers import AutoProcessor, AutoTokenizer, AutoImageProcessor, PreTrainedTokenizerFast
    import importlib.util
    import sys

    # Try AutoProcessor first
    try:
        return AutoProcessor.from_pretrained(
            model_id,
            trust_remote_code=trust_remote_code,
            revision=revision,
        )
    except (ValueError, OSError, KeyError) as e:
        logger.debug(f"AutoProcessor failed: {e}")

    # Try loading custom processor from model's code (for models like DeepSeek-OCR)
    try:
        from huggingface_hub import hf_hub_download
        
        # Check if model has custom processing code
        try:
            processing_file = hf_hub_download(
                model_id, "processing_deepseekocr.py", revision=revision
            )
            spec = importlib.util.spec_from_file_location("processing_deepseekocr", processing_file)
            module = importlib.util.module_from_spec(spec)
            sys.modules["processing_deepseekocr"] = module
            spec.loader.exec_module(module)
            if hasattr(module, "DeepSeekOCRProcessor"):
                return module.DeepSeekOCRProcessor.from_pretrained(
                    model_id, trust_remote_code=trust_remote_code, revision=revision
                )
        except Exception:
            pass
    except ImportError:
        pass

    # Fallback: Try to load tokenizer and image processor separately
    tokenizer = None
    
    # Try AutoTokenizer first
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_id, trust_remote_code=trust_remote_code, revision=revision
        )
    except Exception as e:
        logger.debug(f"AutoTokenizer failed: {e}")
        
        # Try PreTrainedTokenizerFast directly (handles custom tokenizer_class issues)
        try:
            from huggingface_hub import hf_hub_download
            tokenizer_file = hf_hub_download(model_id, "tokenizer.json", revision=revision)
            tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_file)
            logger.info(f"Loaded tokenizer using PreTrainedTokenizerFast for {model_id}")
        except Exception as e2:
            logger.debug(f"PreTrainedTokenizerFast failed: {e2}")
    
    if tokenizer is None:
        raise ValueError(
            f"Could not load processor for {model_id}. "
            f"This model may require custom processing code or a specific transformers version."
        )
    
    # Try to add image processor if available
    try:
        image_processor = AutoImageProcessor.from_pretrained(
            model_id, trust_remote_code=trust_remote_code, revision=revision
        )
        # Create a simple combined processor
        class CombinedProcessor:
            def __init__(self, tokenizer, image_processor):
                self.tokenizer = tokenizer
                self.image_processor = image_processor
            
            def __call__(self, images=None, text=None, **kwargs):
                result = {}
                if images is not None:
                    result.update(self.image_processor(images, **kwargs))
                if text is not None:
                    result.update(self.tokenizer(text, **kwargs))
                return result
        
        return CombinedProcessor(tokenizer, image_processor)
    except Exception:
        return tokenizer



def _load_model(
    model_id: str,
    trust_remote_code: bool,
    torch_dtype,
    revision: str | None,
    attn_implementation: str = "auto",
    hw: HardwareInfo | None = None,
):
    """Load a HuggingFace model with appropriate attention implementation.
    
    Args:
        model_id: HuggingFace model ID
        trust_remote_code: Whether to trust remote code
        torch_dtype: Torch dtype for model weights
        revision: Model revision
        attn_implementation: Attention implementation ("auto", "flash_attention_2", "sdpa", "eager")
        hw: Hardware info (auto-detected if None)
    """
    import warnings
    import torch
    from transformers import AutoModel, AutoConfig
    
    # Detect hardware if not provided
    if hw is None:
        hw = detect_hardware()
    
    # Determine attention implementations to try
    if attn_implementation == "auto":
        # Order based on hardware
        if hw.has_cuda:
            attn_implementations = ["flash_attention_2", "sdpa", "eager"]
        elif hw.has_mps:
            attn_implementations = ["sdpa", "eager"]
        else:
            attn_implementations = ["eager", "sdpa"]
    else:
        # Use specified implementation with fallbacks
        attn_implementations = [attn_implementation]
        if attn_implementation != "eager":
            attn_implementations.append("eager")
    
    # Try to load config first
    config = None
    try:
        config = AutoConfig.from_pretrained(
            model_id,
            trust_remote_code=trust_remote_code,
            revision=revision,
        )
    except Exception:
        pass
    
    last_error = None
    for attn_impl in attn_implementations:
        try:
            logger.info(f"Attempting to load model with attn_implementation={attn_impl}")
            
            # Suppress model type mismatch warnings for custom architectures
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=".*to instantiate a model of type.*")
                warnings.filterwarnings("ignore", message=".*FlashAttention.*")
                
                kwargs = {
                    "trust_remote_code": trust_remote_code,
                    "torch_dtype": torch_dtype,
                    "revision": revision,
                    "attn_implementation": attn_impl,
                    "use_safetensors": True,
                    "low_cpu_mem_usage": True,
                }
                if config is not None:
                    kwargs["config"] = config
                
                model = AutoModel.from_pretrained(model_id, **kwargs)
                
                # Move to appropriate device
                if hw.has_cuda:
                    model = model.to("cuda")
                    logger.info("Model loaded on CUDA")
                elif hw.has_mps:
                    try:
                        model = model.to("mps")
                        logger.info("Model loaded on MPS (Apple Silicon)")
                    except Exception:
                        logger.warning("Failed to move model to MPS, using CPU")
                else:
                    logger.info("Model loaded on CPU")
                
                logger.info(f"Successfully loaded model with {attn_impl} attention")
                return model
                
        except ImportError as e:
            err_str = str(e)
            # Flash attention not available - try next implementation
            if "FlashAttention" in err_str or "flash_attn" in err_str or "flash-attn" in err_str:
                logger.info(f"Flash attention not available: {e}")
                last_error = e
                continue
            raise
        except Exception as e:
            logger.warning(f"Failed to load with {attn_impl}: {e}")
            last_error = e
            continue
    
    raise RuntimeError(f"Failed to load model {model_id}: {last_error}")


def _get_tokenizer(processor):
    tokenizer = getattr(processor, "tokenizer", None)
    return tokenizer or processor


def _prepare_inputs(processor, images, prompt: str):
    try:
        return processor(images=images, text=[prompt] * len(images), padding=True, return_tensors="pt")
    except TypeError:
        return processor(images=images, return_tensors="pt")


def _build_labels(tokenizer, texts: list[str], max_target_length: int | None):
    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_target_length,
        return_tensors="pt",
    )
    labels = encoded.input_ids
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if pad_token_id is not None:
        labels = labels.clone()
        labels[labels == pad_token_id] = -100
    return labels


class _ImageTextCollator:
    def __init__(self, processor, prompt: str, max_target_length: int | None):
        self.processor = processor
        self.prompt = prompt
        self.max_target_length = max_target_length

    def __call__(self, batch: list[dict[str, str]]):
        from PIL import Image

        images = [Image.open(item["image_path"]).convert("RGB") for item in batch]
        texts = [item["text"] for item in batch]
        model_inputs = _prepare_inputs(self.processor, images, self.prompt)
        tokenizer = _get_tokenizer(self.processor)
        model_inputs["labels"] = _build_labels(tokenizer, texts, self.max_target_length)
        return model_inputs


def _evaluate(
    model,
    processor,
    entries: list[LabelEntry],
    prompt: str,
    max_new_tokens: int,
    batch_size: int,
    device,
) -> dict[str, float]:
    from PIL import Image

    if not entries:
        return {}

    tokenizer = _get_tokenizer(processor)
    predictions: list[str] = []
    labels: list[str] = []

    for idx in range(0, len(entries), batch_size):
        batch = entries[idx : idx + batch_size]
        images = [Image.open(item.image_path).convert("RGB") for item in batch]
        inputs = _prepare_inputs(processor, images, prompt)
        inputs = {key: value.to(device) for key, value in inputs.items()}
        generated = model.generate(**inputs, max_new_tokens=max_new_tokens)
        decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)
        for pred, entry in zip(decoded, batch, strict=True):
            predictions.append(postprocess_vin(pred))
            labels.append(normalize_vin(entry.label))

    metrics = compute_metrics(predictions, labels)
    metrics["sample_count"] = float(len(entries))
    return metrics


def _export_onnx(model_dir: Path, onnx_task: str, onnx_output: Path) -> Path:
    from vin_ocr.utils.cli import run_command

    export_dir = model_dir / "onnx"
    export_dir.mkdir(parents=True, exist_ok=True)

    run_command(
        [
            "optimum-cli",
            "export",
            "onnx",
            "--model",
            str(model_dir),
            "--task",
            onnx_task,
            str(export_dir),
        ]
    )

    onnx_files = sorted(export_dir.rglob("*.onnx"))
    if not onnx_files:
        raise FileNotFoundError(f"No ONNX file produced under {export_dir}")

    onnx_output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(onnx_files[0], onnx_output)
    return onnx_output


def build_arg_parser(default_model_id: str | None = None, default_model_config: str | None = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fine-tune HF OCR model.")
    
    # Model specification (config takes precedence over model-id)
    model_group = parser.add_mutually_exclusive_group(required=(default_model_id is None and default_model_config is None))
    model_group.add_argument(
        "--model-config",
        default=default_model_config,
        help="Model config name (e.g., 'deepseek', 'liquidai') from configs/models/",
    )
    model_group.add_argument(
        "--model-id",
        default=default_model_id,
        help="HuggingFace model ID (e.g., 'deepseek-ai/DeepSeek-OCR')",
    )
    
    parser.add_argument("--train-labels", required=True)
    parser.add_argument("--val-labels")
    parser.add_argument("--test-labels")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--metrics-file")
    parser.add_argument("--max-train-samples", type=int, default=DEFAULT_MAX_TRAIN_SAMPLES)
    parser.add_argument("--max-eval-samples", type=int, default=DEFAULT_MAX_EVAL_SAMPLES)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--eval-batch-size", type=int, default=DEFAULT_EVAL_BATCH_SIZE)
    parser.add_argument("--num-epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LR)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--prompt", default="")
    parser.add_argument("--max-target-length", type=int)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument(
        "--model-revision",
        default=None,
        help="HF model revision (commit hash, tag, or branch) for reproducible runs.",
    )
    parser.add_argument("--export-onnx", action="store_true")
    parser.add_argument("--onnx-task", default=DEFAULT_ONNX_TASK)
    parser.add_argument("--onnx-output")
    return parser


def main(
    default_model_id: str | None = None,
    default_model_config: str | None = None,
) -> None:
    """Main entry point for HF fine-tuning.
    
    Args:
        default_model_id: Default HuggingFace model ID (legacy support)
        default_model_config: Default model config name (e.g., 'deepseek', 'liquidai')
    """
    logging.basicConfig(level=logging.INFO)
    _require_deps()

    parser = build_arg_parser(default_model_id, default_model_config)
    args = parser.parse_args()

    # Load model configuration
    model_config = load_model_config(
        config_name=args.model_config,
        model_id=args.model_id,
    )
    
    # Detect hardware and log info
    hw = detect_hardware()
    log_hardware_info(hw, model_config)
    
    # Get effective model_id and settings from config
    model_id = model_config.model_id
    trust_remote_code = args.trust_remote_code or model_config.training.get("trust_remote_code", False)
    attn_implementation = model_config.get_attn_implementation(hw)
    
    logger.info("Using model config: %s (model_id=%s)", model_config.model_name, model_id)

    from datasets import Dataset
    from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, set_seed
    import torch

    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_entries = _read_entries(args.train_labels, args.max_train_samples)
    val_entries = _read_entries(args.val_labels, args.max_eval_samples) if args.val_labels else []
    test_entries = _read_entries(args.test_labels, args.max_eval_samples) if args.test_labels else []

    # Try to load processor and model, with fallback support
    try:
        processor = _load_processor(model_id, trust_remote_code, args.model_revision)
        # Set dtype based on hardware
        # - CUDA: fp16 if requested
        # - MPS: fp16 works on Apple Silicon M1+
        # - CPU: use default (fp32)
        if args.fp16 and (hw.has_cuda or hw.has_mps):
            torch_dtype = torch.float16
        else:
            torch_dtype = None
        model = _load_model(
            model_id,
            trust_remote_code,
            torch_dtype,
            args.model_revision,
            attn_implementation=attn_implementation,
            hw=hw,
        )
    except (ValueError, ImportError, RuntimeError, OSError) as e:
        fallback_config = model_config.fallback
        if fallback_config.get("enabled") and fallback_config.get("type") == "baseline":
            logger.warning(
                "Failed to load model %s: %s. Using baseline fallback.",
                model_id, e
            )
            from vin_ocr.training import baseline_trainer
            baseline_trainer.run_baseline_training(
                model_name=fallback_config.get("model_name", f"{model_config.model_name}-baseline"),
                train_labels=args.train_labels,
                val_labels=args.val_labels,
                test_labels=args.test_labels,
                output_dir=args.output_dir,
                metrics_file=args.metrics_file,
                model_file=None,
                max_samples=args.max_train_samples,
            )
            return
        raise

    train_dataset = Dataset.from_list(_build_dataset(train_entries))
    eval_dataset = Dataset.from_list(_build_dataset(val_entries)) if val_entries else None

    # Configure training precision based on hardware
    use_fp16 = args.fp16 and (hw.has_cuda or hw.has_mps)
    
    training_args = Seq2SeqTrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_epochs,
        evaluation_strategy="epoch" if eval_dataset else "no",
        save_strategy="epoch",
        logging_steps=10,
        predict_with_generate=False,
        remove_unused_columns=False,
        report_to=[],
        fp16=use_fp16 and hw.has_cuda,  # fp16 for CUDA
        # MPS uses its own mixed precision via torch_dtype
        use_mps_device=hw.has_mps,  # Enable MPS device
    )

    collator = _ImageTextCollator(processor, args.prompt, args.max_target_length)

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
    )

    logger.info("Starting training for %s", model_id)
    trainer.train()
    trainer.save_model(str(output_dir))

    # Use detected hardware for evaluation device
    if hw.has_cuda:
        device = torch.device("cuda")
    elif hw.has_mps:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    model.to(device)
    model.eval()

    metrics: dict[str, float] = {
        "train_samples": float(len(train_entries)),
        "val_samples": float(len(val_entries)),
        "test_samples": float(len(test_entries)),
    }

    if val_entries:
        metrics.update(
            {f"val_{key}": value for key, value in _evaluate(
                model,
                processor,
                val_entries,
                args.prompt,
                args.max_new_tokens,
                args.eval_batch_size,
                device,
            ).items()}
        )

    if test_entries:
        metrics.update(
            {f"test_{key}": value for key, value in _evaluate(
                model,
                processor,
                test_entries,
                args.prompt,
                args.max_new_tokens,
                args.eval_batch_size,
                device,
            ).items()}
        )

    metrics_path = Path(args.metrics_file) if args.metrics_file else output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")

    if args.export_onnx:
        onnx_output = Path(args.onnx_output) if args.onnx_output else output_dir / "export.onnx"
        exported = _export_onnx(output_dir, args.onnx_task, onnx_output)
        logger.info("Exported ONNX to %s", exported)


if __name__ == "__main__":
    main()
