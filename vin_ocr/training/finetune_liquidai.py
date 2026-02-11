"""LiquidAI OCR fine-tuning wrapper using Unsloth.

LiquidAI models use a custom architecture (`lfm2_vl`) that requires Unsloth's
FastVisionModel for loading (not standard transformers).

See: https://github.com/Liquid4All/cookbook/blob/main/finetuning/notebooks/sft_for_vision_language_model.ipynb

REQUIREMENTS:
    pip install unsloth
    # Or for Colab: pip install --no-deps bitsandbytes accelerate xformers peft trl triton unsloth_zoo

HARDWARE NOTE:
    LiquidAI LFM2.5-VL-1.6B (~1.6B params) works on:
    - NVIDIA GPU (CUDA) - best performance with Unsloth
    - Apple Silicon (MPS) - may need adjustments
    - CPU - slow but works
"""

import logging
import argparse
import torch
from pathlib import Path

logger = logging.getLogger(__name__)


def check_unsloth_available() -> bool:
    """Check if Unsloth is installed."""
    try:
        from unsloth import FastVisionModel
        return True
    except ImportError:
        return False


def main() -> None:
    """Run LiquidAI fine-tuning using Unsloth's FastVisionModel.
    
    Unsloth requires CUDA - falls back to TrOCR on Apple Silicon/CPU.
    """
    logging.basicConfig(level=logging.INFO)
    
    has_cuda = torch.cuda.is_available()
    has_mps = torch.backends.mps.is_available()
    
    if has_cuda:
        logger.info("CUDA detected")
    elif has_mps:
        logger.info("Apple Silicon (MPS) detected")
    else:
        logger.info("CPU detected")
    
    # Unsloth requires CUDA - it uses triton which is NVIDIA-only
    if not has_cuda:
        logger.warning(
            "LiquidAI models require Unsloth which only works on NVIDIA GPUs (CUDA). "
            "Detected: %s. Falling back to TrOCR.",
            "Apple Silicon (MPS)" if has_mps else "CPU"
        )
        from vin_ocr.training import hf_finetune
        hf_finetune.main(
            default_model_id="microsoft/trocr-base-printed",
            default_model_config="trocr",
        )
        return
    
    if not check_unsloth_available():
        logger.warning(
            "Unsloth not installed. LiquidAI models require Unsloth. "
            "Install with: pip install unsloth triton "
            "Falling back to TrOCR."
        )
        from vin_ocr.training import hf_finetune
        hf_finetune.main(
            default_model_id="microsoft/trocr-base-printed",
            default_model_config="trocr",
        )
        return
    
    # Parse args
    parser = argparse.ArgumentParser(description="LiquidAI VLM Fine-tuning with Unsloth")
    parser.add_argument("--train-labels", required=True, help="Training labels file")
    parser.add_argument("--val-labels", help="Validation labels file")
    parser.add_argument("--test-labels", help="Test labels file")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--metrics-file", help="Metrics output file")
    parser.add_argument("--model-id", default="LiquidAI/LFM2.5-VL-1.6B", help="Model ID")
    parser.add_argument("--max-seq-length", type=int, default=2048, help="Max sequence length")
    parser.add_argument("--load-in-4bit", action="store_true", help="Load in 4-bit quantization")
    parser.add_argument("--num-epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--lora-r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=16, help="LoRA alpha")
    # Ignore unknown args for compatibility with zenml pipeline
    args, _ = parser.parse_known_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("Loading LiquidAI model with Unsloth FastVisionModel...")
    
    from unsloth import FastVisionModel
    
    # Load model with Unsloth
    model, tokenizer = FastVisionModel.from_pretrained(
        model_name=args.model_id,
        max_seq_length=args.max_seq_length,
        load_in_4bit=args.load_in_4bit,
    )
    
    # Apply LoRA
    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=False,  # Keep vision frozen for OCR
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.0,
        bias="none",
        use_rslora=False,
        loftq_config=None,
    )
    
    logger.info("Model loaded successfully with LoRA adapters")
    
    # Read training data
    def read_entries(label_file: str) -> list[tuple[str, str]]:
        entries = []
        with open(label_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split("\t")
                if len(parts) >= 2:
                    entries.append((parts[0], parts[1]))
        return entries
    
    train_entries = read_entries(args.train_labels)
    val_entries = read_entries(args.val_labels) if args.val_labels else []
    
    logger.info(f"Training samples: {len(train_entries)}, Validation samples: {len(val_entries)}")
    
    # Prepare dataset in chat format for VLM
    from PIL import Image
    from datasets import Dataset
    
    def prepare_dataset(entries: list[tuple[str, str]]) -> list[dict]:
        data = []
        for img_path, label in entries:
            if not Path(img_path).exists():
                continue
            data.append({
                "image": img_path,
                "conversations": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {"type": "text", "text": "Read the VIN number from this image."}
                        ]
                    },
                    {"role": "assistant", "content": label}
                ]
            })
        return data
    
    train_data = prepare_dataset(train_entries)
    
    if not train_data:
        logger.error("No valid training data found")
        return
    
    # Use TRL for training
    from trl import SFTTrainer, SFTConfig
    from unsloth import is_bfloat16_supported
    
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=Dataset.from_list(train_data),
        eval_dataset=Dataset.from_list(prepare_dataset(val_entries)) if val_entries else None,
        args=SFTConfig(
            output_dir=str(output_dir),
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=4,
            warmup_steps=5,
            num_train_epochs=args.num_epochs,
            learning_rate=args.learning_rate,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=10,
            save_strategy="epoch",
            evaluation_strategy="epoch" if val_entries else "no",
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=42,
            report_to=[],
            # VLM specific
            dataset_text_field="",
            dataset_kwargs={"skip_prepare_dataset": True},
            remove_unused_columns=False,
        ),
    )
    
    logger.info("Starting training...")
    trainer.train()
    
    # Save model
    model.save_pretrained(str(output_dir / "lora_model"))
    tokenizer.save_pretrained(str(output_dir / "lora_model"))
    logger.info(f"LoRA model saved to {output_dir / 'lora_model'}")
    
    # Save metrics
    import json
    metrics = {
        "model_id": args.model_id,
        "train_samples": len(train_entries),
        "val_samples": len(val_entries),
        "epochs": args.num_epochs,
        "lora_r": args.lora_r,
        "framework": "unsloth",
    }
    
    metrics_file = args.metrics_file or str(output_dir / "metrics.json")
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Metrics saved to {metrics_file}")


if __name__ == "__main__":
    main()
