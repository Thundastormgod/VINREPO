"""DeepSeek OCR fine-tuning wrapper (config-based).

All model-specific configuration is now in:
    vin_ocr/configs/models/deepseek.yaml

This includes:
    - Model ID
    - Environment workarounds (OMP fix, flash_attn mock)
    - Fallback configuration
    - Training defaults

HARDWARE NOTE:
    DeepSeek-OCR (~7B params) requires NVIDIA GPU with CUDA and flash_attention_2.
    On Apple Silicon (MPS), we use microsoft/trocr-base-printed as a smaller
    alternative that runs efficiently on MPS backend.
"""

import logging
import argparse
import torch

logger = logging.getLogger(__name__)

# Alternative model for Apple Silicon (TrOCR is ~300M params vs DeepSeek's ~7B)
APPLE_SILICON_ALTERNATIVE = "microsoft/trocr-base-printed"


def main() -> None:
    """Run DeepSeek-OCR fine-tuning using config-based approach.
    
    On Apple Silicon, uses TrOCR as a smaller alternative via MPS backend.
    On CPU, uses baseline training.
    """
    logging.basicConfig(level=logging.INFO)
    
    has_cuda = torch.cuda.is_available()
    has_mps = torch.backends.mps.is_available()
    
    if has_cuda:
        # CUDA available - use full DeepSeek-OCR
        logger.info("CUDA detected - using DeepSeek-OCR")
        from vin_ocr.training import hf_finetune
        hf_finetune.main(default_model_config="deepseek")
    
    elif has_mps:
        # Apple Silicon - use TrOCR as smaller alternative on MPS
        logger.info(
            "Apple Silicon (MPS) detected. DeepSeek-OCR (~7B params) is too large. "
            "Using %s as alternative.", APPLE_SILICON_ALTERNATIVE
        )
        from vin_ocr.training import hf_finetune
        # Override to use TrOCR on MPS
        hf_finetune.main(
            default_model_id=APPLE_SILICON_ALTERNATIVE,
            default_model_config=None,  # Don't use deepseek config
        )
    
    else:
        # CPU only - use baseline (too slow for real training)
        logger.warning(
            "CPU only detected. Using baseline training (no model loading)."
        )
        
        # Parse minimal args for baseline training
        parser = argparse.ArgumentParser()
        parser.add_argument("--train-labels", required=True)
        parser.add_argument("--val-labels")
        parser.add_argument("--test-labels")
        parser.add_argument("--output-dir", required=True)
        parser.add_argument("--metrics-file")
        args, _ = parser.parse_known_args()
        
        from vin_ocr.training import baseline_trainer
        baseline_trainer.run_baseline_training(
            model_name="deepseek-ocr-baseline",
            train_labels=args.train_labels,
            val_labels=args.val_labels,
            test_labels=args.test_labels,
            output_dir=args.output_dir,
            metrics_file=args.metrics_file,
            model_file=None,
            max_samples=0,
        )


if __name__ == "__main__":
    main()
