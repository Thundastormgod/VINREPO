# VIN OCR Training (PaddleOCR, DeepSeek, LiquidAI)

This subproject plugs a VIN OCR workflow into the ZenML experiment pipeline.
It prepares VIN datasets, then runs external training commands for PaddleOCR,
DeepSeek OCR, and LiquidAI LFM2.5-VL-1.6B.

## Quick Start

1) Install dependencies

```bash
pip install -r vin_ocr/requirements-cpu.txt
```

For GPU training (CUDA builds), use:

```bash
pip install -r vin_ocr/requirements-gpu.txt
```

If you prefer a single combined file (CPU defaults + GPU notes), use:

```bash
pip install -r vin_ocr/requirements.txt
```

Or use the Make target:

```bash
make vin-ocr-install
```

2) Export DagsHub credentials (if required)

```bash
export DAGSHUB_USER="your-username"
export DAGSHUB_TOKEN="your-token"
```

3) Fetch data from DagsHub

```bash
python vin_ocr/tools/dagshub_fetch.py \
  --repo-url https://dagshub.com/Thundastormgod/jlr-vin-ocr \
  --repo-data-path data \
  --output-dir vin_ocr_data/raw \
  --clone-dir vin_ocr_data/.dagshub_repo \
  --clean
```

For public repos without auth, add:

```bash
python vin_ocr/tools/dagshub_fetch.py \
  --repo-url https://dagshub.com/Thundastormgod/jlr-vin-ocr \
  --repo-data-path data \
  --output-dir vin_ocr_data/raw \
  --public
```

To download only train images and sample 100 into the raw folder for quick testing:

```bash
python vin_ocr/tools/dagshub_fetch.py \
  --repo-url https://dagshub.com/Thundastormgod/jlr-vin-ocr \
  --repo-data-path data \
  --output-dir vin_ocr_data/raw \
  --sample-from train/images \
  --sample-count 100 \
  --public
```

Or use the Make target:

```bash
make vin-ocr-fetch
```

If your data lives in the DagsHub bucket (not tracked by DVC), use the bucket download:

```bash
python vin_ocr/tools/dagshub_fetch.py \
  --dagshub-bucket Thundastormgod/jlr-vin-ocr \
  --dagshub-bucket-path data_fixed/train/images \
  --output-dir vin_ocr_data/raw
```

To also track the downloaded data with DVC in this repo:

```bash
python vin_ocr/tools/dagshub_fetch.py \
  --dagshub-bucket Thundastormgod/jlr-vin-ocr \
  --dagshub-bucket-path data_fixed/train/images \
  --output-dir vin_ocr_data/raw \
  --dvc-track
```

4) Configure training commands

Update the command field in these files with your training commands:
- vin_ocr/configs/paddleocr_finetune.yaml
- vin_ocr/configs/deepseek_finetune.yaml
- vin_ocr/configs/liquidai_finetune.yaml

The default configs point at starter training scripts that run a baseline
trainer so the pipeline executes end-to-end without external repositories.
Swap them with your actual PaddleOCR/DeepSeek/LiquidAI trainers when ready.

Each command can interpolate:
- {dataset_dir}
- {train_labels}
- {val_labels}
- {test_labels}
- {output_dir}

5) Run a single-model pipeline (clean runs)

```bash
python run_vin_paddleocr.py --fetch-from-dagshub
python run_vin_deepseek.py --fetch-from-dagshub
python run_vin_liquidai.py --fetch-from-dagshub
```

For remote-only data (no local download), supply a remote URI and set data mode:

```bash
python run_vin_paddleocr.py \
  --data-mode remote \
  --remote-data-uri s3://bucket/path/to/images
```

To run all models in one go, use:

```bash
python run_vin_ocr.py --fetch-from-dagshub
```

## Notes

- The pipeline will create a train/val/test split if your raw data is a flat
  image folder with VINs embedded in the filenames.
- If your data is already in train/val/test format, place it under
  vin_ocr_data/raw with train/images and train_labels.txt, etc.
- LiquidAI model reference: https://huggingface.co/LiquidAI/LFM2.5-VL-1.6B
