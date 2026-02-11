# Hardware Requirements for VIN OCR Training

This document describes the hardware requirements for training VIN OCR models
on different platforms.

## Supported Platforms

| Platform | Attention | Performance | Notes |
|----------|-----------|-------------|-------|
| **NVIDIA GPU (CUDA)** | flash_attention_2 | ⭐⭐⭐⭐⭐ | **Recommended for production training** |
| **NVIDIA GPU (CUDA)** | sdpa | ⭐⭐⭐⭐ | Good if flash-attn not installed |
| **Apple Silicon (MPS)** | sdpa | ⭐⭐⭐ | Good for development/testing |
| **CPU** | eager | ⭐ | Very slow - for testing only |

## Model-Specific Requirements

### DeepSeek-OCR (`deepseek-ai/DeepSeek-OCR`)

**Optimal Hardware:**
- NVIDIA GPU with CUDA 11.8+
- 16GB+ VRAM (recommended)
- 8GB+ system RAM
- `flash-attn==2.7.3` for best performance

**Minimum Requirements:**
- NVIDIA GPU with 8GB VRAM
- OR Apple Silicon with 16GB unified memory (uses SDPA, slower)
- OR CPU with 16GB RAM (uses eager attention, very slow)

**Dependencies:**
```bash
# For NVIDIA GPU (recommended)
pip install torch==2.6.0 transformers==4.46.3 tokenizers==0.20.3
pip install flash-attn==2.7.3 --no-build-isolation

# For Apple Silicon / CPU (install stub)
pip install torch==2.6.0 transformers==4.46.3 tokenizers==0.20.3
pip install -e ./stubs/flash_attn_stub
```

### LiquidAI (`LiquidAI/LFM2.5-VL-1.6B`)

**Optimal Hardware:**
- NVIDIA GPU with CUDA
- 8GB+ VRAM

**Minimum Requirements:**
- 4GB VRAM (GPU) or 8GB RAM (CPU)
- Works on Apple Silicon with SDPA

**Dependencies:**
```bash
pip install torch>=2.0.0 transformers>=4.40.0
```

### PaddleOCR

**Optimal Hardware:**
- NVIDIA GPU with CUDA
- 4GB+ VRAM

**Minimum Requirements:**
- CPU only supported (slower)
- 8GB+ RAM

**Dependencies:**
```bash
# GPU
pip install paddlepaddle-gpu==3.0.0 paddleocr==3.3.3

# CPU (Apple Silicon / x86)
pip install paddlepaddle==3.0.0 paddleocr==3.3.3
```

## Cloud Deployment Recommendations

### AWS
- **g4dn.xlarge**: T4 GPU, 16GB VRAM - Good for inference
- **g5.xlarge**: A10G GPU, 24GB VRAM - Good for training
- **p3.2xlarge**: V100 GPU, 16GB VRAM - Production training

### GCP
- **n1-standard-8 + T4**: Development/inference
- **a2-highgpu-1g**: A100 GPU - Production training

### Azure
- **Standard_NC6s_v3**: V100 GPU - Production training
- **Standard_NV6ads_A10_v5**: A10 GPU - Good for training

## Local Development Setup

### macOS (Apple Silicon)

1. Install the flash-attn stub:
```bash
pip install -e ./stubs/flash_attn_stub
```

2. The training code will automatically:
   - Detect MPS (Metal Performance Shaders)
   - Use SDPA attention (optimized for Apple Silicon)
   - Move models to MPS device

3. Note: Some models may fall back to baseline training if they have hard dependencies on CUDA.

### macOS / Linux (CPU only)

1. Install the flash-attn stub:
```bash
pip install -e ./stubs/flash_attn_stub
```

2. The training code will:
   - Use eager attention (slowest but most compatible)
   - Run on CPU

3. **Warning**: Training on CPU is very slow. Use for testing only.

## Attention Implementation Details

### flash_attention_2
- **Requires**: NVIDIA GPU + `flash-attn` package
- **Performance**: 2-4x faster than standard attention
- **Memory**: 5-20x lower memory usage
- **Compatibility**: CUDA only

### sdpa (Scaled Dot Product Attention)
- **Requires**: PyTorch 2.0+
- **Performance**: 1.5-2x faster than eager
- **Memory**: Better than eager
- **Compatibility**: CUDA, MPS, CPU

### eager
- **Requires**: Nothing special
- **Performance**: Baseline
- **Memory**: High
- **Compatibility**: All platforms

## Troubleshooting

### "flash_attn not available"
Install the stub for non-CUDA systems:
```bash
pip install -e ./stubs/flash_attn_stub
```

### "MPS out of memory"
Reduce batch size or use CPU:
```bash
python -m vin_ocr.training.hf_finetune --batch-size 1
```

### OpenMP library conflict on macOS
This is automatically handled by using `psutil` for RAM detection.
If issues persist, set:
```bash
export KMP_DUPLICATE_LIB_OK=TRUE
```

## Hardware Auto-Detection

The training code automatically detects your hardware and selects:
1. **Platform**: CUDA > MPS > CPU
2. **Attention**: flash_attention_2 > sdpa > eager
3. **Device**: Automatically moves model to best available device

You can override attention implementation in the model config:
```yaml
training:
  attn_implementation: "sdpa"  # Force SDPA even on CUDA
```
