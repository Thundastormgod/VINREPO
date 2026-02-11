# Flash Attention Stub Package
# ============================
# This provides a CPU/MPS-compatible stub for flash_attn on non-CUDA systems.
# 
# Install: pip install -e ./stubs/flash_attn_stub
#
# This stub allows models that import flash_attn to load without errors,
# but will raise NotImplementedError if flash attention functions are actually called.
# The HuggingFace transformers library will automatically fall back to SDPA or eager
# attention when attn_implementation is set appropriately.
