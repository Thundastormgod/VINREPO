"""Flash Attention Stub for non-CUDA systems.

This module provides stub implementations of flash_attn functions that raise
NotImplementedError when called. It allows models that import flash_attn to
load on CPU/MPS systems without import errors.

For actual training on non-CUDA systems, use:
    - attn_implementation="sdpa" (PyTorch Scaled Dot Product Attention)
    - attn_implementation="eager" (Standard attention, slowest but most compatible)

For CUDA systems with flash_attn installed:
    - attn_implementation="flash_attention_2" (Fastest, requires NVIDIA GPU)
"""

__version__ = "0.1.0"

_STUB_ERROR_MSG = (
    "flash_attn is not available on this platform. "
    "This stub allows model loading but not execution of flash attention. "
    "Use attn_implementation='sdpa' or 'eager' in model config, "
    "or run on an NVIDIA GPU with flash-attn installed."
)


class FlashAttnNotAvailableError(NotImplementedError):
    """Raised when flash_attn functions are called on non-CUDA systems."""
    pass


def flash_attn_func(
    q, k, v,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
):
    """Stub for flash_attn_func - raises NotImplementedError."""
    raise FlashAttnNotAvailableError(_STUB_ERROR_MSG)


def flash_attn_varlen_func(
    q, k, v,
    cu_seqlens_q, cu_seqlens_k,
    max_seqlen_q, max_seqlen_k,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
):
    """Stub for flash_attn_varlen_func - raises NotImplementedError."""
    raise FlashAttnNotAvailableError(_STUB_ERROR_MSG)


def flash_attn_qkvpacked_func(
    qkv,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
):
    """Stub for flash_attn_qkvpacked_func - raises NotImplementedError."""
    raise FlashAttnNotAvailableError(_STUB_ERROR_MSG)


def flash_attn_kvpacked_func(
    q, kv,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
):
    """Stub for flash_attn_kvpacked_func - raises NotImplementedError."""
    raise FlashAttnNotAvailableError(_STUB_ERROR_MSG)


def flash_attn_with_kvcache(
    q, k_cache, v_cache,
    k=None, v=None,
    rotary_cos=None, rotary_sin=None,
    cache_seqlens=None,
    cache_batch_idx=None,
    block_table=None,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),
    rotary_interleaved=True,
    alibi_slopes=None,
    num_splits=0,
):
    """Stub for flash_attn_with_kvcache - raises NotImplementedError."""
    raise FlashAttnNotAvailableError(_STUB_ERROR_MSG)
