"""
MXFP4 and MXFP8 quantization and dequantization utilities.

Provides:
  - dynamic_mxfp4_quant : fp16/bf16 tensor -> packed e2m1 uint8 + e8m0 scales
  - dynamic_mxfp8_quant : fp16/bf16 tensor -> float8_e4m3fn + e8m0 scales
  - mxfp4_to_f32        : packed e2m1 uint8 -> fp32 (for reference / testing)
  - mxfp8_to_f32        : float8_e4m3fn + e8m0 scales -> fp32
  - e8m0_to_f32          : e8m0 uint8 scales -> fp32 multipliers
"""

import torch
import triton

from fast_loraq.kernels import _mxfp4_quant_kernel, _mxfp8_quant_kernel


# -- quantize ---------------------------------------------------------------

def dynamic_mxfp4_quant(
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize a 2-D tensor to MXFP4 e2m1 with e8m0 block scales.

    Every group of 32 contiguous elements along the last axis shares
    one e8m0 scale (a pure power-of-two exponent).

    Parameters
    ----------
    x : (M, N) tensor, fp16 / bf16 / fp32, on CUDA.
        N must be divisible by 32.

    Returns
    -------
    x_fp4 : (M, N // 2) uint8
        Two e2m1 nibbles packed per byte (low nibble = even index).
    scales : (M, N // 32) uint8
        One e8m0 exponent per 32-element group.
    """
    assert x.ndim == 2, f"Expected 2-D tensor, got {x.ndim}-D"
    M, N = x.shape
    assert N % 32 == 0, f"N={N} must be divisible by 32"

    QUANT_GROUP = 32
    BLOCK_SIZE = 128  # rows per program

    x_fp4 = torch.empty((M, N // 2), dtype=torch.uint8, device=x.device)
    n_scale_cols = triton.cdiv(N, QUANT_GROUP)
    scales = torch.empty((M, n_scale_cols), dtype=torch.uint8, device=x.device)

    grid = (triton.cdiv(M, BLOCK_SIZE), n_scale_cols)
    _mxfp4_quant_kernel[grid](
        x,
        x_fp4,
        scales,
        x.stride(0), x.stride(1),
        x_fp4.stride(0), x_fp4.stride(1),
        scales.stride(0), scales.stride(1),
        M=M, N=N,
        BLOCK_SIZE=BLOCK_SIZE,
        QUANT_GROUP=QUANT_GROUP,
    )
    return x_fp4, scales



def dynamic_mxfp8_quant(
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize a 2-D tensor to MXFP8 e4m3 with e8m0 block scales.

    Every group of 32 contiguous elements along the last axis shares
    one e8m0 scale (a pure power-of-two exponent).

    Parameters
    ----------
    x : (M, N) tensor, fp16 / bf16 / fp32, on CUDA.
        N must be divisible by 32.

    Returns
    -------
    x_fp8 : (M, N) float8_e4m3fn
        One float8_e4m3fn value per byte (no packing needed).
    scales : (M, N // 32) uint8
        One e8m0 exponent per 32-element group.
    """
    assert x.ndim == 2, f"Expected 2-D tensor, got {x.ndim}-D"
    M, N = x.shape
    assert N % 32 == 0, f"N={N} must be divisible by 32"

    QUANT_GROUP = 32
    BLOCK_SIZE = 128  # rows per program

    x_fp8 = torch.empty((M, N), dtype=torch.uint8, device=x.device)
    n_scale_cols = triton.cdiv(N, QUANT_GROUP)
    scales = torch.empty((M, n_scale_cols), dtype=torch.uint8, device=x.device)

    grid = (triton.cdiv(M, BLOCK_SIZE), n_scale_cols)
    _mxfp8_quant_kernel[grid](
        x,
        x_fp8,
        scales,
        x.stride(0), x.stride(1),
        x_fp8.stride(0), x_fp8.stride(1),
        scales.stride(0), scales.stride(1),
        M=M, N=N,
        BLOCK_SIZE=BLOCK_SIZE,
        QUANT_GROUP=QUANT_GROUP,
    )
    # Reinterpret output bytes as float8_e4m3fn
    x_fp8 = x_fp8.view(torch.float8_e4m3fn)
    return x_fp8, scales


# -- dequantize (CPU / reference, no Triton needed) -------------------------

# Lookup table: 4-bit index -> fp32 value for e2m1
_MXFP4_LUT = [
    0.0,   # 0x0
    0.5,   # 0x1
    1.0,   # 0x2
    1.5,   # 0x3
    2.0,   # 0x4
    3.0,   # 0x5
    4.0,   # 0x6
    6.0,   # 0x7
    -0.0,  # 0x8
    -0.5,  # 0x9
    -1.0,  # 0xA
    -1.5,  # 0xB
    -2.0,  # 0xC
    -3.0,  # 0xD
    -4.0,  # 0xE
    -6.0,  # 0xF
]


def mxfp4_to_f32(x: torch.Tensor) -> torch.Tensor:
    """
    Unpack a packed e2m1 uint8 tensor to fp32.

    Parameters
    ----------
    x : (..., N_packed) uint8
        Each byte holds two e2m1 nibbles.

    Returns
    -------
    (..., N_packed * 2) fp32
    """
    if x.dtype == getattr(torch, "float4_e2m1fn_x2", None):
        x = x.view(torch.uint8)

    # Duplicate each byte, mask low / high nibble
    x2 = x.repeat_interleave(2, dim=-1)
    x2[..., ::2] = x2[..., ::2] & 0xF
    x2[..., 1::2] = x2[..., 1::2] >> 4

    lut = torch.tensor(_MXFP4_LUT, dtype=torch.float32, device=x.device)
    return lut[x2.long()]


def mxfp8_to_f32(
    x_fp8: torch.Tensor,
    scales: torch.Tensor,
) -> torch.Tensor:
    """
    Dequantize MXFP8 e4m3 tensor with e8m0 block scales to fp32.

    Parameters
    ----------
    x_fp8 : (M, N) float8_e4m3fn or uint8 (viewed as e4m3 bytes)
    scales : (M, N // 32) uint8  -- e8m0 block scales

    Returns
    -------
    (M, N) fp32
    """
    if x_fp8.dtype == torch.uint8:
        x_fp8 = x_fp8.view(torch.float8_e4m3fn)
    x_f32 = x_fp8.to(torch.float32)
    s_f32 = e8m0_to_f32(scales)                        # (M, N//32)
    s_f32 = s_f32.repeat_interleave(32, dim=-1)         # (M, N)
    return x_f32 * s_f32


def e8m0_to_f32(scale_e8m0: torch.Tensor) -> torch.Tensor:
    """
    Convert e8m0 scales to fp32 multipliers.

    E8M0 stores only the exponent (8 bits, biased by 127).
    The value is ``2 ** (exponent - 127)``.

    Special cases:
        0x00 -> 2**(-126)  (minimum normal)
        0xFF -> NaN
    """
    raw = scale_e8m0.view(torch.uint8).to(torch.int32)

    result = raw << 23  # place exponent in fp32 position

    # Handle special values
    result[scale_e8m0.view(torch.uint8) == 0] = 0x00400000    # 2^-126
    result[scale_e8m0.view(torch.uint8) == 0xFF] = 0x7F800001  # NaN

    return result.view(torch.float32)
