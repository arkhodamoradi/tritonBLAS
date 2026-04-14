"""
MXFP4 / MXFP8 Quantization
===========================
Triton kernels and Python wrappers for converting FP32 tensors to
MXFP8 (E4M3 / E5M2) and MXFP4 (E2M1) with either

  * RTN  – round-to-nearest-even  (hardware packed instruction)
  * SR   – stochastic rounding     (hardware SR instruction)

All public functions follow a common signature::

    fp_quantized, scales = quantize_<fmt>_<rounding>(x, group_size=32, ...)

``fp_quantized`` is a uint8 tensor (packed for FP4, byte-per-element for FP8).
``scales`` is a uint8 tensor of shape ``[M, K // group_size]`` holding E8M0
exponents.

Importable example::

    from mxfp4fp8_quantization import quantize_mxfp8e4_rtne, quantize_mxfp4_rtne
    fp8, scales = quantize_mxfp8e4_rtne(x)
    fp4, scales = quantize_mxfp4_rtne(x)
"""

import torch
import triton
import triton.language as tl

__all__ = [
    # helpers
    "fp4_e2m1_to_fp32",
    # FP8 E4M3 (e4m3fn)
    "quantize_mxfp8e4_rtne",
    "quantize_mxfp8e4_sr",
    # FP8 E5M2
    "quantize_mxfp8e5_rtne",
    "quantize_mxfp8e5_sr",
    # FP4 E2M1
    "quantize_mxfp4_rtne",
]

# ---------------------------------------------------------------------------
# Shared helper: compute E8M0 scale exponent from absmax of a group
# ---------------------------------------------------------------------------

@triton.jit
def _get_exponent(x, offset):
    """Return E8M0 exponent (uint32) for each row of x [GROUPS, GROUP_SIZE]."""
    absmax = tl.max(tl.abs(x), axis=1)
    absmax_bits = absmax.to(tl.int32, bitcast=True)
    f32_exp = (absmax_bits >> 23) & 0xFF
    exp = f32_exp - offset
    exp = tl.maximum(exp, 0)
    exp = tl.minimum(exp, 255)
    return exp


# ---------------------------------------------------------------------------
# MXFP8 E4M3 – RTNE (round-to-nearest-even) via hardware packed instruction
# ---------------------------------------------------------------------------

@triton.jit
def _f32_to_mxfp8e4_rtne_kernel(
    x_ptr, out_ptr, scale_ptr,
    M, K,
    stride_xm, stride_xk,
    stride_outm, stride_outk,
    stride_sm, stride_sg,
    GROUP_SIZE: tl.constexpr,
    GROUPS_PER_BLOCK: tl.constexpr,
):
    """
    Convert FP32 → MXFP8 E4M3 with RTNE using v_cvt_scalef32_pk_fp8_f32.
    Each program handles GROUPS_PER_BLOCK groups along the K dimension.
    """
    pid_m = tl.program_id(0)
    pid_g = tl.program_id(1)

    BLOCK_SIZE: tl.constexpr = GROUP_SIZE * GROUPS_PER_BLOCK

    block_start = pid_g * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    x = tl.load(x_ptr + pid_m * stride_xm + offsets * stride_xk,
                 mask=offsets < K, other=0.0)

    x_grouped = tl.reshape(x, (GROUPS_PER_BLOCK, GROUP_SIZE))
    scale_exp = _get_exponent(x_grouped, 8)

    group_indices = tl.arange(0, GROUPS_PER_BLOCK)
    g_abs = pid_g * GROUPS_PER_BLOCK + group_indices
    tl.store(scale_ptr + pid_m * stride_sm + g_abs * stride_sg,
             scale_exp.to(tl.uint8),
             mask=g_abs < (K // GROUP_SIZE))

    # Broadcast scale to pairs of elements
    scale_exp_broad = tl.broadcast_to(
        tl.reshape(scale_exp, (GROUPS_PER_BLOCK, 1)),
        (GROUPS_PER_BLOCK, GROUP_SIZE // 2),
    )
    scale_f32 = (tl.reshape(scale_exp_broad, (BLOCK_SIZE // 2,)).to(tl.uint32) << 23)

    x_pairs = tl.reshape(x, (BLOCK_SIZE // 2, 2))
    x_even, x_odd = tl.split(x_pairs)

    fp8_packed = tl.inline_asm_elementwise(
        "v_cvt_scalef32_pk_fp8_f32 $0, $1, $2, $3",
        "=v,v,v,v",
        args=[x_even, x_odd, scale_f32],
        dtype=tl.uint16, is_pure=True, pack=1,
    )

    fp8_0 = (fp8_packed & 0xFF).to(tl.uint8)
    fp8_1 = ((fp8_packed >> 8) & 0xFF).to(tl.uint8)
    # clamp NaN (0x7F / 0xFF) → max finite (0x7E / 0xFE)
    fp8_0 = tl.where((fp8_0 & 0x7F) == 0x7F, fp8_0 - 1, fp8_0)
    fp8_1 = tl.where((fp8_1 & 0x7F) == 0x7F, fp8_1 - 1, fp8_1)

    fp8_interleaved = tl.interleave(fp8_0, fp8_1)
    out_offsets = block_start + tl.arange(0, BLOCK_SIZE)
    tl.store(out_ptr + pid_m * stride_outm + out_offsets * stride_outk,
             fp8_interleaved, mask=out_offsets < K)


# ---------------------------------------------------------------------------
# MXFP8 E4M3 – SR (stochastic rounding) via v_cvt_scalef32_sr_fp8_f32
# ---------------------------------------------------------------------------

@triton.jit
def _f32_to_mxfp8e4_sr_kernel(
    x_ptr, out_ptr, scale_ptr,
    M, K,
    stride_xm, stride_xk,
    stride_outm, stride_outk,
    stride_sm, stride_sg,
    GROUP_SIZE: tl.constexpr,
    GROUPS_PER_BLOCK: tl.constexpr,
):
    """
    Convert FP32 → MXFP8 E4M3 with stochastic rounding via
    v_cvt_scalef32_sr_fp8_f32.
    """
    pid_m = tl.program_id(0)
    pid_g = tl.program_id(1)

    BLOCK_SIZE: tl.constexpr = GROUP_SIZE * GROUPS_PER_BLOCK

    block_start = pid_g * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    x = tl.load(x_ptr + pid_m * stride_xm + offsets * stride_xk,
                 mask=offsets < K, other=0.0)

    x_grouped = tl.reshape(x, (GROUPS_PER_BLOCK, GROUP_SIZE))
    scale_exp = _get_exponent(x_grouped, 8)

    group_indices = tl.arange(0, GROUPS_PER_BLOCK)
    g_abs = pid_g * GROUPS_PER_BLOCK + group_indices
    tl.store(scale_ptr + pid_m * stride_sm + g_abs * stride_sg,
             scale_exp.to(tl.uint8),
             mask=g_abs < (K // GROUP_SIZE))

    scale_exp_broad = tl.broadcast_to(
        tl.reshape(scale_exp, (GROUPS_PER_BLOCK, 1)),
        (GROUPS_PER_BLOCK, GROUP_SIZE),
    )
    scale_f32 = (tl.reshape(scale_exp_broad, (BLOCK_SIZE,)).to(tl.uint32) << 23)

    sr_seed = 0.0
    fp8 = tl.inline_asm_elementwise(
        "v_cvt_scalef32_sr_fp8_f32 $0, $1, $2, $3",
        "=v,v,v,v",
        args=[x, sr_seed, scale_f32],
        dtype=tl.uint16, is_pure=True, pack=1,
    )
    fp8 = fp8.to(tl.uint8)
    fp8 = tl.where((fp8 & 0x7F) == 0x7F, fp8 - 1, fp8)

    tl.store(out_ptr + pid_m * stride_outm + offsets * stride_outk,
             fp8, mask=offsets < K)


# ---------------------------------------------------------------------------
# MXFP8 E5M2 – RTNE via v_cvt_scalef32_pk_bf8_f32
# ---------------------------------------------------------------------------

@triton.jit
def _f32_to_mxfp8e5_rtne_kernel(
    x_ptr, out_ptr, scale_ptr,
    M, K,
    stride_xm, stride_xk,
    stride_outm, stride_outk,
    stride_sm, stride_sg,
    GROUP_SIZE: tl.constexpr,
    GROUPS_PER_BLOCK: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_g = tl.program_id(1)

    BLOCK_SIZE: tl.constexpr = GROUP_SIZE * GROUPS_PER_BLOCK

    block_start = pid_g * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    x = tl.load(x_ptr + pid_m * stride_xm + offsets * stride_xk,
                 mask=offsets < K, other=0.0)

    x_grouped = tl.reshape(x, (GROUPS_PER_BLOCK, GROUP_SIZE))
    scale_exp = _get_exponent(x_grouped, 15)

    group_indices = tl.arange(0, GROUPS_PER_BLOCK)
    g_abs = pid_g * GROUPS_PER_BLOCK + group_indices
    tl.store(scale_ptr + pid_m * stride_sm + g_abs * stride_sg,
             scale_exp.to(tl.uint8),
             mask=g_abs < (K // GROUP_SIZE))

    scale_exp_broad = tl.broadcast_to(
        tl.reshape(scale_exp, (GROUPS_PER_BLOCK, 1)),
        (GROUPS_PER_BLOCK, GROUP_SIZE // 2),
    )
    scale_f32 = (tl.reshape(scale_exp_broad, (BLOCK_SIZE // 2,)).to(tl.uint32) << 23)

    x_pairs = tl.reshape(x, (BLOCK_SIZE // 2, 2))
    x_even, x_odd = tl.split(x_pairs)

    fp8_packed = tl.inline_asm_elementwise(
        "v_cvt_scalef32_pk_bf8_f32 $0, $1, $2, $3",
        "=v,v,v,v",
        args=[x_even, x_odd, scale_f32],
        dtype=tl.uint16, is_pure=True, pack=1,
    )

    fp8_0 = (fp8_packed & 0xFF).to(tl.uint8)
    fp8_1 = ((fp8_packed >> 8) & 0xFF).to(tl.uint8)
    # clamp Inf/NaN to max finite
    fp8_0 = tl.where((fp8_0 >= 0x7C) & (fp8_0 < 0x80), 0x7B, fp8_0)
    fp8_0 = tl.where(fp8_0 >= 0xFC, 0xFB, fp8_0)
    fp8_1 = tl.where((fp8_1 >= 0x7C) & (fp8_1 < 0x80), 0x7B, fp8_1)
    fp8_1 = tl.where(fp8_1 >= 0xFC, 0xFB, fp8_1)

    fp8_interleaved = tl.interleave(fp8_0, fp8_1)
    out_offsets = block_start + tl.arange(0, BLOCK_SIZE)
    tl.store(out_ptr + pid_m * stride_outm + out_offsets * stride_outk,
             fp8_interleaved, mask=out_offsets < K)


# ---------------------------------------------------------------------------
# MXFP8 E5M2 – SR via v_cvt_scalef32_sr_bf8_f32
# ---------------------------------------------------------------------------

@triton.jit
def _f32_to_mxfp8e5_sr_kernel(
    x_ptr, out_ptr, scale_ptr,
    M, K,
    stride_xm, stride_xk,
    stride_outm, stride_outk,
    stride_sm, stride_sg,
    GROUP_SIZE: tl.constexpr,
    GROUPS_PER_BLOCK: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_g = tl.program_id(1)

    BLOCK_SIZE: tl.constexpr = GROUP_SIZE * GROUPS_PER_BLOCK

    block_start = pid_g * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    x = tl.load(x_ptr + pid_m * stride_xm + offsets * stride_xk,
                 mask=offsets < K, other=0.0)

    x_grouped = tl.reshape(x, (GROUPS_PER_BLOCK, GROUP_SIZE))
    scale_exp = _get_exponent(x_grouped, 15)

    group_indices = tl.arange(0, GROUPS_PER_BLOCK)
    g_abs = pid_g * GROUPS_PER_BLOCK + group_indices
    tl.store(scale_ptr + pid_m * stride_sm + g_abs * stride_sg,
             scale_exp.to(tl.uint8),
             mask=g_abs < (K // GROUP_SIZE))

    scale_exp_broad = tl.broadcast_to(
        tl.reshape(scale_exp, (GROUPS_PER_BLOCK, 1)),
        (GROUPS_PER_BLOCK, GROUP_SIZE),
    )
    scale_f32 = (tl.reshape(scale_exp_broad, (BLOCK_SIZE,)).to(tl.uint32) << 23)

    sr_seed = 0.0
    fp8 = tl.inline_asm_elementwise(
        "v_cvt_scalef32_sr_bf8_f32 $0, $1, $2, $3",
        "=v,v,v,v",
        args=[x, sr_seed, scale_f32],
        dtype=tl.uint16, is_pure=True, pack=1,
    )
    fp8 = fp8.to(tl.uint8)
    # clamp Inf/NaN
    fp8 = tl.where((fp8 >= 0x7C) & (fp8 < 0x80), 0x7B, fp8)
    fp8 = tl.where(fp8 >= 0xFC, 0xFB, fp8)

    tl.store(out_ptr + pid_m * stride_outm + offsets * stride_outk,
             fp8, mask=offsets < K)


# ---------------------------------------------------------------------------
# MXFP4 E2M1 – RTNE via v_cvt_scalef32_pk_fp4_f32
# ---------------------------------------------------------------------------

@triton.jit
def _f32_to_mxfp4_rtne_kernel(
    x_ptr, out_ptr, scale_ptr,
    M, K,
    stride_xm, stride_xk,
    stride_outm, stride_outk,
    stride_sm, stride_sg,
    GROUP_SIZE: tl.constexpr,
    GROUPS_PER_BLOCK: tl.constexpr,
):
    """
    Convert FP32 → MXFP4 E2M1 with RTNE using v_cvt_scalef32_pk_fp4_f32.
    Each output byte holds two FP4 nibbles (low=even, high=odd).
    """
    pid_m = tl.program_id(0)
    pid_g = tl.program_id(1)

    BLOCK_SIZE: tl.constexpr = GROUP_SIZE * GROUPS_PER_BLOCK

    block_start = pid_g * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    x = tl.load(x_ptr + pid_m * stride_xm + offsets * stride_xk,
                 mask=offsets < K, other=0.0)

    x_grouped = tl.reshape(x, (GROUPS_PER_BLOCK, GROUP_SIZE))
    scale_exp = _get_exponent(x_grouped, 2)

    group_indices = tl.arange(0, GROUPS_PER_BLOCK)
    g_abs = pid_g * GROUPS_PER_BLOCK + group_indices
    tl.store(scale_ptr + pid_m * stride_sm + g_abs * stride_sg,
             scale_exp.to(tl.uint8),
             mask=g_abs < (K // GROUP_SIZE))

    scale_exp_broad = tl.broadcast_to(
        tl.reshape(scale_exp, (GROUPS_PER_BLOCK, 1)),
        (GROUPS_PER_BLOCK, GROUP_SIZE // 2),
    )
    scale_f32 = (tl.reshape(scale_exp_broad, (BLOCK_SIZE // 2,)).to(tl.uint32) << 23)

    x_pairs = tl.reshape(x, (BLOCK_SIZE // 2, 2))
    x_even, x_odd = tl.split(x_pairs)

    fp4_packed = tl.inline_asm_elementwise(
        "v_cvt_scalef32_pk_fp4_f32 $0, $1, $2, $3",
        "=v,v,v,v",
        args=[x_even, x_odd, scale_f32],
        dtype=tl.uint16, is_pure=True, pack=1,
    )

    fp4 = fp4_packed.to(tl.uint8)

    pair_offsets = tl.arange(0, BLOCK_SIZE // 2)
    out_byte_offsets = block_start // 2 + pair_offsets
    tl.store(out_ptr + pid_m * stride_outm + out_byte_offsets * stride_outk,
             fp4, mask=out_byte_offsets < (K // 2))


# ---------------------------------------------------------------------------
# Public Python wrappers
# ---------------------------------------------------------------------------

def _check_shape(x: torch.Tensor, group_size: int):
    assert x.ndim == 2, "Input must be 2-D [M, K]"
    M, K = x.shape
    assert K % group_size == 0, f"K ({K}) must be divisible by group_size ({group_size})"
    return M, K


def quantize_mxfp8e4_rtne(
    x: torch.Tensor,
    group_size: int = 32,
    groups_per_block: int = 16,
    num_warps: int = 4,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    FP32 → MXFP8 E4M3 (round-to-nearest-even).

    Returns
    -------
    fp8   : torch.float8_e4m3fn tensor [M, K]
    scales: torch.uint8 tensor [M, K // group_size]  (E8M0 exponents)
    """
    M, K = _check_shape(x, group_size)
    n_groups = K // group_size
    out = torch.empty((M, K), dtype=torch.uint8, device=x.device)
    scales = torch.empty((M, n_groups), dtype=torch.uint8, device=x.device)

    grid = (M, n_groups // groups_per_block)
    _f32_to_mxfp8e4_rtne_kernel[grid](
        x, out, scales, M, K,
        x.stride(0), x.stride(1),
        out.stride(0), out.stride(1),
        scales.stride(0), scales.stride(1),
        GROUP_SIZE=group_size,
        GROUPS_PER_BLOCK=groups_per_block,
        num_warps=num_warps,
    )
    return out.view(torch.float8_e4m3fn), scales


def quantize_mxfp8e4_sr(
    x: torch.Tensor,
    group_size: int = 32,
    groups_per_block: int = 256,
    num_warps: int = 4,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    FP32 → MXFP8 E4M3 (stochastic rounding).

    Returns
    -------
    fp8   : torch.float8_e4m3fn tensor [M, K]
    scales: torch.uint8 tensor [M, K // group_size]
    """
    M, K = _check_shape(x, group_size)
    n_groups = K // group_size
    out = torch.empty((M, K), dtype=torch.uint8, device=x.device)
    scales = torch.empty((M, n_groups), dtype=torch.uint8, device=x.device)

    grid = (M, n_groups // groups_per_block)
    _f32_to_mxfp8e4_sr_kernel[grid](
        x, out, scales, M, K,
        x.stride(0), x.stride(1),
        out.stride(0), out.stride(1),
        scales.stride(0), scales.stride(1),
        GROUP_SIZE=group_size,
        GROUPS_PER_BLOCK=groups_per_block,
        num_warps=num_warps,
    )
    return out.view(torch.float8_e4m3fn), scales


def quantize_mxfp8e5_rtne(
    x: torch.Tensor,
    group_size: int = 32,
    groups_per_block: int = 16,
    num_warps: int = 4,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    FP32 → MXFP8 E5M2 (round-to-nearest-even).

    Returns
    -------
    fp8   : torch.float8_e5m2 tensor [M, K]
    scales: torch.uint8 tensor [M, K // group_size]
    """
    M, K = _check_shape(x, group_size)
    n_groups = K // group_size
    out = torch.empty((M, K), dtype=torch.uint8, device=x.device)
    scales = torch.empty((M, n_groups), dtype=torch.uint8, device=x.device)

    grid = (M, n_groups // groups_per_block)
    _f32_to_mxfp8e5_rtne_kernel[grid](
        x, out, scales, M, K,
        x.stride(0), x.stride(1),
        out.stride(0), out.stride(1),
        scales.stride(0), scales.stride(1),
        GROUP_SIZE=group_size,
        GROUPS_PER_BLOCK=groups_per_block,
        num_warps=num_warps,
    )
    return out.view(torch.float8_e5m2), scales


def quantize_mxfp8e5_sr(
    x: torch.Tensor,
    group_size: int = 32,
    groups_per_block: int = 256,
    num_warps: int = 4,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    FP32 → MXFP8 E5M2 (stochastic rounding).

    Returns
    -------
    fp8   : torch.float8_e5m2 tensor [M, K]
    scales: torch.uint8 tensor [M, K // group_size]
    """
    M, K = _check_shape(x, group_size)
    n_groups = K // group_size
    out = torch.empty((M, K), dtype=torch.uint8, device=x.device)
    scales = torch.empty((M, n_groups), dtype=torch.uint8, device=x.device)

    grid = (M, n_groups // groups_per_block)
    _f32_to_mxfp8e5_sr_kernel[grid](
        x, out, scales, M, K,
        x.stride(0), x.stride(1),
        out.stride(0), out.stride(1),
        scales.stride(0), scales.stride(1),
        GROUP_SIZE=group_size,
        GROUPS_PER_BLOCK=groups_per_block,
        num_warps=num_warps,
    )
    return out.view(torch.float8_e5m2), scales


def quantize_mxfp4_rtne(
    x: torch.Tensor,
    group_size: int = 32,
    groups_per_block: int = 16,
    num_warps: int = 4,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    FP32 → MXFP4 E2M1 (round-to-nearest-even).

    Returns
    -------
    fp4   : torch.uint8 tensor [M, K // 2]  (two nibbles packed per byte)
    scales: torch.uint8 tensor [M, K // group_size]
    """
    M, K = _check_shape(x, group_size)
    n_groups = K // group_size
    out = torch.empty((M, K // 2), dtype=torch.uint8, device=x.device)
    scales = torch.empty((M, n_groups), dtype=torch.uint8, device=x.device)

    grid = (M, n_groups // groups_per_block)
    _f32_to_mxfp4_rtne_kernel[grid](
        x, out, scales, M, K,
        x.stride(0), x.stride(1),
        out.stride(0), out.stride(1),
        scales.stride(0), scales.stride(1),
        GROUP_SIZE=group_size,
        GROUPS_PER_BLOCK=groups_per_block,
        num_warps=num_warps,
    )
    return out, scales


# ---------------------------------------------------------------------------
# Decode helpers
# ---------------------------------------------------------------------------

def fp4_e2m1_to_fp32(packed: torch.Tensor) -> torch.Tensor:
    """
    Decode packed FP4 E2M1 bytes back to float32.

    Parameters
    ----------
    packed : uint8 tensor [..., K//2]  (low nibble = even element)

    Returns
    -------
    float32 tensor [..., K]
    """
    if packed.dtype != torch.uint8:
        raise TypeError(f"expected torch.uint8, got {packed.dtype}")

    lo = packed & 0x0F
    hi = (packed >> 4) & 0x0F
    codes = torch.stack((lo, hi), dim=-1).reshape(*packed.shape[:-1], -1)

    sign = (codes >> 3) & 0x1
    exp  = (codes >> 1) & 0x3
    mant = codes & 0x1

    out = torch.empty_like(codes, dtype=torch.float32)
    sub_mask = exp == 0
    out[sub_mask]  = mant[sub_mask].float() * 0.5
    out[~sub_mask] = (1.0 + 0.5 * mant[~sub_mask].float()) * torch.pow(
        2.0, exp[~sub_mask].float() - 1.0
    )
    out = torch.where(sign.bool(), -out, out)
    return out


# ---------------------------------------------------------------------------
# Test bench
# ---------------------------------------------------------------------------

def _run_tests():
    import triton.testing as tt

    try:
        import tcast
        _TCAST_DICT = {
            "e4m3": tcast.mxfp8e4,
            "e5m2": tcast.mxfp8e5,
            "e2m1": tcast.mxfp4e2,
        }
        _tcast_available = True
    except ImportError:
        _tcast_available = False

    def _tcast_quantize(x, fmt):
        """Return (quantized_fp32_values, scales_uint8) from tcast, matching our layout."""
        tc = tcast.cast(x, _TCAST_DICT[fmt])
        tc_scale = tc.scaledata.scale.to(torch.uint8).T.reshape(x.shape[0], -1)
        tc_s = 2.0 ** ((tc.scaledata.scale - 127).float()).T  # [M, n_groups]
        tc_s_broad = tc_s.reshape(x.shape[0], -1).repeat_interleave(32, dim=1)
        if fmt == "e4m3":
            tc_q = (tc.tensor.view(x.shape[0], -1, 32) /
                    tc_s.view(x.shape[0], -1).unsqueeze(-1)
                    ).to(torch.float8_e4m3fn).reshape(x.shape)
        elif fmt == "e5m2":
            tc_q = (tc.tensor.view(x.shape[0], -1, 32) /
                    tc_s.view(x.shape[0], -1).unsqueeze(-1)
                    ).to(torch.float8_e5m2).reshape(x.shape)
        else:  # e2m1 – tcast stores as bfloat16
            tc_q = (tc.tensor.view(x.shape[0], -1, 32) /
                    tc_s.view(x.shape[0], -1).unsqueeze(-1)
                    ).to(torch.bfloat16).reshape(x.shape)
        return tc_q, tc_scale, tc_s_broad

    torch.manual_seed(42)
    M, K = 4096, 4096
    GROUP_SIZE = 32

    print(f"=== mxfp4fp8_quantization test bench  M={M} K={K} group={GROUP_SIZE} ===")
    print(f"    tcast available: {_tcast_available}\n")

    x = torch.randn((M, K), device="cuda", dtype=torch.float32)

    # (label, fn, kwargs, tcast_fmt)
    configs = [
        ("MXFP8 E4M3  RTNE", quantize_mxfp8e4_rtne, dict(group_size=GROUP_SIZE, groups_per_block=16),  "e4m3"),
        ("MXFP8 E4M3  SR  ", quantize_mxfp8e4_sr,   dict(group_size=GROUP_SIZE, groups_per_block=256), "e4m3"),
        ("MXFP8 E5M2  RTNE", quantize_mxfp8e5_rtne, dict(group_size=GROUP_SIZE, groups_per_block=16),  "e5m2"),
        ("MXFP8 E5M2  SR  ", quantize_mxfp8e5_sr,   dict(group_size=GROUP_SIZE, groups_per_block=256), "e5m2"),
        ("MXFP4 E2M1  RTNE", quantize_mxfp4_rtne,   dict(group_size=GROUP_SIZE, groups_per_block=16),  "e2m1"),
    ]

    for name, fn, kwargs, tc_fmt in configs:
        try:
            q, s = fn(x, **kwargs)
            assert s.shape == (M, K // GROUP_SIZE), f"scale shape mismatch: {s.shape}"
            assert s.dtype == torch.uint8

            if "E4M3" in name:
                assert q.dtype == torch.float8_e4m3fn, f"wrong dtype {q.dtype}"
                assert q.shape == (M, K)
            elif "E5M2" in name:
                assert q.dtype == torch.float8_e5m2, f"wrong dtype {q.dtype}"
                assert q.shape == (M, K)
            else:
                assert q.dtype == torch.uint8
                assert q.shape == (M, K // 2)

            # Round-trip L_inf vs original FP32
            s_f32 = (2.0 ** (s.float() - 127)).repeat_interleave(GROUP_SIZE, dim=1)
            if "E2M1" in name:
                x_recon = fp4_e2m1_to_fp32(q) * s_f32
            else:
                x_recon = q.float() * s_f32
            l_inf_fp32 = torch.max(torch.abs(x_recon - x)).item()

            # Optional: L_inf vs tcast reference
            tc_suffix = ""
            if _tcast_available:
                tc_q, tc_scale, tc_s_broad = _tcast_quantize(x, tc_fmt)
                # compare scales
                l_inf_scale = torch.max(torch.abs(s.float() - tc_scale.float())).item()
                # compare dequantized values
                if "E2M1" in name:
                    # tcast returns bfloat16; compare dequantized fp32
                    tc_recon = tc_q.float() * tc_s_broad
                    x_recon_tc = fp4_e2m1_to_fp32(q) * s_f32
                    l_inf_vs_tc = torch.max(torch.abs(x_recon_tc - tc_recon)).item()
                else:
                    tc_recon = tc_q.float() * tc_s_broad
                    l_inf_vs_tc = torch.max(torch.abs(x_recon - tc_recon)).item()
                tc_suffix = f"  vs_tcast(L_inf scale={l_inf_scale:.1f}, L_inf val={l_inf_vs_tc:.4f})"

            ms = tt.do_bench(lambda: fn(x, **kwargs), warmup=25, rep=500)
            gbps = x.numel() * 4 / (ms * 1e-3) / 1e9

            print(f"  {name}: L_inf={l_inf_fp32:.4f}  time={ms:.4f} ms  read_bw={gbps:.1f} GB/s{tc_suffix}")

        except Exception as e:
            import traceback
            print(f"  {name}: FAILED — {e}")
            traceback.print_exc()

    print()


if __name__ == "__main__":
    _run_tests()
