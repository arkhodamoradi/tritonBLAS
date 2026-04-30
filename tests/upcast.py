"""
MXFP Upcasting Kernels - Converts MXFP4/MXFP8 data back to FP32 using hardware instructions.
"""

import torch
import triton
import triton.language as tl
import triton.testing as tt

from downcast_gemm import f32_to_mxfp8_triton, f32_to_mxfp4_triton, f32_to_mxfp6_triton

_TORCH_TO_TL_DTYPE = {
    torch.float32:  tl.float32,
    torch.bfloat16: tl.bfloat16,
    torch.float16:  tl.float16,
}

# HW instruction to use for each (format, out_dtype) combination
_FP8E4_ASM = {
    torch.float32:  "v_cvt_scalef32_f32_fp8",
    torch.float16:  "v_cvt_scalef32_f16_fp8",
}
_FP8E5_ASM = {
    torch.float32:  "v_cvt_scalef32_f32_bf8",
    torch.float16:  "v_cvt_scalef32_f16_bf8",
}
# FP4: f32 path is hard-coded in the kernel; only 16-bit variants go via ASM_INSTR
_FP4_ASM = {
    torch.float16:  "v_cvt_scalef32_pk_f16_fp4",
    torch.bfloat16: "v_cvt_scalef32_pk_bf16_fp4",
}
# FP6 E2M3 (fp6 suffix): f32 is hard-coded; 16-bit variants via ASM_INSTR
_FP6E2_ASM = {
    torch.float16:  "v_cvt_scalef32_pk32_f16_fp6",
    torch.bfloat16: "v_cvt_scalef32_pk32_bf16_fp6",
}
# FP6 E3M2 (bf6 suffix): f32 is hard-coded; 16-bit variants via ASM_INSTR
_FP6E3_ASM = {
    torch.float16:  "v_cvt_scalef32_pk32_f16_bf6",
    torch.bfloat16: "v_cvt_scalef32_pk32_bf16_bf6",
}

@triton.jit
def mxfp8e4_to_f32_kernel_hw(
    fp8_ptr,         # Input: FP8 E4M3 data (uint8)
    scale_ptr,       # Input: E8M0 scales (uint8)
    out_ptr,         # Output values
    M,               # Number of rows
    K,               # Number of columns
    stride_fp8_m,    # Stride for FP8 input (row)
    stride_fp8_k,    # Stride for FP8 input (col)
    stride_scale_m,  # Stride for scales (row)
    stride_scale_g,  # Stride for scales (group)
    stride_out_m,    # Stride for output (row)
    stride_out_k,    # Stride for output (col)
    GROUP_SIZE: tl.constexpr,
    GROUPS_PER_BLOCK: tl.constexpr,
    OUT_DTYPE: tl.constexpr = tl.float32,
    ASM_INSTR: tl.constexpr = "v_cvt_scalef32_f32_fp8",
):
    """
    Hardware-accelerated MXFP8 E4M3 upcast kernel.
    Instruction is selected by ASM_INSTR (f32/f16 output variants).
    
    V_CVT_SCALEF32_F32_FP8 instruction:
    - Converts FP8 (E4M3) float input to single-precision float
    - Scales the value using the exponent from the second single-precision float input
    - Uses OPSEL[1:0] to determine which byte to read from the 32-bit input
    
    scale = 32'U(exponent(S1.f32));
    srcbyte = OPSEL[1:0].i32 * 8;
    src = VGPR[laneId][SRC0.u32][srcbyte + 7 : srcbyte].fp8;
    tmp = fp8_to_f32_scale(src, scale.u8);
    D0 = tmp.b32
    """
    pid_m = tl.program_id(0)  # Row index
    pid_g = tl.program_id(1)  # Group block index
    
    # Total elements per block = GROUP_SIZE * GROUPS_PER_BLOCK
    BLOCK_SIZE: tl.constexpr = GROUP_SIZE * GROUPS_PER_BLOCK
    
    block_start = pid_g * BLOCK_SIZE
    
    # Load FP8 values (each is 1 byte)
    offsets = tl.arange(0, BLOCK_SIZE)
    fp8_indices = block_start + offsets
    
    # Load FP8 bytes
    fp8_ptrs = fp8_ptr + pid_m * stride_fp8_m + fp8_indices * stride_fp8_k
    mask = fp8_indices < K
    fp8_bytes = tl.load(fp8_ptrs, mask=mask, other=0)
    
    # Load scales for each group
    # Each group of GROUP_SIZE values shares one scale
    n_groups_per_block = BLOCK_SIZE // GROUP_SIZE
    group_base = pid_g * n_groups_per_block
    
    # Calculate which group each element belongs to
    group_indices = offsets // GROUP_SIZE
    
    # Load scales for each element
    scale_ptrs = scale_ptr + pid_m * stride_scale_m + (group_base + group_indices) * stride_scale_g
    scales = tl.load(scale_ptrs, mask=mask, other=0).to(tl.uint32)
    
    # Convert scale to F32 format: scale_f32 = 2^scale (as F32 bits)
    # The instruction expects the scale as an F32 value where only the exponent matters
    scale_f32 = (scales << 23)
    
    fp8_u32 = fp8_bytes.to(tl.uint32)

    # inline_asm dtype must be a literal — constexpr if selects it at compile time
    if OUT_DTYPE == tl.float32:
        result = tl.inline_asm_elementwise(
            ASM_INSTR + " $0, $1, $2", "=v,v,v",
            args=[fp8_u32, scale_f32], dtype=tl.float32, is_pure=True, pack=1,
        )
    else:  # float16
        result = tl.inline_asm_elementwise(
            ASM_INSTR + " $0, $1, $2", "=v,v,v",
            args=[fp8_u32, scale_f32], dtype=tl.float16, is_pure=True, pack=1,
        )

    out_ptrs = out_ptr + pid_m * stride_out_m + fp8_indices * stride_out_k
    tl.store(out_ptrs, result, mask=mask)


@triton.jit
def mxfp8e5_to_f32_kernel_hw(
    fp8_ptr,         # Input: FP8 E5M2 data (uint8)
    scale_ptr,       # Input: E8M0 scales (uint8)
    out_ptr,         # Output values
    M,               # Number of rows
    K,               # Number of columns
    stride_fp8_m,    # Stride for FP8 input (row)
    stride_fp8_k,    # Stride for FP8 input (col)
    stride_scale_m,  # Stride for scales (row)
    stride_scale_g,  # Stride for scales (group)
    stride_out_m,    # Stride for output (row)
    stride_out_k,    # Stride for output (col)
    GROUP_SIZE: tl.constexpr,
    GROUPS_PER_BLOCK: tl.constexpr,
    OUT_DTYPE: tl.constexpr = tl.float32,
    ASM_INSTR: tl.constexpr = "v_cvt_scalef32_f32_bf8",
):
    """
    Hardware-accelerated MXFP8 E5M2 (BF8) upcast kernel.
    Instruction is selected by ASM_INSTR (f32/f16 output variants).
    
    V_CVT_SCALEF32_F32_BF8 instruction:
    - Converts BF8 (E5M2) float input to single-precision float
    - Scales the value using the exponent from the second single-precision float input
    - Uses OPSEL[1:0] to determine which byte to read from the 32-bit input
    
    scale = 32'U(exponent(S1.f32));
    srcbyte = OPSEL[1:0].i32 * 8;
    src = VGPR[laneId][SRC0.u32][srcbyte + 7 : srcbyte].bf8;
    tmp = bf8_to_f32_scale(src, scale.u8);
    D0 = tmp.b32
    """
    pid_m = tl.program_id(0)  # Row index
    pid_g = tl.program_id(1)  # Group block index
    
    # Total elements per block = GROUP_SIZE * GROUPS_PER_BLOCK
    BLOCK_SIZE: tl.constexpr = GROUP_SIZE * GROUPS_PER_BLOCK
    
    block_start = pid_g * BLOCK_SIZE
    
    # Load FP8 values (each is 1 byte)
    offsets = tl.arange(0, BLOCK_SIZE)
    fp8_indices = block_start + offsets
    
    # Load FP8 bytes
    fp8_ptrs = fp8_ptr + pid_m * stride_fp8_m + fp8_indices * stride_fp8_k
    mask = fp8_indices < K
    fp8_bytes = tl.load(fp8_ptrs, mask=mask, other=0)
    
    # Load scales for each group
    # Each group of GROUP_SIZE values shares one scale
    n_groups_per_block = BLOCK_SIZE // GROUP_SIZE
    group_base = pid_g * n_groups_per_block
    
    # Calculate which group each element belongs to
    group_indices = offsets // GROUP_SIZE
    
    # Load scales for each element
    scale_ptrs = scale_ptr + pid_m * stride_scale_m + (group_base + group_indices) * stride_scale_g
    scales = tl.load(scale_ptrs, mask=mask, other=0).to(tl.uint32)
    
    # Convert scale to F32 format: scale_f32 = 2^scale (as F32 bits)
    # The instruction expects the scale as an F32 value where only the exponent matters
    scale_f32 = (scales << 23)
    
    fp8_u32 = fp8_bytes.to(tl.uint32)

    if OUT_DTYPE == tl.float32:
        result = tl.inline_asm_elementwise(
            ASM_INSTR + " $0, $1, $2", "=v,v,v",
            args=[fp8_u32, scale_f32], dtype=tl.float32, is_pure=True, pack=1,
        )
    else:  # float16
        result = tl.inline_asm_elementwise(
            ASM_INSTR + " $0, $1, $2", "=v,v,v",
            args=[fp8_u32, scale_f32], dtype=tl.float16, is_pure=True, pack=1,
        )

    out_ptrs = out_ptr + pid_m * stride_out_m + fp8_indices * stride_out_k
    tl.store(out_ptrs, result, mask=mask)


@triton.jit
def mxfp4e2_to_f32_kernel_hw(
    fp4_ptr,         # Input: Packed FP4 E2M1 data (uint8, each byte = 2 FP4 values)
    scale_ptr,       # Input: E8M0 scales (uint8)
    out_ptr,         # Output values
    M,               # Number of rows
    K,               # Number of columns (unpacked, i.e., 2x the packed size)
    stride_fp4_m,    # Stride for FP4 input (row)
    stride_fp4_k,    # Stride for FP4 input (col, in packed bytes)
    stride_scale_m,  # Stride for scales (row)
    stride_scale_g,  # Stride for scales (group)
    stride_out_m,    # Stride for output (row)
    stride_out_k,    # Stride for output (col)
    GROUP_SIZE: tl.constexpr,
    GROUPS_PER_BLOCK: tl.constexpr,
    OUT_DTYPE: tl.constexpr = tl.float32,
    ASM_INSTR: tl.constexpr = "v_cvt_scalef32_pk_f16_fp4",
):
    """
    Hardware-accelerated MXFP4 E2M1 upcast kernel.
    f32 path: hard-coded v_cvt_scalef32_pk_f32_fp4 (uint64 → 2×f32).
    f16/bf16 path: ASM_INSTR selects the instruction (uint32 → 2×16-bit).
    
    V_CVT_SCALEF32_PK_F32_FP4 instruction:
    - Converts packed 2-component FP4 (E2M1) input to packed single-precision floats
    - Scales the values using the exponent from the second single-precision float input
    - Each byte contains 2 FP4 values: low nibble (bits 3:0) and high nibble (bits 7:4)
    - Outputs 2 F32 values per byte (64 bits total)
    
    scale = 32'U(exponent(S1.f32));
    srcbyte = OPSEL[1:0].i32 * 8;
    src = VGPR[laneId][SRC0.u32][srcbyte + 7 : srcbyte].b8;
    tmp0 = fp4_to_f32_scale(src[3:0].fp4, scale.u8);
    tmp1 = fp4_to_f32_scale(src[7:4].fp4, scale.u8);
    D0[31:0].f32 = tmp0;
    D0[63:32].f32 = tmp1
    """
    pid_m = tl.program_id(0)  # Row index
    pid_g = tl.program_id(1)  # Group block index
    
    # Total unpacked elements per block = GROUP_SIZE * GROUPS_PER_BLOCK
    BLOCK_SIZE: tl.constexpr = GROUP_SIZE * GROUPS_PER_BLOCK
    # Packed bytes per block (2 FP4 values per byte)
    PACKED_BLOCK_SIZE: tl.constexpr = BLOCK_SIZE // 2
    
    block_start = pid_g * BLOCK_SIZE
    packed_block_start = pid_g * PACKED_BLOCK_SIZE
    
    # Load packed FP4 bytes (each byte = 2 FP4 values)
    packed_offsets = tl.arange(0, PACKED_BLOCK_SIZE)
    fp4_packed_indices = packed_block_start + packed_offsets
    
    # Load FP4 packed bytes
    fp4_ptrs = fp4_ptr + pid_m * stride_fp4_m + fp4_packed_indices * stride_fp4_k
    packed_mask = fp4_packed_indices < (K // 2)
    fp4_bytes = tl.load(fp4_ptrs, mask=packed_mask, other=0)
    
    # Load scales for each group
    # Each group of GROUP_SIZE unpacked values shares one scale
    # That's GROUP_SIZE // 2 packed bytes per group
    n_groups_per_block = BLOCK_SIZE // GROUP_SIZE
    group_base = pid_g * n_groups_per_block
    
    # Calculate which group each packed byte belongs to
    # Each packed byte contains 2 values, so we need to map to groups correctly
    # packed_offsets * 2 gives us the unpacked index of the first value in each byte
    unpacked_indices = packed_offsets * 2
    group_indices = unpacked_indices // GROUP_SIZE
    
    # Load scales for each packed byte (same scale for both FP4 values in the byte)
    scale_ptrs = scale_ptr + pid_m * stride_scale_m + (group_base + group_indices) * stride_scale_g
    scales = tl.load(scale_ptrs, mask=packed_mask, other=0).to(tl.uint32)
    
    # Convert scale to F32 format: scale_f32 = 2^scale (as F32 bits)
    # The instruction expects the scale as an F32 value where only the exponent matters
    scale_f32 = (scales << 23)
    
    fp4_u32 = fp4_bytes.to(tl.uint32)
    out_indices = block_start + tl.arange(0, BLOCK_SIZE)
    out_mask = out_indices < K

    if OUT_DTYPE == tl.float32:
        # f32: outputs 2 packed f32 in a uint64
        f32_packed = tl.inline_asm_elementwise(
            "v_cvt_scalef32_pk_f32_fp4 $0, $1, $2",
            "=v,v,v",
            args=[fp4_u32, scale_f32],
            dtype=tl.uint64,
            is_pure=True,
            pack=1,
        )
        lo = (f32_packed & 0xFFFFFFFF).to(tl.uint32).to(tl.float32, bitcast=True)
        hi = ((f32_packed >> 32) & 0xFFFFFFFF).to(tl.uint32).to(tl.float32, bitcast=True)
    else:
        # f16/bf16: ASM_INSTR selects the instruction; outputs 2 packed 16-bit values in a uint32
        pk = tl.inline_asm_elementwise(
            ASM_INSTR + " $0, $1, $2",
            "=v,v,v",
            args=[fp4_u32, scale_f32],
            dtype=tl.uint32,
            is_pure=True,
            pack=1,
        )
        lo = (pk & 0xFFFF).to(tl.uint16).to(OUT_DTYPE, bitcast=True)
        hi = ((pk >> 16) & 0xFFFF).to(tl.uint16).to(OUT_DTYPE, bitcast=True)

    tl.store(out_ptr + pid_m * stride_out_m + out_indices * stride_out_k,
             tl.interleave(lo, hi), mask=out_mask)


def mxfp4_to_f32_triton_hw(
    fp4_data: torch.Tensor,
    scales: torch.Tensor,
    M: int,
    K: int,
    group_size: int = 32,
    num_warps: int = 4,
    out_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    assert fp4_data.dtype == torch.uint8, f"Expected uint8 fp4_data, got {fp4_data.dtype}"
    assert scales.dtype == torch.uint8, f"Expected uint8 scales, got {scales.dtype}"
    assert K == fp4_data.shape[1] * 2, f"K ({K}) must be 2x packed size ({fp4_data.shape[1]})"

    out = torch.empty((M, K), dtype=out_dtype, device=fp4_data.device)

    GROUPS_PER_BLOCK = 16
    n_groups = K // group_size
    grid = (M, n_groups // GROUPS_PER_BLOCK)

    # f32 path uses a hard-coded instruction in the kernel; f16/bf16 need ASM_INSTR
    asm_instr = _FP4_ASM.get(out_dtype, _FP4_ASM[torch.float16])  # dummy for f32 path

    mxfp4e2_to_f32_kernel_hw[grid](
        fp4_data, scales, out,
        M, K,
        fp4_data.stride(0), fp4_data.stride(1),
        scales.stride(0), scales.stride(1),
        out.stride(0), out.stride(1),
        GROUP_SIZE=group_size,
        GROUPS_PER_BLOCK=GROUPS_PER_BLOCK,
        OUT_DTYPE=_TORCH_TO_TL_DTYPE[out_dtype],
        ASM_INSTR=asm_instr,
        num_warps=num_warps,
    )

    return out


def mxfp8_to_f32_triton_hw(
    fp8_data: torch.Tensor,
    scales: torch.Tensor,
    M: int,
    K: int,
    fmt: str = "e4m3",
    group_size: int = 32,
    num_warps: int = 4,
    out_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    if fp8_data.dtype != torch.uint8:
        fp8_data = fp8_data.view(torch.uint8)

    assert scales.dtype == torch.uint8, f"Expected uint8 scales, got {scales.dtype}"

    out = torch.empty((M, K), dtype=out_dtype, device=fp8_data.device)

    GROUPS_PER_BLOCK = 16
    n_groups = K // group_size
    grid = (M, n_groups // GROUPS_PER_BLOCK)

    if fmt == "e4m3":
        kernel = mxfp8e4_to_f32_kernel_hw
        asm_map = _FP8E4_ASM
    else:
        kernel = mxfp8e5_to_f32_kernel_hw
        asm_map = _FP8E5_ASM
    assert out_dtype in asm_map, f"No HW instruction for FP8 {fmt} → {out_dtype}"

    kernel[grid](
        fp8_data, scales, out,
        M, K,
        fp8_data.stride(0), fp8_data.stride(1),
        scales.stride(0), scales.stride(1),
        out.stride(0), out.stride(1),
        GROUP_SIZE=group_size,
        GROUPS_PER_BLOCK=GROUPS_PER_BLOCK,
        OUT_DTYPE=_TORCH_TO_TL_DTYPE[out_dtype],
        ASM_INSTR=asm_map[out_dtype],
        num_warps=num_warps,
    )

    return out


def fp8_e4m3_to_fp32_sw(fp8_data: torch.Tensor) -> torch.Tensor:
    """
    Software decode of FP8 E4M3 values to float32.
    
    FP8 E4M3 format: [S][EEEE][MMM] - 1 sign, 4 exponent, 3 mantissa bits
    Bias = 7
    
    Args:
        fp8_data: uint8 tensor containing FP8 E4M3 values
    
    Returns:
        float32 tensor with decoded values
    """
    if fp8_data.dtype != torch.uint8:
        fp8_data = fp8_data.view(torch.uint8)
    
    # Extract bit fields
    sign = (fp8_data >> 7) & 0x1
    exp = (fp8_data >> 3) & 0xF  # 4 bits
    mant = fp8_data & 0x7  # 3 bits
    
    # Output buffer
    out = torch.empty_like(fp8_data, dtype=torch.float32)
    
    # exp == 0 -> subnormal/zero: value = mant/8 * 2^(-6)
    # exp > 0 -> normal: value = (1 + mant/8) * 2^(exp-7)
    sub_mask = (exp == 0)
    out[sub_mask] = mant[sub_mask].to(torch.float32) / 8.0 * (2.0 ** -6)
    
    norm_mask = ~sub_mask
    out[norm_mask] = (
        (1.0 + mant[norm_mask].to(torch.float32) / 8.0) *
        torch.pow(2.0, exp[norm_mask].to(torch.float32) - 7.0)
    )
    
    # Apply sign
    out = torch.where(sign.bool(), -out, out)
    return out


def fp8_e5m2_to_fp32_sw(fp8_data: torch.Tensor) -> torch.Tensor:
    """
    Software decode of FP8 E5M2 (BF8) values to float32.
    
    FP8 E5M2 format: [S][EEEEE][MM] - 1 sign, 5 exponent, 2 mantissa bits
    Bias = 15
    
    Args:
        fp8_data: uint8 tensor containing FP8 E5M2 values
    
    Returns:
        float32 tensor with decoded values
    """
    if fp8_data.dtype != torch.uint8:
        fp8_data = fp8_data.view(torch.uint8)
    
    # Extract bit fields
    sign = (fp8_data >> 7) & 0x1
    exp = (fp8_data >> 2) & 0x1F  # 5 bits
    mant = fp8_data & 0x3  # 2 bits
    
    # Output buffer
    out = torch.empty_like(fp8_data, dtype=torch.float32)
    
    # exp == 0 -> subnormal/zero: value = mant/4 * 2^(-14)
    # exp > 0 -> normal: value = (1 + mant/4) * 2^(exp-15)
    sub_mask = (exp == 0)
    out[sub_mask] = mant[sub_mask].to(torch.float32) / 4.0 * (2.0 ** -14)
    
    norm_mask = ~sub_mask
    out[norm_mask] = (
        (1.0 + mant[norm_mask].to(torch.float32) / 4.0) *
        torch.pow(2.0, exp[norm_mask].to(torch.float32) - 15.0)
    )
    
    # Apply sign
    out = torch.where(sign.bool(), -out, out)
    return out


def fp4_e2m1_to_fp32_sw(packed: torch.Tensor) -> torch.Tensor:
    """
    Software decode of packed FP4 E2M1 values to float32.
    
    FP4 E2M1 format: [S][EE][M] - 1 sign, 2 exponent, 1 mantissa bit
    Each byte contains 2 FP4 values: low nibble (bits 3:0) and high nibble (bits 7:4)
    
    Args:
        packed: uint8 tensor, each byte contains 2 FP4 values
                low nibble first, high nibble second
    
    Returns:
        float32 tensor with shape (..., 2 * packed.shape[-1])
    """
    if packed.dtype != torch.uint8:
        raise TypeError(f"expected torch.uint8, got {packed.dtype}")

    # Split each byte into low/high 4-bit codes
    lo = packed & 0x0F
    hi = (packed >> 4) & 0x0F

    # Interleave low/high nibbles so output order matches packed order
    codes = torch.stack((lo, hi), dim=-1).reshape(*packed.shape[:-1], -1)

    # Bit fields: [sign | exp(2) | mant(1)]
    sign = (codes >> 3) & 0x1
    exp = (codes >> 1) & 0x3
    mant = codes & 0x1

    # Output buffer
    out = torch.empty_like(codes, dtype=torch.float32)

    # exp == 0  -> subnormal/zero: value = mant * 0.5
    # exp > 0   -> normal: value = (1 + mant*0.5) * 2^(exp-1)
    sub_mask = (exp == 0)
    out[sub_mask] = mant[sub_mask].to(torch.float32) * 0.5

    norm_mask = ~sub_mask
    out[norm_mask] = (
        (1.0 + 0.5 * mant[norm_mask].to(torch.float32)) *
        torch.pow(2.0, (exp[norm_mask].to(torch.float32) - 1.0))
    )

    # Apply sign
    out = torch.where(sign.bool(), -out, out)
    return out


def fp6_e2m3_to_fp32_sw(packed: torch.Tensor) -> torch.Tensor:
    """
    Software decode of packed MXFP6 E2M3 values to float32.

    FP6 E2M3 format: [S][EE][MMM] - 1 sign, 2 exponent, 3 mantissa bits
    Exponent bias = 1. No NaN or Inf (all bit patterns are finite).

    Args:
        packed: uint32 tensor, shape (M, n_groups*6)
                6 uint32s encode 32 FP6 values (192 bits) per group

    Returns:
        float32 tensor of shape (M, K) where K = n_groups * 32
    """
    if packed.dtype != torch.uint32:
        raise TypeError(f"expected torch.uint32, got {packed.dtype}")

    M = packed.shape[0]
    n_packed = packed.shape[1]   # n_groups * 6
    n_groups = n_packed // 6
    K = n_groups * 32

    # View as uint8, reshape to (M, n_groups*8, 3): 3 bytes -> 4 FP6 codes
    packed_bytes = packed.view(torch.uint8).reshape(M, n_groups * 8, 3)

    b0 = packed_bytes[:, :, 0].to(torch.int64)
    b1 = packed_bytes[:, :, 1].to(torch.int64)
    b2 = packed_bytes[:, :, 2].to(torch.int64)
    bits24 = b0 | (b1 << 8) | (b2 << 16)   # (M, n_groups*8)

    # Extract 4 FP6 codes per 24-bit word
    shifts = torch.tensor([0, 6, 12, 18], device=packed.device, dtype=torch.int64)
    codes = ((bits24.unsqueeze(-1) >> shifts) & 0x3F).to(torch.int32)  # (M, n_groups*8, 4)
    codes = codes.reshape(M, K)

    # Bit fields: [S1][E2][M3]
    sign = (codes >> 5) & 0x1
    exp  = (codes >> 3) & 0x3   # 2 bits
    mant =  codes       & 0x7   # 3 bits

    out = torch.empty((M, K), dtype=torch.float32, device=packed.device)

    # E=0: subnormal = M/8 * 2^(1-bias) = M/8 * 2^0 = M/8
    # E>0: normal    = (1 + M/8) * 2^(E-1)
    sub_mask  = (exp == 0)
    norm_mask = ~sub_mask

    out[sub_mask]  = mant[sub_mask].float() / 8.0
    out[norm_mask] = (1.0 + mant[norm_mask].float() / 8.0) * torch.pow(
        2.0, exp[norm_mask].float() - 1.0
    )

    out = torch.where(sign.bool(), -out, out)
    return out


def fp6_e3m2_to_fp32_sw(packed: torch.Tensor) -> torch.Tensor:
    """
    Software decode of packed MXFP6 E3M2 values to float32.

    FP6 E3M2 format: [S][EEE][MM] - 1 sign, 3 exponent, 2 mantissa bits
    Exponent bias = 3. No NaN or Inf (all bit patterns are finite).

    Args:
        packed: uint32 tensor, shape (M, n_groups*6)
                6 uint32s encode 32 FP6 values (192 bits) per group

    Returns:
        float32 tensor of shape (M, K) where K = n_groups * 32
    """
    if packed.dtype != torch.uint32:
        raise TypeError(f"expected torch.uint32, got {packed.dtype}")

    M = packed.shape[0]
    n_packed = packed.shape[1]   # n_groups * 6
    n_groups = n_packed // 6
    K = n_groups * 32

    # View as uint8, reshape to (M, n_groups*8, 3): 3 bytes -> 4 FP6 codes
    packed_bytes = packed.view(torch.uint8).reshape(M, n_groups * 8, 3)

    b0 = packed_bytes[:, :, 0].to(torch.int64)
    b1 = packed_bytes[:, :, 1].to(torch.int64)
    b2 = packed_bytes[:, :, 2].to(torch.int64)
    bits24 = b0 | (b1 << 8) | (b2 << 16)   # (M, n_groups*8)

    # Extract 4 FP6 codes per 24-bit word
    shifts = torch.tensor([0, 6, 12, 18], device=packed.device, dtype=torch.int64)
    codes = ((bits24.unsqueeze(-1) >> shifts) & 0x3F).to(torch.int32)  # (M, n_groups*8, 4)
    codes = codes.reshape(M, K)

    # Bit fields: [S1][E3][M2]
    sign = (codes >> 5) & 0x1
    exp  = (codes >> 2) & 0x7   # 3 bits
    mant =  codes       & 0x3   # 2 bits

    out = torch.empty((M, K), dtype=torch.float32, device=packed.device)

    # E=0: subnormal = M/4 * 2^(1-bias) = M/4 * 2^(-2) = M/16
    # E>0: normal    = (1 + M/4) * 2^(E-3)
    sub_mask  = (exp == 0)
    norm_mask = ~sub_mask

    out[sub_mask]  = mant[sub_mask].float() / 4.0 * (2.0 ** -2)
    out[norm_mask] = (1.0 + mant[norm_mask].float() / 4.0) * torch.pow(
        2.0, exp[norm_mask].float() - 3.0
    )

    out = torch.where(sign.bool(), -out, out)
    return out


@triton.jit
def mxfp6e2_to_f32_kernel_hw(
    fp6_ptr,         # Input: packed FP6 E2M3 data (uint32), shape (M, n_groups*6)
    scale_ptr,       # Input: E8M0 scales (uint8), shape (M, n_groups)
    out_ptr,         # Output values, shape (M, K)
    M,
    K,
    stride_fp6_m,    # Row stride for fp6 input (in uint32 units)
    stride_scale_m,  # Row stride for scales
    stride_scale_g,  # Group stride for scales
    stride_out_m,    # Row stride for output
    GROUP_SIZE: tl.constexpr,  # Must be 32
    OUT_DTYPE: tl.constexpr = tl.float32,
    ASM_INSTR: tl.constexpr = "v_cvt_scalef32_pk32_f16_fp6",
):
    """
    Hardware-accelerated MXFP6 E2M3 upcast kernel.
    f32 path: hard-coded v_cvt_scalef32_pk32_f32_fp6 (32 VGPRs → 32×f32).
    f16/bf16 path: ASM_INSTR selects the instruction (16 VGPRs → 32×16-bit packed).

    V_CVT_SCALEF32_PK32_F32_FP6 (opcode 598):
    - S0: 6 consecutive VGPRs — 32 packed FP6 E2M3 values (192 bits)
    - S1: 1 VGPR — scale (f32, only exponent used)
    - D0: 32 consecutive VGPRs — 32 FP32 results (1024 bits)

    scale = 32'U(exponent(S1.f32));
    for pass in 0:31:
        tmp[pass*32+31:pass*32].f32 = fp6_to_f32_scale(S0[pass*6+5:pass*6].fp6, scale.u8)
    D0[1023:0] = tmp.b1024
    """
    pid_m = tl.program_id(0)
    pid_g = tl.program_id(1)

    # Load 6 packed uint32s for this group (192 bits = 32 FP6 E2M3 values)
    fp6_base = fp6_ptr + pid_m * stride_fp6_m + pid_g * 6
    r0 = tl.load(fp6_base + 0).to(tl.uint32)
    r1 = tl.load(fp6_base + 1).to(tl.uint32)
    r2 = tl.load(fp6_base + 2).to(tl.uint32)
    r3 = tl.load(fp6_base + 3).to(tl.uint32)
    r4 = tl.load(fp6_base + 4).to(tl.uint32)
    r5 = tl.load(fp6_base + 5).to(tl.uint32)

    # Load scale and convert to F32 format (scale_exp << 23)
    scale_raw = tl.load(scale_ptr + pid_m * stride_scale_m + pid_g * stride_scale_g).to(tl.uint32)
    scale_f32 = (scale_raw << 23)

    out_base = out_ptr + pid_m * stride_out_m + pid_g * GROUP_SIZE

    if OUT_DTYPE == tl.float32:
        # f32 path: 32 VGPRs output (v56-v87), one f32 per element
        (o0,  o1,  o2,  o3,  o4,  o5,  o6,  o7,
         o8,  o9,  o10, o11, o12, o13, o14, o15,
         o16, o17, o18, o19, o20, o21, o22, o23,
         o24, o25, o26, o27, o28, o29, o30, o31) = tl.inline_asm_elementwise(
            asm="""
            v_mov_b32 v50, $32
            v_mov_b32 v51, $33
            v_mov_b32 v52, $34
            v_mov_b32 v53, $35
            v_mov_b32 v54, $36
            v_mov_b32 v55, $37
            v_cvt_scalef32_pk32_f32_fp6 v[56:87], v[50:55], $38
            v_mov_b32 $0,  v56
            v_mov_b32 $1,  v57
            v_mov_b32 $2,  v58
            v_mov_b32 $3,  v59
            v_mov_b32 $4,  v60
            v_mov_b32 $5,  v61
            v_mov_b32 $6,  v62
            v_mov_b32 $7,  v63
            v_mov_b32 $8,  v64
            v_mov_b32 $9,  v65
            v_mov_b32 $10, v66
            v_mov_b32 $11, v67
            v_mov_b32 $12, v68
            v_mov_b32 $13, v69
            v_mov_b32 $14, v70
            v_mov_b32 $15, v71
            v_mov_b32 $16, v72
            v_mov_b32 $17, v73
            v_mov_b32 $18, v74
            v_mov_b32 $19, v75
            v_mov_b32 $20, v76
            v_mov_b32 $21, v77
            v_mov_b32 $22, v78
            v_mov_b32 $23, v79
            v_mov_b32 $24, v80
            v_mov_b32 $25, v81
            v_mov_b32 $26, v82
            v_mov_b32 $27, v83
            v_mov_b32 $28, v84
            v_mov_b32 $29, v85
            v_mov_b32 $30, v86
            v_mov_b32 $31, v87
            """,
            constraints=(
                "=v,=v,=v,=v,=v,=v,=v,=v,"
                "=v,=v,=v,=v,=v,=v,=v,=v,"
                "=v,=v,=v,=v,=v,=v,=v,=v,"
                "=v,=v,=v,=v,=v,=v,=v,=v,"
                "v,v,v,v,v,v,v,"
                "~{v50},~{v51},~{v52},~{v53},~{v54},~{v55},"
                "~{v56},~{v57},~{v58},~{v59},~{v60},~{v61},~{v62},~{v63},"
                "~{v64},~{v65},~{v66},~{v67},~{v68},~{v69},~{v70},~{v71},"
                "~{v72},~{v73},~{v74},~{v75},~{v76},~{v77},~{v78},~{v79},"
                "~{v80},~{v81},~{v82},~{v83},~{v84},~{v85},~{v86},~{v87}"
            ),
            args=[r0, r1, r2, r3, r4, r5, scale_f32],
            dtype=(tl.float32,) * 32,
            is_pure=True,
            pack=1,
        )
        tl.store(out_base + 0,  o0);  tl.store(out_base + 1,  o1)
        tl.store(out_base + 2,  o2);  tl.store(out_base + 3,  o3)
        tl.store(out_base + 4,  o4);  tl.store(out_base + 5,  o5)
        tl.store(out_base + 6,  o6);  tl.store(out_base + 7,  o7)
        tl.store(out_base + 8,  o8);  tl.store(out_base + 9,  o9)
        tl.store(out_base + 10, o10); tl.store(out_base + 11, o11)
        tl.store(out_base + 12, o12); tl.store(out_base + 13, o13)
        tl.store(out_base + 14, o14); tl.store(out_base + 15, o15)
        tl.store(out_base + 16, o16); tl.store(out_base + 17, o17)
        tl.store(out_base + 18, o18); tl.store(out_base + 19, o19)
        tl.store(out_base + 20, o20); tl.store(out_base + 21, o21)
        tl.store(out_base + 22, o22); tl.store(out_base + 23, o23)
        tl.store(out_base + 24, o24); tl.store(out_base + 25, o25)
        tl.store(out_base + 26, o26); tl.store(out_base + 27, o27)
        tl.store(out_base + 28, o28); tl.store(out_base + 29, o29)
        tl.store(out_base + 30, o30); tl.store(out_base + 31, o31)
    else:
        # f16/bf16: ASM_INSTR selects the instruction; 16 VGPRs (v56-v71),
        # each holding 2 packed 16-bit values.
        (p0,  p1,  p2,  p3,  p4,  p5,  p6,  p7,
         p8,  p9,  p10, p11, p12, p13, p14, p15) = tl.inline_asm_elementwise(
            asm=f"""
            v_mov_b32 v50, $16
            v_mov_b32 v51, $17
            v_mov_b32 v52, $18
            v_mov_b32 v53, $19
            v_mov_b32 v54, $20
            v_mov_b32 v55, $21
            {ASM_INSTR} v[56:71], v[50:55], $22
            v_mov_b32 $0,  v56
            v_mov_b32 $1,  v57
            v_mov_b32 $2,  v58
            v_mov_b32 $3,  v59
            v_mov_b32 $4,  v60
            v_mov_b32 $5,  v61
            v_mov_b32 $6,  v62
            v_mov_b32 $7,  v63
            v_mov_b32 $8,  v64
            v_mov_b32 $9,  v65
            v_mov_b32 $10, v66
            v_mov_b32 $11, v67
            v_mov_b32 $12, v68
            v_mov_b32 $13, v69
            v_mov_b32 $14, v70
            v_mov_b32 $15, v71
            """,
            constraints=(
                "=v,=v,=v,=v,=v,=v,=v,=v,"
                "=v,=v,=v,=v,=v,=v,=v,=v,"
                "v,v,v,v,v,v,v,"
                "~{v50},~{v51},~{v52},~{v53},~{v54},~{v55},"
                "~{v56},~{v57},~{v58},~{v59},~{v60},~{v61},~{v62},~{v63},"
                "~{v64},~{v65},~{v66},~{v67},~{v68},~{v69},~{v70},~{v71}"
            ),
            args=[r0, r1, r2, r3, r4, r5, scale_f32],
            dtype=(tl.uint32,) * 16,
            is_pure=True,
            pack=1,
        )
        tl.store(out_base +  0, (p0  & 0xFFFF).to(tl.uint16).to(OUT_DTYPE, bitcast=True))
        tl.store(out_base +  1, ((p0  >> 16) & 0xFFFF).to(tl.uint16).to(OUT_DTYPE, bitcast=True))
        tl.store(out_base +  2, (p1  & 0xFFFF).to(tl.uint16).to(OUT_DTYPE, bitcast=True))
        tl.store(out_base +  3, ((p1  >> 16) & 0xFFFF).to(tl.uint16).to(OUT_DTYPE, bitcast=True))
        tl.store(out_base +  4, (p2  & 0xFFFF).to(tl.uint16).to(OUT_DTYPE, bitcast=True))
        tl.store(out_base +  5, ((p2  >> 16) & 0xFFFF).to(tl.uint16).to(OUT_DTYPE, bitcast=True))
        tl.store(out_base +  6, (p3  & 0xFFFF).to(tl.uint16).to(OUT_DTYPE, bitcast=True))
        tl.store(out_base +  7, ((p3  >> 16) & 0xFFFF).to(tl.uint16).to(OUT_DTYPE, bitcast=True))
        tl.store(out_base +  8, (p4  & 0xFFFF).to(tl.uint16).to(OUT_DTYPE, bitcast=True))
        tl.store(out_base +  9, ((p4  >> 16) & 0xFFFF).to(tl.uint16).to(OUT_DTYPE, bitcast=True))
        tl.store(out_base + 10, (p5  & 0xFFFF).to(tl.uint16).to(OUT_DTYPE, bitcast=True))
        tl.store(out_base + 11, ((p5  >> 16) & 0xFFFF).to(tl.uint16).to(OUT_DTYPE, bitcast=True))
        tl.store(out_base + 12, (p6  & 0xFFFF).to(tl.uint16).to(OUT_DTYPE, bitcast=True))
        tl.store(out_base + 13, ((p6  >> 16) & 0xFFFF).to(tl.uint16).to(OUT_DTYPE, bitcast=True))
        tl.store(out_base + 14, (p7  & 0xFFFF).to(tl.uint16).to(OUT_DTYPE, bitcast=True))
        tl.store(out_base + 15, ((p7  >> 16) & 0xFFFF).to(tl.uint16).to(OUT_DTYPE, bitcast=True))
        tl.store(out_base + 16, (p8  & 0xFFFF).to(tl.uint16).to(OUT_DTYPE, bitcast=True))
        tl.store(out_base + 17, ((p8  >> 16) & 0xFFFF).to(tl.uint16).to(OUT_DTYPE, bitcast=True))
        tl.store(out_base + 18, (p9  & 0xFFFF).to(tl.uint16).to(OUT_DTYPE, bitcast=True))
        tl.store(out_base + 19, ((p9  >> 16) & 0xFFFF).to(tl.uint16).to(OUT_DTYPE, bitcast=True))
        tl.store(out_base + 20, (p10 & 0xFFFF).to(tl.uint16).to(OUT_DTYPE, bitcast=True))
        tl.store(out_base + 21, ((p10 >> 16) & 0xFFFF).to(tl.uint16).to(OUT_DTYPE, bitcast=True))
        tl.store(out_base + 22, (p11 & 0xFFFF).to(tl.uint16).to(OUT_DTYPE, bitcast=True))
        tl.store(out_base + 23, ((p11 >> 16) & 0xFFFF).to(tl.uint16).to(OUT_DTYPE, bitcast=True))
        tl.store(out_base + 24, (p12 & 0xFFFF).to(tl.uint16).to(OUT_DTYPE, bitcast=True))
        tl.store(out_base + 25, ((p12 >> 16) & 0xFFFF).to(tl.uint16).to(OUT_DTYPE, bitcast=True))
        tl.store(out_base + 26, (p13 & 0xFFFF).to(tl.uint16).to(OUT_DTYPE, bitcast=True))
        tl.store(out_base + 27, ((p13 >> 16) & 0xFFFF).to(tl.uint16).to(OUT_DTYPE, bitcast=True))
        tl.store(out_base + 28, (p14 & 0xFFFF).to(tl.uint16).to(OUT_DTYPE, bitcast=True))
        tl.store(out_base + 29, ((p14 >> 16) & 0xFFFF).to(tl.uint16).to(OUT_DTYPE, bitcast=True))
        tl.store(out_base + 30, (p15 & 0xFFFF).to(tl.uint16).to(OUT_DTYPE, bitcast=True))
        tl.store(out_base + 31, ((p15 >> 16) & 0xFFFF).to(tl.uint16).to(OUT_DTYPE, bitcast=True))


@triton.jit
def mxfp6e3_to_f32_kernel_hw(
    fp6_ptr,         # Input: packed FP6 E3M2 data (uint32), shape (M, n_groups*6)
    scale_ptr,       # Input: E8M0 scales (uint8), shape (M, n_groups)
    out_ptr,         # Output values, shape (M, K)
    M,
    K,
    stride_fp6_m,    # Row stride for fp6 input (in uint32 units)
    stride_scale_m,  # Row stride for scales
    stride_scale_g,  # Group stride for scales
    stride_out_m,    # Row stride for output
    GROUP_SIZE: tl.constexpr,  # Must be 32
    OUT_DTYPE: tl.constexpr = tl.float32,
    ASM_INSTR: tl.constexpr = "v_cvt_scalef32_pk32_f16_bf6",
):
    """
    Hardware-accelerated MXFP6 E3M2 upcast kernel.
    f32 path: hard-coded v_cvt_scalef32_pk32_f32_bf6 (32 VGPRs → 32×f32).
    f16/bf16 path: ASM_INSTR selects the instruction (16 VGPRs → 32×16-bit packed).

    V_CVT_SCALEF32_PK32_F32_BF6 (opcode 599):
    - S0: 6 consecutive VGPRs — 32 packed FP6 E3M2 values (192 bits)
    - S1: 1 VGPR — scale (f32, only exponent used)
    - D0: 32 consecutive VGPRs — 32 FP32 results (1024 bits)

    scale = 32'U(exponent(S1.f32));
    for pass in 0:31:
        tmp[pass*32+31:pass*32].f32 = bf6_to_f32_scale(S0[pass*6+5:pass*6].bf6, scale.u8)
    D0[1023:0] = tmp.b1024
    """
    pid_m = tl.program_id(0)
    pid_g = tl.program_id(1)

    # Load 6 packed uint32s for this group (192 bits = 32 FP6 E3M2 values)
    fp6_base = fp6_ptr + pid_m * stride_fp6_m + pid_g * 6
    r0 = tl.load(fp6_base + 0).to(tl.uint32)
    r1 = tl.load(fp6_base + 1).to(tl.uint32)
    r2 = tl.load(fp6_base + 2).to(tl.uint32)
    r3 = tl.load(fp6_base + 3).to(tl.uint32)
    r4 = tl.load(fp6_base + 4).to(tl.uint32)
    r5 = tl.load(fp6_base + 5).to(tl.uint32)

    # Load scale and convert to F32 format (scale_exp << 23)
    scale_raw = tl.load(scale_ptr + pid_m * stride_scale_m + pid_g * stride_scale_g).to(tl.uint32)
    scale_f32 = (scale_raw << 23)

    out_base = out_ptr + pid_m * stride_out_m + pid_g * GROUP_SIZE

    if OUT_DTYPE == tl.float32:
        (o0,  o1,  o2,  o3,  o4,  o5,  o6,  o7,
         o8,  o9,  o10, o11, o12, o13, o14, o15,
         o16, o17, o18, o19, o20, o21, o22, o23,
         o24, o25, o26, o27, o28, o29, o30, o31) = tl.inline_asm_elementwise(
            asm="""
            v_mov_b32 v50, $32
            v_mov_b32 v51, $33
            v_mov_b32 v52, $34
            v_mov_b32 v53, $35
            v_mov_b32 v54, $36
            v_mov_b32 v55, $37
            v_cvt_scalef32_pk32_f32_bf6 v[56:87], v[50:55], $38
            v_mov_b32 $0,  v56
            v_mov_b32 $1,  v57
            v_mov_b32 $2,  v58
            v_mov_b32 $3,  v59
            v_mov_b32 $4,  v60
            v_mov_b32 $5,  v61
            v_mov_b32 $6,  v62
            v_mov_b32 $7,  v63
            v_mov_b32 $8,  v64
            v_mov_b32 $9,  v65
            v_mov_b32 $10, v66
            v_mov_b32 $11, v67
            v_mov_b32 $12, v68
            v_mov_b32 $13, v69
            v_mov_b32 $14, v70
            v_mov_b32 $15, v71
            v_mov_b32 $16, v72
            v_mov_b32 $17, v73
            v_mov_b32 $18, v74
            v_mov_b32 $19, v75
            v_mov_b32 $20, v76
            v_mov_b32 $21, v77
            v_mov_b32 $22, v78
            v_mov_b32 $23, v79
            v_mov_b32 $24, v80
            v_mov_b32 $25, v81
            v_mov_b32 $26, v82
            v_mov_b32 $27, v83
            v_mov_b32 $28, v84
            v_mov_b32 $29, v85
            v_mov_b32 $30, v86
            v_mov_b32 $31, v87
            """,
            constraints=(
                "=v,=v,=v,=v,=v,=v,=v,=v,"
                "=v,=v,=v,=v,=v,=v,=v,=v,"
                "=v,=v,=v,=v,=v,=v,=v,=v,"
                "=v,=v,=v,=v,=v,=v,=v,=v,"
                "v,v,v,v,v,v,v,"
                "~{v50},~{v51},~{v52},~{v53},~{v54},~{v55},"
                "~{v56},~{v57},~{v58},~{v59},~{v60},~{v61},~{v62},~{v63},"
                "~{v64},~{v65},~{v66},~{v67},~{v68},~{v69},~{v70},~{v71},"
                "~{v72},~{v73},~{v74},~{v75},~{v76},~{v77},~{v78},~{v79},"
                "~{v80},~{v81},~{v82},~{v83},~{v84},~{v85},~{v86},~{v87}"
            ),
            args=[r0, r1, r2, r3, r4, r5, scale_f32],
            dtype=(tl.float32,) * 32,
            is_pure=True,
            pack=1,
        )
        tl.store(out_base + 0,  o0);  tl.store(out_base + 1,  o1)
        tl.store(out_base + 2,  o2);  tl.store(out_base + 3,  o3)
        tl.store(out_base + 4,  o4);  tl.store(out_base + 5,  o5)
        tl.store(out_base + 6,  o6);  tl.store(out_base + 7,  o7)
        tl.store(out_base + 8,  o8);  tl.store(out_base + 9,  o9)
        tl.store(out_base + 10, o10); tl.store(out_base + 11, o11)
        tl.store(out_base + 12, o12); tl.store(out_base + 13, o13)
        tl.store(out_base + 14, o14); tl.store(out_base + 15, o15)
        tl.store(out_base + 16, o16); tl.store(out_base + 17, o17)
        tl.store(out_base + 18, o18); tl.store(out_base + 19, o19)
        tl.store(out_base + 20, o20); tl.store(out_base + 21, o21)
        tl.store(out_base + 22, o22); tl.store(out_base + 23, o23)
        tl.store(out_base + 24, o24); tl.store(out_base + 25, o25)
        tl.store(out_base + 26, o26); tl.store(out_base + 27, o27)
        tl.store(out_base + 28, o28); tl.store(out_base + 29, o29)
        tl.store(out_base + 30, o30); tl.store(out_base + 31, o31)
    else:
        # f16/bf16: ASM_INSTR selects the instruction; 16 VGPRs (v56-v71),
        # each holding 2 packed 16-bit values.
        (p0,  p1,  p2,  p3,  p4,  p5,  p6,  p7,
         p8,  p9,  p10, p11, p12, p13, p14, p15) = tl.inline_asm_elementwise(
            asm=f"""
            v_mov_b32 v50, $16
            v_mov_b32 v51, $17
            v_mov_b32 v52, $18
            v_mov_b32 v53, $19
            v_mov_b32 v54, $20
            v_mov_b32 v55, $21
            {ASM_INSTR} v[56:71], v[50:55], $22
            v_mov_b32 $0,  v56
            v_mov_b32 $1,  v57
            v_mov_b32 $2,  v58
            v_mov_b32 $3,  v59
            v_mov_b32 $4,  v60
            v_mov_b32 $5,  v61
            v_mov_b32 $6,  v62
            v_mov_b32 $7,  v63
            v_mov_b32 $8,  v64
            v_mov_b32 $9,  v65
            v_mov_b32 $10, v66
            v_mov_b32 $11, v67
            v_mov_b32 $12, v68
            v_mov_b32 $13, v69
            v_mov_b32 $14, v70
            v_mov_b32 $15, v71
            """,
            constraints=(
                "=v,=v,=v,=v,=v,=v,=v,=v,"
                "=v,=v,=v,=v,=v,=v,=v,=v,"
                "v,v,v,v,v,v,v,"
                "~{v50},~{v51},~{v52},~{v53},~{v54},~{v55},"
                "~{v56},~{v57},~{v58},~{v59},~{v60},~{v61},~{v62},~{v63},"
                "~{v64},~{v65},~{v66},~{v67},~{v68},~{v69},~{v70},~{v71}"
            ),
            args=[r0, r1, r2, r3, r4, r5, scale_f32],
            dtype=(tl.uint32,) * 16,
            is_pure=True,
            pack=1,
        )
        tl.store(out_base +  0, (p0  & 0xFFFF).to(tl.uint16).to(OUT_DTYPE, bitcast=True))
        tl.store(out_base +  1, ((p0  >> 16) & 0xFFFF).to(tl.uint16).to(OUT_DTYPE, bitcast=True))
        tl.store(out_base +  2, (p1  & 0xFFFF).to(tl.uint16).to(OUT_DTYPE, bitcast=True))
        tl.store(out_base +  3, ((p1  >> 16) & 0xFFFF).to(tl.uint16).to(OUT_DTYPE, bitcast=True))
        tl.store(out_base +  4, (p2  & 0xFFFF).to(tl.uint16).to(OUT_DTYPE, bitcast=True))
        tl.store(out_base +  5, ((p2  >> 16) & 0xFFFF).to(tl.uint16).to(OUT_DTYPE, bitcast=True))
        tl.store(out_base +  6, (p3  & 0xFFFF).to(tl.uint16).to(OUT_DTYPE, bitcast=True))
        tl.store(out_base +  7, ((p3  >> 16) & 0xFFFF).to(tl.uint16).to(OUT_DTYPE, bitcast=True))
        tl.store(out_base +  8, (p4  & 0xFFFF).to(tl.uint16).to(OUT_DTYPE, bitcast=True))
        tl.store(out_base +  9, ((p4  >> 16) & 0xFFFF).to(tl.uint16).to(OUT_DTYPE, bitcast=True))
        tl.store(out_base + 10, (p5  & 0xFFFF).to(tl.uint16).to(OUT_DTYPE, bitcast=True))
        tl.store(out_base + 11, ((p5  >> 16) & 0xFFFF).to(tl.uint16).to(OUT_DTYPE, bitcast=True))
        tl.store(out_base + 12, (p6  & 0xFFFF).to(tl.uint16).to(OUT_DTYPE, bitcast=True))
        tl.store(out_base + 13, ((p6  >> 16) & 0xFFFF).to(tl.uint16).to(OUT_DTYPE, bitcast=True))
        tl.store(out_base + 14, (p7  & 0xFFFF).to(tl.uint16).to(OUT_DTYPE, bitcast=True))
        tl.store(out_base + 15, ((p7  >> 16) & 0xFFFF).to(tl.uint16).to(OUT_DTYPE, bitcast=True))
        tl.store(out_base + 16, (p8  & 0xFFFF).to(tl.uint16).to(OUT_DTYPE, bitcast=True))
        tl.store(out_base + 17, ((p8  >> 16) & 0xFFFF).to(tl.uint16).to(OUT_DTYPE, bitcast=True))
        tl.store(out_base + 18, (p9  & 0xFFFF).to(tl.uint16).to(OUT_DTYPE, bitcast=True))
        tl.store(out_base + 19, ((p9  >> 16) & 0xFFFF).to(tl.uint16).to(OUT_DTYPE, bitcast=True))
        tl.store(out_base + 20, (p10 & 0xFFFF).to(tl.uint16).to(OUT_DTYPE, bitcast=True))
        tl.store(out_base + 21, ((p10 >> 16) & 0xFFFF).to(tl.uint16).to(OUT_DTYPE, bitcast=True))
        tl.store(out_base + 22, (p11 & 0xFFFF).to(tl.uint16).to(OUT_DTYPE, bitcast=True))
        tl.store(out_base + 23, ((p11 >> 16) & 0xFFFF).to(tl.uint16).to(OUT_DTYPE, bitcast=True))
        tl.store(out_base + 24, (p12 & 0xFFFF).to(tl.uint16).to(OUT_DTYPE, bitcast=True))
        tl.store(out_base + 25, ((p12 >> 16) & 0xFFFF).to(tl.uint16).to(OUT_DTYPE, bitcast=True))
        tl.store(out_base + 26, (p13 & 0xFFFF).to(tl.uint16).to(OUT_DTYPE, bitcast=True))
        tl.store(out_base + 27, ((p13 >> 16) & 0xFFFF).to(tl.uint16).to(OUT_DTYPE, bitcast=True))
        tl.store(out_base + 28, (p14 & 0xFFFF).to(tl.uint16).to(OUT_DTYPE, bitcast=True))
        tl.store(out_base + 29, ((p14 >> 16) & 0xFFFF).to(tl.uint16).to(OUT_DTYPE, bitcast=True))
        tl.store(out_base + 30, (p15 & 0xFFFF).to(tl.uint16).to(OUT_DTYPE, bitcast=True))
        tl.store(out_base + 31, ((p15 >> 16) & 0xFFFF).to(tl.uint16).to(OUT_DTYPE, bitcast=True))


def mxfp6_to_f32_triton_hw(
    fp6_data: torch.Tensor,
    scales: torch.Tensor,
    M: int,
    K: int,
    fmt: str = "e2m3",
    group_size: int = 32,
    num_warps: int = 4,
    out_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    assert fmt in ("e2m3", "e3m2"), f"Unsupported MXFP6 format: {fmt}"
    assert fp6_data.dtype == torch.uint32, f"Expected uint32 fp6_data, got {fp6_data.dtype}"
    assert scales.dtype == torch.uint8, f"Expected uint8 scales, got {scales.dtype}"
    assert group_size == 32, "MXFP6 upcast requires group_size=32"

    n_groups = K // group_size
    assert fp6_data.shape == (M, n_groups * 6), \
        f"Expected fp6_data shape ({M}, {n_groups * 6}), got {fp6_data.shape}"

    out = torch.empty((M, K), dtype=out_dtype, device=fp6_data.device)
    grid = (M, n_groups)

    if fmt == "e2m3":
        kernel = mxfp6e2_to_f32_kernel_hw
        asm_map = _FP6E2_ASM
    else:
        kernel = mxfp6e3_to_f32_kernel_hw
        asm_map = _FP6E3_ASM
    # f32 path uses a hard-coded instruction in the kernel; pick a valid entry as a dummy
    asm_instr = asm_map.get(out_dtype, next(iter(asm_map.values())))

    kernel[grid](
        fp6_data, scales, out,
        M, K,
        fp6_data.stride(0),
        scales.stride(0), scales.stride(1),
        out.stride(0),
        GROUP_SIZE=group_size,
        OUT_DTYPE=_TORCH_TO_TL_DTYPE[out_dtype],
        ASM_INSTR=asm_instr,
        num_warps=num_warps,
    )

    return out


def _apply_scales(A_ref_f32, scales, M, K, GROUP_SIZE):
    """Apply E8M0 scales to a per-element SW-decoded tensor."""
    n_groups = K // GROUP_SIZE
    scale_factors = torch.pow(2.0, scales.to(torch.float32).unsqueeze(-1) - 127)
    return (A_ref_f32.view(M, n_groups, GROUP_SIZE) * scale_factors).view(M, K)


def _print_errors(A_f32_hw, A_ref_scaled, A, fmt_label):
    hw_vs_ref_max  = torch.max(torch.abs(A_f32_hw - A_ref_scaled)).item()
    hw_vs_ref_mean = torch.mean(torch.abs(A_f32_hw - A_ref_scaled)).item()
    print(f"   HW vs SW Reference:")
    print(f"      Max error:  {hw_vs_ref_max:.6f}")
    print(f"      Mean error: {hw_vs_ref_mean:.6f}")

    quant_max  = torch.max(torch.abs(A_f32_hw - A)).item()
    quant_mean = torch.mean(torch.abs(A_f32_hw - A)).item()
    print(f"   Quantization error ({fmt_label} roundtrip vs original FP32):")
    print(f"      Max error:  {quant_max:.6f}")
    print(f"      Mean error: {quant_mean:.6f}")


def main():
    """Test MXFP4/MXFP8/MXFP6 downcast -> upcast pipeline with hardware acceleration."""
    torch.manual_seed(42)

    M, K = 8192, 8192
    GROUP_SIZE = 32

    A = torch.randn((M, K), device="cuda", dtype=torch.float32)

    # (label, fmt, downcast_fn, upcast_fn, sw_ref_fn, preprocess_fn)
    # upcast_fn signature: (data, scales, fmt, out_dtype) -> Tensor
    configs = [
        ("MXFP4", "e2m1",
         lambda fmt: f32_to_mxfp4_triton(A, fmt=fmt, group_size=GROUP_SIZE, method='hw'),
         lambda data, scales, fmt, odt: mxfp4_to_f32_triton_hw(data, scales, M, K, group_size=GROUP_SIZE, out_dtype=odt),
         lambda data, fmt: fp4_e2m1_to_fp32_sw(data),
         lambda data: data),
        ("MXFP8", "e4m3",
         lambda fmt: f32_to_mxfp8_triton(A, fmt=fmt, group_size=GROUP_SIZE, method='hw'),
         lambda data, scales, fmt, odt: mxfp8_to_f32_triton_hw(data, scales, M, K, fmt=fmt, group_size=GROUP_SIZE, out_dtype=odt),
         lambda data, fmt: fp8_e4m3_to_fp32_sw(data),
         lambda data: data.view(torch.uint8)),
        ("MXFP8", "e5m2",
         lambda fmt: f32_to_mxfp8_triton(A, fmt=fmt, group_size=GROUP_SIZE, method='hw'),
         lambda data, scales, fmt, odt: mxfp8_to_f32_triton_hw(data, scales, M, K, fmt=fmt, group_size=GROUP_SIZE, out_dtype=odt),
         lambda data, fmt: fp8_e5m2_to_fp32_sw(data),
         lambda data: data.view(torch.uint8)),
        ("MXFP6", "e2m3",
         lambda fmt: f32_to_mxfp6_triton(A, fmt=fmt, group_size=GROUP_SIZE, method='hw'),
         lambda data, scales, fmt, odt: mxfp6_to_f32_triton_hw(data, scales, M, K, fmt=fmt, group_size=GROUP_SIZE, out_dtype=odt),
         lambda data, fmt: fp6_e2m3_to_fp32_sw(data),
         lambda data: data),
        ("MXFP6", "e3m2",
         lambda fmt: f32_to_mxfp6_triton(A, fmt=fmt, group_size=GROUP_SIZE, method='hw'),
         lambda data, scales, fmt, odt: mxfp6_to_f32_triton_hw(data, scales, M, K, fmt=fmt, group_size=GROUP_SIZE, out_dtype=odt),
         lambda data, fmt: fp6_e3m2_to_fp32_sw(data),
         lambda data: data),
    ]

    out_dtypes = [torch.float32, torch.float16, torch.bfloat16]

    for label, fmt, downcast_fn, upcast_fn, sw_ref_fn, preprocess_fn in configs:
        downcast_time = tt.do_bench(lambda: downcast_fn(fmt), warmup=10, rep=100)
        raw_data, scales = downcast_fn(fmt)
        data = preprocess_fn(raw_data)
        A_ref_scaled = _apply_scales(sw_ref_fn(data, fmt), scales, M, K, GROUP_SIZE)

        for out_dtype in out_dtypes:
            dtype_name = {torch.float32: "f32", torch.float16: "f16", torch.bfloat16: "bf16"}[out_dtype]
            print(f"\n── {label} {fmt} → {dtype_name} ──")
            try:
                upcast_hw_time = tt.do_bench(lambda: upcast_fn(data, scales, fmt, out_dtype), warmup=10, rep=100)
                A_hw = upcast_fn(data, scales, fmt, out_dtype)
            except AssertionError as e:
                print(f"   Skipped: {e}")
                continue
            print(f"   Downcast time:  {downcast_time:.4f} ms")
            print(f"   HW Upcast time: {upcast_hw_time:.4f} ms")
            print(f"   Output shape: {A_hw.shape}, dtype: {A_hw.dtype}")
            # Cast reference and original through out_dtype so the comparison
            # is in the same precision space as the HW output.
            ref = A_ref_scaled.to(out_dtype).float()
            orig = A.to(out_dtype).float()
            _print_errors(A_hw.float(), ref, orig, f"{label} {fmt}")

if __name__ == "__main__":
    main()
