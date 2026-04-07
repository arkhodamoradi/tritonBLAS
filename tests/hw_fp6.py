"""
Triton kernel for F32 to MXFP8 conversion using absmax scaling.

This implementation uses V_CVT_SCALEF32_PK_FP8_F32 and V_CVT_SCALEF32_PK_BF8_F32 
hardware instructions for accelerated FP8 conversion with proper rounding.
"""

import torch
import triton
import triton.language as tl

@triton.jit
def get_exponent(x, offset):
    absmax = tl.max(tl.abs(x), axis=1)
    
    # Extract F32 exponent from absmax
    absmax_bits = absmax.to(tl.int32, bitcast=True)
    f32_exp = (absmax_bits >> 23) & 0xFF
    
    # Compute E8M0 scale: scale = f32_exp - 8
    _exp = f32_exp - offset
    _exp = tl.maximum(_exp, 0)
    _exo = tl.minimum(_exp, 255)
    return _exp

@triton.jit
def f32_to_mxfp8e4_sr_kernel_hw(
    x_ptr,           
    out_ptr,        
    scale_ptr,      
    M,              
    K,              
    stride_xm,       
    stride_xk,      
    stride_outm,     
    stride_outk,    
    stride_sm,      
    stride_sg,      
    GROUP_SIZE: tl.constexpr,
    GROUPS_PER_BLOCK: tl.constexpr,
):
    """
    Triton kernel for F32 to MXFP8 conversion using hardware instructions.
    Processes multiple groups per workgroup for 100% thread utilization.
    With GROUPS_PER_BLOCK=2 and GROUP_SIZE=32, we process 64 elements with 64 threads.
    """
    pid_m = tl.program_id(0) 
    pid_g = tl.program_id(1)  

    # Total elements per block = GROUP_SIZE * GROUPS_PER_BLOCK 
    BLOCK_SIZE: tl.constexpr = GROUP_SIZE * GROUPS_PER_BLOCK
    
    block_start = pid_g * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    x_ptrs = x_ptr + pid_m * stride_xm + offsets * stride_xk
    mask = offsets < K
    x = tl.load(x_ptrs, mask=mask, other=0.0)
    
    # Reshape to [GROUPS_PER_BLOCK, GROUP_SIZE] to compute per-group absmax
    x_grouped = tl.reshape(x, (GROUPS_PER_BLOCK, GROUP_SIZE))
    
    scale_exp = get_exponent(x_grouped, 8)
    
    # Store scales for each group
    group_indices = tl.arange(0, GROUPS_PER_BLOCK)
    scale_ptrs = scale_ptr + pid_m * stride_sm + (pid_g * GROUPS_PER_BLOCK + group_indices) * stride_sg
    tl.store(scale_ptrs, scale_exp.to(tl.uint8), mask=(pid_g * GROUPS_PER_BLOCK + group_indices) < (K // GROUP_SIZE))
    
    # Broadcast scale_exp to match elements
    scale_exp_expanded = tl.reshape(scale_exp, (GROUPS_PER_BLOCK, 1))
    scale_exp_broadcast = tl.broadcast_to(scale_exp_expanded, (GROUPS_PER_BLOCK, GROUP_SIZE))
    scale_exp_flat = tl.reshape(scale_exp_broadcast, (BLOCK_SIZE,)) 
    
    scale_f32 = (scale_exp_flat.to(tl.uint32) << 23)
    
    sr_seed = 0.0
    fp8 = tl.inline_asm_elementwise(
            "v_cvt_scalef32_sr_fp8_f32 $0, $1, $2, $3",
            "=v,v,v,v",
            args=[x, sr_seed, scale_f32],
            dtype=tl.uint16,
            is_pure=True,
            pack=1,
        )
    
    fp8 = fp8.to(tl.uint8)
    
    # e4m3: NaN is 0x7F (positive) or 0xFF (negative)
    # Replace with max value 0x7E (448) or 0xFE (-448)
    fp8 = tl.where((fp8 & 0x7F) == 0x7F, fp8 - 1, fp8)

    out_ptrs = out_ptr + pid_m * stride_outm + offsets * stride_outk
    tl.store(out_ptrs, fp8, mask=mask)

@triton.jit
def f32_to_mxfp8e5_sr_kernel_hw(
    x_ptr,           
    out_ptr,        
    scale_ptr,      
    M,              
    K,              
    stride_xm,       
    stride_xk,      
    stride_outm,     
    stride_outk,    
    stride_sm,      
    stride_sg,      
    GROUP_SIZE: tl.constexpr,
    GROUPS_PER_BLOCK: tl.constexpr,
):
    """
    Triton kernel for F32 to MXFP8 conversion using hardware instructions.
    Processes multiple groups per workgroup for 100% thread utilization.
    With GROUPS_PER_BLOCK=2 and GROUP_SIZE=32, we process 64 elements with 64 threads.
    """
    pid_m = tl.program_id(0) 
    pid_g = tl.program_id(1)  

    # Total elements per block = GROUP_SIZE * GROUPS_PER_BLOCK 
    BLOCK_SIZE: tl.constexpr = GROUP_SIZE * GROUPS_PER_BLOCK
    
    block_start = pid_g * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    x_ptrs = x_ptr + pid_m * stride_xm + offsets * stride_xk
    mask = offsets < K
    x = tl.load(x_ptrs, mask=mask, other=0.0)
    
    # Reshape to [GROUPS_PER_BLOCK, GROUP_SIZE] to compute per-group absmax
    x_grouped = tl.reshape(x, (GROUPS_PER_BLOCK, GROUP_SIZE))
    
    scale_exp = get_exponent(x_grouped, 15)
    
    # Store scales for each group
    group_indices = tl.arange(0, GROUPS_PER_BLOCK)
    scale_ptrs = scale_ptr + pid_m * stride_sm + (pid_g * GROUPS_PER_BLOCK + group_indices) * stride_sg
    tl.store(scale_ptrs, scale_exp.to(tl.uint8), mask=(pid_g * GROUPS_PER_BLOCK + group_indices) < (K // GROUP_SIZE))
    
    # Broadcast scale_exp to match elements
    scale_exp_expanded = tl.reshape(scale_exp, (GROUPS_PER_BLOCK, 1))
    scale_exp_broadcast = tl.broadcast_to(scale_exp_expanded, (GROUPS_PER_BLOCK, GROUP_SIZE))
    scale_exp_flat = tl.reshape(scale_exp_broadcast, (BLOCK_SIZE,)) 
    
    scale_f32 = (scale_exp_flat.to(tl.uint32) << 23)
    
    sr_seed = 0.0
    fp8 = tl.inline_asm_elementwise(
            "v_cvt_scalef32_sr_bf8_f32 $0, $1, $2, $3",
            "=v,v,v,v",
            args=[x, sr_seed, scale_f32],
            dtype=tl.uint16,
            is_pure=True,
            pack=1,
        )
    
    fp8 = fp8.to(tl.uint8)

    fp8 = tl.where((fp8 >= 0x7C) & (fp8 < 0x80), 0x7B, fp8)
    fp8 = tl.where(fp8 >= 0xFC, 0xFB, fp8)

    out_ptrs = out_ptr + pid_m * stride_outm + offsets * stride_outk
    tl.store(out_ptrs, fp8, mask=mask)

@triton.jit
def f32_to_mxfp8e4_rtne_kernel_hw(
    x_ptr,           
    out_ptr,        
    scale_ptr,      
    M,              
    K,              
    stride_xm,       
    stride_xk,      
    stride_outm,     
    stride_outk,    
    stride_sm,      
    stride_sg,      
    GROUP_SIZE: tl.constexpr,
    GROUPS_PER_BLOCK: tl.constexpr,  
):
    """
    Triton kernel for F32 to MXFP8E4M3 conversion using hardware instructions.
    Processes multiple groups per workgroup for better thread utilization.
    """
    pid_m = tl.program_id(0) 
    pid_g = tl.program_id(1)  
    
    # Total elements per block = GROUP_SIZE * GROUPS_PER_BLOCK (e.g., 32 * 2 = 64)
    BLOCK_SIZE: tl.constexpr = GROUP_SIZE * GROUPS_PER_BLOCK
    
    block_start = pid_g * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    x_ptrs = x_ptr + pid_m * stride_xm + offsets * stride_xk
    mask = offsets < K
    x = tl.load(x_ptrs, mask=mask, other=0.0)
    
    # Reshape to [GROUPS_PER_BLOCK, GROUP_SIZE] to compute per-group absmax
    x_grouped = tl.reshape(x, (GROUPS_PER_BLOCK, GROUP_SIZE))
    
    scale_exp = get_exponent(x_grouped, 8)
    
    # Store scales for each group
    group_indices = tl.arange(0, GROUPS_PER_BLOCK)
    scale_ptrs = scale_ptr + pid_m * stride_sm + (pid_g * GROUPS_PER_BLOCK + group_indices) * stride_sg
    tl.store(scale_ptrs, scale_exp.to(tl.uint8), mask=(pid_g * GROUPS_PER_BLOCK + group_indices) < (K // GROUP_SIZE))
    
    # Broadcast scale_exp to match elements
    # Each group has GROUP_SIZE//2 pairs
    scale_exp_expanded = tl.reshape(scale_exp, (GROUPS_PER_BLOCK, 1))
    scale_exp_broadcast = tl.broadcast_to(scale_exp_expanded, (GROUPS_PER_BLOCK, GROUP_SIZE // 2))
    scale_exp_flat = tl.reshape(scale_exp_broadcast, (BLOCK_SIZE // 2,))
    
    scale_f32 = (scale_exp_flat.to(tl.uint32) << 23)
    
    # Get pairs of values from x
    pair_offsets = tl.arange(0, BLOCK_SIZE // 2)
    x_pairs = tl.reshape(x, (BLOCK_SIZE // 2, 2))
    x_even, x_odd = tl.split(x_pairs) 

    fp8_packed = tl.inline_asm_elementwise(
        "v_cvt_scalef32_pk_fp8_f32 $0, $1, $2, $3",
        "=v,v,v,v",
        args=[x_even, x_odd, scale_f32],
        dtype=tl.uint16,
        is_pure=True,
        pack=1,
    )
    
    # Extract individual FP8 bytes
    fp8_0 = (fp8_packed & 0xFF).to(tl.uint8)
    fp8_1 = ((fp8_packed >> 8) & 0xFF).to(tl.uint8)
    
    # Clamp NaN to max valid FP8 values
    fp8_0 = tl.where((fp8_0 & 0x7F) == 0x7F, fp8_0 - 1, fp8_0)
    fp8_1 = tl.where((fp8_1 & 0x7F) == 0x7F, fp8_1 - 1, fp8_1)

    fp8_interleaved = tl.interleave(fp8_0, fp8_1) 

    # Store the interleaved uint8 values directly
    out_ptrs = out_ptr + pid_m * stride_outm + (block_start + tl.arange(0, BLOCK_SIZE)) * stride_outk
    tl.store(out_ptrs, fp8_interleaved, mask=(block_start + tl.arange(0, BLOCK_SIZE)) < K)

@triton.jit
def f32_to_mxfp8e5_rtne_kernel_hw(
    x_ptr,           
    out_ptr,        
    scale_ptr,      
    M,              
    K,              
    stride_xm,       
    stride_xk,      
    stride_outm,     
    stride_outk,    
    stride_sm,      
    stride_sg,      
    GROUP_SIZE: tl.constexpr,  
    GROUPS_PER_BLOCK: tl.constexpr,
):
    """
    Triton kernel for F32 to MXFP8 conversion using hardware instructions.
    """
    pid_m = tl.program_id(0) 
    pid_g = tl.program_id(1) 
    
    # Total elements per block = GROUP_SIZE * GROUPS_PER_BLOCK (e.g., 32 * 2 = 64)
    BLOCK_SIZE: tl.constexpr = GROUP_SIZE * GROUPS_PER_BLOCK

    block_start = pid_g * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    x_ptrs = x_ptr + pid_m * stride_xm + offsets * stride_xk
    mask = offsets < K
    x = tl.load(x_ptrs, mask=mask, other=0.0)
    
    # Reshape to [GROUPS_PER_BLOCK, GROUP_SIZE] to compute per-group absmax
    x_grouped = tl.reshape(x, (GROUPS_PER_BLOCK, GROUP_SIZE))
    
    scale_exp = get_exponent(x_grouped, 15)
    
    # Store scale (as uint8)
    group_indices = tl.arange(0, GROUPS_PER_BLOCK)
    scale_ptrs = scale_ptr + pid_m * stride_sm + (pid_g * GROUPS_PER_BLOCK + group_indices) * stride_sg
    tl.store(scale_ptrs, scale_exp.to(tl.uint8), mask=(pid_g * GROUPS_PER_BLOCK + group_indices) < (K // GROUP_SIZE))

    # Broadcast scale_exp to match elements
    # Each group has GROUP_SIZE//2 pairs
    scale_exp_expanded = tl.reshape(scale_exp, (GROUPS_PER_BLOCK, 1))
    scale_exp_broadcast = tl.broadcast_to(scale_exp_expanded, (GROUPS_PER_BLOCK, GROUP_SIZE // 2))
    scale_exp_flat = tl.reshape(scale_exp_broadcast, (BLOCK_SIZE // 2,))
    
    scale_f32 = (scale_exp_flat.to(tl.uint32) << 23)
    
    
    # Get pairs of values from x
    pair_offsets = tl.arange(0, BLOCK_SIZE // 2)
    x_pairs = tl.reshape(x, (BLOCK_SIZE // 2, 2))
    x_even, x_odd = tl.split(x_pairs) 
    
    fp8_packed = tl.inline_asm_elementwise(
        "v_cvt_scalef32_pk_bf8_f32 $0, $1, $2, $3",
        "=v,v,v,v",
        args=[x_even, x_odd, scale_f32],
        dtype=tl.uint16,
        is_pure=True,
        pack=1,
    )
    
    # Extract individual FP8 bytes from the packed result
    # Lower byte = fp8(src0), Upper byte = fp8(src1)
    fp8_0 = (fp8_packed & 0xFF).to(tl.uint8)
    fp8_1 = ((fp8_packed >> 8) & 0xFF).to(tl.uint8)
    
    # Clamp NaN/Inf to max valid FP8 values
    # e5m2: Inf is 0x7C/0xFC, NaN is 0x7D-7F/0xFD-FF
    # Replace with max value 0x7B (57344) or 0xFB (-57344)
    fp8_0 = tl.where((fp8_0 >= 0x7C) & (fp8_0 < 0x80), 0x7B, fp8_0)
    fp8_0 = tl.where(fp8_0 >= 0xFC, 0xFB, fp8_0)
    fp8_1 = tl.where((fp8_1 >= 0x7C) & (fp8_1 < 0x80), 0x7B, fp8_1)
    fp8_1 = tl.where(fp8_1 >= 0xFC, 0xFB, fp8_1)

    fp8_interleaved = tl.interleave(fp8_0, fp8_1) 

    # Store the interleaved uint8 values directly
    out_ptrs = out_ptr + pid_m * stride_outm + (block_start + tl.arange(0, BLOCK_SIZE)) * stride_outk
    tl.store(out_ptrs, fp8_interleaved, mask=(block_start + tl.arange(0, BLOCK_SIZE)) < K)


@triton.jit
def f32_to_mxfp6e2_rtne_kernel_hw(
    x_ptr,
    out_ptr,
    scale_ptr,
    M,
    K,
    stride_xm,
    stride_xk,
    stride_outm,
    stride_outk,
    stride_sm,
    stride_sg,
    GROUP_SIZE: tl.constexpr,
    GROUPS_PER_BLOCK: tl.constexpr,
):
    """
    FP6 E2M3 conversion kernel using v_cvt_scalef32_2xpk16_fp6_f32 hardware instruction.
    This instruction uses RTNE (Round to Nearest Even) rounding mode.
    Uses v_mov_b32 to arrange registers into consecutive VGPRs before calling the instruction.
    
    The instruction requires:
    - Output: v[0:5] - 6 consecutive VGPRs (192 bits = 32 FP6 values)
    - Input1: v[6:21] - 16 consecutive VGPRs (16 float values for first half)
    - Input2: v[22:37] - 16 consecutive VGPRs (16 float values for second half)
    - Scale: v38 - 1 VGPR (f32)
    """
    pid_m = tl.program_id(0)
    pid_g = tl.program_id(1)

    # Total elements per block = GROUP_SIZE * GROUPS_PER_BLOCK
    BLOCK_SIZE: tl.constexpr = GROUP_SIZE * GROUPS_PER_BLOCK

    block_start = pid_g * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    x_ptrs = x_ptr + pid_m * stride_xm + offsets * stride_xk
    mask = offsets < K
    x = tl.load(x_ptrs, mask=mask, other=0.0)

    # Reshape to [GROUPS_PER_BLOCK, GROUP_SIZE] to compute per-group absmax
    x_grouped = tl.reshape(x, (GROUPS_PER_BLOCK, GROUP_SIZE))

    # FP6 E2M3 offset is 2 (not 8 like FP8)
    scale_exp = get_exponent(x_grouped, 2)

    # Store scales for each group
    group_indices = tl.arange(0, GROUPS_PER_BLOCK)
    scale_ptrs = scale_ptr + pid_m * stride_sm + (pid_g * GROUPS_PER_BLOCK + group_indices) * stride_sg
    tl.store(scale_ptrs, scale_exp.to(tl.uint8), mask=(pid_g * GROUPS_PER_BLOCK + group_indices) < (K // GROUP_SIZE))

    # For HW instruction, we need to process 32 elements at a time
    # Load 32 floats as individual scalars for the inline asm
    a0 = tl.load(x_ptr + pid_m * stride_xm + block_start * stride_xk + 0)
    a1 = tl.load(x_ptr + pid_m * stride_xm + block_start * stride_xk + 1)
    a2 = tl.load(x_ptr + pid_m * stride_xm + block_start * stride_xk + 2)
    a3 = tl.load(x_ptr + pid_m * stride_xm + block_start * stride_xk + 3)
    a4 = tl.load(x_ptr + pid_m * stride_xm + block_start * stride_xk + 4)
    a5 = tl.load(x_ptr + pid_m * stride_xm + block_start * stride_xk + 5)
    a6 = tl.load(x_ptr + pid_m * stride_xm + block_start * stride_xk + 6)
    a7 = tl.load(x_ptr + pid_m * stride_xm + block_start * stride_xk + 7)
    a8 = tl.load(x_ptr + pid_m * stride_xm + block_start * stride_xk + 8)
    a9 = tl.load(x_ptr + pid_m * stride_xm + block_start * stride_xk + 9)
    a10 = tl.load(x_ptr + pid_m * stride_xm + block_start * stride_xk + 10)
    a11 = tl.load(x_ptr + pid_m * stride_xm + block_start * stride_xk + 11)
    a12 = tl.load(x_ptr + pid_m * stride_xm + block_start * stride_xk + 12)
    a13 = tl.load(x_ptr + pid_m * stride_xm + block_start * stride_xk + 13)
    a14 = tl.load(x_ptr + pid_m * stride_xm + block_start * stride_xk + 14)
    a15 = tl.load(x_ptr + pid_m * stride_xm + block_start * stride_xk + 15)
    a16 = tl.load(x_ptr + pid_m * stride_xm + block_start * stride_xk + 16)
    a17 = tl.load(x_ptr + pid_m * stride_xm + block_start * stride_xk + 17)
    a18 = tl.load(x_ptr + pid_m * stride_xm + block_start * stride_xk + 18)
    a19 = tl.load(x_ptr + pid_m * stride_xm + block_start * stride_xk + 19)
    a20 = tl.load(x_ptr + pid_m * stride_xm + block_start * stride_xk + 20)
    a21 = tl.load(x_ptr + pid_m * stride_xm + block_start * stride_xk + 21)
    a22 = tl.load(x_ptr + pid_m * stride_xm + block_start * stride_xk + 22)
    a23 = tl.load(x_ptr + pid_m * stride_xm + block_start * stride_xk + 23)
    a24 = tl.load(x_ptr + pid_m * stride_xm + block_start * stride_xk + 24)
    a25 = tl.load(x_ptr + pid_m * stride_xm + block_start * stride_xk + 25)
    a26 = tl.load(x_ptr + pid_m * stride_xm + block_start * stride_xk + 26)
    a27 = tl.load(x_ptr + pid_m * stride_xm + block_start * stride_xk + 27)
    a28 = tl.load(x_ptr + pid_m * stride_xm + block_start * stride_xk + 28)
    a29 = tl.load(x_ptr + pid_m * stride_xm + block_start * stride_xk + 29)
    a30 = tl.load(x_ptr + pid_m * stride_xm + block_start * stride_xk + 30)
    a31 = tl.load(x_ptr + pid_m * stride_xm + block_start * stride_xk + 31)
    
    # Get scale for this group (first group in block)
    scale_val = scale_exp  # This is already computed above
    scale_f32 = (scale_val.to(tl.uint32) << 23)
    
    # Load scale as scalar
    scale_scalar = tl.load(scale_ptr + pid_m * stride_sm + pid_g * stride_sg).to(tl.uint32)
    scale_f32_scalar = (scale_scalar << 23)
        
    # Use v_mov_b32 to arrange registers into consecutive VGPRs, call FP6 instruction, move results back
    # Using v_cvt_scalef32_2xpk16_fp6_f32 which uses RTNE rounding (not stochastic)
    # This instruction takes two sets of 16 floats and produces 32 FP6 values
    (r0, r1, r2, r3, r4, r5) = tl.inline_asm_elementwise(
        asm="""
        // Move first 16 input floats to consecutive registers v40-v55
        v_mov_b32 v40, $6
        v_mov_b32 v41, $7
        v_mov_b32 v42, $8
        v_mov_b32 v43, $9
        v_mov_b32 v44, $10
        v_mov_b32 v45, $11
        v_mov_b32 v46, $12
        v_mov_b32 v47, $13
        v_mov_b32 v48, $14
        v_mov_b32 v49, $15
        v_mov_b32 v50, $16
        v_mov_b32 v51, $17
        v_mov_b32 v52, $18
        v_mov_b32 v53, $19
        v_mov_b32 v54, $20
        v_mov_b32 v55, $21
        
        // Move second 16 input floats to consecutive registers v56-v71
        v_mov_b32 v56, $22
        v_mov_b32 v57, $23
        v_mov_b32 v58, $24
        v_mov_b32 v59, $25
        v_mov_b32 v60, $26
        v_mov_b32 v61, $27
        v_mov_b32 v62, $28
        v_mov_b32 v63, $29
        v_mov_b32 v64, $30
        v_mov_b32 v65, $31
        v_mov_b32 v66, $32
        v_mov_b32 v67, $33
        v_mov_b32 v68, $34
        v_mov_b32 v69, $35
        v_mov_b32 v70, $36
        v_mov_b32 v71, $37
        
        // Call FP6 conversion with RTNE rounding (2xpk16 version)
        // Output: v[34:39], Input1: v[40:55], Input2: v[56:71], Scale: $38
        v_cvt_scalef32_2xpk16_fp6_f32 v[34:39], v[40:55], v[56:71], $38
        
        // Move 6 output i32s back to output registers
        v_mov_b32 $0, v34
        v_mov_b32 $1, v35
        v_mov_b32 $2, v36
        v_mov_b32 $3, v37
        v_mov_b32 $4, v38
        v_mov_b32 $5, v39
        """,
        constraints=(
            "=v,=v,=v,=v,=v,=v,"  # 6 outputs ($0-$5)
            "v,v,v,v,v,v,v,v,v,v,v,v,v,v,v,v,"  # 16 inputs ($6-$21) - first half
            "v,v,v,v,v,v,v,v,v,v,v,v,v,v,v,v,"  # 16 more inputs ($22-$37) - second half
            "v,"  # scale ($38)
            "~{v34},~{v35},~{v36},~{v37},~{v38},~{v39},"  # clobber output registers
            "~{v40},~{v41},~{v42},~{v43},~{v44},~{v45},~{v46},~{v47},"  
            "~{v48},~{v49},~{v50},~{v51},~{v52},~{v53},~{v54},~{v55}," 
            "~{v56},~{v57},~{v58},~{v59},~{v60},~{v61},~{v62},~{v63}," 
            "~{v64},~{v65},~{v66},~{v67},~{v68},~{v69},~{v70},~{v71}" 
        ),
        args=[ # Interleave the inputs to match the instruction requirements
            a0, a2, a4, a6, a8, a10, a12, a14,
            a16, a18, a20, a22, a24, a26, a28, a30,
            a1, a3, a5, a7, a9, a11, a13, a15,
            a17, a19, a21, a23, a25, a27, a29, a31,
            scale_f32_scalar
        ],
        dtype=(tl.int32, tl.int32, tl.int32, tl.int32, tl.int32, tl.int32),
        is_pure=False,
        pack=1,
    )
    
    # Store the 6 packed i32 results (192 bits = 32 FP6 values)
    # Each i32 contains ~5.33 FP6 values, total 6 i32 = 192 bits = 32 FP6 values
    tl.store(out_ptr + pid_m * stride_outm + (block_start // 32) * 6 + 0, r0)
    tl.store(out_ptr + pid_m * stride_outm + (block_start // 32) * 6 + 1, r1)
    tl.store(out_ptr + pid_m * stride_outm + (block_start // 32) * 6 + 2, r2)
    tl.store(out_ptr + pid_m * stride_outm + (block_start // 32) * 6 + 3, r3)
    tl.store(out_ptr + pid_m * stride_outm + (block_start // 32) * 6 + 4, r4)
    tl.store(out_ptr + pid_m * stride_outm + (block_start // 32) * 6 + 5, r5)

@triton.jit
def f32_to_mxfp6e3_rtne_kernel_hw(
    x_ptr,
    out_ptr,
    scale_ptr,
    M,
    K,
    stride_xm,
    stride_xk,
    stride_outm,
    stride_outk,
    stride_sm,
    stride_sg,
    GROUP_SIZE: tl.constexpr,
    GROUPS_PER_BLOCK: tl.constexpr,
):
    """
    FP6 E3M2 conversion kernel using v_cvt_scalef32_2xpk16_bf6_f32 hardware instruction.
    This instruction uses RTNE (Round to Nearest Even) rounding mode.
    Uses v_mov_b32 to arrange registers into consecutive VGPRs before calling the instruction.
    
    The instruction requires:
    - Output: v[0:5] - 6 consecutive VGPRs (192 bits = 32 FP6 values)
    - Input1: v[6:21] - 16 consecutive VGPRs (16 float values for first half)
    - Input2: v[22:37] - 16 consecutive VGPRs (16 float values for second half)
    - Scale: v38 - 1 VGPR (f32)
    """
    pid_m = tl.program_id(0)
    pid_g = tl.program_id(1)

    # Total elements per block = GROUP_SIZE * GROUPS_PER_BLOCK
    BLOCK_SIZE: tl.constexpr = GROUP_SIZE * GROUPS_PER_BLOCK

    block_start = pid_g * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    x_ptrs = x_ptr + pid_m * stride_xm + offsets * stride_xk
    mask = offsets < K
    x = tl.load(x_ptrs, mask=mask, other=0.0)

    # Reshape to [GROUPS_PER_BLOCK, GROUP_SIZE] to compute per-group absmax
    x_grouped = tl.reshape(x, (GROUPS_PER_BLOCK, GROUP_SIZE))

    # FP6 E3M2 offset is 4 (different from E2M3 which is 2)
    scale_exp = get_exponent(x_grouped, 4)

    # Store scales for each group
    group_indices = tl.arange(0, GROUPS_PER_BLOCK)
    scale_ptrs = scale_ptr + pid_m * stride_sm + (pid_g * GROUPS_PER_BLOCK + group_indices) * stride_sg
    tl.store(scale_ptrs, scale_exp.to(tl.uint8), mask=(pid_g * GROUPS_PER_BLOCK + group_indices) < (K // GROUP_SIZE))

    # For HW instruction, we need to process 32 elements at a time
    # Load 32 floats as individual scalars for the inline asm
    a0 = tl.load(x_ptr + pid_m * stride_xm + block_start * stride_xk + 0)
    a1 = tl.load(x_ptr + pid_m * stride_xm + block_start * stride_xk + 1)
    a2 = tl.load(x_ptr + pid_m * stride_xm + block_start * stride_xk + 2)
    a3 = tl.load(x_ptr + pid_m * stride_xm + block_start * stride_xk + 3)
    a4 = tl.load(x_ptr + pid_m * stride_xm + block_start * stride_xk + 4)
    a5 = tl.load(x_ptr + pid_m * stride_xm + block_start * stride_xk + 5)
    a6 = tl.load(x_ptr + pid_m * stride_xm + block_start * stride_xk + 6)
    a7 = tl.load(x_ptr + pid_m * stride_xm + block_start * stride_xk + 7)
    a8 = tl.load(x_ptr + pid_m * stride_xm + block_start * stride_xk + 8)
    a9 = tl.load(x_ptr + pid_m * stride_xm + block_start * stride_xk + 9)
    a10 = tl.load(x_ptr + pid_m * stride_xm + block_start * stride_xk + 10)
    a11 = tl.load(x_ptr + pid_m * stride_xm + block_start * stride_xk + 11)
    a12 = tl.load(x_ptr + pid_m * stride_xm + block_start * stride_xk + 12)
    a13 = tl.load(x_ptr + pid_m * stride_xm + block_start * stride_xk + 13)
    a14 = tl.load(x_ptr + pid_m * stride_xm + block_start * stride_xk + 14)
    a15 = tl.load(x_ptr + pid_m * stride_xm + block_start * stride_xk + 15)
    a16 = tl.load(x_ptr + pid_m * stride_xm + block_start * stride_xk + 16)
    a17 = tl.load(x_ptr + pid_m * stride_xm + block_start * stride_xk + 17)
    a18 = tl.load(x_ptr + pid_m * stride_xm + block_start * stride_xk + 18)
    a19 = tl.load(x_ptr + pid_m * stride_xm + block_start * stride_xk + 19)
    a20 = tl.load(x_ptr + pid_m * stride_xm + block_start * stride_xk + 20)
    a21 = tl.load(x_ptr + pid_m * stride_xm + block_start * stride_xk + 21)
    a22 = tl.load(x_ptr + pid_m * stride_xm + block_start * stride_xk + 22)
    a23 = tl.load(x_ptr + pid_m * stride_xm + block_start * stride_xk + 23)
    a24 = tl.load(x_ptr + pid_m * stride_xm + block_start * stride_xk + 24)
    a25 = tl.load(x_ptr + pid_m * stride_xm + block_start * stride_xk + 25)
    a26 = tl.load(x_ptr + pid_m * stride_xm + block_start * stride_xk + 26)
    a27 = tl.load(x_ptr + pid_m * stride_xm + block_start * stride_xk + 27)
    a28 = tl.load(x_ptr + pid_m * stride_xm + block_start * stride_xk + 28)
    a29 = tl.load(x_ptr + pid_m * stride_xm + block_start * stride_xk + 29)
    a30 = tl.load(x_ptr + pid_m * stride_xm + block_start * stride_xk + 30)
    a31 = tl.load(x_ptr + pid_m * stride_xm + block_start * stride_xk + 31)
    
    # Load scale as scalar
    scale_scalar = tl.load(scale_ptr + pid_m * stride_sm + pid_g * stride_sg).to(tl.uint32)
    scale_f32_scalar = (scale_scalar << 23)
    
    # Use v_mov_b32 to arrange registers into consecutive VGPRs, call FP6 instruction, move results back
    # Using v_cvt_scalef32_2xpk16_bf6_f32 which uses RTNE rounding (not stochastic)
    # This instruction takes two sets of 16 floats and produces 32 FP6 E3M2 values
    (r0, r1, r2, r3, r4, r5) = tl.inline_asm_elementwise(
        asm="""
        // Move first 16 input floats to consecutive registers v40-v55
        v_mov_b32 v40, $6
        v_mov_b32 v41, $7
        v_mov_b32 v42, $8
        v_mov_b32 v43, $9
        v_mov_b32 v44, $10
        v_mov_b32 v45, $11
        v_mov_b32 v46, $12
        v_mov_b32 v47, $13
        v_mov_b32 v48, $14
        v_mov_b32 v49, $15
        v_mov_b32 v50, $16
        v_mov_b32 v51, $17
        v_mov_b32 v52, $18
        v_mov_b32 v53, $19
        v_mov_b32 v54, $20
        v_mov_b32 v55, $21
        
        // Move second 16 input floats to consecutive registers v56-v71
        v_mov_b32 v56, $22
        v_mov_b32 v57, $23
        v_mov_b32 v58, $24
        v_mov_b32 v59, $25
        v_mov_b32 v60, $26
        v_mov_b32 v61, $27
        v_mov_b32 v62, $28
        v_mov_b32 v63, $29
        v_mov_b32 v64, $30
        v_mov_b32 v65, $31
        v_mov_b32 v66, $32
        v_mov_b32 v67, $33
        v_mov_b32 v68, $34
        v_mov_b32 v69, $35
        v_mov_b32 v70, $36
        v_mov_b32 v71, $37
        
        // Call FP6 E3M2 conversion with RTNE rounding (2xpk16 version)
        // Output: v[34:39], Input1: v[40:55], Input2: v[56:71], Scale: $38
        v_cvt_scalef32_2xpk16_bf6_f32 v[34:39], v[40:55], v[56:71], $38
        
        // Move 6 output i32s back to output registers
        v_mov_b32 $0, v34
        v_mov_b32 $1, v35
        v_mov_b32 $2, v36
        v_mov_b32 $3, v37
        v_mov_b32 $4, v38
        v_mov_b32 $5, v39
        """,
        constraints=(
            "=v,=v,=v,=v,=v,=v,"  # 6 outputs ($0-$5)
            "v,v,v,v,v,v,v,v,v,v,v,v,v,v,v,v,"  # 16 inputs ($6-$21) - first half
            "v,v,v,v,v,v,v,v,v,v,v,v,v,v,v,v,"  # 16 more inputs ($22-$37) - second half
            "v,"  # scale ($38)
            "~{v34},~{v35},~{v36},~{v37},~{v38},~{v39},"  # clobber output registers
            "~{v40},~{v41},~{v42},~{v43},~{v44},~{v45},~{v46},~{v47},"  
            "~{v48},~{v49},~{v50},~{v51},~{v52},~{v53},~{v54},~{v55},"  
            "~{v56},~{v57},~{v58},~{v59},~{v60},~{v61},~{v62},~{v63}," 
            "~{v64},~{v65},~{v66},~{v67},~{v68},~{v69},~{v70},~{v71}"  
        ),
        args=[ # Interleave the inputs to match the instruction requirements
            a0, a2, a4, a6, a8, a10, a12, a14,
            a16, a18, a20, a22, a24, a26, a28, a30,
            a1, a3, a5, a7, a9, a11, a13, a15,
            a17, a19, a21, a23, a25, a27, a29, a31,
            scale_f32_scalar
        ],
        dtype=(tl.int32, tl.int32, tl.int32, tl.int32, tl.int32, tl.int32),
        is_pure=False,
        pack=1,
    )
    
    # Store the 6 packed i32 results (192 bits = 32 FP6 values)
    tl.store(out_ptr + pid_m * stride_outm + (block_start // 32) * 6 + 0, r0)
    tl.store(out_ptr + pid_m * stride_outm + (block_start // 32) * 6 + 1, r1)
    tl.store(out_ptr + pid_m * stride_outm + (block_start // 32) * 6 + 2, r2)
    tl.store(out_ptr + pid_m * stride_outm + (block_start // 32) * 6 + 3, r3)
    tl.store(out_ptr + pid_m * stride_outm + (block_start // 32) * 6 + 4, r4)
    tl.store(out_ptr + pid_m * stride_outm + (block_start // 32) * 6 + 5, r5)

@triton.jit
def f32_to_mxfp8_kernel_sw(
    x_ptr,           
    out_ptr,        
    scale_ptr,      
    M,              
    K,              
    stride_xm,      
    stride_xk,      
    stride_outm,    
    stride_outk,    
    stride_sm,      
    stride_sg,      
    GROUP_SIZE: tl.constexpr, 
    FP8_MAX: tl.constexpr,    
    FP8_EXP_OFFSET: tl.constexpr,  
    BLOCK_M: tl.constexpr,    
):
    pid_m = tl.program_id(0)  
    pid_g = tl.program_id(1)  
    
    group_start = pid_g * GROUP_SIZE
    offsets = group_start + tl.arange(0, GROUP_SIZE)
    
    x_ptrs = x_ptr + pid_m * stride_xm + offsets * stride_xk
    mask = offsets < K
    x = tl.load(x_ptrs, mask=mask, other=0.0)
    
    absmax = tl.max(tl.abs(x))
    
    absmax_bits = absmax.to(tl.int32, bitcast=True)
    f32_exp = (absmax_bits >> 23) & 0xFF
    
    scale_exp = f32_exp - FP8_EXP_OFFSET
    scale_exp = tl.maximum(scale_exp, 0)
    scale_exp = tl.minimum(scale_exp, 255)
    
    scale_ptr_out = scale_ptr + pid_m * stride_sm + pid_g * stride_sg
    tl.store(scale_ptr_out, scale_exp.to(tl.uint8))
    
    # Compute scale factor: 2^(scale_exp - 127)
    scale_factor = tl.exp2((scale_exp - 127).to(tl.float32))
    
    # Divide input by scale factor to normalize
    x_scaled = x / scale_factor
    
    # Clamp to FP8 range
    x_scaled = tl.maximum(x_scaled, -FP8_MAX)
    x_scaled = tl.minimum(x_scaled, FP8_MAX)
    
    # Store scaled F32 values
    out_ptrs = out_ptr + pid_m * stride_outm + offsets * stride_outk
    tl.store(out_ptrs, x_scaled, mask=mask)

@triton.jit
def f32_to_mxfp4e2_rtne_kernel_hw(
    x_ptr,           
    out_ptr,        
    scale_ptr,      
    M,              
    K,              
    stride_xm,       
    stride_xk,      
    stride_outm,     
    stride_outk,    
    stride_sm,      
    stride_sg,      
    GROUP_SIZE: tl.constexpr,
    GROUPS_PER_BLOCK: tl.constexpr,  
):
    """
    Triton kernel for F32 to MXFP4E2M1 conversion using hardware instructions.
    Processes multiple groups per workgroup for better thread utilization.
    """
    pid_m = tl.program_id(0) 
    pid_g = tl.program_id(1)  
    
    # Total elements per block = GROUP_SIZE * GROUPS_PER_BLOCK (e.g., 32 * 2 = 64)
    BLOCK_SIZE: tl.constexpr = GROUP_SIZE * GROUPS_PER_BLOCK
    
    block_start = pid_g * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    x_ptrs = x_ptr + pid_m * stride_xm + offsets * stride_xk
    mask = offsets < K
    x = tl.load(x_ptrs, mask=mask, other=0.0)
    
    # Reshape to [GROUPS_PER_BLOCK, GROUP_SIZE] to compute per-group absmax
    x_grouped = tl.reshape(x, (GROUPS_PER_BLOCK, GROUP_SIZE))
    
    scale_exp = get_exponent(x_grouped, 2)
    
    # Store scales for each group
    group_indices = tl.arange(0, GROUPS_PER_BLOCK)
    scale_ptrs = scale_ptr + pid_m * stride_sm + (pid_g * GROUPS_PER_BLOCK + group_indices) * stride_sg
    tl.store(scale_ptrs, scale_exp.to(tl.uint8), mask=(pid_g * GROUPS_PER_BLOCK + group_indices) < (K // GROUP_SIZE))
    
    # Broadcast scale_exp to match elements
    # Each group has GROUP_SIZE//2 pairs
    scale_exp_expanded = tl.reshape(scale_exp, (GROUPS_PER_BLOCK, 1))
    scale_exp_broadcast = tl.broadcast_to(scale_exp_expanded, (GROUPS_PER_BLOCK, GROUP_SIZE // 2))
    scale_exp_flat = tl.reshape(scale_exp_broadcast, (BLOCK_SIZE // 2,))
    
    scale_f32 = (scale_exp_flat.to(tl.uint32) << 23)
    
    # Get pairs of values from x
    pair_offsets = tl.arange(0, BLOCK_SIZE // 2)
    x_pairs = tl.reshape(x, (BLOCK_SIZE // 2, 2))
    x_even, x_odd = tl.split(x_pairs) 

    fp4_packed = tl.inline_asm_elementwise(
        "v_cvt_scalef32_pk_fp4_f32 $0, $1, $2, $3",
        "=v,v,v,v",
        args=[x_even, x_odd, scale_f32],
        dtype=tl.uint16,
        is_pure=True,
        pack=1,
    )

    fp4 = fp4_packed.to(tl.uint8)

    out_ptrs = out_ptr + pid_m * stride_outm + (block_start // 2 + pair_offsets) * stride_outk #offsets * stride_outk
    tl.store(out_ptrs, fp4, mask=(block_start // 2 + pair_offsets) < (K // 2))

def f32_to_mxfp4_triton(x: torch.Tensor, fmt: str = "e2m1", group_size: int = 32, method: str = 'sw', num_warps: int = 1):
    """
    Convert F32 tensor to MXFP4 format using Triton kernel.
   """
    M, K = x.shape
    assert K % group_size == 0, f"K ({K}) must be divisible by group_size ({group_size})"
    
    n_groups = K // group_size
    
    scales = torch.empty((M, n_groups), dtype=torch.uint8, device=x.device)
    
    # For packed HW kernel, process multiple groups per workgroup
    GROUPS_PER_BLOCK = 16 
    
    if method == 'hw':
        out_fp4 = torch.empty((M, K//2), dtype=torch.uint8, device=x.device)
        grid_hw = (M, n_groups // GROUPS_PER_BLOCK)
        f32_to_mxfp4e2_rtne_kernel_hw[grid_hw](
            x, out_fp4, scales,
            M, K,
            x.stride(0), x.stride(1),
            out_fp4.stride(0), out_fp4.stride(1),
            scales.stride(0), scales.stride(1),
            GROUP_SIZE=group_size,
            GROUPS_PER_BLOCK=GROUPS_PER_BLOCK,
            num_warps=num_warps,
        )    
    else:
        raise NotImplementedError("Only HW method is implemented for MXFP4 conversion.")
    
    return out_fp4, scales

def f32_to_mxfp6_triton(x: torch.Tensor, fmt: str = "e2m3", group_size: int = 32, method: str = 'sw', num_warps: int = 1):
    """
    Convert F32 tensor to MXFP6 format using Triton kernel.
    
    Output is packed as 6 uint32 values per 32 FP6 values (192 bits).
    Output shape: (M, K * 6 // 32) as uint32
    """
    M, K = x.shape
    assert K % group_size == 0, f"K ({K}) must be divisible by group_size ({group_size})"
    assert group_size == 32, "FP6 conversion requires group_size=32 for V_CVT_SCALEF32_SR_PK32_FP6_F32 instruction"
    
    n_groups = K // group_size

    scales = torch.empty((M, n_groups), dtype=torch.uint8, device=x.device)

    # For FP6, use GROUPS_PER_BLOCK for proper thread utilization
    GROUPS_PER_BLOCK = 1

    if method == 'hw':
        out_fp6 = torch.empty((M, n_groups * 6), dtype=torch.uint32, device=x.device)
        grid_hw = (M, n_groups // GROUPS_PER_BLOCK)

        if fmt == "e2m3":
            f32_to_mxfp6e2_rtne_kernel_hw[grid_hw](
                x, out_fp6, scales,
                M, K,
                x.stride(0), x.stride(1),
                out_fp6.stride(0), out_fp6.stride(1),
                scales.stride(0), scales.stride(1),
                GROUP_SIZE=group_size,
                GROUPS_PER_BLOCK=GROUPS_PER_BLOCK,
                num_warps=num_warps,
            )
        else:  # e3m2
            f32_to_mxfp6e3_rtne_kernel_hw[grid_hw](
                x, out_fp6, scales,
                M, K,
                x.stride(0), x.stride(1),
                out_fp6.stride(0), out_fp6.stride(1),
                scales.stride(0), scales.stride(1),
                GROUP_SIZE=group_size,
                GROUPS_PER_BLOCK=GROUPS_PER_BLOCK,
                num_warps=num_warps,
            )
        
        return out_fp6, scales
    else:
        raise NotImplementedError("Only HW method is implemented for MXFP6 conversion.")

def f32_to_mxfp8_triton(x: torch.Tensor, fmt: str = "e4m3", group_size: int = 32, method: str = 'sw', num_warps: int = 1):
    """
    Convert F32 tensor to MXFP8 format using Triton kernel.
   """
    M, K = x.shape
    assert K % group_size == 0, f"K ({K}) must be divisible by group_size ({group_size})"
    
    n_groups = K // group_size
    
    if fmt == "e4m3":
        fp8_max = 448.0
        fp8_exp_offset = 8
        fp8_dtype = torch.float8_e4m3fn
    else:  # e5m2
        fp8_max = 57344.0
        fp8_exp_offset = 15
        fp8_dtype = torch.float8_e5m2
    
    scales = torch.empty((M, n_groups), dtype=torch.uint8, device=x.device)
    
    # For packed HW kernel, process multiple groups per workgroup
    GROUPS_PER_BLOCK = 16 
    
    if method == 'hw':
        out_fp8 = torch.empty((M, K), dtype=torch.uint8, device=x.device)
        if fmt == "e4m3":
            grid_hw = (M, n_groups // GROUPS_PER_BLOCK)
            f32_to_mxfp8e4_rtne_kernel_hw[grid_hw](
                x, out_fp8, scales,
                M, K,
                x.stride(0), x.stride(1),
                out_fp8.stride(0), out_fp8.stride(1),
                scales.stride(0), scales.stride(1),
                GROUP_SIZE=group_size,
                GROUPS_PER_BLOCK=GROUPS_PER_BLOCK,
                num_warps=num_warps,
            )
        else:
            grid_e5m2 = (M, n_groups)
            f32_to_mxfp8e5_rtne_kernel_hw[grid_e5m2](
                x, out_fp8, scales,
                M, K,
                x.stride(0), x.stride(1),
                out_fp8.stride(0), out_fp8.stride(1),
                scales.stride(0), scales.stride(1),
                GROUP_SIZE=group_size,
                GROUPS_PER_BLOCK=GROUPS_PER_BLOCK,
                num_warps=num_warps,
            )
        
        fp8 = out_fp8.view(fp8_dtype)

    elif method == 'hw_sr':
        out_fp8 = torch.empty((M, K), dtype=torch.uint8, device=x.device)
        GROUPS_PER_BLOCK_SR = 256
        grid_sr = (M, n_groups // GROUPS_PER_BLOCK_SR)
        
        if fmt == "e4m3":
            f32_to_mxfp8e4_sr_kernel_hw[grid_sr](
                x, out_fp8, scales,
                M, K,
                x.stride(0), x.stride(1),
                out_fp8.stride(0), out_fp8.stride(1),
                scales.stride(0), scales.stride(1),
                GROUP_SIZE=group_size,
                GROUPS_PER_BLOCK=GROUPS_PER_BLOCK_SR,
                num_warps=num_warps,
            )
        else:
            f32_to_mxfp8e5_sr_kernel_hw[grid_sr](
                x, out_fp8, scales,
                M, K,
                x.stride(0), x.stride(1),
                out_fp8.stride(0), out_fp8.stride(1),
                scales.stride(0), scales.stride(1),
                GROUP_SIZE=group_size,
                GROUPS_PER_BLOCK=GROUPS_PER_BLOCK_SR,
                num_warps=num_warps,
            )
        
        fp8 = out_fp8.view(fp8_dtype)

    else:
        out_f32 = torch.empty((M, K), dtype=torch.float32, device=x.device)
        grid_sw = (M, n_groups)
        
        f32_to_mxfp8_kernel_sw[grid_sw](
            x, out_f32, scales,
            M, K,
            x.stride(0), x.stride(1),
            out_f32.stride(0), out_f32.stride(1),
            scales.stride(0), scales.stride(1),
            GROUP_SIZE=group_size,
            FP8_MAX=fp8_max,
            FP8_EXP_OFFSET=fp8_exp_offset,
            BLOCK_M=1,
            num_warps=num_warps,
        )
        
        fp8 = out_f32.to(fp8_dtype)
    
    return fp8, scales

def fp4_e2m1_to_fp32(packed: torch.Tensor) -> torch.Tensor:
    """
    Decode packed FP4 E2M1 values into float32.

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
    exp  = (codes >> 1) & 0x3
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

def fp6_packed_to_fp32(packed: torch.Tensor, fmt: str = "e2m3") -> torch.Tensor:
    if packed.dtype != torch.uint32:
        raise TypeError(f"expected torch.uint32, got {packed.dtype}")
    
    M = packed.shape[0]
    n_packed = packed.shape[1]
    K = n_packed * 32 // 6
    n_groups = n_packed // 6  # 256 groups of 32 FP6 values
    
    # View as bytes: (M, 1536) uint32 -> (M, 1536*4) uint8 -> (M, 256, 24) uint8
    packed_bytes = packed.view(torch.uint8).reshape(M, n_groups, 24)
    
    out = torch.empty((M, K), dtype=torch.float32, device=packed.device)
    
    # Determine bit layout based on format
    # E2M3: [S][EE][MMM] - 1 sign, 2 exponent, 3 mantissa bits
    # E3M2: [S][EEE][MM] - 1 sign, 3 exponent, 2 mantissa bits
    if fmt == "e2m3":
        exp_bits = 2
        mant_bits = 3
        exp_mask = 0x3   # 2 bits
        mant_mask = 0x7  # 3 bits
    else:  # e3m2
        exp_bits = 3
        mant_bits = 2
        exp_mask = 0x7   # 3 bits
        mant_mask = 0x3  # 2 bits
    
    for g in range(n_groups):
        group_bytes = packed_bytes[:, g, :]  # (M, 24) = 192 bits = 32 FP6 values
        
        # Process 3 bytes at a time -> 4 FP6 values
        for i in range(8):  # 8 iterations * 3 bytes = 24 bytes, 8 * 4 = 32 FP6 values
            # Load 3 bytes - use int64 for bitwise operations
            byte0 = group_bytes[:, i*3 + 0].to(torch.int64)
            byte1 = group_bytes[:, i*3 + 1].to(torch.int64)
            byte2 = group_bytes[:, i*3 + 2].to(torch.int64)
            
            # Reconstruct 24-bit value
            packed_24bit = byte0 | (byte1 << 8) | (byte2 << 16)
            
            # Extract 4 FP6 values from 24 bits
            for j in range(4):
                shift = j * 6
                fp6_val = (packed_24bit >> shift) & 0x3F
                
                # Extract FP6 components based on format
                sign = (fp6_val >> 5) & 0x1
                fp6_exponent = (fp6_val >> mant_bits) & exp_mask
                fp6_mantissa = fp6_val & mant_mask
                
                if fmt == "e2m3":
                    # E2M3: Convert to FP8 E4M3 (from mxfp468_gemm.py logic)
                    fp8_exponent = torch.where((fp6_exponent == 0) & (fp6_mantissa == 0), 0,
                                    torch.where((fp6_exponent == 0) & (fp6_mantissa == 1), 4,
                                    torch.where((fp6_exponent == 0) & (fp6_mantissa > 1) & (fp6_mantissa < 4), 5,
                                    torch.where((fp6_exponent == 0) & (fp6_mantissa > 3), 6,
                                    fp6_exponent + 6))))
                    
                    fp8_mantissa = torch.where((fp6_exponent == 0) & (fp6_mantissa == 0), 0,
                                    torch.where((fp6_exponent == 0) & (fp6_mantissa == 1), 0,
                                    torch.where((fp6_exponent == 0) & (fp6_mantissa > 1) & (fp6_mantissa < 4), (fp6_mantissa & 1) << 2,
                                    torch.where((fp6_exponent == 0) & (fp6_mantissa > 3), (fp6_mantissa & 3) << 1,
                                    fp6_mantissa))))
                    
                    # Convert FP8 E4M3 to float32 (bias = 7)
                    val = torch.where(fp8_exponent == 0,
                        fp8_mantissa.float() / 8.0 * (2.0 ** -6),  # subnormal
                        (1.0 + fp8_mantissa.float() / 8.0) * torch.pow(2.0, fp8_exponent.float() - 7.0)
                    )
                else:
                    # E3M2: Convert to FP8 E5M2 (similar logic)
                    fp8_exponent = torch.where((fp6_exponent == 0) & (fp6_mantissa == 0), 0,
                                    torch.where((fp6_exponent == 0) & (fp6_mantissa == 1), 11,
                                    torch.where((fp6_exponent == 0) & (fp6_mantissa > 1), 12,
                                    fp6_exponent + 12)))
                    
                    fp8_mantissa = torch.where((fp6_exponent == 0) & (fp6_mantissa == 0), 0,
                                    torch.where((fp6_exponent == 0) & (fp6_mantissa == 1), 0,
                                    torch.where((fp6_exponent == 0) & (fp6_mantissa > 1), (fp6_mantissa & 1) << 1,
                                    fp6_mantissa)))
                    
                    # Convert FP8 E5M2 to float32 (bias = 15)
                    val = torch.where(fp8_exponent == 0,
                        fp8_mantissa.float() / 4.0 * (2.0 ** -14),  # subnormal
                        (1.0 + fp8_mantissa.float() / 4.0) * torch.pow(2.0, fp8_exponent.float() - 15.0)
                    )
                
                val = torch.where(sign.bool(), -val, val)
                
                out[:, g * 32 + i * 4 + j] = val
    
    return out

def fp6_e2m3_to_fp32(packed: torch.Tensor) -> torch.Tensor:
    """
    Decode packed FP6 E2M3 (BF6) values into float32.
    Wrapper for fp6_packed_to_fp32 with fmt="e2m3".
    """
    # [8192, 1536] = 8192/32-bit = 256 = 1536/6-bit
    return fp6_packed_to_fp32(packed, fmt="e2m3")


def fp6_e3m2_to_fp32(packed: torch.Tensor) -> torch.Tensor:
    """
    Decode packed FP6 E3M2 (BF6) values into float32.
    Wrapper for fp6_packed_to_fp32 with fmt="e3m2".
    """
    return fp6_packed_to_fp32(packed, fmt="e3m2")

def main():
    """Test MXFP GEMM: HW-accelerated vs TCAST vs Torch."""
    import tcast
    import triton.testing as tt
    
    torch.manual_seed(123)
    
    M, N, K = 8192, 8192, 8192
    BM, BN, BK = 256, 256, 128
    NUM_WARPS = 1
    
    # Create random F32 tensors (activations and weights)
    A = torch.randn((M, K), device="cuda", dtype=torch.float32)
        
    fmt = "e2m3"
    CASTDICT = {"e4m3": tcast.mxfp8e4, "e5m2": tcast.mxfp8e5, "e2m3": tcast.mxfp6e2, "e3m2": tcast.mxfp6e3, "e2m1": tcast.mxfp4e2}
    
    # Helper function for TCAST conversion
    def tcast_convert(x, cast_fmt):
        x_tcast = tcast.cast(x, cast_fmt)
        scale = x_tcast.scaledata.scale.to(torch.uint8).T.reshape(x.shape[0], -1)
        s = 2**((x_tcast.scaledata.scale - 127)).to(torch.float32).T
        fp8 = (x_tcast.tensor.view(x.shape[0], -1, 32) / s.view(x.shape[0], -1).unsqueeze(-1)).to(torch.float8_e4m3fn).reshape(x.shape[0], x.shape[1])
        return fp8, scale
    
    # Benchmark TCAST conversion time
    tcast_conv_time = tt.do_bench(lambda: tcast_convert(A, CASTDICT[fmt]), warmup=10, rep=1000)
    print(f"TCAST conversion time: {tcast_conv_time:.4f} ms")
    
    # Convert using tcast
    A_tcast = tcast.cast(A, CASTDICT[fmt])
    
    # Get scales
    A_scale_tcast = A_tcast.scaledata.scale.to(torch.uint8).T.reshape(M, -1)
    
    # Get FP8 values
    A_s = 2**((A_tcast.scaledata.scale - 127)).to(torch.float32).T
    A_fp_tcast = (A_tcast.tensor.view(M, -1, 32) / A_s.view(M, -1).unsqueeze(-1)).reshape(M, K)
    
    # HW-Accelerated MXFP GEMM
    for METHOD in ["hw"]: # ["hw", "hw_sr", "sw"]:
        try:
            if fmt == "e2m1":
                triton_func = f32_to_mxfp4_triton
            elif fmt == "e2m3" or fmt == "e3m2":
                triton_func = f32_to_mxfp6_triton
            else:
                triton_func = f32_to_mxfp8_triton

            # Benchmark HW conversion time
            hw_conv_time = tt.do_bench(lambda: triton_func(A, fmt=fmt, group_size=32, method=METHOD, num_warps=1), warmup=10, rep=100)
            print(f"HW {METHOD} conversion time: {hw_conv_time:.4f} ms")
            
            # Convert using HW-accelerated kernel
            A_fp_hw, A_scale_hw = triton_func(A, fmt=fmt, group_size=32, method=METHOD, num_warps=1)

            if fmt == "e2m1":
                A_fp_hw = fp4_e2m1_to_fp32(A_fp_hw)
            elif fmt == "e2m3": 
                A_fp_hw = fp6_e2m3_to_fp32(A_fp_hw)
            elif fmt == "e3m2":
                A_fp_hw = fp6_e3m2_to_fp32(A_fp_hw)
            else:
                raise("Only MXFP4, MXFP6 and MXFP8 datatypes are supported.")
            # compare to tcast results
            print(f"\tL_inf errors: A_fp: {torch.max(torch.abs(A_fp_hw.float() - A_fp_tcast.float())):.4f}, Scale: {torch.max(torch.abs(A_scale_hw.float() - A_scale_tcast.float())):.4f}")

        except Exception as e:
            print(f"HW-accelerated path failed: {e}")
            import traceback
            traceback.print_exc()
    

if __name__ == "__main__":
    main()
