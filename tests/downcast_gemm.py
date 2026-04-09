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
        
    fmt = "e2m1"
    CASTDICT = {"e4m3": tcast.mxfp8e4, "e5m2": tcast.mxfp8e5, "e2m1": tcast.mxfp4e2}
    
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
            else:
                triton_func = f32_to_mxfp8_triton

            # Benchmark HW conversion time
            hw_conv_time = tt.do_bench(lambda: triton_func(A, fmt=fmt, group_size=32, method=METHOD, num_warps=NUM_WARPS), warmup=10, rep=100)
            print(f"HW {METHOD} conversion time: {hw_conv_time:.4f} ms")
            
            # Convert using HW-accelerated kernel
            A_fp_hw, A_scale_hw = triton_func(A, fmt=fmt, group_size=32, method=METHOD, num_warps=NUM_WARPS)

            if fmt == "e2m1":
                A_fp_hw = fp4_e2m1_to_fp32(A_fp_hw)
            # compare to tcast results
            print(f"\tL_inf errors: A_fp: {torch.max(torch.abs(A_fp_hw.float() - A_fp_tcast.float())):.4f}, Scale: {torch.max(torch.abs(A_scale_hw.float() - A_scale_tcast.float())):.4f}")

        except Exception as e:
            print(f"HW-accelerated path failed: {e}")
            import traceback
            traceback.print_exc()
    

if __name__ == "__main__":
    main()
