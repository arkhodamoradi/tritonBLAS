"""
Triton kernel for F32 to MXFP8 conversion using absmax scaling.

This implementation uses V_CVT_SCALEF32_PK_FP8_F32 and V_CVT_SCALEF32_PK_BF8_F32 
hardware instructions for accelerated FP8 conversion with proper rounding.
"""

import torch
import triton
import triton.language as tl

@triton.jit
def f32_to_mxfp8_sr_kernel_hw(
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
    FP8_EXP_OFFSET: tl.constexpr, 
    IS_E4M3: tl.constexpr,     # True for e4m3, False for e5m2
):
    """
    Triton kernel for F32 to MXFP8 conversion using hardware instructions.
    """
    pid_m = tl.program_id(0) 
    pid_g = tl.program_id(1) 
    
    group_start = pid_g * GROUP_SIZE
    offsets = group_start + tl.arange(0, GROUP_SIZE)
    
    x_ptrs = x_ptr + pid_m * stride_xm + offsets * stride_xk
    mask = offsets < K
    x = tl.load(x_ptrs, mask=mask, other=0.0)
    
    absmax = tl.max(tl.abs(x))
    
    # Extract F32 exponent from absmax
    # F32 format: sign(1) | exponent(8) | mantissa(23)
    absmax_bits = absmax.to(tl.int32, bitcast=True)
    f32_exp = (absmax_bits >> 23) & 0xFF  # Biased exponent [0, 255]
    
    # Compute E8M0 scale: scale = f32_exp - FP8_EXP_OFFSET
    scale_exp = f32_exp - FP8_EXP_OFFSET
    
    # Clamp to valid E8M0 range [0, 255]
    scale_exp = tl.maximum(scale_exp, 0)
    scale_exp = tl.minimum(scale_exp, 255)
    
    # Store scale (as uint8)
    scale_ptr_out = scale_ptr + pid_m * stride_sm + pid_g * stride_sg
    tl.store(scale_ptr_out, scale_exp.to(tl.uint8))

    scale_f32 = (scale_exp.to(tl.uint32) << 23)
    
    sr_seed = 0.0
    fp8 = tl.inline_asm_elementwise(
            "v_cvt_scalef32_sr_fp8_f32 $0, $1, $2, $3",
            "=v,v,v,v",
            args=[x, sr_seed, scale_f32],
            dtype=tl.uint32,
            is_pure=True,
            pack=1,
        )
    
    fp8 = fp8.to(tl.uint8)
    if IS_E4M3:
        # e4m3: NaN is 0x7F (positive) or 0xFF (negative)
        # Replace with max value 0x7E (448) or 0xFE (-448)
        fp8 = tl.where(fp8 == 0x7F, 0x7E, fp8)
        fp8 = tl.where(fp8 == 0xFF, 0xFE, fp8)

    out_ptr = out_ptr + pid_m * stride_outm + offsets * stride_outk
    tl.store(out_ptr, fp8, mask=mask)

@triton.jit
def f32_to_mxfp8_kernel_hw(
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
    FP8_EXP_OFFSET: tl.constexpr, 
    IS_E4M3: tl.constexpr,     # True for e4m3, False for e5m2
):
    """
    Triton kernel for F32 to MXFP8 conversion using hardware instructions.
    """
    pid_m = tl.program_id(0) 
    pid_g = tl.program_id(1) 
    
    group_start = pid_g * GROUP_SIZE
    offsets = group_start + tl.arange(0, GROUP_SIZE)
    
    x_ptrs = x_ptr + pid_m * stride_xm + offsets * stride_xk
    mask = offsets < K
    x = tl.load(x_ptrs, mask=mask, other=0.0)
    
    absmax = tl.max(tl.abs(x))
    
    # Extract F32 exponent from absmax
    # F32 format: sign(1) | exponent(8) | mantissa(23)
    absmax_bits = absmax.to(tl.int32, bitcast=True)
    f32_exp = (absmax_bits >> 23) & 0xFF  # Biased exponent [0, 255]
    
    # Compute E8M0 scale: scale = f32_exp - FP8_EXP_OFFSET
    scale_exp = f32_exp - FP8_EXP_OFFSET
    
    # Clamp to valid E8M0 range [0, 255]
    scale_exp = tl.maximum(scale_exp, 0)
    scale_exp = tl.minimum(scale_exp, 255)
    
    # Store scale (as uint8)
    scale_ptr_out = scale_ptr + pid_m * stride_sm + pid_g * stride_sg
    tl.store(scale_ptr_out, scale_exp.to(tl.uint8))

    scale_f32 = (scale_exp.to(tl.uint32) << 23)
    
    
    # Get pairs of values from x
    pair_offsets = tl.arange(0, GROUP_SIZE // 2)
    x_pairs = tl.reshape(x, (GROUP_SIZE // 2, 2))  # Shape: [16, 2]

    # Split along last dim (size 2) → two tensors of shape [16]
    x_even, x_odd = tl.split(x_pairs)  # x_even: [16], x_odd: [16]
    
    if IS_E4M3:
        
        fp8_packed = tl.inline_asm_elementwise(
            "v_cvt_scalef32_pk_fp8_f32 $0, $1, $2, $3",
            "=v,v,v,v",
            args=[x_even, x_odd, scale_f32],
            dtype=tl.uint32,
            is_pure=True,
            pack=1,
        )
    else:
        fp8_packed = tl.inline_asm_elementwise(
            "v_cvt_scalef32_pk_bf8_f32 $0, $1, $2, $3",
            "=v,v,v,v",
            args=[x_even, x_odd, scale_f32],
            dtype=tl.uint32,
            is_pure=True,
            pack=1,
        )
    
    
    # Extract individual FP8 bytes from the packed result
    # Lower byte = fp8(src0), Upper byte = fp8(src1)
    fp8_0 = (fp8_packed & 0xFF).to(tl.uint8)
    fp8_1 = ((fp8_packed >> 8) & 0xFF).to(tl.uint8)
    
    # Clamp NaN/Inf to max valid FP8 values
    if IS_E4M3:
        # e4m3: NaN is 0x7F (positive) or 0xFF (negative)
        # Replace with max value 0x7E (448) or 0xFE (-448)
        fp8_0 = tl.where(fp8_0 == 0x7F, 0x7E, fp8_0)
        fp8_0 = tl.where(fp8_0 == 0xFF, 0xFE, fp8_0)
        fp8_1 = tl.where(fp8_1 == 0x7F, 0x7E, fp8_1)
        fp8_1 = tl.where(fp8_1 == 0xFF, 0xFE, fp8_1)
    else:
        # e5m2: Inf is 0x7C/0xFC, NaN is 0x7D-7F/0xFD-FF
        # Replace with max value 0x7B (57344) or 0xFB (-57344)
        fp8_0 = tl.where((fp8_0 >= 0x7C) & (fp8_0 < 0x80), 0x7B, fp8_0)
        fp8_0 = tl.where(fp8_0 >= 0xFC, 0xFB, fp8_0)
        fp8_1 = tl.where((fp8_1 >= 0x7C) & (fp8_1 < 0x80), 0x7B, fp8_1)
        fp8_1 = tl.where(fp8_1 >= 0xFC, 0xFB, fp8_1)

    out_even_ptrs = out_ptr + pid_m * stride_outm + (group_start + pair_offsets * 2) * stride_outk
    out_odd_ptrs = out_ptr + pid_m * stride_outm + (group_start + pair_offsets * 2 + 1) * stride_outk
    tl.store(out_even_ptrs, fp8_0, mask=(group_start + pair_offsets * 2) < K)
    tl.store(out_odd_ptrs, fp8_1, mask=(group_start + pair_offsets * 2 + 1) < K)


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
    
    grid = (M, n_groups)
    if method == 'hw':
        out_fp8 = torch.empty((M, K), dtype=torch.uint8, device=x.device)
        
        f32_to_mxfp8_kernel_hw[grid](
            x, out_fp8, scales,
            M, K,
            x.stride(0), x.stride(1),
            out_fp8.stride(0), out_fp8.stride(1),
            scales.stride(0), scales.stride(1),
            GROUP_SIZE=group_size,
            FP8_EXP_OFFSET=fp8_exp_offset,
            IS_E4M3=(fmt == "e4m3"),
            num_warps=num_warps,
        )
        
        fp8 = out_fp8.view(fp8_dtype)

    elif method == 'hw_sr':
        out_fp8 = torch.empty((M, K), dtype=torch.uint8, device=x.device)
        
        f32_to_mxfp8_sr_kernel_hw[grid](
            x, out_fp8, scales,
            M, K,
            x.stride(0), x.stride(1),
            out_fp8.stride(0), out_fp8.stride(1),
            scales.stride(0), scales.stride(1),
            GROUP_SIZE=group_size,
            FP8_EXP_OFFSET=fp8_exp_offset,
            IS_E4M3=(fmt == "e4m3"),
            num_warps=num_warps,
        )
        
        fp8 = out_fp8.view(fp8_dtype)

    else:
        out_f32 = torch.empty((M, K), dtype=torch.float32, device=x.device)
        
        f32_to_mxfp8_kernel_sw[grid](
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
        
    fmt = "e4m3"
    CASTDICT = {"e4m3": tcast.mxfp8e4, "e5m2": tcast.mxfp8e5}
    
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
    A_fp8_tcast = (A_tcast.tensor.view(M, -1, 32) / A_s.view(M, -1).unsqueeze(-1)).to(torch.float8_e4m3fn).reshape(M, K)
    
    # HW-Accelerated MXFP GEMM
    for METHOD in ["hw", "hw_sr", "sw"]:
        try:
            # Benchmark HW conversion time
            hw_conv_time = tt.do_bench(lambda: f32_to_mxfp8_triton(A, fmt=fmt, group_size=32, method=METHOD, num_warps=NUM_WARPS), warmup=10, rep=100)
            print(f"HW {METHOD} conversion time: {hw_conv_time:.4f} ms")
            
            # Convert using HW-accelerated kernel
            A_fp8_hw, A_scale_hw = f32_to_mxfp8_triton(A, fmt=fmt, group_size=32, method=METHOD, num_warps=NUM_WARPS)

            # compare to tcast results
            print(f"\tL_inf errors: A_fp8: {torch.max(torch.abs(A_fp8_hw.float() - A_fp8_tcast.float())):.4f}, Scale: {torch.max(torch.abs(A_scale_hw.float() - A_scale_tcast.float())):.4f}")
            
        except Exception as e:
            print(f"HW-accelerated path failed: {e}")
            import traceback
            traceback.print_exc()
    

if __name__ == "__main__":
    main()
