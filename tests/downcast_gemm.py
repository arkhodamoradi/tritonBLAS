"""
Triton kernel for F32 to MXFP8 conversion using absmax scaling.

This implementation uses V_CVT_SCALEF32_PK_FP8_F32 and V_CVT_SCALEF32_PK_BF8_F32 
hardware instructions for accelerated FP8 conversion with proper rounding.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def mxfp468_dot_scaled_gemm(
    A_ptr, B_ptr, C_ptr,
    As_ptr, Bs_ptr,                       # e8m0 scales (uint8)
    M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,

    # A: [M, K]
    stride_am: tl.constexpr, stride_ak: tl.constexpr,
    # B: [N, K]
    stride_bn: tl.constexpr, stride_bk: tl.constexpr,
    # C: [M, N]
    stride_cm: tl.constexpr, stride_cn: tl.constexpr,

    # As: [M, K//32]
    stride_asm: tl.constexpr, stride_askg: tl.constexpr,
    # Bs: [N, K//32]  
    stride_bsn: tl.constexpr, stride_bskg: tl.constexpr,

    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr = 8,
    A_FMT: tl.constexpr = "e4m3",
    B_FMT: tl.constexpr = "e4m3",
    A_DIV_K: tl.constexpr = 1,
    B_DIV_K: tl.constexpr = 1,
    OUT_DTYPE: tl.constexpr = tl.float16, # tl.float16 / tl.bfloat16 / tl.float32
):
    # dot_scaled for MX formats uses group_size=32 for e8m0 scales
    GROUP_SIZE: tl.constexpr = 32
    tl.static_assert(BLOCK_K % GROUP_SIZE == 0, "BLOCK_K must be multiple of 32 for e8m0 scales")
    tl.static_assert(K % GROUP_SIZE == 0, "K must be multiple of 32 for e8m0 scales (MXFP)")

    pid = tl.program_id(0)
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)

    # simple grouped swizzle on M for L2 locality
    group_id = pid // (GROUP_M * grid_n)
    first_m = group_id * GROUP_M
    pid_in_group = pid % (GROUP_M * grid_n)
    pid_m = first_m + (pid_in_group % GROUP_M)
    pid_n = pid_in_group // GROUP_M

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    offs_kg_tile = tl.arange(0, BLOCK_K // GROUP_SIZE)
    
    # iterate over K tiles
    for k0 in range(0, K, BLOCK_K):
        offs_k_a = k0//A_DIV_K + tl.arange(0, BLOCK_K//A_DIV_K)
        offs_k_b = k0//B_DIV_K + tl.arange(0, BLOCK_K//B_DIV_K)
        offs_kg = (k0 // GROUP_SIZE) + offs_kg_tile

        # ----- load FP8 tiles -----
        # A: [BM, BK]
        a = tl.load(
            A_ptr + offs_m[:, None] * stride_am + offs_k_a[None, :] * stride_ak,
            mask=(offs_m[:, None] < M) & (offs_k_a[None, :] < K//A_DIV_K),
            other=0.0,
        )

        # B: [BN, BK]
        b = tl.load(
            B_ptr + offs_n[None, :] * stride_bn + offs_k_b[:, None] * stride_bk,
            mask=(offs_k_b[:, None] < K//B_DIV_K) & (offs_n[None, :] < N),
            other=0.0,
        )

        # ----- load e8m0 scales (uint8) -----
        # As: [BM, BK/32]
        a_scale = tl.load(
            As_ptr + offs_m[:, None] * stride_asm + offs_kg[None, :] * stride_askg,
            mask=(offs_m[:, None] < M) & (offs_kg[None, :] < (K // GROUP_SIZE)),
            other=0,
        )

        # Bs: [BN, BK/32]  (IMPORTANT: Bs is indexed by N then K-group; do NOT transpose)
        b_scale = tl.load(
            Bs_ptr + offs_n[:, None] * stride_bsn + offs_kg[None, :] * stride_bskg,
            mask=(offs_n[:, None] < N) & (offs_kg[None, :] < (K // GROUP_SIZE)),
            other=0,
        )

        # ----- scaled dot (native MX path when supported) -----
        acc = tl.dot_scaled(
            a, a_scale, A_FMT,
            b, b_scale, B_FMT,
            acc=acc,
            out_dtype=tl.float32,
        )

    # store
    tl.store(
        C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn,
        acc.to(OUT_DTYPE),
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )

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
    
    pair_offsets = tl.arange(0, GROUP_SIZE // 2)
    
    # Get pairs of values from x
    x_even = tl.load(x_ptr + pid_m * stride_xm + (group_start + pair_offsets * 2) * stride_xk,
                     mask=(group_start + pair_offsets * 2) < K, other=0.0)
    x_odd = tl.load(x_ptr + pid_m * stride_xm + (group_start + pair_offsets * 2 + 1) * stride_xk,
                    mask=(group_start + pair_offsets * 2 + 1) < K, other=0.0)
    
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


def f32_to_mxfp8_triton(x: torch.Tensor, fmt: str = "e4m3", group_size: int = 32, use_hw: bool = True):
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
    
    if use_hw:
        out_fp8 = torch.empty((M, K), dtype=torch.uint8, device=x.device)
        
        grid = (M, n_groups)
        f32_to_mxfp8_kernel_hw[grid](
            x, out_fp8, scales,
            M, K,
            x.stride(0), x.stride(1),
            out_fp8.stride(0), out_fp8.stride(1),
            scales.stride(0), scales.stride(1),
            GROUP_SIZE=group_size,
            FP8_EXP_OFFSET=fp8_exp_offset,
            IS_E4M3=(fmt == "e4m3"),
        )
        
        fp8 = out_fp8.view(fp8_dtype)
    else:
        out_f32 = torch.empty((M, K), dtype=torch.float32, device=x.device)
        
        grid = (M, n_groups)
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
        )
        
        fp8 = out_f32.to(fp8_dtype)
    
    return fp8, scales


def mxfp_gemm(A_fp8, B_fp8, A_scale, B_scale, fmt="e4m3",
              BM=128, BN=128, BK=128, num_warps=8, group_m=8, num_stages=1, nonkdim=32,
              out_dtype=torch.float32):
    """
    MXFP GEMM using pre-converted FP8 inputs with scales.
    
    Args:
        A_fp8: FP8 tensor [M, K]
        B_fp8: FP8 tensor [N, K] (note: B is [N, K], not [K, N])
        A_scale: E8M0 scale tensor [M, K//32]
        B_scale: E8M0 scale tensor [N, K//32]
        fmt: "e4m3" or "e5m2"
    
    Returns:
        C: Output tensor [M, N]
    """
    M, KA = A_fp8.shape
    N, KB = B_fp8.shape
    assert KA == KB, "Inner dimensions must match"
    K = KA
    
    assert A_scale.dtype == torch.uint8 and B_scale.dtype == torch.uint8
    assert A_scale.shape == (M, K // 32)
    assert B_scale.shape == (N, K // 32)
    
    C = torch.empty((M, N), device=A_fp8.device, dtype=out_dtype)
    grid = (triton.cdiv(M, BM) * triton.cdiv(N, BN),)
    
    A_scale = A_scale.contiguous()
    B_scale = B_scale.contiguous()
    
    kernel_kwargs = {"matrix_instr_nonkdim": nonkdim}
    
    mxfp468_dot_scaled_gemm[grid](
        A_fp8, B_fp8, C, A_scale, B_scale,
        M, N, K,
        A_fp8.stride(0), A_fp8.stride(1),
        B_fp8.stride(0), B_fp8.stride(1),
        C.stride(0), C.stride(1),
        A_scale.stride(0), A_scale.stride(1),
        B_scale.stride(0), B_scale.stride(1),
        BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK,
        GROUP_M=group_m,
        A_FMT=fmt, B_FMT=fmt,
        A_DIV_K=1,
        B_DIV_K=1,
        OUT_DTYPE=tl.float16 if out_dtype == torch.float16 else (tl.bfloat16 if out_dtype == torch.bfloat16 else tl.float32),
        num_warps=num_warps,
        num_stages=num_stages,
        **kernel_kwargs,
    )
    
    return C


def main():
    """Test MXFP GEMM: HW-accelerated vs TCAST vs Torch."""
    import tcast
    import triton.testing as tt
    
    torch.manual_seed(123)
    
    M, N, K = 4096, 4096, 4096
    BM, BN, BK = 256, 256, 128
    GROUP_M = 1
    NUM_WARPS = 8
    NUM_STAGES = 2
    NONKDIM = 16 if BK % 128 == 0 else 32
    
    # Create random F32 tensors (activations and weights)
    A = torch.randn((M, K), device="cuda", dtype=torch.float32)  
    B = torch.randn((N, K), device="cuda", dtype=torch.float32) 
    
    print("=" * 70)
    print("MXFP GEMM Comparison: HW-Accelerated vs TCAST vs Torch")
    print("=" * 70)
    print(f"Matrix sizes: A[{M}, {K}] x B[{N}, {K}].T = C[{M}, {N}]")
    print(f"Block sizes: BM={BM}, BN={BN}, BK={BK}")
    
    fmt = "e4m3"
    CASTDICT = {"e4m3": tcast.mxfp8e4, "e5m2": tcast.mxfp8e5}
    
    print(f"\n{'='*50}")
    print(f"Format: {fmt}")
    print(f"{'='*50}")
    
    # 1. Torch Reference (F32 GEMM)
    print("\n--- 1. Torch Reference (F32 GEMM) ---")
    C_torch = A @ B.T
    torch_time = tt.do_bench(lambda: A @ B.T, warmup=10, rep=100)
    print(f"Torch F32 GEMM time: {torch_time:.4f} ms")
    print(f"Torch TFLOPS: {(2*M*N*K)/(torch_time*1e9):.2f}")
    
    # 2. TCAST-based MXFP GEMM
    print("\n--- 2. TCAST-based MXFP GEMM ---")
    
    # Helper function for TCAST conversion
    def tcast_convert(x, cast_fmt):
        x_tcast = tcast.cast(x, cast_fmt)
        scale = x_tcast.scaledata.scale.to(torch.uint8).T.reshape(x.shape[0], -1)
        s = 2**((x_tcast.scaledata.scale - 127)).to(torch.float32).T
        fp8 = (x_tcast.tensor.view(x.shape[0], -1, 32) / s.view(x.shape[0], -1).unsqueeze(-1)).to(torch.float8_e4m3fn).reshape(x.shape[0], x.shape[1])
        return fp8, scale
    
    # Benchmark TCAST conversion time
    tcast_conv_time = tt.do_bench(lambda: tcast_convert(A, CASTDICT[fmt]), warmup=10, rep=100)
    print(f"TCAST conversion time (per matrix): {tcast_conv_time:.4f} ms")
    
    # Convert using tcast
    A_tcast = tcast.cast(A, CASTDICT[fmt])
    B_tcast = tcast.cast(B, CASTDICT[fmt])
    
    # Get scales
    A_scale_tcast = A_tcast.scaledata.scale.to(torch.uint8).T.reshape(M, -1)
    B_scale_tcast = B_tcast.scaledata.scale.to(torch.uint8).T.reshape(N, -1)
    
    # Get FP8 values
    A_s = 2**((A_tcast.scaledata.scale - 127)).to(torch.float32).T
    A_fp8_tcast = (A_tcast.tensor.view(M, -1, 32) / A_s.view(M, -1).unsqueeze(-1)).to(torch.float8_e4m3fn).reshape(M, K)
    
    B_s = 2**((B_tcast.scaledata.scale - 127)).to(torch.float32).T
    B_fp8_tcast = (B_tcast.tensor.view(N, -1, 32) / B_s.view(N, -1).unsqueeze(-1)).to(torch.float8_e4m3fn).reshape(N, K)
    
    # Run MXFP GEMM
    C_tcast = mxfp_gemm(A_fp8_tcast, B_fp8_tcast, A_scale_tcast, B_scale_tcast, fmt=fmt,
                        BM=BM, BN=BN, BK=BK, num_warps=NUM_WARPS, num_stages=NUM_STAGES, nonkdim=NONKDIM)
    
    tcast_gemm_time = tt.do_bench(lambda: mxfp_gemm(A_fp8_tcast, B_fp8_tcast, A_scale_tcast, B_scale_tcast, fmt=fmt,
                                                BM=BM, BN=BN, BK=BK, num_warps=NUM_WARPS, num_stages=NUM_STAGES, nonkdim=NONKDIM),
                              warmup=10, rep=100)
    tcast_total_time = tcast_conv_time * 2 + tcast_gemm_time  # 2 matrices to convert
    print(f"TCAST MXFP GEMM time: {tcast_gemm_time:.4f} ms")
    print(f"TCAST Total time (conv + gemm): {tcast_total_time:.4f} ms")
    print(f"TCAST TFLOPS (gemm only): {(2*M*N*K)/(tcast_gemm_time*1e9):.2f}")
    
    # Compare with Torch
    tcast_error = (C_torch - C_tcast).abs().max().item()
    print(f"Max error vs Torch: {tcast_error:.6f}")
    
    # 3. HW-Accelerated MXFP GEMM
    print("\n--- 3. HW-Accelerated MXFP GEMM ---")
    
    try:
        # Benchmark HW conversion time
        hw_conv_time = tt.do_bench(lambda: f32_to_mxfp8_triton(A, fmt=fmt, group_size=32, use_hw=True), warmup=10, rep=100)
        print(f"HW conversion time (per matrix): {hw_conv_time:.4f} ms")
        
        # Convert using HW-accelerated kernel
        A_fp8_hw, A_scale_hw = f32_to_mxfp8_triton(A, fmt=fmt, group_size=32, use_hw=True)
        B_fp8_hw, B_scale_hw = f32_to_mxfp8_triton(B, fmt=fmt, group_size=32, use_hw=True)
        
        # Run MXFP GEMM
        C_hw = mxfp_gemm(A_fp8_hw, B_fp8_hw, A_scale_hw, B_scale_hw, fmt=fmt,
                         BM=BM, BN=BN, BK=BK, num_warps=NUM_WARPS, num_stages=NUM_STAGES, nonkdim=NONKDIM)
        
        hw_gemm_time = tt.do_bench(lambda: mxfp_gemm(A_fp8_hw, B_fp8_hw, A_scale_hw, B_scale_hw, fmt=fmt,
                                                 BM=BM, BN=BN, BK=BK, num_warps=NUM_WARPS, num_stages=NUM_STAGES, nonkdim=NONKDIM),
                               warmup=10, rep=100)
        hw_total_time = hw_conv_time * 2 + hw_gemm_time  # 2 matrices to convert
        print(f"HW MXFP GEMM time: {hw_gemm_time:.4f} ms")
        print(f"HW Total time (conv + gemm): {hw_total_time:.4f} ms")
        print(f"HW TFLOPS (gemm only): {(2*M*N*K)/(hw_gemm_time*1e9):.2f}")
        
        # Compare with Torch
        hw_error = (C_torch - C_hw).abs().max().item()
        print(f"Max error vs Torch: {hw_error:.6f}")
        
        # Compare HW vs TCAST
        hw_vs_tcast_error = (C_tcast - C_hw).abs().max().item()
        print(f"Max error HW vs TCAST: {hw_vs_tcast_error:.6f}")
        
    except Exception as e:
        print(f"HW-accelerated path failed: {e}")
        import traceback
        traceback.print_exc()
    
    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"{'Method':<25} {'Conv (ms)':<12} {'GEMM (ms)':<12} {'Total (ms)':<12} {'Error':<15}")
    print("-" * 80)
    print(f"{'Torch F32':<25} {'N/A':<12} {torch_time:<12.4f} {torch_time:<12.4f} {'N/A':<15}")
    print(f"{'TCAST MXFP':<25} {tcast_conv_time*2:<12.4f} {tcast_gemm_time:<12.4f} {tcast_total_time:<12.4f} {tcast_error:<15.6f}")
    try:
        print(f"{'HW-Accelerated MXFP':<25} {hw_conv_time*2:<12.4f} {hw_gemm_time:<12.4f} {hw_total_time:<12.4f} {hw_error:<15.6f}")
    except:
        pass


if __name__ == "__main__":
    main()
