"""
MXFP4 / MXFP8 Mixed-Precision GeMM
=====================================
Provides a single ``mxfp4fp8_gemm`` function that accepts pre-quantized
MXFP4 or MXFP8 tensors (in any combination) and dispatches to the
``dot_scaled`` Triton kernel from ``mxfp468_gemm``.

Also exposes end-to-end ``gemm_fp32_*`` helpers that quantize on the fly
and call the GEMM in one shot.

Importable example::

    from mxfp4fp8_gemm import mxfp4fp8_gemm, gemm_fp32_e4m3_e4m3
    C = gemm_fp32_e4m3_e4m3(A_f32, B_f32)

Test bench::

    python mxfp4fp8_gemm.py
"""

import torch
import triton
import triton.language as tl
import triton.testing as tt

# Re-use the dot_scaled kernel defined in mxfp468_gemm
from mxfp468_gemm import mxfp468_dot_scaled_gemm   # Triton JIT kernel

# Quantization wrappers from the companion module
from mxfp4fp8_quantization import (
    quantize_mxfp8e4_rtn,
    quantize_mxfp8e4_sr,
    quantize_mxfp8e5_rtn,
    quantize_mxfp8e5_sr,
    quantize_mxfp4_rtn,
    fp4_e2m1_to_fp32,
)

__all__ = [
    "mxfp4fp8_gemm",
    "gemm_fp32_e4m3_e4m3",
    "gemm_fp32_e5m2_e5m2",
    "gemm_fp32_e4m3_e5m2",
    "gemm_fp32_e4m3_e2m1",
    "gemm_fp32_e2m1_e4m3",
    "gemm_fp32_e2m1_e2m1",
]

# Map format string → (Triton constexpr string, element bytes, packed divisor)
# packed_div: how much smaller the K dim of the quantized tensor is vs logical K
_FMT_INFO = {
    "e4m3": ("e4m3", 1),   # (triton_fmt, k_divisor)
    "e5m2": ("e5m2", 1),
    "e2m1": ("e2m1", 2),   # FP4: two values per byte
}


def mxfp4fp8_gemm(
    A: torch.Tensor,
    B: torch.Tensor,
    As: torch.Tensor,
    Bs: torch.Tensor,
    afmt: str = "e4m3",
    bfmt: str = "e4m3",
    BM: int = 128,
    BN: int = 128,
    BK: int = 128,
    group_m: int = 8,
    num_warps: int = 8,
    num_stages: int = 1,
    nonkdim: int = 32,
    out_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    MX-format GeMM  C = A @ B^T  using ``tl.dot_scaled``.

    Parameters
    ----------
    A     : quantized activations  [M, K_A]
            dtype float8_e4m3fn / float8_e5m2 / uint8(FP4-packed)
    B     : quantized weights      [N, K_B]
            dtype float8_e4m3fn / float8_e5m2 / uint8(FP4-packed)
    As    : E8M0 scales for A      [M, K // 32]  dtype uint8
    Bs    : E8M0 scales for B      [N, K // 32]  dtype uint8
    afmt  : one of "e4m3", "e5m2", "e2m1"
    bfmt  : one of "e4m3", "e5m2", "e2m1"

    Returns
    -------
    C : torch.Tensor [M, N] in ``out_dtype``
    """
    assert afmt in _FMT_INFO and bfmt in _FMT_INFO, \
        f"Unsupported formats: {afmt}, {bfmt}"
    assert As.dtype == torch.uint8 and Bs.dtype == torch.uint8

    _, a_kdiv = _FMT_INFO[afmt]
    _, b_kdiv = _FMT_INFO[bfmt]

    M, KA = A.shape
    N, KB = B.shape

    # Logical K inferred from the non-packed side; both sides must agree
    K = KA * a_kdiv
    assert KB * b_kdiv == K, \
        f"K mismatch: A implies K={K}, B implies K={KB * b_kdiv}"

    GROUP_SIZE = 32
    assert K % GROUP_SIZE == 0
    assert As.shape == (M, K // GROUP_SIZE), f"As shape {As.shape} != ({M}, {K // GROUP_SIZE})"
    assert Bs.shape == (N, K // GROUP_SIZE), f"Bs shape {Bs.shape} != ({N}, {K // GROUP_SIZE})"

    C = torch.empty((M, N), device=A.device, dtype=out_dtype)
    grid = (triton.cdiv(M, BM) * triton.cdiv(N, BN),)

    As = As.contiguous()
    Bs = Bs.contiguous()

    _out_tl = (
        tl.float16 if out_dtype == torch.float16 else
        tl.bfloat16 if out_dtype == torch.bfloat16 else
        tl.float32
    )

    mxfp468_dot_scaled_gemm[grid](
        A, B, C, As, Bs,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        As.stride(0), As.stride(1),
        Bs.stride(0), Bs.stride(1),
        BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK,
        GROUP_M=group_m,
        A_FMT=afmt, B_FMT=bfmt,
        A_DIV_K=a_kdiv,
        B_DIV_K=b_kdiv,
        OUT_DTYPE=_out_tl,
        num_warps=num_warps,
        num_stages=num_stages,
        matrix_instr_nonkdim=nonkdim,
    )
    return C


# ---------------------------------------------------------------------------
# End-to-end helpers: FP32 in, quantize on the fly, run GEMM
# ---------------------------------------------------------------------------

def _e2e_gemm(A_f32, B_f32, afmt, bfmt, quant_A, quant_B,
              BM=128, BN=128, BK=128, group_m=8,
              num_warps=8, num_stages=1, nonkdim=32,
              out_dtype=torch.float32):
    A_q, As = quant_A(A_f32)
    B_q, Bs = quant_B(B_f32)
    return mxfp4fp8_gemm(
        A_q, B_q, As, Bs,
        afmt=afmt, bfmt=bfmt,
        BM=BM, BN=BN, BK=BK,
        group_m=group_m, num_warps=num_warps, num_stages=num_stages,
        nonkdim=nonkdim, out_dtype=out_dtype,
    )


def gemm_fp32_e4m3_e4m3(A, B, rounding="rtn", **kw):
    """FP32 → MXFP8 E4M3 × MXFP8 E4M3 GeMM."""
    q = quantize_mxfp8e4_rtn if rounding == "rtn" else quantize_mxfp8e4_sr
    return _e2e_gemm(A, B, "e4m3", "e4m3", q, q, **kw)


def gemm_fp32_e5m2_e5m2(A, B, rounding="rtn", **kw):
    """FP32 → MXFP8 E5M2 × MXFP8 E5M2 GeMM."""
    q = quantize_mxfp8e5_rtn if rounding == "rtn" else quantize_mxfp8e5_sr
    return _e2e_gemm(A, B, "e5m2", "e5m2", q, q, **kw)


def gemm_fp32_e4m3_e5m2(A, B, rounding="rtn", **kw):
    """FP32 → MXFP8 E4M3 × MXFP8 E5M2 GeMM (mixed)."""
    qa = quantize_mxfp8e4_rtn if rounding == "rtn" else quantize_mxfp8e4_sr
    qb = quantize_mxfp8e5_rtn if rounding == "rtn" else quantize_mxfp8e5_sr
    return _e2e_gemm(A, B, "e4m3", "e5m2", qa, qb, **kw)


def gemm_fp32_e4m3_e2m1(A, B, **kw):
    """FP32 → MXFP8 E4M3 × MXFP4 E2M1 GeMM (mixed precision)."""
    return _e2e_gemm(A, B, "e4m3", "e2m1",
                     quantize_mxfp8e4_rtn, quantize_mxfp4_rtn, **kw)


def gemm_fp32_e2m1_e4m3(A, B, **kw):
    """FP32 → MXFP4 E2M1 × MXFP8 E4M3 GeMM (mixed precision)."""
    return _e2e_gemm(A, B, "e2m1", "e4m3",
                     quantize_mxfp4_rtn, quantize_mxfp8e4_rtn, **kw)


def gemm_fp32_e2m1_e2m1(A, B, **kw):
    """FP32 → MXFP4 E2M1 × MXFP4 E2M1 GeMM."""
    return _e2e_gemm(A, B, "e2m1", "e2m1",
                     quantize_mxfp4_rtn, quantize_mxfp4_rtn, **kw)


# ---------------------------------------------------------------------------
# Test bench
# ---------------------------------------------------------------------------

def _run_tests():
    torch.manual_seed(42)

    M = N = K = 4096
    BM = BN = 256
    BK = 128
    GROUP_M = 1
    NUM_WARPS = 8
    NUM_STAGES = 2
    NONKDIM = 16

    print(f"=== mxfp4fp8_gemm test bench  M=N=K={M} ===\n")

    A = torch.randn((M, K), device="cuda", dtype=torch.float32)
    B = torch.randn((N, K), device="cuda", dtype=torch.float32)

    # Reference: FP32 matmul
    C_ref = A @ B.T

    gemm_configs = [
        ("E4M3 × E4M3  RTN", gemm_fp32_e4m3_e4m3,
         dict(rounding="rtn")),
        ("E4M3 × E4M3  SR ", gemm_fp32_e4m3_e4m3,
         dict(rounding="sr")),
        ("E5M2 × E5M2  RTN", gemm_fp32_e5m2_e5m2,
         dict(rounding="rtn")),
        ("E4M3 × E5M2  RTN", gemm_fp32_e4m3_e5m2,
         dict(rounding="rtn")),
        ("E4M3 × E2M1  RTN", gemm_fp32_e4m3_e2m1,
         dict()),
        ("E2M1 × E4M3  RTN", gemm_fp32_e2m1_e4m3,
         dict()),
        ("E2M1 × E2M1  RTN", gemm_fp32_e2m1_e2m1,
         dict()),
    ]

    common = dict(BM=BM, BN=BN, BK=BK, group_m=GROUP_M,
                  num_warps=NUM_WARPS, num_stages=NUM_STAGES,
                  nonkdim=NONKDIM, out_dtype=torch.float32)

    for name, fn, extra_kw in gemm_configs:
        kw = {**common, **extra_kw}
        try:
            C = fn(A, B, **kw)
            l_inf = torch.max(torch.abs(C - C_ref)).item()
            rel   = (l_inf / torch.max(torch.abs(C_ref)).item()) * 100

            ms = tt.do_bench(lambda: fn(A, B, **kw), warmup=10, rep=500)
            tflops = 2 * M * N * K / (ms * 1e-3) / 1e12

            print(f"  {name}: L_inf={l_inf:.3f} ({rel:.1f}%)  "
                  f"time={ms:.4f} ms  {tflops:.3f} TFLOPS")
        except Exception as e:
            import traceback
            print(f"  {name}: FAILED — {e}")
            traceback.print_exc()

    print()

    # ---- standalone pre-quantized GEMM example ----
    print("--- Pre-quantized path (mxfp4fp8_gemm directly) ---")
    A_q, As = quantize_mxfp8e4_rtn(A)
    B_q, Bs = quantize_mxfp4_rtn(B)

    C_mixed = mxfp4fp8_gemm(
        A_q, B_q, As, Bs,
        afmt="e4m3", bfmt="e2m1",
        BM=BM, BN=BN, BK=BK,
        group_m=GROUP_M, num_warps=NUM_WARPS, num_stages=NUM_STAGES,
        nonkdim=NONKDIM, out_dtype=torch.float32,
    )
    l_inf = torch.max(torch.abs(C_mixed - C_ref)).item()
    print(f"  E4M3(pre-quant) × E2M1(pre-quant): L_inf={l_inf:.3f}\n")


if __name__ == "__main__":
    _run_tests()
