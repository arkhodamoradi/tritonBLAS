"""
Benchmark: TritonLinear vs nn.Linear vs TritonLinearFP4 vs TritonLinearLoRA.

Measures throughput in TFLOPs and latency in µs for various (M, K, N) configs.
Results are printed as tables and optionally exported as JSON.

Usage:
    python -m benchmarks.bench_linear [--json results.json]
"""

import argparse
import json
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.testing as tt

from loraq.linear import TritonLinear, TritonLinearFP4, TritonLinearLoRA
from loraq.linear import TritonLinearLoRaQ, TritonLinearLoRaQFP8
from loraq.linear import (
    TritonLinearLoRaQ_8_8, TritonLinearLoRaQ_8_16,
    TritonLinearLoRaQ_16_8, TritonLinearLoRaQ_16_16,
)
from loraq.quant import dynamic_mxfp8_quant, dynamic_mxfp4_quant, mxfp4_to_f32, e8m0_to_f32
from loraq.autotune_configs import (
    AutotunedLoRaQ, AutotunedLoRaQ_8_16, AutotunedLoRaQ_16_8, AutotunedLoRaQ_16_16,
    AutotunedLoRaQ4, AutotunedLoRaQ4_16,
    AutotunedLoRaQ3, AutotunedLoRaQFP16LR, AutotunedDualGEMM, AutotunedProjectAndQuant, LORAQ_Q8_CONFIGS,
)
from loraq.updated_kernels import (
    loraq_fused_q8_kernel,
    loraq_fused_q8_scaled_kernel_8_8,
    loraq_fused_q8_scaled_kernel_8_8_RL,
    loraq_fused_q4_kernel,
    loraq_fused_q4_kernel_4_16,
    loraq_fused_q8_scaled_kernel_8_16,
    loraq_fused_q8_scaled_kernel_16_8,
    loraq_fused_q8_scaled_kernel_16_16,
    loraq_dual_gemm_kernel, loraq_project_and_quant_kernel,
)
from loraq.kernels import loraq_fused_q8_fp16lr_kernel

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SIZES = [
    # (M,     K,     N)      -- representative workloads
    #(1,     4096,  4096),    # single-token decode
    #(8,     4096,  4096),    # small batch decode
    #(32,    4096,  4096),
    #(64,    4096,  4096),
    #(128,   4096,  4096),
    #(256,   4096,  4096),
    #(512,   4096,  4096),
    #(1024,  4096,  4096),
    #(2048,  4096,  4096),
    #(4096,  4096,  4096),
    #(1,     8192,  8192),    # single-token decode
    #(8,     8192,  8192),    # small batch decode
    #(32,    8192,  8192),
    #(64,    8192,  8192),
    #(128,   8192,  8192),
    #(256,   8192,  8192),
    #(512,   8192,  8192),
    #(1024,  8192,  8192),
    #(2048,  8192,  8192),
    #(4096,  8192,  8192),
    #(8192,  8192,  8192),
    #(128,   4096,  11008),   # LLaMA-7B FFN up
    #(128,   11008, 4096),    # LLaMA-7B FFN down
    #(128,   5120,  5120),    # LLaMA-13B hidden
    #(128,   8192,  8192),    # LLaMA-65B hidden
    #(256,   4096,  14336),   # LLaMA-2 70B FFN up
    #(1024,  4096,  14336),
    #(2048,  4096,  4096),    # large batch
    (4096, 1152, 1152), #Pixart qkv
    (4096, 1152, 3*1152), #Pixart qkv
    (4096, 4608, 1152), # pixart ffn down
    (4096, 1152, 4608), # pixart ffn up
    (512, 3072, 3072), # flux
    (512,3072,12288),
    (4096,3072,3072),
    (4096,12288,3072),
    (4096,3072,12288),
    (4096,  4096,  4096),    # large batch
]

WARMUP = 25
ITERS = 100

DTYPES_FP = [torch.float16, torch.bfloat16]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def tflops(m: int, n: int, k: int, time_s: float) -> float:
    """Compute TFLOPs for a GEMM of shape (M, K) @ (K, N)."""
    return 2.0 * m * n * k / time_s / 1e12


def us(time_s: float) -> float:
    """Convert seconds to microseconds."""
    return time_s * 1e6


def benchmark_fn(fn, warmup: int = WARMUP, iters: int = ITERS) -> float:
    """Run *fn* and return median wall-clock time in seconds."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    times = []
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append(t1 - t0)

    times.sort()
    return times[len(times) // 2]  # median


# ---------------------------------------------------------------------------
# fp16 / bf16 benchmark
# ---------------------------------------------------------------------------

def bench_fp(sizes, dtypes):
    rows = []
    W = 110
    print("\n" + "=" * W)
    print("  fp16 / bf16  Benchmark: TritonLinear  vs  nn.Linear")
    print("=" * W)
    header = (
        f"{'M':>6} {'K':>6} {'N':>6} {'dtype':>7}  "
        f"{'Triton µs':>10} {'Triton TFLOPS':>14}  "
        f"{'Torch µs':>10} {'Torch TFLOPS':>13}  "
        f"{'Speedup':>8}"
    )
    print(header)
    print("-" * W)

    for M, K, N in sizes:
        for dtype in dtypes:
            x = torch.randn(M, K, device="cuda", dtype=dtype)

            # Create reference first, then copy its weight into Triton layer
            # so both operate on real (non-zero) data for a fair comparison.
            tl_torch = nn.Linear(K, N, bias=False, device="cuda", dtype=dtype)

            tl = TritonLinear(K, N, bias=False, dtype=dtype)
            tl.weight.copy_(tl_torch.weight)

            # Correctness check
            with torch.no_grad():
                out_triton = tl(x)
                out_torch = tl_torch(x)
            max_err = (out_triton - out_torch).abs().max().item()

            t_triton = benchmark_fn(lambda: tl(x))
            t_torch = benchmark_fn(lambda: tl_torch(x))

            tf_triton = tflops(M, N, K, t_triton)
            tf_torch = tflops(M, N, K, t_torch)
            speedup = tf_triton / tf_torch if tf_torch > 0 else float("inf")

            label = "fp16" if dtype == torch.float16 else "bf16"
            err_flag = " ⚠" if max_err > 1e-1 else ""
            print(
                f"{M:>6} {K:>6} {N:>6} {label:>7}  "
                f"{us(t_triton):>9.1f}µ {tf_triton:>13.2f}  "
                f"{us(t_torch):>9.1f}µ {tf_torch:>12.2f}  "
                f"{speedup:>7.2f}x  err={max_err:.2e}{err_flag}"
            )
            rows.append({
                "M": M, "K": K, "N": N, "dtype": label,
                "triton_us": round(us(t_triton), 1),
                "triton_tflops": round(tf_triton, 3),
                "torch_us": round(us(t_torch), 1),
                "torch_tflops": round(tf_torch, 3),
                "speedup": round(speedup, 3),
            })

    return rows


# ---------------------------------------------------------------------------
# FP4 benchmark
# ---------------------------------------------------------------------------

def bench_fp4(sizes):
    rows = []
    W = 110
    print("\n" + "=" * W)
    print("  MXFP4 (e2m1)  Benchmark: TritonLinearFP4  vs  nn.Linear (fp16)")
    print("=" * W)
    header = (
        f"{'M':>6} {'K':>6} {'N':>6}  "
        f"{'FP4 µs':>9} {'FP4 TFLOPS':>12}  "
        f"{'fp16 µs':>9} {'fp16 TFLOPS':>12}  "
        f"{'Speedup':>8}  {'Mem saved':>10}"
    )
    print(header)
    print("-" * W)

    for M, K, N in sizes:
        if K % 32 != 0 or N % 32 != 0:
            continue

        x = torch.randn(M, K, device="cuda", dtype=torch.float16)
        ref_linear = nn.Linear(K, N, bias=False, device="cuda", dtype=torch.float16)
        fp4_layer = TritonLinearFP4.from_float(ref_linear)

        t_fp4 = benchmark_fn(lambda: fp4_layer(x))
        t_fp16 = benchmark_fn(lambda: ref_linear(x))

        tf_fp4 = tflops(M, N, K, t_fp4)
        tf_fp16 = tflops(M, N, K, t_fp16)
        speedup = tf_fp4 / tf_fp16 if tf_fp16 > 0 else float("inf")

        fp16_bytes = N * K * 2
        fp4_bytes = N * (K // 2) + N * (K // 32)
        mem_ratio = fp4_bytes / fp16_bytes

        print(
            f"{M:>6} {K:>6} {N:>6}  "
            f"{us(t_fp4):>8.1f}µ {tf_fp4:>11.2f}  "
            f"{us(t_fp16):>8.1f}µ {tf_fp16:>11.2f}  "
            f"{speedup:>7.2f}x  "
            f"{(1 - mem_ratio) * 100:>8.1f}%"
        )
        rows.append({
            "M": M, "K": K, "N": N,
            "format": "MXFP4_e2m1",
            "fp4_us": round(us(t_fp4), 1),
            "fp4_tflops": round(tf_fp4, 3),
            "fp16_us": round(us(t_fp16), 1),
            "fp16_tflops": round(tf_fp16, 3),
            "speedup": round(speedup, 3),
            "weight_memory_saved_pct": round((1 - mem_ratio) * 100, 1),
        })

    return rows


# ---------------------------------------------------------------------------
# SVDQ benchmark
# ---------------------------------------------------------------------------

def bench_svdq(sizes):
    rows = []
    W = 130
    print("\n" + "=" * W)
    print("  SVDQ  Benchmark: TritonLinearLoRA  vs  TritonLinearFP4  vs  nn.Linear (fp16)")
    print("=" * W)
    header = (
        f"{'M':>6} {'K':>6} {'N':>6}  "
        f"{'SVDQ µs':>10} {'SVDQ TFLOPS':>14}  "
        f"{'FP4 µs':>8} {'FP4 TFLOPS':>11}  "
        f"{'fp16 µs':>8} {'fp16 TFLOPS':>11}  "
        f"{'vs fp16':>8} {'vs FP4':>7}  "
        f"{'Wt mem':>10}"
    )
    print(header)
    print("-" * W)

    for M, K, N in sizes:
        if K % 32 != 0 or N % 32 != 0:
            continue

        x = torch.randn(M, K, device="cuda", dtype=torch.float16)
        ref_linear = nn.Linear(K, N, bias=False, device="cuda", dtype=torch.float16)

        loraq_layer = TritonLinearLoRA.from_float(ref_linear)
        fp4_layer = TritonLinearFP4.from_float(ref_linear)

        t_loraq = benchmark_fn(lambda: loraq_layer(x))
        t_fp4 = benchmark_fn(lambda: fp4_layer(x))
        t_fp16 = benchmark_fn(lambda: ref_linear(x))

        tf_loraq = tflops(M, N, K, t_loraq)
        tf_fp4 = tflops(M, N, K, t_fp4)
        tf_fp16 = tflops(M, N, K, t_fp16)
        speedup_fp16 = tf_loraq / tf_fp16 if tf_fp16 > 0 else float("inf")
        speedup_fp4 = tf_loraq / tf_fp4 if tf_fp4 > 0 else float("inf")

        fp4_bytes = N * (K // 2) + N * (K // 32)
        lr_bytes = N * 32 * 2 + 32 * K * 2
        loraq_bytes = fp4_bytes + lr_bytes

        print(
            f"{M:>6} {K:>6} {N:>6}  "
            f"{us(t_loraq):>9.1f}µ {tf_loraq:>13.2f}  "
            f"{us(t_fp4):>7.1f}µ {tf_fp4:>10.2f}  "
            f"{us(t_fp16):>7.1f}µ {tf_fp16:>10.2f}  "
            f"{speedup_fp16:>7.2f}x {speedup_fp4:>6.2f}x  "
            f"{loraq_bytes / 1024:>8.1f} KB"
        )
        rows.append({
            "M": M, "K": K, "N": N,
            "format": "SVDMXFP4_e2m1",
            "loraq_us": round(us(t_loraq), 1),
            "loraq_tflops": round(tf_loraq, 3),
            "fp4_us": round(us(t_fp4), 1),
            "fp4_tflops": round(tf_fp4, 3),
            "fp16_us": round(us(t_fp16), 1),
            "fp16_tflops": round(tf_fp16, 3),
            "speedup_vs_fp16": round(speedup_fp16, 3),
            "speedup_vs_fp4": round(speedup_fp4, 3),
            "weight_memory_bytes": loraq_bytes,
        })

    return rows


# ---------------------------------------------------------------------------
# LoRaQ FP8 benchmark:  LoRaQ.1 (tl.dot fp16)  vs  V2 (dot_scaled fp8)
# ---------------------------------------------------------------------------

def bench_loraq_q8(sizes):
    rows = []
    W = 155
    print("\n" + "=" * W)
    print("  LoRaQ FP8  Benchmark: LoRaQ.1 (tl.dot fp16)  vs  V2 (dot_scaled fp8)  vs  TritonLinearFP4  vs  nn.Linear")
    print("  Activation is pre-quantized to MXFP8 before timing (online quant excluded by design)")
    print("=" * W)
    header = (
        f"{'M':>6} {'K':>6} {'N':>6}  "
        f"{'LoRaQ.1 µs':>8} {'LoRaQ.1 TFLOPS':>10}  "
        f"{'V2 µs':>8} {'V2 TFLOPS':>10}  "
        f"{'FP4 µs':>8} {'FP4 TFLOPS':>10}  "
        f"{'fp16 µs':>8} {'fp16 TFLOPS':>11}  "
        f"{'V2/LoRaQ.1':>6} {'LoRaQ.1/fp16':>8} {'V2/fp16':>8}  "
        f"{'Wt mem':>9}"
    )
    print(header)
    print("-" * W)

    at_v1 = AutotunedLoRaQ(loraq_fused_q8_kernel,        LORAQ_Q8_CONFIGS, warmup=5, rep=25)
    at_v2 = AutotunedLoRaQ(loraq_fused_q8_scaled_kernel_8_8, LORAQ_Q8_CONFIGS, warmup=5, rep=25)

    for M, K, N in sizes:
        if K % 64 != 0 or N % 32 != 0:
            continue

        x = torch.randn(M, K, device="cuda", dtype=torch.float16)
        a_fp8, a_scale = dynamic_mxfp8_quant(x)

        ref_linear = nn.Linear(K, N, bias=False, device="cuda", dtype=torch.float16)

        v1_layer  = TritonLinearLoRaQ.from_float(ref_linear)
        fp4_layer = TritonLinearFP4.from_float(ref_linear)
        w_fp4_t   = v1_layer.weight_fp4.t().contiguous()

        # Autotune V1 and V2 (with wpe sweep) then bench with tt.do_bench
        _ = at_v1(a_fp8, a_scale, v1_layer.R_fp8, v1_layer.R_scale,
                  v1_layer.L_fp8, v1_layer.L_scale, w_fp4_t, v1_layer.weight_scale, M, N, K)
        _ = at_v2(a_fp8, a_scale, v1_layer.R_fp8, v1_layer.R_scale,
                  v1_layer.L_fp8, v1_layer.L_scale, w_fp4_t, v1_layer.weight_scale, M, N, K)

        t_v1_ms = tt.do_bench(
            lambda: at_v1(a_fp8, a_scale, v1_layer.R_fp8, v1_layer.R_scale,
                          v1_layer.L_fp8, v1_layer.L_scale, w_fp4_t, v1_layer.weight_scale, M, N, K),
            warmup=WARMUP, rep=ITERS,
        )
        t_v2_ms = tt.do_bench(
            lambda: at_v2(a_fp8, a_scale, v1_layer.R_fp8, v1_layer.R_scale,
                          v1_layer.L_fp8, v1_layer.L_scale, w_fp4_t, v1_layer.weight_scale, M, N, K),
            warmup=WARMUP, rep=ITERS,
        )
        t_v1  = t_v1_ms / 1000.0
        t_v2  = t_v2_ms / 1000.0
        t_fp4 = benchmark_fn(lambda: fp4_layer(x))
        t_fp16 = benchmark_fn(lambda: ref_linear(x))

        tf_v1  = tflops(M, N, K, t_v1)
        tf_v2  = tflops(M, N, K, t_v2)
        tf_fp4 = tflops(M, N, K, t_fp4)
        tf_fp16 = tflops(M, N, K, t_fp16)

        v2_over_v1   = tf_v2  / tf_v1  if tf_v1  > 0 else float("inf")
        v1_over_fp16 = tf_v1  / tf_fp16 if tf_fp16 > 0 else float("inf")
        v2_over_fp16 = tf_v2  / tf_fp16 if tf_fp16 > 0 else float("inf")

        fp4_bytes = N * (K // 2) + N * (K // 32)
        lr_bytes  = 64 * K + 64 * (K // 32) + N * 64 + N * 2
        total_bytes = fp4_bytes + lr_bytes

        print(
            f"{M:>6} {K:>6} {N:>6}  "
            f"{us(t_v1):>7.1f}µ {tf_v1:>9.2f}  "
            f"{us(t_v2):>7.1f}µ {tf_v2:>9.2f}  "
            f"{us(t_fp4):>7.1f}µ {tf_fp4:>9.2f}  "
            f"{us(t_fp16):>7.1f}µ {tf_fp16:>10.2f}  "
            f"{v2_over_v1:>5.2f}x {v1_over_fp16:>7.2f}x {v2_over_fp16:>7.2f}x  "
            f"{total_bytes / 1024:>7.1f} KB"
        )
        rows.append({
            "M": M, "K": K, "N": N,
            "v1_us":  round(us(t_v1),  1),
            "v1_tflops":  round(tf_v1,  3),
            "v2_us":  round(us(t_v2),  1),
            "v2_tflops":  round(tf_v2,  3),
            "fp4_us": round(us(t_fp4), 1),
            "fp4_tflops": round(tf_fp4, 3),
            "fp16_us": round(us(t_fp16), 1),
            "fp16_tflops": round(tf_fp16, 3),
            "v2_over_v1":   round(v2_over_v1, 3),
            "v1_over_fp16": round(v1_over_fp16, 3),
            "v2_over_fp16": round(v2_over_fp16, 3),
            "weight_memory_bytes": total_bytes,
        })

    return rows


# ---------------------------------------------------------------------------
# LoRaQ.1 vs SVDQ comparison
# ---------------------------------------------------------------------------

def bench_v1_vs_loraq(sizes):
    """
    Direct comparison: V1 and V2 (rank=64, MXFP8 L/R, 1 fused kernel)
    vs SVDQ (rank=32, fp16 L/R, 2 kernels: project+quant + dual_gemm).

    V1 Phase 2 = tl.dot fp16; V2 Phase 2 = dot_scaled fp8 (constexpr dims).
    Both V1/V2 take pre-quantized MXFP8 activation; SVDQ takes raw fp16.
    Online quantization is excluded from V1/V2 timing by design.
    """
    rows = []
    W = 160
    print("\n" + "=" * W)
    print("  V1 (tl.dot fp16)  vs  V2 (dot_scaled fp8)  vs  SVDQ (rank=32, fp16, 2 kernels)")
    print("  NOTE: V1/V2 input is pre-quantized MXFP8 — online quant cost excluded by design.")
    print("        SVDQ input is raw fp16.  All store W as MXFP4 (quantized from same nn.Linear).")
    print("=" * W)
    header = (
        f"{'M':>6} {'K':>6} {'N':>6}  "
        f"{'SVDQ µs':>9} {'SVDQ TF':>8}  "
        f"{'V1 µs':>9} {'V1 TF':>8}  "
        f"{'V2 µs':>9} {'V2 TF':>8}  "
        f"{'V1/SVDQ':>8} {'V2/SVDQ':>8} {'V2/V1':>7}  "
        f"{'FP4 µs':>7} {'fp16 µs':>8}"
    )
    print(header)
    print("-" * W)

    at_v1 = AutotunedLoRaQ(loraq_fused_q8_kernel,        LORAQ_Q8_CONFIGS, warmup=5, rep=25)
    at_v2 = AutotunedLoRaQ(loraq_fused_q8_scaled_kernel_8_8, LORAQ_Q8_CONFIGS, warmup=5, rep=25)

    for M, K, N in sizes:
        if K % 64 != 0 or N % 32 != 0:
            continue

        x = torch.randn(M, K, device="cuda", dtype=torch.float16)
        a_fp8, a_scale = dynamic_mxfp8_quant(x)

        ref_linear  = nn.Linear(K, N, bias=False, device="cuda", dtype=torch.float16)
        fp4_layer   = TritonLinearFP4.from_float(ref_linear)
        loraq_layer = TritonLinearLoRA.from_float(ref_linear)   # SVDQ: rank=32, fp16 L/R
        v1_layer    = TritonLinearLoRaQ.from_float(ref_linear)  # V1/V2: rank=64, MXFP8 L/R
        w_fp4_t     = v1_layer.weight_fp4.t().contiguous()

        _ = at_v1(a_fp8, a_scale, v1_layer.R_fp8, v1_layer.R_scale,
                  v1_layer.L_fp8, v1_layer.L_scale, w_fp4_t, v1_layer.weight_scale, M, N, K)
        _ = at_v2(a_fp8, a_scale, v1_layer.R_fp8, v1_layer.R_scale,
                  v1_layer.L_fp8, v1_layer.L_scale, w_fp4_t, v1_layer.weight_scale, M, N, K)

        t_loraq = benchmark_fn(lambda: loraq_layer(x))
        t_v1 = tt.do_bench(
            lambda: at_v1(a_fp8, a_scale, v1_layer.R_fp8, v1_layer.R_scale,
                          v1_layer.L_fp8, v1_layer.L_scale, w_fp4_t, v1_layer.weight_scale, M, N, K),
            warmup=WARMUP, rep=ITERS,
        ) / 1000.0
        t_v2 = tt.do_bench(
            lambda: at_v2(a_fp8, a_scale, v1_layer.R_fp8, v1_layer.R_scale,
                          v1_layer.L_fp8, v1_layer.L_scale, w_fp4_t, v1_layer.weight_scale, M, N, K),
            warmup=WARMUP, rep=ITERS,
        ) / 1000.0
        t_fp4  = benchmark_fn(lambda: fp4_layer(x))
        t_fp16 = benchmark_fn(lambda: ref_linear(x))

        tf_loraq = tflops(M, N, K, t_loraq)
        tf_v1    = tflops(M, N, K, t_v1)
        tf_v2    = tflops(M, N, K, t_v2)

        v1_over_loraq = t_loraq / t_v1 if t_v1 > 0 else float("inf")
        v2_over_loraq = t_loraq / t_v2 if t_v2 > 0 else float("inf")
        v2_over_v1    = t_v1    / t_v2 if t_v2 > 0 else float("inf")

        print(
            f"{M:>6} {K:>6} {N:>6}  "
            f"{us(t_loraq):>8.1f}µ {tf_loraq:>7.2f}  "
            f"{us(t_v1):>8.1f}µ {tf_v1:>7.2f}  "
            f"{us(t_v2):>8.1f}µ {tf_v2:>7.2f}  "
            f"{v1_over_loraq:>7.2f}x {v2_over_loraq:>7.2f}x {v2_over_v1:>6.2f}x  "
            f"{us(t_fp4):>6.1f}µ {us(t_fp16):>7.1f}µ"
        )
        rows.append({
            "M": M, "K": K, "N": N,
            "svdq_us": round(us(t_loraq), 1), "svdq_tflops": round(tf_loraq, 3),
            "v1_us":   round(us(t_v1),    1), "v1_tflops":   round(tf_v1,    3),
            "v2_us":   round(us(t_v2),    1), "v2_tflops":   round(tf_v2,    3),
            "v1_over_svdq": round(v1_over_loraq, 3),
            "v2_over_svdq": round(v2_over_loraq, 3),
            "v2_over_v1":   round(v2_over_v1,    3),
            "fp4_us":  round(us(t_fp4),  1),
            "fp16_us": round(us(t_fp16), 1),
        })

    return rows


# ---------------------------------------------------------------------------
# LoRaQ.1-autotuned vs SVDQ comparison
# ---------------------------------------------------------------------------

def bench_v1_vs_loraq_autotuned(sizes):
    """
    V1 / V2 / V3 / V4 autotuned vs SVDQ autotuned — all at best configs with wpe sweep.

    V1: AutotunedLoRaQ       — kernel 7      (tl.dot fp16 Phase 2, FP8 in / FP8 out)
    V2: AutotunedLoRaQ       — kernel 8_8    (dot_scaled fp8 Phase 2, FP8 in / FP8 out)
    V3: AutotunedLoRaQ_8_16  — kernel 8_16   (FP8 in / FP16 out, no output quant)
    V4: AutotunedLoRaQ_16_8  — kernel 16_8   (FP16 in + channel scale / FP8 out)
    V5: AutotunedLoRaQ_16_16 — kernel 16_16  (FP16 in / FP16 out, in-register quant)
    V6: AutotunedLoRaQ       — kernel 8_8_RL (Phase1=AR only, Phase2=AW+(AR)L)
    SVDQ: AutotunedProjectAndQuant + AutotunedDualGEMM (kernels 5+6)

    V1-V6 share the same MXFP8 L/R and FP4 W weight tensors.
    V4/V5 take raw fp16 A with a per-column channel scale (ones here).
    All sweep waves_per_eu in [0,1,2] via their autotuners.
    """
    rows = []
    W = 230
    print("\n" + "=" * W)
    print("  V1 / V2 / V3 / V4 / V5 / V6 / SVDQ — all fully autotuned (wpe sweep)")
    print("  V1: K7 8→8 fp16-Ph2    V2: K8_8 8→8    V3: K8_16 8→16    V4: K16_8 16→8    V5: K16_16 16→16    V6: K8_8_RL (split loops)")
    print("  SVDQ: rank=32, fp16 L/R, kernels 5+6")
    print("=" * W)
    header = (
        f"{'M':>6} {'K':>6} {'N':>6}  "
        f"{'SVDQ µs':>10} {'SVDQ TF':>8}  "
        f"{'V1 µs':>9} {'V1 TF':>7}  "
        f"{'V2 µs':>9} {'V2 TF':>7}  "
        f"{'V3 µs':>9} {'V3 TF':>7}  "
        f"{'V4 µs':>9} {'V4 TF':>7}  "
        f"{'V5 µs':>9} {'V5 TF':>7}  "
        f"{'V6 µs':>9} {'V6 TF':>7}  "
        f"{'V1/SVD':>7} {'V2/SVD':>7} {'V3/SVD':>7} {'V4/SVD':>7} {'V5/SVD':>7} {'V6/SVD':>7}"
    )
    print(header)
    print("-" * W)

    at_v1 = AutotunedLoRaQ(loraq_fused_q8_kernel,                LORAQ_Q8_CONFIGS, warmup=5, rep=25)
    at_v2 = AutotunedLoRaQ(loraq_fused_q8_scaled_kernel_8_8,     LORAQ_Q8_CONFIGS, warmup=5, rep=25)
    at_v3 = AutotunedLoRaQ_8_16(loraq_fused_q8_scaled_kernel_8_16,   LORAQ_Q8_CONFIGS, warmup=5, rep=25)
    at_v4 = AutotunedLoRaQ_16_8(loraq_fused_q8_scaled_kernel_16_8,   LORAQ_Q8_CONFIGS, warmup=5, rep=25)
    at_v5 = AutotunedLoRaQ_16_16(loraq_fused_q8_scaled_kernel_16_16, LORAQ_Q8_CONFIGS, warmup=5, rep=25)
    at_v6 = AutotunedLoRaQ(loraq_fused_q8_scaled_kernel_8_8_RL,  LORAQ_Q8_CONFIGS, warmup=5, rep=25)
    at_pq = AutotunedProjectAndQuant(loraq_project_and_quant_kernel, warmup=5, rep=25)
    at_dg = AutotunedDualGEMM(loraq_dual_gemm_kernel, LORAQ_Q8_CONFIGS, warmup=5, rep=25)

    for M, K, N in sizes:
        if K % 64 != 0 or N % 32 != 0:
            continue

        x = torch.randn(M, K, device="cuda", dtype=torch.float16)
        a_fp8, a_scale = dynamic_mxfp8_quant(x)
        ref_linear  = nn.Linear(K, N, bias=False, device="cuda", dtype=torch.float16)

        loraq_layer = TritonLinearLoRA.from_float(ref_linear)   # SVDQ: rank=32, fp16 L/R
        v1_layer    = TritonLinearLoRaQ.from_float(ref_linear)  # V1-V4: rank=64, MXFP8 L/R

        # V4/V5: fp16 A with per-column input (K,) and output (N,) channel scales.
        channel_scale_in  = torch.ones(K, dtype=torch.float16, device="cuda")
        channel_scale_out = torch.ones(N, dtype=torch.float16, device="cuda")

        # ---- SVDQ autotuned ----
        w_fp4_t_svd = loraq_layer.weight_fp4.t().contiguous()
        P, a_fp4, a_scale_q = at_pq(x, loraq_layer.R, loraq_layer.channel_scale)
        _ = at_dg(P, loraq_layer.L, a_fp4, a_scale_q,
                  w_fp4_t_svd, loraq_layer.weight_scale, M, N, K)

        def svdq_e2e():
            P_, a_fp4_, a_sq_ = at_pq(x, loraq_layer.R, loraq_layer.channel_scale)
            at_dg(P_, loraq_layer.L, a_fp4_, a_sq_,
                  w_fp4_t_svd, loraq_layer.weight_scale, M, N, K)

        t_svd_ms = tt.do_bench(svdq_e2e, warmup=WARMUP, rep=ITERS)

        # ---- V1 / V2 / V3 / V6 autotuned (share FP8 L/R weights, same call signature) ----
        w_fp4_t = v1_layer.weight_fp4.t().contiguous()
        for at in (at_v1, at_v2, at_v3, at_v6):
            at(a_fp8, a_scale,
               v1_layer.R_fp8, v1_layer.R_scale,
               v1_layer.L_fp8, v1_layer.L_scale,
               w_fp4_t, v1_layer.weight_scale, M, N, K)

        t_v1_ms = tt.do_bench(
            lambda: at_v1(a_fp8, a_scale,
                          v1_layer.R_fp8, v1_layer.R_scale,
                          v1_layer.L_fp8, v1_layer.L_scale,
                          w_fp4_t, v1_layer.weight_scale, M, N, K),
            warmup=WARMUP, rep=ITERS,
        )
        t_v2_ms = tt.do_bench(
            lambda: at_v2(a_fp8, a_scale,
                          v1_layer.R_fp8, v1_layer.R_scale,
                          v1_layer.L_fp8, v1_layer.L_scale,
                          w_fp4_t, v1_layer.weight_scale, M, N, K),
            warmup=WARMUP, rep=ITERS,
        )
        t_v3_ms = tt.do_bench(
            lambda: at_v3(a_fp8, a_scale,
                          v1_layer.R_fp8, v1_layer.R_scale,
                          v1_layer.L_fp8, v1_layer.L_scale,
                          w_fp4_t, v1_layer.weight_scale, M, N, K),
            warmup=WARMUP, rep=ITERS,
        )

        t_v6_ms = tt.do_bench(
            lambda: at_v6(a_fp8, a_scale,
                          v1_layer.R_fp8, v1_layer.R_scale,
                          v1_layer.L_fp8, v1_layer.L_scale,
                          w_fp4_t, v1_layer.weight_scale, M, N, K),
            warmup=WARMUP, rep=ITERS,
        )

        # ---- V4 autotuned (fp16 A + input/output channel scales, FP8 out) ----
        at_v4(x, channel_scale_in, channel_scale_out,
              v1_layer.R_fp8, v1_layer.R_scale,
              v1_layer.L_fp8, v1_layer.L_scale,
              w_fp4_t, v1_layer.weight_scale, M, N, K)
        t_v4_ms = tt.do_bench(
            lambda: at_v4(x, channel_scale_in, channel_scale_out,
                          v1_layer.R_fp8, v1_layer.R_scale,
                          v1_layer.L_fp8, v1_layer.L_scale,
                          w_fp4_t, v1_layer.weight_scale, M, N, K),
            warmup=WARMUP, rep=ITERS,
        )

        # ---- V5 autotuned (fp16 A + input channel scale, fp16 out) ----
        at_v5(x, channel_scale_in,
              v1_layer.R_fp8, v1_layer.R_scale,
              v1_layer.L_fp8, v1_layer.L_scale,
              w_fp4_t, v1_layer.weight_scale, M, N, K)
        t_v5_ms = tt.do_bench(
            lambda: at_v5(x, channel_scale_in,
                          v1_layer.R_fp8, v1_layer.R_scale,
                          v1_layer.L_fp8, v1_layer.L_scale,
                          w_fp4_t, v1_layer.weight_scale, M, N, K),
            warmup=WARMUP, rep=ITERS,
        )

        t_svd = t_svd_ms / 1000.0
        t_v1  = t_v1_ms  / 1000.0
        t_v2  = t_v2_ms  / 1000.0
        t_v3  = t_v3_ms  / 1000.0
        t_v4  = t_v4_ms  / 1000.0
        t_v5  = t_v5_ms  / 1000.0
        t_v6  = t_v6_ms  / 1000.0

        tf_svd = tflops(M, N, K, t_svd)
        tf_v1  = tflops(M, N, K, t_v1)
        tf_v2  = tflops(M, N, K, t_v2)
        tf_v3  = tflops(M, N, K, t_v3)
        tf_v4  = tflops(M, N, K, t_v4)
        tf_v5  = tflops(M, N, K, t_v5)
        tf_v6  = tflops(M, N, K, t_v6)

        v1_over_svd = t_svd / t_v1 if t_v1 > 0 else float("inf")
        v2_over_svd = t_svd / t_v2 if t_v2 > 0 else float("inf")
        v3_over_svd = t_svd / t_v3 if t_v3 > 0 else float("inf")
        v4_over_svd = t_svd / t_v4 if t_v4 > 0 else float("inf")
        v5_over_svd = t_svd / t_v5 if t_v5 > 0 else float("inf")
        v6_over_svd = t_svd / t_v6 if t_v6 > 0 else float("inf")

        print(
            f"{M:>6} {K:>6} {N:>6}  "
            f"{us(t_svd):>9.1f}µ {tf_svd:>7.2f}  "
            f"{us(t_v1):>8.1f}µ {tf_v1:>6.2f}  "
            f"{us(t_v2):>8.1f}µ {tf_v2:>6.2f}  "
            f"{us(t_v3):>8.1f}µ {tf_v3:>6.2f}  "
            f"{us(t_v4):>8.1f}µ {tf_v4:>6.2f}  "
            f"{us(t_v5):>8.1f}µ {tf_v5:>6.2f}  "
            f"{us(t_v6):>8.1f}µ {tf_v6:>6.2f}  "
            f"{v1_over_svd:>6.2f}x {v2_over_svd:>6.2f}x {v3_over_svd:>6.2f}x "
            f"{v4_over_svd:>6.2f}x {v5_over_svd:>6.2f}x {v6_over_svd:>6.2f}x"
        )
        rows.append({
            "M": M, "K": K, "N": N,
            "svdq_us": round(us(t_svd), 1), "svdq_tflops": round(tf_svd, 3),
            "v1_us":   round(us(t_v1),  1), "v1_tflops":   round(tf_v1,  3),
            "v2_us":   round(us(t_v2),  1), "v2_tflops":   round(tf_v2,  3),
            "v3_us":   round(us(t_v3),  1), "v3_tflops":   round(tf_v3,  3),
            "v4_us":   round(us(t_v4),  1), "v4_tflops":   round(tf_v4,  3),
            "v5_us":   round(us(t_v5),  1), "v5_tflops":   round(tf_v5,  3),
            "v6_us":   round(us(t_v6),  1), "v6_tflops":   round(tf_v6,  3),
            "v1_over_svdq": round(v1_over_svd, 3),
            "v2_over_svdq": round(v2_over_svd, 3),
            "v3_over_svdq": round(v3_over_svd, 3),
            "v4_over_svdq": round(v4_over_svd, 3),
            "v5_over_svdq": round(v5_over_svd, 3),
            "v6_over_svdq": round(v6_over_svd, 3),
        })

    return rows


# ---------------------------------------------------------------------------
# LoRaQ.1-autotuned vs SVDQ (wall-clock benchmark_fn version)
# ---------------------------------------------------------------------------

def bench_v1_vs_loraq_wallclock(sizes):
    """
    LoRaQ.1 vs SVDQ using actual module forward() calls with benchmark_fn.

    Both modules now autotune internally on first call, so this benchmarks
    the fully-optimized modules including all Python dispatch overhead.
    """
    rows = []
    W = 130
    print("\n" + "=" * W)
    print("  LoRaQ.1 vs SVDQ  (actual modules, wall-clock benchmark_fn)")
    print("  LoRaQ.1: TritonLinearLoRaQ.forward() — autotuned kernel 7")
    print("  SVDQ: TritonLinearLoRA.forward() — autotuned kernels 5+6")
    print("  NOTE: first call triggers autotuning; subsequent calls use cached config")
    print("=" * W)
    header = (
        f"{'M':>6} {'K':>6} {'N':>6}  "
        f"{'SVDQ µs':>10} {'LQ TFLOPS':>10}  "
        f"{'LoRaQ.1 µs':>10} {'LoRaQ.1 TFLOPS':>10}  "
        f"{'LoRaQ.1/LQ':>6}  "
        f"{'fp16 µs':>9}"
    )
    print(header)
    print("-" * W)

    for M, K, N in sizes:
        if K % 64 != 0 or N % 32 != 0:
            continue

        x = torch.randn(M, K, device="cuda", dtype=torch.float16)
        a_fp8, a_scale = dynamic_mxfp8_quant(x)
        ref_linear = nn.Linear(K, N, bias=False, device="cuda", dtype=torch.float16)

        loraq_layer = TritonLinearLoRA.from_float(ref_linear)
        v1_layer    = TritonLinearLoRaQ.from_float(ref_linear)

        # Warmup / trigger autotuning (first call tunes)
        _ = loraq_layer(x)
        _ = v1_layer(a_fp8, a_scale)

        # Measure actual module forward() with benchmark_fn
        t_loraq = benchmark_fn(lambda: loraq_layer(x))
        t_v1    = benchmark_fn(lambda: v1_layer(a_fp8, a_scale))
        t_fp16  = benchmark_fn(lambda: ref_linear(x))

        tf_loraq = tflops(M, N, K, t_loraq)
        tf_v1    = tflops(M, N, K, t_v1)

        v1_over_loraq = t_loraq / t_v1 if t_v1 > 0 else float("inf")

        print(
            f"{M:>6} {K:>6} {N:>6}  "
            f"{us(t_loraq):>9.1f}µ {tf_loraq:>9.2f}  "
            f"{us(t_v1):>9.1f}µ {tf_v1:>9.2f}  "
            f"{v1_over_loraq:>5.2f}x  "
            f"{us(t_fp16):>8.1f}µ"
        )
        rows.append({
            "M": M, "K": K, "N": N,
            "loraq_us":    round(us(t_loraq), 1),
            "loraq_tflops": round(tf_loraq, 3),
            "v1_us":       round(us(t_v1), 1),
            "v1_tflops":   round(tf_v1, 3),
            "v1_over_loraq": round(v1_over_loraq, 3),
            "fp16_us":     round(us(t_fp16), 1),
        })

    return rows


# ---------------------------------------------------------------------------
# Autotuned vs fixed-config comparison
# ---------------------------------------------------------------------------

def bench_autotuned(sizes):
    rows = []
    W = 80
    print("\n" + "=" * W)
    print("  Autotuned Benchmark: LoRaQ.1 vs LoRaQ.2  (all fully autotuned)")
    print("  Note: first run for each (M,K,N) triggers autotuning (31 configs × do_bench)")
    print("=" * W)
    header = (
        f"{'M':>6} {'K':>6} {'N':>6}  "
        f"{'LoRaQ.1 µs':>14}  "
        f"{'LoRaQ.2 µs':>14}  "
        f"{'LoRaQ.2 / LoRaQ.1':>19}"
    )
    print(header)
    print("-" * W)

    at_v1 = AutotunedLoRaQ(loraq_fused_q8_kernel, LORAQ_Q8_CONFIGS, warmup=5, rep=25)
    at_v2 = AutotunedLoRaQ(loraq_fused_q8_scaled_kernel_8_8, LORAQ_Q8_CONFIGS, warmup=5, rep=25)

    for M, K, N in sizes:
        if K % 64 != 0 or N % 32 != 0:
            continue

        x = torch.randn(M, K, device="cuda", dtype=torch.float16)
        a_fp8, a_scale = dynamic_mxfp8_quant(x)
        ref_linear = nn.Linear(K, N, bias=False, device="cuda", dtype=torch.float16)

        v1_layer = TritonLinearLoRaQ.from_float(ref_linear)
        w_fp4_t  = v1_layer.weight_fp4.t().contiguous()

        t_v1_tuned = tt.do_bench(
            lambda: at_v1(
                a_fp8, a_scale,
                v1_layer.R_fp8, v1_layer.R_scale,
                v1_layer.L_fp8, v1_layer.L_scale,
                w_fp4_t, v1_layer.weight_scale,
                M, N, K,
            ),
            warmup=WARMUP, rep=ITERS,
        ) / 1000.0
        t_v2_tuned = tt.do_bench(
            lambda: at_v2(
                a_fp8, a_scale,
                v1_layer.R_fp8, v1_layer.R_scale,
                v1_layer.L_fp8, v1_layer.L_scale,
                w_fp4_t, v1_layer.weight_scale,
                M, N, K,
            ),
            warmup=WARMUP, rep=ITERS,
        ) / 1000.0

        v2_over_v1 = t_v2_tuned / t_v1_tuned if t_v1_tuned > 0 else float("inf")

        print(
            f"{M:>6} {K:>6} {N:>6}  "
            f"{us(t_v1_tuned):>13.1f}µ  "
            f"{us(t_v2_tuned):>13.1f}µ  "
            f"{v2_over_v1:>18.2f}x"
        )
        rows.append({
            "M": M, "K": K, "N": N,
            "v1_tuned_us":  round(us(t_v1_tuned),  1),
            "v2_tuned_us":  round(us(t_v2_tuned),  1),
            "v2_over_v1":   round(v2_over_v1, 3),
        })

    return rows


# ---------------------------------------------------------------------------
# LoRaQ.3 (fp16io) vs LoRaQ.1 comparison
# ---------------------------------------------------------------------------

def bench_loraq3(sizes):
    """
    LoRaQ.3 (fp16 in, fp16 out, fused input quant) vs LoRaQ.1 (fp8 in, fp8 out).
    Both fully autotuned.

    LoRaQ.3 avoids output quantization cost (beneficial when N >> K).
    LoRaQ.1 is autotuned via AutotunedLoRaQ; timing excludes input quant.
    """
    rows = []
    W = 85
    print("\n" + "=" * W)
    print("  LoRaQ.3 (fp16io, autotuned) vs LoRaQ.1 (fp8 in/out, autotuned)")
    print("  LoRaQ.3: fp16 in, fused input quant, fp16 out (kernel 9)")
    print("  LoRaQ.1: fp8 in (pre-quant excluded), fp8 out (kernel 7)")
    print("  Both swept over LORAQ_Q8_CONFIGS, best config cached per (M,N,K)")
    print("=" * W)
    header = (
        f"{'M':>6} {'K':>6} {'N':>6}  "
        f"{'LoRaQ.1 µs':>12}  "
        f"{'LoRaQ.3 µs':>12}  "
        f"{'LoRaQ.3/LoRaQ.1':>17}"
    )
    print(header)
    print("-" * W)

    at_v1 = AutotunedLoRaQ(loraq_fused_q8_kernel, LORAQ_Q8_CONFIGS, warmup=5, rep=25)
    at_v3 = AutotunedLoRaQ3(loraq_fused_fp16io_kernel, LORAQ_Q8_CONFIGS, warmup=5, rep=25)

    for M, K, N in sizes:
        if K % 64 != 0 or N % 32 != 0:
            continue

        x = torch.randn(M, K, device="cuda", dtype=torch.float16)
        a_fp8, a_scale = dynamic_mxfp8_quant(x)
        ref_linear = nn.Linear(K, N, bias=False, device="cuda", dtype=torch.float16)

        v1_layer = TritonLinearLoRaQ.from_float(ref_linear)
        v3_layer = TritonLinearLoRaQ3.from_float(ref_linear)

        # ---- LoRaQ.1 autotuned (kernel 7) ----
        w_fp4_t_v1 = v1_layer.weight_fp4.t().contiguous()
        _ = at_v1(a_fp8, a_scale,
                  v1_layer.R_fp8, v1_layer.R_scale,
                  v1_layer.L_fp8, v1_layer.L_scale,
                  w_fp4_t_v1, v1_layer.weight_scale, M, N, K)

        t_v1_ms = tt.do_bench(
            lambda: at_v1(
                a_fp8, a_scale,
                v1_layer.R_fp8, v1_layer.R_scale,
                v1_layer.L_fp8, v1_layer.L_scale,
                w_fp4_t_v1, v1_layer.weight_scale, M, N, K,
            ),
            warmup=WARMUP, rep=ITERS,
        )
        t_v1 = t_v1_ms / 1000.0

        # ---- LoRaQ.3 autotuned (kernel 9) ----
        w_fp4_t_v3 = v3_layer.weight_fp4.t().contiguous()
        _ = at_v3(x, v3_layer.R_fp8, v3_layer.R_scale,
                  v3_layer.L_fp8, v3_layer.L_scale,
                  w_fp4_t_v3, v3_layer.weight_scale, M, N, K)

        t_v3_ms = tt.do_bench(
            lambda: at_v3(
                x, v3_layer.R_fp8, v3_layer.R_scale,
                v3_layer.L_fp8, v3_layer.L_scale,
                w_fp4_t_v3, v3_layer.weight_scale, M, N, K,
            ),
            warmup=WARMUP, rep=ITERS,
        )
        t_v3 = t_v3_ms / 1000.0

        # Speedup: > 1 means LoRaQ.3 is faster
        v3_speedup = t_v1 / t_v3 if t_v3 > 0 else float("inf")

        print(
            f"{M:>6} {K:>6} {N:>6}  "
            f"{us(t_v1):>11.1f}µ  "
            f"{us(t_v3):>11.1f}µ  "
            f"{v3_speedup:>16.2f}x"
        )
        rows.append({
            "M": M, "K": K, "N": N,
            "v1_tuned_us": round(us(t_v1), 1),
            "v3_tuned_us": round(us(t_v3), 1),
            "v3_speedup":  round(v3_speedup, 3),
        })

    return rows


# ---------------------------------------------------------------------------
# K7 (FP8) vs K13 (FP4) vs SVDQ comparison
# ---------------------------------------------------------------------------

def bench_v1_vs_q4_vs_svdq(sizes):
    """
    Three-way comparison:
      K7   (loraq_fused_q8_kernel)  — MXFP8 activation, FP8 R/L, FP4 W
      K13  (loraq_fused_q4_kernel)  — MXFP8 activation, FP8 R, FP4 L/W, FP8 out
      K14  (loraq_fused_q4_kernel_4_16) — MXFP8 activation, FP8 R, FP4 L/W, FP16 out
      SVDQ (kernels 5+6)            — FP16 activation, rank=32 FP16 L/R

    K7 uses rank=64 FP8 L/R. K13/K14 use rank=128 with FP4 R/L.
    Online activation quantisation is excluded from K7/K13/K14 timing by design.
    """
    rows = []
    W = 175
    print("\n" + "=" * W)
    print("  K7 (FP8 in/out) vs K13 (FP8 A, FP4 R/L/W, FP8 out) vs K14 (FP8 A, FP4 R/L/W, FP16 out) vs SVDQ — all fully autotuned")
    print("  K7:   MXFP8 activation, FP8 R/L (rank=64),  FP4 W")
    print("  K13:  MXFP8 activation, FP4 R/L (rank=128), FP4 W, FP8 out")
    print("  K14:  MXFP8 activation, FP4 R/L (rank=128), FP4 W, FP16 out")
    print("  SVDQ: FP16 activation,  FP16 R/L (rank=32),  FP4 W — kernels 5+6")
    print("=" * W)
    header = (
        f"{'M':>6} {'K':>6} {'N':>6}  "
        f"{'SVDQ µs':>10} {'SVDQ TF':>8}  "
        f"{'K7 µs':>9} {'K7 TF':>7}  "
        f"{'K13 µs':>9} {'K13 TF':>7}  "
        f"{'K14 µs':>9} {'K14 TF':>7}  "
        f"{'K7/SVD':>7} {'K13/SVD':>8} {'K14/SVD':>8} {'K13/K7':>7} {'K14/K7':>7}"
    )
    print(header)
    print("-" * W)

    at_k7   = AutotunedLoRaQ(loraq_fused_q8_kernel,       LORAQ_Q8_CONFIGS, warmup=5, rep=25)
    at_k13  = AutotunedLoRaQ4(loraq_fused_q4_kernel,      LORAQ_Q8_CONFIGS, warmup=5, rep=25)
    at_k14  = AutotunedLoRaQ4_16(loraq_fused_q4_kernel_4_16, LORAQ_Q8_CONFIGS, warmup=5, rep=25)
    at_pq   = AutotunedProjectAndQuant(loraq_project_and_quant_kernel, warmup=5, rep=25)
    at_dg   = AutotunedDualGEMM(loraq_dual_gemm_kernel, LORAQ_Q8_CONFIGS, warmup=5, rep=25)

    for M, K, N in sizes:
        if K % 64 != 0 or N % 32 != 0:
            continue

        x = torch.randn(M, K, device="cuda", dtype=torch.float16)

        # Activations: FP8 for K7/K13/K14 (all pre-quantised, cost excluded)
        a_fp8, a_scale_fp8 = dynamic_mxfp8_quant(x)

        ref_linear  = nn.Linear(K, N, bias=False, device="cuda", dtype=torch.float16)
        loraq_layer = TritonLinearLoRA.from_float(ref_linear)   # SVDQ rank=32 FP16 L/R
        v1_layer    = TritonLinearLoRaQ.from_float(ref_linear)  # K7/K13 rank=64 FP8 L/R

        # Compute rank-128 FP4 R and L for K13/K14 via direct SVD of the FP4 weight residual.
        # rank must match at_k13.rank (128) — cannot reuse K7's rank-64 FP8 tensors.
        rank_q4  = at_k13.rank   # 128
        w_deq    = mxfp4_to_f32(v1_layer.weight_fp4)                            # (N, K) normalised fp32
        s_exp    = e8m0_to_f32(v1_layer.weight_scale).repeat_interleave(32, dim=-1)  # (N, K) fp32
        residual = (ref_linear.weight.to(torch.float16).cuda().float() - w_deq * s_exp)
        U, S, Vh = torch.linalg.svd(residual, full_matrices=False)
        sqrt_S   = S[:rank_q4].sqrt()
        R_fp16   = (sqrt_S[:, None] * Vh[:rank_q4, :]).to(torch.float16)        # (128, K)
        L_fp16   = (U[:, :rank_q4]  * sqrt_S[None, :]).to(torch.float16)        # (N, 128)
        R_fp4, R_scale_fp4 = dynamic_mxfp4_quant(R_fp16)    # (128, K//2), (128, K//32)
        L_fp4, L_scale_fp4 = dynamic_mxfp4_quant(L_fp16)    # (N, 64),    (N, 4)

        w_fp4_t     = v1_layer.weight_fp4.t().contiguous()
        w_fp4_t_svd = loraq_layer.weight_fp4.t().contiguous()

        # ---- SVDQ autotuned ----
        P, a_fp4_svd, a_sq_svd = at_pq(x, loraq_layer.R, loraq_layer.channel_scale)
        _ = at_dg(P, loraq_layer.L, a_fp4_svd, a_sq_svd,
                  w_fp4_t_svd, loraq_layer.weight_scale, M, N, K)

        def svdq_e2e():
            P_, a_, s_ = at_pq(x, loraq_layer.R, loraq_layer.channel_scale)
            at_dg(P_, loraq_layer.L, a_, s_,
                  w_fp4_t_svd, loraq_layer.weight_scale, M, N, K)

        t_svd_ms = tt.do_bench(svdq_e2e, warmup=WARMUP, rep=ITERS)

        # ---- K7 autotuned ----
        _ = at_k7(a_fp8, a_scale_fp8,
                  v1_layer.R_fp8, v1_layer.R_scale,
                  v1_layer.L_fp8, v1_layer.L_scale,
                  w_fp4_t, v1_layer.weight_scale, M, N, K)
        t_k7_ms = tt.do_bench(
            lambda: at_k7(a_fp8, a_scale_fp8,
                          v1_layer.R_fp8, v1_layer.R_scale,
                          v1_layer.L_fp8, v1_layer.L_scale,
                          w_fp4_t, v1_layer.weight_scale, M, N, K),
            warmup=WARMUP, rep=ITERS,
        )

        # ---- K13 autotuned (MXFP8 A, MXFP4 R/L/W, MXFP8 out) ----
        _ = at_k13(a_fp8, a_scale_fp8,
                   R_fp4, R_scale_fp4,
                   L_fp4, L_scale_fp4,
                   w_fp4_t, v1_layer.weight_scale, M, N, K)
        t_k13_ms = tt.do_bench(
            lambda: at_k13(a_fp8, a_scale_fp8,
                           R_fp4, R_scale_fp4,
                           L_fp4, L_scale_fp4,
                           w_fp4_t, v1_layer.weight_scale, M, N, K),
            warmup=WARMUP, rep=ITERS,
        )

        # ---- K14 autotuned (MXFP8 A, MXFP4 R/L/W, FP16 out) ----
        _ = at_k14(a_fp8, a_scale_fp8,
                   R_fp4, R_scale_fp4,
                   L_fp4, L_scale_fp4,
                   w_fp4_t, v1_layer.weight_scale, M, N, K)
        t_k14_ms = tt.do_bench(
            lambda: at_k14(a_fp8, a_scale_fp8,
                           R_fp4, R_scale_fp4,
                           L_fp4, L_scale_fp4,
                           w_fp4_t, v1_layer.weight_scale, M, N, K),
            warmup=WARMUP, rep=ITERS,
        )

        t_svd = t_svd_ms  / 1000.0
        t_k7  = t_k7_ms   / 1000.0
        t_k13 = t_k13_ms  / 1000.0
        t_k14 = t_k14_ms  / 1000.0

        tf_svd = tflops(M, N, K, t_svd)
        tf_k7  = tflops(M, N, K, t_k7)
        tf_k13 = tflops(M, N, K, t_k13)
        tf_k14 = tflops(M, N, K, t_k14)

        k7_over_svd  = t_svd / t_k7  if t_k7  > 0 else float("inf")
        k13_over_svd = t_svd / t_k13 if t_k13 > 0 else float("inf")
        k14_over_svd = t_svd / t_k14 if t_k14 > 0 else float("inf")
        k13_over_k7  = t_k7  / t_k13 if t_k13 > 0 else float("inf")
        k14_over_k7  = t_k7  / t_k14 if t_k14 > 0 else float("inf")

        print(
            f"{M:>6} {K:>6} {N:>6}  "
            f"{us(t_svd):>9.1f}µ {tf_svd:>7.2f}  "
            f"{us(t_k7):>8.1f}µ {tf_k7:>6.2f}  "
            f"{us(t_k13):>8.1f}µ {tf_k13:>6.2f}  "
            f"{us(t_k14):>8.1f}µ {tf_k14:>6.2f}  "
            f"{k7_over_svd:>6.2f}x {k13_over_svd:>7.2f}x {k14_over_svd:>7.2f}x "
            f"{k13_over_k7:>6.2f}x {k14_over_k7:>6.2f}x"
        )
        rows.append({
            "M": M, "K": K, "N": N,
            "svdq_us": round(us(t_svd),  1), "svdq_tflops": round(tf_svd,  3),
            "k7_us":   round(us(t_k7),   1), "k7_tflops":   round(tf_k7,   3),
            "k13_us":  round(us(t_k13),  1), "k13_tflops":  round(tf_k13,  3),
            "k14_us":  round(us(t_k14),  1), "k14_tflops":  round(tf_k14,  3),
            "k7_over_svdq":  round(k7_over_svd,  3),
            "k13_over_svdq": round(k13_over_svd, 3),
            "k14_over_svdq": round(k14_over_svd, 3),
            "k13_over_k7":   round(k13_over_k7,  3),
            "k14_over_k7":   round(k14_over_k7,  3),
        })

    return rows


# ---------------------------------------------------------------------------
# Pixart-σ transformer block benchmark
# ---------------------------------------------------------------------------

PIXART_HIDDEN   = 1152
PIXART_FFN      = 4608
PIXART_SEQ_LENS = [512, 1024, 2048, 4096]


def _dequant_mxfp8(fp8_data: torch.Tensor, scale_e8m0: torch.Tensor) -> torch.Tensor:
    """(M, N) float8_e4m3fn + (M, N//32) uint8 e8m0 → (M, N) fp16."""
    scale = (scale_e8m0.float() - 127).exp2_().repeat_interleave(32, dim=1)
    return (fp8_data.float() * scale).to(torch.float16)


def bench_svdq_vs_loraq_pixarts_tblock():
    """
    End-to-end Pixart-σ transformer block comparison.

    One block = QKV proj + O proj + FFN-up + FFN-down + 2× LayerNorm.
    Attention is mocked (V pass-through) so linear layer data movement dominates.

    SVDQ block     — all FP16 I/O via TritonLinearLoRA (rank=32, FP4 W, FP16 L/R):
        FP16 → QKV(FP16) → O(FP16) → FFN-up(FP16) → GELU → FFN-dn(FP16)

    LoRaQ-Mixed — all 16→16 via TritonLinearLoRaQ_16_16 (rank=64, FP4 W, FP8 L/R):
        FP16 → QKV(FP16) → O(FP16) → FFN-up(FP16) → GELU → FFN-dn(FP16)
        (in-register MXFP8 quant per K-tile; no explicit quant/dequant between layers)
    """
    hidden  = PIXART_HIDDEN
    ffn_dim = PIXART_FFN
    dev     = "cuda"
    dtype   = torch.float16

    def _make_linear(k, n):
        return nn.Linear(k, n, bias=False, device=dev, dtype=dtype)

    print("\nInitialising SVDQ layers (SVD of quantisation residual) ...")
    qkv_svdq = TritonLinearLoRA.from_float(_make_linear(hidden, 3 * hidden), out_dtype=dtype)
    o_svdq   = TritonLinearLoRA.from_float(_make_linear(hidden, hidden),     out_dtype=dtype)
    up_svdq  = TritonLinearLoRA.from_float(_make_linear(hidden, ffn_dim),    out_dtype=dtype)
    dn_svdq  = TritonLinearLoRA.from_float(_make_linear(ffn_dim, hidden),    out_dtype=dtype)

    print("Initialising LoRaQ-Mixed layers (all 16→16) ...")
    qkv_lq = TritonLinearLoRaQ_16_16.from_float(_make_linear(hidden, 3 * hidden))
    o_lq   = TritonLinearLoRaQ_16_16.from_float(_make_linear(hidden, hidden))
    up_lq  = TritonLinearLoRaQ_16_16.from_float(_make_linear(hidden, ffn_dim))
    dn_lq  = TritonLinearLoRaQ_16_16.from_float(_make_linear(ffn_dim, hidden))

    norm1 = nn.LayerNorm(hidden, device=dev, dtype=dtype)
    norm2 = nn.LayerNorm(hidden, device=dev, dtype=dtype)

    rows = []
    W = 90
    print("\n" + "=" * W)
    print("  SVDQ vs LoRaQ-Mixed — Pixart-σ transformer block")
    print("  SVDQ : all FP16 I/O (TritonLinearLoRA, rank=32, FP4 W, FP16 L/R)")
    print("  LoRaQ: all 16→16  (TritonLinearLoRaQ_16_16, rank=64, FP4 W, FP8 L/R, in-register quant)")
    print("=" * W)
    header = (
        f"{'M':>6}  "
        f"{'SVDQ µs':>10}  {'LoRaQ µs':>10}  {'Speedup':>9}"
    )
    print(header)
    print("-" * W)

    for M in PIXART_SEQ_LENS:
        x = torch.randn(M, hidden, device=dev, dtype=dtype)

        def svdq_fwd():
            h    = norm1(x)
            qkv  = qkv_svdq(h)
            v    = qkv[:, 2 * hidden:]
            o    = o_svdq(v)
            r    = x + o
            h2   = norm2(r)
            u    = up_svdq(h2)
            u    = F.gelu(u)
            d    = dn_svdq(u)
            return r + d

        def loraq_fwd():
            h   = norm1(x)
            qkv = qkv_lq(h)
            v   = qkv[:, 2 * hidden:]
            o   = o_lq(v)
            r   = x + o
            h2  = norm2(r)
            u   = up_lq(h2)
            u   = F.gelu(u)
            d   = dn_lq(u)
            return r + d

        _ = svdq_fwd()
        _ = loraq_fwd()

        t_svdq_ms  = tt.do_bench(svdq_fwd,  warmup=WARMUP, rep=ITERS)
        t_loraq_ms = tt.do_bench(loraq_fwd, warmup=WARMUP, rep=ITERS)

        t_svdq  = t_svdq_ms  / 1000.0
        t_loraq = t_loraq_ms / 1000.0
        speedup = t_svdq / t_loraq if t_loraq > 0 else float("inf")

        print(
            f"{M:>6}  "
            f"{us(t_svdq):>9.1f}µ  {us(t_loraq):>9.1f}µ  {speedup:>8.2f}x"
        )
        rows.append({
            "M": M, "hidden": hidden, "ffn_dim": ffn_dim,
            "svdq_us":  round(us(t_svdq),  1),
            "loraq_us": round(us(t_loraq), 1),
            "speedup":  round(speedup, 3),
        })

    return rows


# ---------------------------------------------------------------------------
# LoRaQ.6 (fp16 L,R) vs LoRaQ.1 (fp8 L,R) comparison
# ---------------------------------------------------------------------------

def bench_loraq_fp16lr(sizes):
    """
    LoRaQ.6 (K13: fp16 L,R, dequant A in low-rank branch) vs LoRaQ.1 (K7: fp8 L,R).
    Both autotuned. Measures overhead of using fp16 L/R vs fp8 L/R.
    """
    rows = []
    W = 85
    print("\n" + "=" * W)
    print("  LoRaQ.6 (fp16 L,R) vs LoRaQ.1 (fp8 L,R)  — autotuned")
    print("  LoRaQ.6: K13, dequant A→fp16 for A×R^T, L loaded as fp16")
    print("  LoRaQ.1: K7, dot_scaled for A×R^T, L dequanted from fp8")
    print("=" * W)
    header = (
        f"{'M':>6} {'K':>6} {'N':>6}  "
        f"{'LoRaQ.1 µs':>12}  "
        f"{'LoRaQ.6 µs':>12}  "
        f"{'LoRaQ.6/LoRaQ.1':>17}"
    )
    print(header)
    print("-" * W)

    at_v1 = AutotunedLoRaQ(loraq_fused_q8_kernel, LORAQ_Q8_CONFIGS, warmup=5, rep=25)
    at_v6 = AutotunedLoRaQFP16LR(loraq_fused_q8_fp16lr_kernel, LORAQ_Q8_CONFIGS, warmup=5, rep=25)

    for M, K, N in sizes:
        if K % 64 != 0 or N % 32 != 0:
            continue

        x = torch.randn(M, K, device="cuda", dtype=torch.float16)
        a_fp8, a_scale = dynamic_mxfp8_quant(x)
        ref_linear = nn.Linear(K, N, bias=False, device="cuda", dtype=torch.float16)

        v1_layer = TritonLinearLoRaQ.from_float(ref_linear)

        from loraq.quant import dynamic_mxfp4_quant, mxfp4_to_f32, e8m0_to_f32
        W_full = ref_linear.weight.to(torch.float16)
        w_fp4, w_scale_q = dynamic_mxfp4_quant(W_full)
        w_deq = mxfp4_to_f32(w_fp4)
        s_f32 = e8m0_to_f32(w_scale_q).repeat_interleave(32, dim=-1)
        w_recon = (w_deq * s_f32).to(torch.float16).to("cuda")
        residual = (W_full - w_recon).float()
        U, S, Vh = torch.linalg.svd(residual, full_matrices=False)
        rank = 64
        sqrt_S = S[:rank].sqrt()
        L_fp16 = (U[:, :rank] * sqrt_S[None, :]).to(torch.float16).contiguous()
        R_fp16 = (sqrt_S[:, None] * Vh[:rank, :]).to(torch.float16).contiguous()

        w_fp4_t = v1_layer.weight_fp4.t().contiguous()
        channel_scale = torch.ones(N, dtype=torch.float16, device="cuda")

        _ = at_v1(a_fp8, a_scale,
                  v1_layer.R_fp8, v1_layer.R_scale,
                  v1_layer.L_fp8, v1_layer.L_scale,
                  w_fp4_t, v1_layer.weight_scale, M, N, K)
        t_v1_ms = tt.do_bench(
            lambda: at_v1(
                a_fp8, a_scale,
                v1_layer.R_fp8, v1_layer.R_scale,
                v1_layer.L_fp8, v1_layer.L_scale,
                w_fp4_t, v1_layer.weight_scale, M, N, K,
            ),
            warmup=WARMUP, rep=ITERS,
        )
        t_v1 = t_v1_ms / 1000.0

        _ = at_v6(a_fp8, a_scale, R_fp16, L_fp16,
                  w_fp4_t, v1_layer.weight_scale, M, N, K,
                  channel_scale=channel_scale)
        t_v6_ms = tt.do_bench(
            lambda: at_v6(
                a_fp8, a_scale, R_fp16, L_fp16,
                w_fp4_t, v1_layer.weight_scale, M, N, K,
                channel_scale=channel_scale,
            ),
            warmup=WARMUP, rep=ITERS,
        )
        t_v6 = t_v6_ms / 1000.0

        v6_ratio = t_v6 / t_v1 if t_v1 > 0 else float("inf")

        print(
            f"{M:>6} {K:>6} {N:>6}  "
            f"{us(t_v1):>11.1f}µ  "
            f"{us(t_v6):>11.1f}µ  "
            f"{v6_ratio:>16.2f}x"
        )
        rows.append({
            "M": M, "K": K, "N": N,
            "v1_tuned_us": round(us(t_v1), 1),
            "v6_tuned_us": round(us(t_v6), 1),
            "v6_ratio":    round(v6_ratio, 3),
        })

    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Benchmark fast_loraq layers")
    parser.add_argument("--json",            type=str,  default=None, help="Export JSON path")
    parser.add_argument("--fp4-only",        action="store_true", help="Run only FP4 benchmarks")
    parser.add_argument("--fp-only",         action="store_true", help="Run only fp16/bf16 benchmarks")
    parser.add_argument("--loraq-only",      action="store_true", help="Run only SVDQ benchmarks")
    parser.add_argument("--loraq-q8-only",   action="store_true", help="Run only LoRaQ FP8 v1/v2 benchmarks")
    parser.add_argument("--v1-vs-loraq",     action="store_true", help="Compare LoRaQ.1 (LoRaQ) vs SVDQ (rank-32)")
    parser.add_argument("--v1-vs-loraq-tuned", action="store_true", help="LoRaQ.1-autotuned vs SVDQ (best configs)")
    parser.add_argument("--autotuned-only",  action="store_true", help="Run autotuned vs fixed-config comparison")
    parser.add_argument("--v1-vs-loraq-wallclock", action="store_true", help="LoRaQ.1 vs SVDQ with wall-clock benchmark_fn")
    parser.add_argument("--loraq3-only",  action="store_true", help="LoRaQ.3 (fp16io) vs LoRaQ.1 comparison")
    parser.add_argument("--v1-vs-q4-vs-svdq", action="store_true",
                        help="K7 (FP8) vs K13 (FP4) vs SVDQ — three-way autotuned comparison")
    parser.add_argument("--svdq-vs-loraq-pixarts-tblock", action="store_true",
                        help="SVDQ vs LoRaQ-Mixed end-to-end Pixart-σ transformer block")
    parser.add_argument("--loraq-fp16lr", action="store_true", help="LoRaQ.6 (fp16 L,R) vs LoRaQ.1 comparison")
    args = parser.parse_args()

    results = {}

    run_all = not (
        args.fp4_only or args.fp_only or args.loraq_only
        or args.loraq_q8_only or args.v1_vs_loraq
        or args.v1_vs_loraq_tuned or args.autotuned_only
        or args.v1_vs_loraq_wallclock
        or args.loraq3_only
        or args.v1_vs_q4_vs_svdq
        or args.svdq_vs_loraq_pixarts_tblock
        or args.loraq_fp16lr
    )

    if run_all or args.fp_only:
        results["fp"] = bench_fp(SIZES, DTYPES_FP)

    if run_all or args.fp4_only:
        results["fp4"] = bench_fp4(SIZES)

    if run_all or args.loraq_only:
        results["loraq"] = bench_svdq(SIZES)

    if run_all or args.loraq_q8_only:
        results["loraq_q8"] = bench_loraq_q8(SIZES)

    if run_all or args.v1_vs_loraq:
        results["v1_vs_loraq"] = bench_v1_vs_loraq(SIZES)

    if run_all or args.v1_vs_loraq_tuned:
        results["v1_vs_loraq_tuned"] = bench_v1_vs_loraq_autotuned(SIZES)

    if args.autotuned_only:
        results["autotuned"] = bench_autotuned(SIZES)

    if args.v1_vs_loraq_wallclock:
        results["v1_vs_loraq_wallclock"] = bench_v1_vs_loraq_wallclock(SIZES)

    if args.loraq3_only:
        results["loraq3"] = bench_loraq3(SIZES)

    if run_all or args.v1_vs_q4_vs_svdq:
        results["v1_vs_q4_vs_svdq"] = bench_v1_vs_q4_vs_svdq(SIZES)

    if run_all or args.svdq_vs_loraq_pixarts_tblock:
        results["svdq_vs_loraq_pixarts_tblock"] = bench_svdq_vs_loraq_pixarts_tblock()

    if args.loraq_fp16lr:
        results["loraq_fp16lr"] = bench_loraq_fp16lr(SIZES)

    if args.json:
        with open(args.json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults exported to {args.json}")


if __name__ == "__main__":
    main()
