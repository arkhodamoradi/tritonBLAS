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
import triton
import triton.testing as tt

from loraq.linear import TritonLinear, TritonLinearFP4, TritonLinearLoRA
from loraq.linear import TritonLinearLoRaQ, TritonLinearLoRaQFP8
from loraq.quant import dynamic_mxfp8_quant, dynamic_mxfp4_quant
from loraq.autotune_configs import (
    AutotunedLoRaQ, AutotunedLoRaQ3, AutotunedLoRaQFP16LR,
    AutotunedDualGEMM, AutotunedProjectAndQuant, LORAQ_Q8_CONFIGS,
)
from loraq.kernels import (
    loraq_fused_q8_kernel, loraq_fused_q8_scaled_kernel,
    loraq_fused_q8_fp16lr_kernel,
    loraq_dual_gemm_kernel, loraq_project_and_quant_kernel,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SIZES = [
    # (M,     K,     N)      -- representative workloads
    (1,     4096,  4096),    # single-token decode
    (8,     4096,  4096),    # small batch decode
    (32,    4096,  4096),
    (64,    4096,  4096),
    (128,   4096,  4096),
    (256,   4096,  4096),
    (512,   4096,  4096),
    (1024,  4096,  4096),
    (2048,  4096,  4096),
    (4096,  4096,  4096),
    (1,     8192,  8192),    # single-token decode
    (8,     8192,  8192),    # small batch decode
    (32,    8192,  8192),
    (64,    8192,  8192),
    (128,   8192,  8192),
    (256,   8192,  8192),
    (512,   8192,  8192),
    (1024,  8192,  8192),
    (2048,  8192,  8192),
    (4096,  8192,  8192),
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
    #(4096,  4096,  4096),    # large batch
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
    at_v2 = AutotunedLoRaQ(loraq_fused_q8_scaled_kernel, LORAQ_Q8_CONFIGS, warmup=5, rep=25)

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
    at_v2 = AutotunedLoRaQ(loraq_fused_q8_scaled_kernel, LORAQ_Q8_CONFIGS, warmup=5, rep=25)

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
    V1 and V2 autotuned vs SVDQ autotuned — all at best configs with wpe sweep.

    V1: AutotunedLoRaQ  — kernel 7 (tl.dot fp16 Phase 2, constexpr dims)
    V2: AutotunedLoRaQ  — kernel 8 (dot_scaled fp8 Phase 2, constexpr dims)
    SVDQ: AutotunedProjectAndQuant + AutotunedDualGEMM (kernels 5+6)

    V1 and V2 share the same weight tensors (same format: MXFP8 L/R, MXFP4 W).
    Both sweep waves_per_eu in [0,1,2] via their autotuners.
    """
    rows = []
    W = 130
    print("\n" + "=" * W)
    print("  V1 / V2 / SVDQ — all fully autotuned (wpe sweep for V1 and V2)")
    print("  V1: rank=64, dot fp16 Phase2 (K7)    V2: rank=64, dot_scaled fp8 Phase2 (K8)")
    print("  SVDQ: rank=32, fp16 L/R, kernels 5+6")
    print("=" * W)
    header = (
        f"{'M':>6} {'K':>6} {'N':>6}  "
        f"{'SVDQ µs':>10} {'SVDQ TF':>8}  "
        f"{'V1 µs':>10} {'V1 TF':>8}  "
        f"{'V2 µs':>10} {'V2 TF':>8}  "
        f"{'V1/SVDQ':>8} {'V2/SVDQ':>8} {'V2/V1':>7}"
    )
    print(header)
    print("-" * W)

    at_v1 = AutotunedLoRaQ(loraq_fused_q8_kernel,        LORAQ_Q8_CONFIGS, warmup=5, rep=25)
    at_v2 = AutotunedLoRaQ(loraq_fused_q8_scaled_kernel, LORAQ_Q8_CONFIGS, warmup=5, rep=25)
    at_pq = AutotunedProjectAndQuant(loraq_project_and_quant_kernel, warmup=5, rep=25)
    at_dg = AutotunedDualGEMM(loraq_dual_gemm_kernel, LORAQ_Q8_CONFIGS, warmup=5, rep=25)

    for M, K, N in sizes:
        if K % 64 != 0 or N % 32 != 0:
            continue

        x = torch.randn(M, K, device="cuda", dtype=torch.float16)
        a_fp8, a_scale = dynamic_mxfp8_quant(x)
        ref_linear  = nn.Linear(K, N, bias=False, device="cuda", dtype=torch.float16)

        loraq_layer = TritonLinearLoRA.from_float(ref_linear)   # SVDQ: rank=32, fp16 L/R
        v1_layer    = TritonLinearLoRaQ.from_float(ref_linear)  # V1/V2: rank=64, MXFP8 L/R

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

        # ---- V1 autotuned ----
        w_fp4_t = v1_layer.weight_fp4.t().contiguous()
        _ = at_v1(a_fp8, a_scale,
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

        # ---- V2 autotuned (same weights as V1, different kernel) ----
        _ = at_v2(a_fp8, a_scale,
                  v1_layer.R_fp8, v1_layer.R_scale,
                  v1_layer.L_fp8, v1_layer.L_scale,
                  w_fp4_t, v1_layer.weight_scale, M, N, K)
        t_v2_ms = tt.do_bench(
            lambda: at_v2(a_fp8, a_scale,
                          v1_layer.R_fp8, v1_layer.R_scale,
                          v1_layer.L_fp8, v1_layer.L_scale,
                          w_fp4_t, v1_layer.weight_scale, M, N, K),
            warmup=WARMUP, rep=ITERS,
        )

        t_svd = t_svd_ms / 1000.0
        t_v1  = t_v1_ms  / 1000.0
        t_v2  = t_v2_ms  / 1000.0

        tf_svd = tflops(M, N, K, t_svd)
        tf_v1  = tflops(M, N, K, t_v1)
        tf_v2  = tflops(M, N, K, t_v2)

        v1_over_svd = t_svd / t_v1 if t_v1 > 0 else float("inf")
        v2_over_svd = t_svd / t_v2 if t_v2 > 0 else float("inf")
        v2_over_v1  = t_v1  / t_v2 if t_v2 > 0 else float("inf")

        print(
            f"{M:>6} {K:>6} {N:>6}  "
            f"{us(t_svd):>9.1f}µ {tf_svd:>7.2f}  "
            f"{us(t_v1):>9.1f}µ {tf_v1:>7.2f}  "
            f"{us(t_v2):>9.1f}µ {tf_v2:>7.2f}  "
            f"{v1_over_svd:>7.2f}x {v2_over_svd:>7.2f}x {v2_over_v1:>6.2f}x"
        )
        rows.append({
            "M": M, "K": K, "N": N,
            "svdq_us": round(us(t_svd), 1), "svdq_tflops": round(tf_svd, 3),
            "v1_us":   round(us(t_v1),  1), "v1_tflops":   round(tf_v1,  3),
            "v2_us":   round(us(t_v2),  1), "v2_tflops":   round(tf_v2,  3),
            "v1_over_svdq": round(v1_over_svd, 3),
            "v2_over_svdq": round(v2_over_svd, 3),
            "v2_over_v1":   round(v2_over_v1,  3),
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
    at_v2 = AutotunedLoRaQ(loraq_fused_q8_scaled_kernel, LORAQ_Q8_CONFIGS, warmup=5, rep=25)

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

        # Prepare fp16 L, R from the SVD (same init as TritonLinearLoRaQ but keep fp16)
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

        # ---- LoRaQ.1 autotuned ----
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

        # ---- LoRaQ.6 autotuned ----
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
    parser.add_argument("--loraq-fp16lr", action="store_true", help="LoRaQ.6 (fp16 L,R) vs LoRaQ.1 comparison")
    args = parser.parse_args()

    results = {}

    run_all = not (
        args.fp4_only or args.fp_only or args.loraq_only
        or args.loraq_q8_only or args.v1_vs_loraq
        or args.v1_vs_loraq_tuned or args.autotuned_only
        or args.v1_vs_loraq_wallclock
        or args.loraq3_only
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

    if args.loraq_fp16lr:
        results["loraq_fp16lr"] = bench_loraq_fp16lr(SIZES)

    if args.json:
        with open(args.json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults exported to {args.json}")


if __name__ == "__main__":
    main()
