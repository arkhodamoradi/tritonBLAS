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

from loraq.linear import TritonLinear, TritonLinearFP4, TritonLinearLoRA
from loraq.linear import TritonLinearLoRaQ, TritonLinearLoRaQFP8
from loraq.quant import dynamic_mxfp8_quant, dynamic_mxfp4_quant
from loraq.autotune_configs import AutotunedLoRaQ, LORAQ_Q8_CONFIGS
from loraq.kernels import loraq_fused_q8_kernel, loraq_fused_q8_scaled_kernel

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
    (512,   4096,  4096),
    (1024,  4096,  4096),
    (2048,  4096,  4096),
    (4096,  4096,  4096),
    (128,   4096,  11008),   # LLaMA-7B FFN up
    (128,   11008, 4096),    # LLaMA-7B FFN down
    (128,   5120,  5120),    # LLaMA-13B hidden
    (128,   8192,  8192),    # LLaMA-65B hidden
    (256,   4096,  14336),   # LLaMA-2 70B FFN up
    (1024,  4096,  14336),
    (2048,  4096,  4096),    # large batch
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
# LoRA+Q benchmark
# ---------------------------------------------------------------------------

def bench_loraq(sizes):
    rows = []
    W = 130
    print("\n" + "=" * W)
    print("  LoRA+Q  Benchmark: TritonLinearLoRA  vs  TritonLinearFP4  vs  nn.Linear (fp16)")
    print("=" * W)
    header = (
        f"{'M':>6} {'K':>6} {'N':>6}  "
        f"{'LoRA+Q µs':>10} {'LoRA+Q TFLOPS':>14}  "
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
            "format": "LoRA+MXFP4_e2m1",
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
# LoRaQ FP8 benchmark:  V1 (tl.dot fp16)  vs  V2 (dot_scaled fp8)
# ---------------------------------------------------------------------------

def bench_loraq_q8(sizes):
    rows = []
    W = 155
    print("\n" + "=" * W)
    print("  LoRaQ FP8  Benchmark: V1 (tl.dot fp16)  vs  V2 (dot_scaled fp8)  vs  TritonLinearFP4  vs  nn.Linear")
    print("  Activation is pre-quantized to MXFP8 before timing (online quant excluded by design)")
    print("=" * W)
    header = (
        f"{'M':>6} {'K':>6} {'N':>6}  "
        f"{'V1 µs':>8} {'V1 TFLOPS':>10}  "
        f"{'V2 µs':>8} {'V2 TFLOPS':>10}  "
        f"{'FP4 µs':>8} {'FP4 TFLOPS':>10}  "
        f"{'fp16 µs':>8} {'fp16 TFLOPS':>11}  "
        f"{'V2/V1':>6} {'V1/fp16':>8} {'V2/fp16':>8}  "
        f"{'Wt mem':>9}"
    )
    print(header)
    print("-" * W)

    for M, K, N in sizes:
        if K % 64 != 0 or N % 32 != 0:
            continue

        x = torch.randn(M, K, device="cuda", dtype=torch.float16)
        a_fp8, a_scale = dynamic_mxfp8_quant(x)

        ref_linear = nn.Linear(K, N, bias=False, device="cuda", dtype=torch.float16)

        v1_layer = TritonLinearLoRaQ.from_float(ref_linear)
        v2_layer = TritonLinearLoRaQFP8.from_float(ref_linear)
        fp4_layer = TritonLinearFP4.from_float(ref_linear)

        t_v1  = benchmark_fn(lambda: v1_layer(a_fp8, a_scale))
        t_v2  = benchmark_fn(lambda: v2_layer(a_fp8, a_scale))
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
# V1 vs LoRA+Q comparison
# ---------------------------------------------------------------------------

def bench_v1_vs_loraq(sizes):
    """
    Direct comparison: TritonLinearLoRaQ V1 (rank=64, MXFP8 L/R, 1 fused kernel)
    vs TritonLinearLoRA  (rank=32, fp16 L/R, 2 kernels: project+quant + dual_gemm).

    Both layers store W as MXFP4 (quantized from the same source nn.Linear).
    V1 takes pre-quantized MXFP8 activation; LoRA+Q takes raw fp16.
    Online quantization is excluded from V1 timing by design.
    """
    rows = []
    W = 135
    print("\n" + "=" * W)
    print("  V1 (LoRaQ, rank=64, MXFP8, 1 kernel)  vs  LoRA+Q (rank=32, fp16, 2 kernels)")
    print("  NOTE: V1 input is pre-quantized MXFP8 — online quant cost excluded by design.")
    print("        LoRA+Q input is raw fp16.  Both store W as MXFP4 (quantized from same nn.Linear).")
    print("=" * W)
    header = (
        f"{'M':>6} {'K':>6} {'N':>6}  "
        f"{'LoRA+Q µs':>10} {'LoRA+Q TFLOPS':>14}  "
        f"{'V1 µs':>8} {'V1 TFLOPS':>10}  "
        f"{'V1/LoRA+Q':>10}  "
        f"{'FP4 µs':>8} {'fp16 µs':>9}  "
        f"{'Rank(LQ)':>9} {'Rank(V1)':>9}"
    )
    print(header)
    print("-" * W)

    for M, K, N in sizes:
        # Both layers require these divisibility constraints
        if K % 64 != 0 or N % 32 != 0:
            continue

        x = torch.randn(M, K, device="cuda", dtype=torch.float16)
        a_fp8, a_scale = dynamic_mxfp8_quant(x)

        ref_linear = nn.Linear(K, N, bias=False, device="cuda", dtype=torch.float16)
        fp4_layer = TritonLinearFP4.from_float(ref_linear)

        loraq_layer = TritonLinearLoRA.from_float(ref_linear)   # rank=32, fp16 L/R
        v1_layer    = TritonLinearLoRaQ.from_float(ref_linear)  # rank=64, MXFP8 L/R

        t_loraq = benchmark_fn(lambda: loraq_layer(x))
        t_v1    = benchmark_fn(lambda: v1_layer(a_fp8, a_scale))
        t_fp4   = benchmark_fn(lambda: fp4_layer(x))
        t_fp16  = benchmark_fn(lambda: ref_linear(x))

        tf_loraq = tflops(M, N, K, t_loraq)
        tf_v1    = tflops(M, N, K, t_v1)

        # Speedup: positive means V1 is faster
        v1_over_loraq = t_loraq / t_v1 if t_v1 > 0 else float("inf")

        print(
            f"{M:>6} {K:>6} {N:>6}  "
            f"{us(t_loraq):>9.1f}µ {tf_loraq:>13.2f}  "
            f"{us(t_v1):>7.1f}µ {tf_v1:>9.2f}  "
            f"{v1_over_loraq:>9.2f}x  "
            f"{us(t_fp4):>7.1f}µ {us(t_fp16):>8.1f}µ  "
            f"{'32':>9} {'64':>9}"
        )
        rows.append({
            "M": M, "K": K, "N": N,
            "loraq_us":    round(us(t_loraq), 1),
            "loraq_tflops": round(tf_loraq, 3),
            "v1_us":       round(us(t_v1), 1),
            "v1_tflops":   round(tf_v1, 3),
            "v1_over_loraq": round(v1_over_loraq, 3),
            "fp4_us":  round(us(t_fp4), 1),
            "fp16_us": round(us(t_fp16), 1),
            "loraq_rank": 32,
            "v1_rank": 64,
        })

    return rows


# ---------------------------------------------------------------------------
# Autotuned vs fixed-config comparison
# ---------------------------------------------------------------------------

def bench_autotuned(sizes):
    rows = []
    W = 130
    print("\n" + "=" * W)
    print("  Autotuned Benchmark: fixed-config  vs  autotuned LoRaQ (V1 & V2)")
    print("  Note: first run for each (M,K,N) triggers autotuning (15 configs × do_bench)")
    print("=" * W)
    header = (
        f"{'M':>6} {'K':>6} {'N':>6}  "
        f"{'V1 fixed µs':>12} {'V1 tuned µs':>12} {'V1 gain':>8}  "
        f"{'V2 fixed µs':>12} {'V2 tuned µs':>12} {'V2 gain':>8}  "
        f"{'Best V1 config':>30}"
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
        v2_layer = TritonLinearLoRaQFP8.from_float(ref_linear)

        t_v1_fixed = benchmark_fn(lambda: v1_layer(a_fp8, a_scale))
        t_v2_fixed = benchmark_fn(lambda: v2_layer(a_fp8, a_scale))

        w_fp4_t  = v1_layer.weight_fp4.t().contiguous()
        w_fp4_t2 = v2_layer.weight_fp4.t().contiguous()

        t_v1_tuned = benchmark_fn(
            lambda: at_v1(
                a_fp8, a_scale,
                v1_layer.R_fp8, v1_layer.R_scale,
                v1_layer.L_fp8, v1_layer.L_scale,
                w_fp4_t, v1_layer.weight_scale,
                M, N, K,
            )
        )
        t_v2_tuned = benchmark_fn(
            lambda: at_v2(
                a_fp8, a_scale,
                v2_layer.R_fp8, v2_layer.R_scale,
                v2_layer.L_fp8, v2_layer.L_scale,
                w_fp4_t2, v2_layer.weight_scale,
                M, N, K,
            )
        )

        v1_gain = t_v1_fixed / t_v1_tuned if t_v1_tuned > 0 else float("inf")
        v2_gain = t_v2_fixed / t_v2_tuned if t_v2_tuned > 0 else float("inf")

        best_v1 = at_v1.get_best_config(M, N, K)
        cfg_str = ""
        if best_v1:
            c = best_v1["config"]
            cfg_str = (f"BM={c.kwargs['BLOCK_M']},"
                       f"BN={c.kwargs['BLOCK_N']},"
                       f"BK={c.kwargs['BLOCK_K']},"
                       f"w={c.num_warps}")

        print(
            f"{M:>6} {K:>6} {N:>6}  "
            f"{us(t_v1_fixed):>11.1f}µ {us(t_v1_tuned):>11.1f}µ {v1_gain:>7.2f}x  "
            f"{us(t_v2_fixed):>11.1f}µ {us(t_v2_tuned):>11.1f}µ {v2_gain:>7.2f}x  "
            f"{cfg_str:>30}"
        )
        rows.append({
            "M": M, "K": K, "N": N,
            "v1_fixed_us":  round(us(t_v1_fixed),  1),
            "v1_tuned_us":  round(us(t_v1_tuned),  1),
            "v1_gain":      round(v1_gain, 3),
            "v2_fixed_us":  round(us(t_v2_fixed),  1),
            "v2_tuned_us":  round(us(t_v2_tuned),  1),
            "v2_gain":      round(v2_gain, 3),
            "best_config":  cfg_str,
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
    parser.add_argument("--loraq-only",      action="store_true", help="Run only LoRA+Q benchmarks")
    parser.add_argument("--loraq-q8-only",   action="store_true", help="Run only LoRaQ FP8 v1/v2 benchmarks")
    parser.add_argument("--v1-vs-loraq",     action="store_true", help="Compare V1 (LoRaQ) vs LoRA+Q (rank-32)")
    parser.add_argument("--autotuned-only",  action="store_true", help="Run autotuned vs fixed-config comparison")
    args = parser.parse_args()

    results = {}

    run_all = not (
        args.fp4_only or args.fp_only or args.loraq_only
        or args.loraq_q8_only or args.v1_vs_loraq or args.autotuned_only
    )

    if run_all or args.fp_only:
        results["fp"] = bench_fp(SIZES, DTYPES_FP)

    if run_all or args.fp4_only:
        results["fp4"] = bench_fp4(SIZES)

    if run_all or args.loraq_only:
        results["loraq"] = bench_loraq(SIZES)

    if run_all or args.loraq_q8_only:
        results["loraq_q8"] = bench_loraq_q8(SIZES)

    if run_all or args.v1_vs_loraq:
        results["v1_vs_loraq"] = bench_v1_vs_loraq(SIZES)

    if args.autotuned_only:
        results["autotuned"] = bench_autotuned(SIZES)

    if args.json:
        with open(args.json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults exported to {args.json}")


if __name__ == "__main__":
    main()
