"""
Proton profiling for LoRaQ kernels.

Generates kernel-level traces for V1 (tl.dot fp16) and V2 (dot_scaled fp8)
variants, viewable in Chrome trace viewer (chrome://tracing) or Perfetto.

Also inspects compiled assembly for MFMA instruction verification.

Usage:
    python -m benchmarks.profile_loraq [--size 128 4096 4096] [--output traces/]
"""

import argparse
import os
import time

import torch
import triton
import triton.testing as tt

from loraq.linear import (
    TritonLinearLoRaQ,
    TritonLinearLoRaQFP8,
    TritonLinearFP4,
    triton_loraq_fused_q8,
    triton_loraq_fused_q8_scaled,
)
from loraq.quant import dynamic_mxfp4_quant, dynamic_mxfp8_quant, mxfp8_to_f32


# ---------------------------------------------------------------------------
# Assembly inspection
# ---------------------------------------------------------------------------

def search_in_triton_cache(search_strings, filename_pattern=None, print_lines=False):
    """Search Triton cache for specific assembly instruction patterns."""
    cache_dirs = [
        os.path.expanduser("~/.triton/cache/"),
        "/root/.triton/cache/",
    ]
    out_dict = {s: 0 for s in search_strings}
    for root_dir in cache_dirs:
        if not os.path.isdir(root_dir):
            continue
        for root, dirs, files in os.walk(root_dir):
            for file in files:
                if filename_pattern and not file.endswith(filename_pattern):
                    continue
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                        for lineno, line in enumerate(f, 1):
                            for s in search_strings:
                                if s in line:
                                    if print_lines:
                                        print(f"  {file_path}:{lineno}: {line.strip()}")
                                    out_dict[s] += 1
                except Exception:
                    pass
    return out_dict


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

RANK = 64

def tflops(m, n, k, time_s):
    flops = 2.0 * m * n * k + 2.0 * m * k * RANK + 2.0 * m * RANK * n
    return flops / time_s / 1e12


def prepare_layer_and_input(M, K, N, device="cuda"):
    """Create layers and pre-quantized inputs."""
    ref = torch.nn.Linear(K, N, bias=False, device=device, dtype=torch.float16)
    v1 = TritonLinearLoRaQ.from_float(ref)
    v2 = TritonLinearLoRaQFP8.from_float(ref)
    fp4 = TritonLinearFP4.from_float(ref)

    x = torch.randn(M, K, device=device, dtype=torch.float16)
    a_fp8, a_scale = dynamic_mxfp8_quant(x)

    return v1, v2, fp4, ref, x, a_fp8, a_scale


# ---------------------------------------------------------------------------
# Profiling
# ---------------------------------------------------------------------------

def profile_with_proton(v1, v2, fp4, ref, x, a_fp8, a_scale, M, K, N,
                        output_dir, warmup=25, iters=50):
    """
    Profile all kernel variants using Triton Proton.

    Triton 3.x API: one session opened with proton.start(), individual
    regions marked with ``with proton.scope("name"):``, closed with
    proton.finalize().  Generates a single .hatchet trace file per
    (M, K, N) size, viewable in https://ui.perfetto.dev/ or as a
    Hatchet tree.
    """
    try:
        import triton.profiler as proton
    except ImportError:
        print("WARNING: triton.profiler (Proton) not available. "
              "Falling back to do_bench timing only.")
        proton = None

    os.makedirs(output_dir, exist_ok=True)
    results = {}

    # ---- Warmup all variants ----
    print("  Warming up kernels...")
    for _ in range(warmup):
        v1(a_fp8, a_scale)
        v2(a_fp8, a_scale)
        fp4(x)
        ref(x)
    torch.cuda.synchronize()

    # ---- Open one Proton session for this (M, K, N) ----
    session_name = f"loraq_{M}x{K}x{N}"
    session_path = os.path.join(output_dir, session_name)
    if proton:
        proton.start(session_name, hook="triton")

    # ---- Profile V1 (tl.dot fp16) ----
    print("  Profiling V1 (LoRaQ, tl.dot fp16)...")
    if proton:
        with proton.scope("v1_loraq"):
            t_v1 = tt.do_bench(lambda: v1(a_fp8, a_scale), warmup=warmup, rep=iters)
    else:
        t_v1 = tt.do_bench(lambda: v1(a_fp8, a_scale), warmup=warmup, rep=iters)

    tf_v1 = tflops(M, N, K, t_v1 / 1000.0)
    results["v1"] = {"time_ms": round(t_v1, 4), "tflops": round(tf_v1, 3)}
    print(f"    V1: {t_v1:.4f} ms  ({tf_v1:.3f} TFLOPS)")

    # ---- Profile V2 (dot_scaled fp8) ----
    print("  Profiling V2 (LoRaQ, dot_scaled fp8)...")
    if proton:
        with proton.scope("v2_loraq_scaled"):
            t_v2 = tt.do_bench(lambda: v2(a_fp8, a_scale), warmup=warmup, rep=iters)
    else:
        t_v2 = tt.do_bench(lambda: v2(a_fp8, a_scale), warmup=warmup, rep=iters)

    tf_v2 = tflops(M, N, K, t_v2 / 1000.0)
    results["v2"] = {"time_ms": round(t_v2, 4), "tflops": round(tf_v2, 3)}
    print(f"    V2: {t_v2:.4f} ms  ({tf_v2:.3f} TFLOPS)")

    # ---- Profile FP4 baseline ----
    print("  Profiling FP4 baseline...")
    if proton:
        with proton.scope("fp4_baseline"):
            t_fp4 = tt.do_bench(lambda: fp4(x), warmup=warmup, rep=iters)
    else:
        t_fp4 = tt.do_bench(lambda: fp4(x), warmup=warmup, rep=iters)

    tf_fp4 = tflops(M, N, K, t_fp4 / 1000.0)
    results["fp4"] = {"time_ms": round(t_fp4, 4), "tflops": round(tf_fp4, 3)}
    print(f"    FP4: {t_fp4:.4f} ms  ({tf_fp4:.3f} TFLOPS)")

    # ---- Profile fp16 reference ----
    print("  Profiling fp16 nn.Linear reference...")
    if proton:
        with proton.scope("fp16_reference"):
            t_fp16 = tt.do_bench(lambda: ref(x), warmup=warmup, rep=iters)
    else:
        t_fp16 = tt.do_bench(lambda: ref(x), warmup=warmup, rep=iters)

    tf_fp16 = tflops(M, N, K, t_fp16 / 1000.0)
    results["fp16"] = {"time_ms": round(t_fp16, 4), "tflops": round(tf_fp16, 3)}
    print(f"    fp16: {t_fp16:.4f} ms  ({tf_fp16:.3f} TFLOPS)")

    # ---- Close Proton session ----
    if proton:
        proton.finalize()

    # ---- Summary ----
    v2_v1 = t_v1 / t_v2 if t_v2 > 0 else float("inf")
    v1_fp16 = t_fp16 / t_v1 if t_v1 > 0 else float("inf")
    v2_fp16 = t_fp16 / t_v2 if t_v2 > 0 else float("inf")

    print(f"\n  Speedup ratios:")
    print(f"    V2/V1:    {v2_v1:.2f}x {'(V2 faster)' if v2_v1 > 1 else '(V1 faster)'}")
    print(f"    V1/fp16:  {v1_fp16:.2f}x")
    print(f"    V2/fp16:  {v2_fp16:.2f}x")

    results["speedup"] = {
        "v2_over_v1": round(v2_v1, 3),
        "v1_over_fp16": round(v1_fp16, 3),
        "v2_over_fp16": round(v2_fp16, 3),
    }

    if proton:
        print(f"\n  Trace saved: {session_path}.hatchet")
        print("  View with: python -m triton.profiler.viewer {session_path}.hatchet")
        print("  or upload to https://ui.perfetto.dev/")

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Profile LoRaQ kernels with Proton")
    parser.add_argument("--size", nargs=3, type=int, default=[128, 4096, 4096],
                        metavar=("M", "K", "N"), help="Problem size (M K N)")
    parser.add_argument("--output", type=str, default="traces",
                        help="Output directory for trace files")
    parser.add_argument("--warmup", type=int, default=25, help="Warmup iterations")
    parser.add_argument("--iters", type=int, default=100, help="Benchmark iterations")
    parser.add_argument("--check-asm", action="store_true",
                        help="Inspect assembly for MFMA instructions")
    parser.add_argument("--all-sizes", action="store_true",
                        help="Profile all standard sizes instead of --size")
    parser.add_argument("--json", type=str, default=None, help="Export JSON path")
    args = parser.parse_args()

    torch.manual_seed(42)

    sizes = [
        #(128, 4096, 4096),
        #(256, 4096, 4096),
        #(1024, 4096, 4096),
        #(128, 4096, 11008),
        (4096, 1152, 4608),
    ] #if args.all_sizes else [tuple(args.size)]

    all_results = {}

    for M, K, N in sizes:
        print(f"\n{'='*80}")
        print(f"  Profiling M={M}, K={K}, N={N}")
        print(f"{'='*80}")

        v1, v2, fp4, ref, x, a_fp8, a_scale = prepare_layer_and_input(M, K, N)

        results = profile_with_proton(
            v1, v2, fp4, ref, x, a_fp8, a_scale,
            M, K, N,
            output_dir=args.output,
            warmup=args.warmup,
            iters=args.iters,
        )
        all_results[f"{M}x{K}x{N}"] = results

    # Assembly inspection
    if args.check_asm:
        print(f"\n{'='*80}")
        print("  Assembly Inspection")
        print(f"{'='*80}")

        strings = [
            "v_mfma_scale_f32_32x32x64_f8f6f4",
            "v_mfma_scale_f32_16x16x128_f8f6f4",
            "v_mfma_f32_32x32x16_f16",
            "v_mfma_f32_16x16x32_f16",
        ]

        print("\n  Kernel 7 (V1 — tl.dot fp16 Phase ②):")
        counts = search_in_triton_cache(strings, print_lines=True)
        for s, c in counts.items():
            marker = " ✓" if c > 0 else " ✗"
            print(f"    {s}: {c}{marker}")

        print("\n  Expected: v_mfma_scale_f32 for Phase ① (A×R^T, A×W^T)")
        print("            v_mfma_f32_*_f16 for Phase ② (P×L^T)")

    # Export
    if args.json:
        import json
        with open(args.json, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults exported to {args.json}")


if __name__ == "__main__":
    main()
