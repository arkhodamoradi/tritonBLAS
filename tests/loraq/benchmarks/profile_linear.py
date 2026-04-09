"""
Proton profiling for TritonLinear (fp16/bf16) vs nn.Linear.

Generates per-kernel traces with Triton Proton and prints timing + TFLOPS
for each (M, K, N) problem size.

Usage:
    python -m benchmarks.profile_linear
    python -m benchmarks.profile_linear --size 1024 4096 4096
    python -m benchmarks.profile_linear --all-sizes --output traces/ --json results.json
    python -m benchmarks.profile_linear --size 128 4096 4096 --check-asm
"""

import argparse
import json
import os
import time

import torch
import torch.nn as nn
import triton.testing as tt

from fast_loraq.linear import TritonLinear, triton_matmul_nt


# ---------------------------------------------------------------------------
# Sizes to sweep
# ---------------------------------------------------------------------------

SIZES = [
    (1,    4096,  4096),
    (8,    4096,  4096),
    (32,   4096,  4096),
    (64,   4096,  4096),
    (128,  4096,  4096),
    (256,  4096,  4096),
    (512,  4096,  4096),
    (1024, 4096,  4096),
    (2048, 4096,  4096),
    (128,  4096,  11008),
    (128,  11008, 4096),
    (128,  5120,  5120),
    (128,  8192,  8192),
    (256,  4096,  14336),
    (1024, 4096,  14336),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def tflops(M: int, N: int, K: int, time_s: float) -> float:
    return 2.0 * M * N * K / time_s / 1e12


def us(time_s: float) -> float:
    return time_s * 1e6


# ---------------------------------------------------------------------------
# Assembly inspection
# ---------------------------------------------------------------------------

def search_triton_cache(search_strings, filename_pattern=None, print_lines=False):
    """Search Triton compilation cache for assembly instruction patterns."""
    cache_dirs = [
        os.path.expanduser("~/.triton/cache/"),
        "/root/.triton/cache/",
    ]
    out = {s: 0 for s in search_strings}
    for root_dir in cache_dirs:
        if not os.path.isdir(root_dir):
            continue
        for root, _, files in os.walk(root_dir):
            for fname in files:
                if filename_pattern and not fname.endswith(filename_pattern):
                    continue
                fpath = os.path.join(root, fname)
                try:
                    with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                        for lineno, line in enumerate(f, 1):
                            for s in search_strings:
                                if s in line:
                                    if print_lines:
                                        print(f"  {fpath}:{lineno}: {line.strip()}")
                                    out[s] += 1
                except Exception:
                    pass
    return out


# ---------------------------------------------------------------------------
# Core profiling function
# ---------------------------------------------------------------------------

def profile_one_size(M, K, N, dtype, output_dir, warmup, iters, proton):
    """
    Profile TritonLinear vs nn.Linear for a single (M, K, N, dtype).

    Returns dict with timing and TFLOPS for both variants.
    """
    label = "fp16" if dtype == torch.float16 else "bf16"

    x = torch.randn(M, K, device="cuda", dtype=dtype)

    tl_layer = TritonLinear(K, N, bias=False, dtype=dtype)
    torch_layer = nn.Linear(K, N, bias=False, device="cuda", dtype=dtype)

    # Warmup
    for _ in range(warmup):
        tl_layer(x)
        torch_layer(x)
    torch.cuda.synchronize()

    # Open one Proton session per size+dtype
    session_name = f"linear_{label}_{M}x{K}x{N}"
    if proton:
        proton.start(session_name, hook="triton")

    # ---- TritonLinear ----
    if proton:
        with proton.scope("triton_linear"):
            t_triton_ms = tt.do_bench(lambda: tl_layer(x), warmup=warmup, rep=iters)
    else:
        t_triton_ms = tt.do_bench(lambda: tl_layer(x), warmup=warmup, rep=iters)

    # ---- nn.Linear ----
    if proton:
        with proton.scope("nn_linear"):
            t_torch_ms = tt.do_bench(lambda: torch_layer(x), warmup=warmup, rep=iters)
    else:
        t_torch_ms = tt.do_bench(lambda: torch_layer(x), warmup=warmup, rep=iters)

    if proton:
        proton.finalize()

    tf_triton = tflops(M, N, K, t_triton_ms / 1000.0)
    tf_torch  = tflops(M, N, K, t_torch_ms  / 1000.0)
    speedup   = tf_triton / tf_torch if tf_torch > 0 else float("inf")

    return {
        "M": M, "K": K, "N": N, "dtype": label,
        "triton_ms":  round(t_triton_ms, 4),
        "triton_tflops": round(tf_triton, 3),
        "torch_ms":   round(t_torch_ms,  4),
        "torch_tflops":  round(tf_torch,  3),
        "speedup":    round(speedup, 3),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Profile TritonLinear (fp16/bf16) vs nn.Linear with Triton Proton"
    )
    parser.add_argument("--size", nargs=3, type=int, default=None,
                        metavar=("M", "K", "N"),
                        help="Single problem size to profile (default: run all sizes)")
    parser.add_argument("--dtype", choices=["fp16", "bf16", "both"], default="fp16",
                        help="Data type(s) to benchmark")
    parser.add_argument("--output", type=str, default="traces",
                        help="Output directory for Proton trace files")
    parser.add_argument("--warmup", type=int, default=25,
                        help="Warmup iterations")
    parser.add_argument("--iters", type=int, default=100,
                        help="Benchmark iterations")
    parser.add_argument("--check-asm", action="store_true",
                        help="Search Triton cache for MFMA instructions after profiling")
    parser.add_argument("--json", type=str, default=None,
                        help="Path to export JSON results")
    parser.add_argument("--all-sizes", action="store_true",
                        help="Run all predefined sizes (overrides --size)")
    args = parser.parse_args()

    torch.manual_seed(42)

    # Resolve sizes
    if args.all_sizes or args.size is None:
        sizes = SIZES
    else:
        sizes = [tuple(args.size)]

    # Resolve dtypes
    if args.dtype == "both":
        dtypes = [torch.float16, torch.bfloat16]
    elif args.dtype == "bf16":
        dtypes = [torch.bfloat16]
    else:
        dtypes = [torch.float16]

    # Try importing Proton
    try:
        import triton.profiler as proton
        print("Triton Proton available — trace files will be saved to:", args.output)
    except ImportError:
        proton = None
        print("triton.profiler (Proton) not available — timing only, no traces.")

    os.makedirs(args.output, exist_ok=True)

    # ---- Header ----
    W = 105
    print(f"\n{'='*W}")
    print("  TritonLinear  vs  nn.Linear  —  fp16/bf16 profiling")
    print(f"{'='*W}")
    hdr = (
        f"{'M':>6} {'K':>6} {'N':>6} {'dtype':>6}  "
        f"{'Triton ms':>10} {'Triton TFLOPS':>14}  "
        f"{'Torch ms':>9} {'Torch TFLOPS':>13}  "
        f"{'Speedup':>8}"
    )
    print(hdr)
    print("-" * W)

    all_results = []

    for M, K, N in sizes:
        for dtype in dtypes:
            row = profile_one_size(
                M, K, N, dtype, args.output,
                warmup=args.warmup, iters=args.iters,
                proton=proton,
            )
            all_results.append(row)
            label = row["dtype"]
            print(
                f"{M:>6} {K:>6} {N:>6} {label:>6}  "
                f"{row['triton_ms']*1000:>9.1f}µ {row['triton_tflops']:>13.2f}  "
                f"{row['torch_ms']*1000:>8.1f}µ {row['torch_tflops']:>12.2f}  "
                f"{row['speedup']:>7.2f}x"
            )

    print(f"\n  Average speedup: "
          f"{sum(r['speedup'] for r in all_results)/len(all_results):.3f}x  "
          f"({'Triton faster' if sum(r['speedup'] for r in all_results)/len(all_results) > 1 else 'Torch faster'})")

    # ---- Assembly inspection ----
    if args.check_asm:
        print(f"\n{'='*W}")
        print("  Assembly Inspection (Triton cache)")
        print(f"{'='*W}")
        strings = [
            "v_mfma_f32_32x32x8_f16",
            "v_mfma_f32_16x16x16_f16",
            "v_mfma_f32_32x32x16_f16",
            "v_mfma_f32_16x16x32_f16",
            "buffer_load_dwordx4",
            "buffer_load_dwordx2",
            "s_waitcnt",
        ]
        counts = search_triton_cache(strings, filename_pattern=".amdgcn",
                                     print_lines=False)
        for s, c in counts.items():
            marker = "✓" if c > 0 else "✗"
            print(f"  {marker}  {s}: {c} occurrences")

        print("\n  Expected for fp16 GEMM:")
        print("    v_mfma_f32_32x32x8_f16 or v_mfma_f32_32x32x16_f16 (matrix_instr_nonkdim=32)")
        print("    buffer_load_dwordx4 (128-bit vector loads for BLOCK_K=128)")

    # ---- Proton trace info ----
    if proton:
        print(f"\n  Trace files saved to: {args.output}/")
        print("  View individual traces:")
        for M, K, N in sizes:
            for dtype in dtypes:
                label = "fp16" if dtype == torch.float16 else "bf16"
                name = f"linear_{label}_{M}x{K}x{N}"
                print(f"    python -m triton.profiler.viewer {args.output}/{name}.hatchet")
        print("  or drag .hatchet files to https://ui.perfetto.dev/")

    # ---- JSON export ----
    if args.json:
        with open(args.json, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\n  Results exported to {args.json}")


if __name__ == "__main__":
    main()
