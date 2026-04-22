"""
Hyperparameter sweep for LoRaQ kernels (Kernel 7 & 8).

Sweeps over (BLOCK_M, BLOCK_N, BLOCK_K, GROUP_SIZE_M, num_warps, num_stages)
to find the optimal configuration for each (M, K, N) problem size.

Also inspects compiled AMDGCN assembly for v_mfma_scale_f32 instructions
to verify hardware-accelerated dot_scaled is being used.

Usage:
    python -m benchmarks.sweep_loraq [--json sweep_results.json]
"""

import argparse
import json
import os
import time

import torch
import torch.nn as nn
import triton
import triton.testing as tt

from loraq.kernels import (
    loraq_fused_q8_kernel, loraq_fused_q8_scaled_kernel
)
from loraq.quant import dynamic_mxfp4_quant, dynamic_mxfp8_quant


# ---------------------------------------------------------------------------
# Sweep configuration
# ---------------------------------------------------------------------------

# Problem sizes to test
SIZES = [
    # (M,     K,     N)
    #(4096,   4096,  4096),
    (4096,   3072,  12288),
    #(256,   4096,  4096),
    #(512,   4096,  4096),
    #(1024,  4096,  4096),
    #(128,   4096,  11008),
    #(128,   8192,  8192),
]

# Hyperparameter grid
BLOCK_MS = [128,256]#[64, 128, 256]
BLOCK_NS = [128,256]#[64, 128, 256]
BLOCK_KS = [128,256]#[64, 128]           # ≥64 required for dot_scaled
GROUP_MS = [1,4,8]#[1, 4, 8]
NUM_WARPS_LIST = [4,8] #[4, 8]
NUM_STAGES_LIST = [2] #[1, 2]

RANK = 64
WARMUP = 10
REP = 100


# ---------------------------------------------------------------------------
# Assembly inspection (from mxfp468_gemm.py pattern)
# ---------------------------------------------------------------------------

def search_in_triton_cache(
    filename_pattern: str | None = None,
    search_strings: list[str] | None = None,
    print_lines: bool = False,
):
    """
    Search compiled kernel assembly in the Triton cache directory
    for specific instruction patterns.
    """
    if search_strings is None:
        search_strings = [
            "v_mfma_scale_f32_32x32x64_f8f6f4",
            "v_mfma_scale_f32_16x16x128_f8f6f4",
            "v_mfma_f32",
            "buffer_load_dword",
        ]

    # Try common Triton cache locations
    cache_dirs = [
        os.path.expanduser("~/.triton/cache/"),
        "/root/.triton/cache/",
        os.path.join(os.getcwd(), ".triton/cache/"),
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

def tflops(m: int, n: int, k: int, time_s: float) -> float:
    """Effective TFLOPS for the LoRaQ computation (A@R^T@L^T + A@W^T)."""
    # Main GEMM: M×K×N (FP4 path)
    # Low-rank: M×K×RANK + M×RANK×N
    flops = 2.0 * m * n * k + 2.0 * m * k * RANK + 2.0 * m * RANK * n
    return flops / time_s / 1e12


def prepare_inputs(M, K, N, device="cuda"):
    """Create pre-quantized inputs for the LoRaQ kernels."""
    A = torch.randn(M, K, device=device, dtype=torch.float16) * 0.5
    R = torch.randn(RANK, K, device=device, dtype=torch.float16) * 0.1
    L = torch.randn(N, RANK, device=device, dtype=torch.float16) * 0.1
    W = torch.randn(N, K, device=device, dtype=torch.float16)

    a_fp8, a_scale = dynamic_mxfp8_quant(A)
    r_fp8, r_scale = dynamic_mxfp8_quant(R)
    l_fp8, l_scale = dynamic_mxfp8_quant(L)
    w_fp4, w_scale = dynamic_mxfp4_quant(W)

    # Transpose W for kernel
    w_fp4_t = w_fp4.t().contiguous()

    # Channel-wise quantization scale (ones = no-op, for benchmarking)
    channel_scale = torch.ones(N, dtype=torch.float16, device=device)

    return a_fp8, a_scale, r_fp8, r_scale, l_fp8, l_scale, w_fp4_t, w_scale, channel_scale


def try_config(kernel_fn, a_fp8, a_scale, r_fp8, r_scale, l_fp8, l_scale,
               w_fp4_t, w_scale, channel_scale, M, N, K,
               block_m, block_n, block_k, group_m, num_warps, num_stages):
    """
    Try a single kernel config and return median time in ms, or None if
    the config is invalid (e.g. block sizes exceed problem dimensions).
    """
    # Validate config
    if block_m > M and M >= 64:
        return None
    if block_n > N and N >= 64:
        return None
    if block_k > K:
        return None
    if N % 32 != 0 or K % 64 != 0:
        return None

    c_fp8 = torch.empty((M, N), dtype=torch.uint8, device=a_fp8.device)
    c_scale = torch.empty((M, N // 32), dtype=torch.uint8, device=a_fp8.device)

    grid = (triton.cdiv(M, block_m) * triton.cdiv(N, block_n),)

    # Use a dummy bias pointer
    bias_ptr = a_fp8

    try:
        ms = tt.do_bench(
            lambda: kernel_fn[grid](
                a_fp8, a_scale,
                r_fp8, r_scale,
                l_fp8, l_scale,
                w_fp4_t, w_scale,
                bias_ptr,
                channel_scale,
                c_fp8, c_scale,
                M, N, K,
                a_fp8.stride(0), a_fp8.stride(1),
                a_scale.stride(0), a_scale.stride(1),
                r_fp8.stride(0), r_fp8.stride(1),
                r_scale.stride(0), r_scale.stride(1),
                l_fp8.stride(0), l_fp8.stride(1),
                l_scale.stride(0), l_scale.stride(1),
                w_fp4_t.stride(0), w_fp4_t.stride(1),
                w_scale.stride(0), w_scale.stride(1),
                c_fp8.stride(0), c_fp8.stride(1),
                c_scale.stride(0), c_scale.stride(1),
                channel_scale.stride(0),
                HAS_BIAS=False,
                RANK=RANK,
                BLOCK_M=block_m,
                BLOCK_N=block_n,
                BLOCK_K=block_k,
                GROUP_SIZE_M=group_m,
                num_warps=num_warps,
                num_stages=num_stages,
                matrix_instr_nonkdim=32,
            ),
            warmup=WARMUP,
            rep=REP,
        )
        return ms
    except Exception as e:
        # Config may be invalid (e.g. tile too large for shared memory)
        return None


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------

def sweep_kernel(kernel_fn, kernel_name, sizes, max_configs=None):
    """Sweep hyperparameters for a single kernel across problem sizes."""
    all_results = []

    for M, K, N in sizes:
        print(f"\n{'='*100}")
        print(f"  {kernel_name}  |  M={M}, K={K}, N={N}")
        print(f"{'='*100}")
        header = (
            f"{'BM':>4} {'BN':>4} {'BK':>4} {'GM':>3} {'NW':>3} {'NS':>3}  "
            f"{'Time (ms)':>10} {'TFLOPS':>10} {'Status':>8}"
        )
        print(header)
        print("-" * 60)

        inputs = prepare_inputs(M, K, N)
        best_ms = float("inf")
        best_cfg = None
        config_count = 0

        for bm in BLOCK_MS:
            for bn in BLOCK_NS:
                for bk in BLOCK_KS:
                    for gm in GROUP_MS:
                        for nw in NUM_WARPS_LIST:
                            for ns in NUM_STAGES_LIST:
                                if max_configs and config_count >= max_configs:
                                    break

                                ms = try_config(
                                    kernel_fn, *inputs, M, N, K,
                                    bm, bn, bk, gm, nw, ns,
                                )

                                if ms is None:
                                    continue

                                config_count += 1
                                tf = tflops(M, N, K, ms / 1000.0)
                                is_best = ms < best_ms
                                if is_best:
                                    best_ms = ms
                                    best_cfg = (bm, bn, bk, gm, nw, ns)

                                status = " *** " if is_best else ""
                                print(
                                    f"{bm:>4} {bn:>4} {bk:>4} {gm:>3} {nw:>3} {ns:>3}  "
                                    f"{ms:>10.4f} {tf:>10.3f} {status:>8}"
                                )

                                all_results.append({
                                    "kernel": kernel_name,
                                    "M": M, "K": K, "N": N,
                                    "BLOCK_M": bm, "BLOCK_N": bn, "BLOCK_K": bk,
                                    "GROUP_SIZE_M": gm, "num_warps": nw, "num_stages": ns,
                                    "time_ms": round(ms, 4),
                                    "tflops": round(tf, 3),
                                })

        if best_cfg:
            bm, bn, bk, gm, nw, ns = best_cfg
            tf = tflops(M, N, K, best_ms / 1000.0)
            print(f"\n  BEST: BLOCK_M={bm}, BLOCK_N={bn}, BLOCK_K={bk}, "
                  f"GROUP_M={gm}, warps={nw}, stages={ns}  "
                  f"→ {best_ms:.4f} ms  ({tf:.3f} TFLOPS)")
        else:
            print("\n  No valid config found!")

    return all_results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="LoRaQ kernel hyperparameter sweep")
    parser.add_argument("--json", type=str, default=None, help="Export JSON path")
    parser.add_argument("--v1-only", action="store_true", help="Sweep only V1 (tl.dot fp16)")
    parser.add_argument("--v2-only", action="store_true", help="Sweep only V2 (dot_scaled fp8)")
    parser.add_argument("--max-configs", type=int, default=None,
                        help="Max configs to test per size (for quick runs)")
    parser.add_argument("--check-asm", action="store_true",
                        help="Inspect Triton cache for MFMA instructions")
    args = parser.parse_args()

    torch.manual_seed(42)
    results = {}

    run_all = not (args.v1_only or args.v2_only)

    if run_all or not args.v2_only:
        print("\n" + "#" * 100)
        print("  SWEEP: LoRaQ V1 (Kernel 7 — tl.dot fp16 for Phase ②)")
        print("#" * 100)
        results["v1"] = sweep_kernel(
            loraq_fused_q8_kernel, "LoRaQ_V1", SIZES,
            max_configs=args.max_configs,
        )

    if run_all or not args.v1_only:
        print("\n" + "#" * 100)
        print("  SWEEP: LoRaQ V2 (Kernel 8 — dot_scaled fp8 for Phase ②)")
        print("#" * 100)
        results["v2"] = sweep_kernel(
            loraq_fused_q8_scaled_kernel, "LoRaQ_V2", SIZES,
            max_configs=args.max_configs,
        )

    # Assembly inspection
    if args.check_asm:
        print("\n" + "=" * 80)
        print("  Assembly Inspection (Triton cache)")
        print("=" * 80)

        strings = [
            "v_mfma_scale_f32_32x32x64_f8f6f4",
            "v_mfma_scale_f32_16x16x128_f8f6f4",
            "v_mfma_f32_32x32x16",
            "v_mfma_f32_16x16x32",
        ]

        for pattern in ["loraq_fused_q8_kernel", "loraq_fused_q8_scaled_kernel"]:
            print(f"\n  Searching for: {pattern}")
            counts = search_in_triton_cache(
                filename_pattern=f"{pattern}.amdgcn",
                search_strings=strings,
                print_lines=False,
            )
            for s, c in counts.items():
                print(f"    {s}: {c} occurrences")

    # Export
    if args.json:
        with open(args.json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults exported to {args.json}")

    # Summary: print best config for each kernel × size
    print("\n" + "=" * 100)
    print("  SUMMARY: Best configs")
    print("=" * 100)
    for kernel_name, kernel_results in results.items():
        print(f"\n  {kernel_name}:")
        # Group by (M, K, N) and find best
        from collections import defaultdict
        by_size = defaultdict(list)
        for r in kernel_results:
            by_size[(r["M"], r["K"], r["N"])].append(r)
        for (m, k, n), cfgs in sorted(by_size.items()):
            best = min(cfgs, key=lambda x: x["time_ms"])
            print(
                f"    M={m:>5}, K={k:>5}, N={n:>5}  →  "
                f"BM={best['BLOCK_M']}, BN={best['BLOCK_N']}, BK={best['BLOCK_K']}, "
                f"GM={best['GROUP_SIZE_M']}, warps={best['num_warps']}, stages={best['num_stages']}  "
                f"({best['time_ms']:.4f} ms, {best['tflops']:.3f} TFLOPS)"
            )


if __name__ == "__main__":
    main()
