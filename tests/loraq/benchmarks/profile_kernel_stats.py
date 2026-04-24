"""
Profile LoRaQ.1 vs SVDQuant kernel statistics.

Extracts from compiled AMDGCN assembly:
  - VGPR/SGPR counts and spill counts
  - Scratch size, LDS size, Occupancy (read directly from .amdgcn)
  - MFMA instruction counts (fp16 and f8f6f4)

Also reports autotuned timing and best config hyperparameters.

Usage:
    python -m benchmarks.profile_kernel_stats --size 4096 1152 4608
    python -m benchmarks.profile_kernel_stats --all-sizes
"""

import argparse
import os
import re
import time

import torch
import torch.nn as nn
import triton
import triton.testing as tt

from loraq.linear import TritonLinearLoRaQ, TritonLinearLoRA
from loraq.quant import dynamic_mxfp8_quant
from loraq.autotune_configs import (
    AutotunedLoRaQ, AutotunedLoRaQProj, AutotunedLoRaQMain,
    AutotunedDualGEMM, AutotunedProjectAndQuant, LORAQ_Q8_CONFIGS,
)
from loraq.kernels import (
    loraq_fused_q8_kernel, loraq_fused_q8_scaled_kernel,
    loraq_dual_gemm_kernel, loraq_project_and_quant_kernel
)


# ---------------------------------------------------------------------------
# AMDGCN metadata extraction
# ---------------------------------------------------------------------------

def _get_triton_cache_dirs():
    """Return all candidate Triton cache directories."""
    dirs = [
        os.path.expanduser("~/.triton/cache/"),
        "/root/.triton/cache/",
    ]
    # Also check TRITON_CACHE_DIR env variable
    env_dir = os.environ.get("TRITON_CACHE_DIR")
    if env_dir:
        dirs.insert(0, env_dir)
    # Check common Docker / relative paths
    for extra in ["cache/", "./cache/", "/tmp/triton_cache/"]:
        if os.path.isdir(extra) and extra not in dirs:
            dirs.append(extra)
    return dirs


def find_amdgcn_by_name(kernel_name):
    """
    Find .amdgcn files matching a kernel function name in Triton cache.

    Searches ALL candidate cache directories for files named
    ``{kernel_name}.amdgcn``.  Returns list sorted by modification time
    (newest first).
    """
    target = f"{kernel_name}.amdgcn"
    files = []
    for cache_dir in _get_triton_cache_dirs():
        if not os.path.isdir(cache_dir):
            continue
        for root, _, filenames in os.walk(cache_dir):
            for f in filenames:
                if f == target:
                    path = os.path.join(root, f)
                    files.append(path)
    return sorted(files, key=os.path.getmtime, reverse=True)


def parse_amdgcn_metadata(filepath):
    """
    Parse kernel metadata from an AMDGCN assembly file.

    Extracts values from both comment-style headers (``; Field: value``)
    and YAML-style metadata (``.field: value``).
    """
    info = {
        # Comment-style fields
        "occupancy": None,
        "total_num_sgprs": None,
        "num_vgprs": None,
        "num_agprs": None,
        "total_num_vgprs": None,
        "scratch_size": None,
        "lds_byte_size": None,
        "max_flat_workgroup_size": None,
        # YAML-style fields
        "sgpr_count": None,
        "sgpr_spill_count": None,
        "vgpr_count": None,
        "vgpr_spill_count": None,
        # MFMA counts
        "mfma_fp16": 0,
        "mfma_f8f6f4": 0,
        # AGPR move counts
        "accvgpr_read": 0,
        "accvgpr_write": 0,
    }

    try:
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
    except Exception:
        return info

    # ---- Comment-style fields ("; Field: value") ----
    m = re.search(r";\s*Occupancy:\s*(\d+)", content)
    if m:
        info["occupancy"] = int(m.group(1))

    m = re.search(r";\s*TotalNumSgprs:\s*(\d+)", content)
    if m:
        info["total_num_sgprs"] = int(m.group(1))

    m = re.search(r";\s*NumVgprs:\s*(\d+)", content)
    if m:
        info["num_vgprs"] = int(m.group(1))

    m = re.search(r";\s*NumAgprs:\s*(\d+)", content)
    if m:
        info["num_agprs"] = int(m.group(1))

    m = re.search(r";\s*TotalNumVgprs:\s*(\d+)", content)
    if m:
        info["total_num_vgprs"] = int(m.group(1))

    m = re.search(r";\s*ScratchSize:\s*(\d+)", content)
    if m:
        info["scratch_size"] = int(m.group(1))

    m = re.search(r";\s*LDSByteSize:\s*(\d+)", content)
    if m:
        info["lds_byte_size"] = int(m.group(1))

    # ---- YAML-style fields (".field: value") ----
    m = re.search(r"\.max_flat_workgroup_size:\s*(\d+)", content)
    if m:
        info["max_flat_workgroup_size"] = int(m.group(1))

    m = re.search(r"\.sgpr_count:\s*(\d+)", content)
    if m:
        info["sgpr_count"] = int(m.group(1))

    m = re.search(r"\.sgpr_spill_count:\s*(\d+)", content)
    if m:
        info["sgpr_spill_count"] = int(m.group(1))

    m = re.search(r"\.vgpr_count:\s*(\d+)", content)
    if m:
        info["vgpr_count"] = int(m.group(1))

    m = re.search(r"\.vgpr_spill_count:\s*(\d+)", content)
    if m:
        info["vgpr_spill_count"] = int(m.group(1))

    # ---- MFMA instruction counts ----
    info["mfma_fp16"] = (
        len(re.findall(r"v_mfma_f32_32x32x16_f16", content))
        + len(re.findall(r"v_mfma_f32_16x16x32_f16", content))
    )
    info["mfma_f8f6f4"] = (
        len(re.findall(r"v_mfma_scale_f32_32x32x64_f8f6f4", content))
        + len(re.findall(r"v_mfma_scale_f32_16x16x128_f8f6f4", content))
    )

    # ---- AGPR read/write instruction counts ----
    info["accvgpr_read"] = len(re.findall(r"v_accvgpr_read_b32", content))
    info["accvgpr_write"] = len(re.findall(r"v_accvgpr_write_b32", content))

    return info


def fmt(v, suffix=""):
    """Format a value, showing '?' if None."""
    return "?" if v is None else f"{v}{suffix}"


# ---------------------------------------------------------------------------
# Kernel profiling
# ---------------------------------------------------------------------------

def profile_loraq1(M, K, N, warmup=25, iters=100):
    """Profile autotuned LoRaQ.1 (Kernel 7)."""
    x = torch.randn(M, K, device="cuda", dtype=torch.float16)
    a_fp8, a_scale = dynamic_mxfp8_quant(x)
    ref = nn.Linear(K, N, bias=False, device="cuda", dtype=torch.float16)
    layer = TritonLinearLoRaQ.from_float(ref)
    w_fp4_t = layer.weight_fp4.t().contiguous()

    at = AutotunedLoRaQ(loraq_fused_q8_kernel, LORAQ_Q8_CONFIGS, warmup=5, rep=25)

    # Trigger autotuning (compiles all configs)
    _ = at(a_fp8, a_scale,
           layer.R_fp8, layer.R_scale,
           layer.L_fp8, layer.L_scale,
           w_fp4_t, layer.weight_scale, M, N, K)
    torch.cuda.synchronize()

    # Benchmark best config
    t_ms = tt.do_bench(
        lambda: at(a_fp8, a_scale,
                   layer.R_fp8, layer.R_scale,
                   layer.L_fp8, layer.L_scale,
                   w_fp4_t, layer.weight_scale, M, N, K),
        warmup=warmup, rep=iters,
    )

    # Get best config
    best = at.get_best_config(M, N, K)
    cfg = best["config"] if best else None
    cfg_str = "?"
    if cfg:
        cfg_str = (f"BM={cfg.kwargs['BLOCK_M']}, BN={cfg.kwargs['BLOCK_N']}, "
                   f"BK={cfg.kwargs['BLOCK_K']}, GM={cfg.kwargs['GROUP_SIZE_M']}, "
                   f"warps={cfg.num_warps}, stages={cfg.num_stages}")

    # Find .amdgcn by kernel function name (most recent = best config)
    amdgcn_files = find_amdgcn_by_name("loraq_fused_q8_kernel")
    metadata = parse_amdgcn_metadata(amdgcn_files[0]) if amdgcn_files else {}

    return {
        "time_us": t_ms * 1000.0,
        "config": cfg_str,
        "metadata": metadata,
        "amdgcn_file": amdgcn_files[0] if amdgcn_files else None,
    }


def profile_loraq2(M, K, N, warmup=25, iters=100):
    """Profile autotuned LoRaQ.2 (Kernel 8 — dot_scaled Phase 2)."""
    x = torch.randn(M, K, device="cuda", dtype=torch.float16)
    a_fp8, a_scale = dynamic_mxfp8_quant(x)
    ref = nn.Linear(K, N, bias=False, device="cuda", dtype=torch.float16)
    layer = TritonLinearLoRaQ.from_float(ref)
    w_fp4_t = layer.weight_fp4.t().contiguous()

    at = AutotunedLoRaQ(loraq_fused_q8_scaled_kernel, LORAQ_Q8_CONFIGS, warmup=5, rep=25)

    # Trigger autotuning
    _ = at(a_fp8, a_scale,
           layer.R_fp8, layer.R_scale,
           layer.L_fp8, layer.L_scale,
           w_fp4_t, layer.weight_scale, M, N, K)
    torch.cuda.synchronize()

    t_ms = tt.do_bench(
        lambda: at(a_fp8, a_scale,
                   layer.R_fp8, layer.R_scale,
                   layer.L_fp8, layer.L_scale,
                   w_fp4_t, layer.weight_scale, M, N, K),
        warmup=warmup, rep=iters,
    )

    best = at.get_best_config(M, N, K)
    cfg = best["config"] if best else None
    cfg_str = "?"
    if cfg:
        cfg_str = (f"BM={cfg.kwargs['BLOCK_M']}, BN={cfg.kwargs['BLOCK_N']}, "
                   f"BK={cfg.kwargs['BLOCK_K']}, GM={cfg.kwargs['GROUP_SIZE_M']}, "
                   f"warps={cfg.num_warps}, stages={cfg.num_stages}")

    amdgcn_files = find_amdgcn_by_name("loraq_fused_q8_scaled_kernel")
    metadata = parse_amdgcn_metadata(amdgcn_files[0]) if amdgcn_files else {}

    return {
        "time_us": t_ms * 1000.0,
        "config": cfg_str,
        "metadata": metadata,
        "amdgcn_file": amdgcn_files[0] if amdgcn_files else None,
    }


def profile_svdquant(M, K, N, warmup=25, iters=100):
    """Profile autotuned SVDQuant (Kernels 5+6)."""
    x = torch.randn(M, K, device="cuda", dtype=torch.float16)
    ref = nn.Linear(K, N, bias=False, device="cuda", dtype=torch.float16)
    layer = TritonLinearLoRA.from_float(ref)
    w_fp4_t = layer.weight_fp4.t().contiguous()

    at_pq = AutotunedProjectAndQuant(loraq_project_and_quant_kernel, warmup=5, rep=25)
    at_dg = AutotunedDualGEMM(loraq_dual_gemm_kernel, LORAQ_Q8_CONFIGS, warmup=5, rep=25)

    # ---- Kernel 5: trigger autotuning ----
    P, a_fp4, a_scale_q = at_pq(x, layer.R, layer.channel_scale)
    torch.cuda.synchronize()

    t_k5_ms = tt.do_bench(
        lambda: at_pq(x, layer.R, layer.channel_scale),
        warmup=warmup, rep=iters,
    )
    amdgcn_k5 = find_amdgcn_by_name("loraq_project_and_quant_kernel")
    meta_k5 = parse_amdgcn_metadata(amdgcn_k5[0]) if amdgcn_k5 else {}

    best_pq = at_pq.get_best_config(M, K)
    k5_cfg = f"BM={best_pq['block_m']}" if best_pq else "?"

    # ---- Kernel 6: trigger autotuning ----
    _ = at_dg(P, layer.L, a_fp4, a_scale_q,
              w_fp4_t, layer.weight_scale, M, N, K)
    torch.cuda.synchronize()

    t_k6_ms = tt.do_bench(
        lambda: at_dg(P, layer.L, a_fp4, a_scale_q,
                      w_fp4_t, layer.weight_scale, M, N, K),
        warmup=warmup, rep=iters,
    )
    amdgcn_k6 = find_amdgcn_by_name("loraq_dual_gemm_kernel")
    meta_k6 = parse_amdgcn_metadata(amdgcn_k6[0]) if amdgcn_k6 else {}

    best_dg = at_dg.get_best_config(M, N, K)
    k6_cfg_str = "?"
    if best_dg:
        c = best_dg["config"]
        k6_cfg_str = (f"BM={c.kwargs['BLOCK_M']}, BN={c.kwargs['BLOCK_N']}, "
                      f"BK={c.kwargs['BLOCK_K']}, GM={c.kwargs['GROUP_SIZE_M']}, "
                      f"warps={c.num_warps}, stages={c.num_stages}")

    # E2E timing
    def svdq_e2e():
        P_, a_fp4_, a_scale_q_ = at_pq(x, layer.R, layer.channel_scale)
        at_dg(P_, layer.L, a_fp4_, a_scale_q_,
              w_fp4_t, layer.weight_scale, M, N, K)

    t_e2e_ms = tt.do_bench(svdq_e2e, warmup=warmup, rep=iters)

    return {
        "k5": {"time_us": t_k5_ms * 1000.0, "config": k5_cfg, "metadata": meta_k5},
        "k6": {"time_us": t_k6_ms * 1000.0, "config": k6_cfg_str, "metadata": meta_k6},
        "e2e_time_us": t_e2e_ms * 1000.0,
    }


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def print_kernel_stats(label, time_us, config_str, meta):
    """Print formatted kernel statistics."""
    print(f"  {label}:")
    print(f"    Config:            {config_str}")
    print(f"    Timing:            {time_us:.1f} µs")
    print(f"    NumVgprs:          {fmt(meta.get('num_vgprs')):<6}  "
          f"TotalNumVgprs:   {fmt(meta.get('total_num_vgprs'))}")
    print(f"    TotalNumSgprs:     {fmt(meta.get('total_num_sgprs')):<6}  "
          f"NumAgprs:        {fmt(meta.get('num_agprs'))}")
    print(f"    .vgpr_count:       {fmt(meta.get('vgpr_count')):<6}  "
          f".sgpr_count:     {fmt(meta.get('sgpr_count'))}")
    print(f"    .vgpr_spill_count: {fmt(meta.get('vgpr_spill_count')):<6}  "
          f".sgpr_spill_count: {fmt(meta.get('sgpr_spill_count'))}")
    print(f"    ScratchSize:       {fmt(meta.get('scratch_size'), ' B'):<10}  "
          f"LDSByteSize:     {fmt(meta.get('lds_byte_size'), ' B')}")
    print(f"    Occupancy:         {fmt(meta.get('occupancy')):<6}  "
          f"MaxWorkgroupSize: {fmt(meta.get('max_flat_workgroup_size'))}")
    print(f"    MFMA fp16:         {meta.get('mfma_fp16', '?'):<6}  "
          f"MFMA f8f6f4:     {meta.get('mfma_f8f6f4', '?')}")
    print(f"    accvgpr_read:      {meta.get('accvgpr_read', '?'):<6}  "
          f"accvgpr_write:   {meta.get('accvgpr_write', '?')}")
    print()


def run_profile(M, K, N):
    """Run full profile for a single (M, K, N) size."""
    W = 80
    print(f"\n{'=' * W}")
    print(f"  Kernel Profile: M={M}, K={K}, N={N}")
    print(f"{'=' * W}\n")

    # ---- LoRaQ.1 ----
    print("  Profiling LoRaQ.1 (Kernel 7) ...")
    r1 = profile_loraq1(M, K, N)
    print_kernel_stats(
        "LoRaQ.1 (Kernel 7 — fused FP8/FP4, Phase2=tl.dot fp16)",
        r1["time_us"], r1["config"], r1["metadata"],
    )

    # ---- LoRaQ.2 ----
    print("  Profiling LoRaQ.2 (Kernel 8) ...")
    r2_loraq = profile_loraq2(M, K, N)
    print_kernel_stats(
        "LoRaQ.2 (Kernel 8 — fused FP8/FP4, Phase2=dot_scaled fp8)",
        r2_loraq["time_us"], r2_loraq["config"], r2_loraq["metadata"],
    )


    # ---- SVDQuant ----
    print("  Profiling SVDQuant (Kernels 5+6) ...")
    r2 = profile_svdquant(M, K, N)
    print_kernel_stats(
        "SVDQuant Kernel 5 (project + quant)",
        r2["k5"]["time_us"], r2["k5"]["config"], r2["k5"]["metadata"],
    )
    print_kernel_stats(
        "SVDQuant Kernel 6 (dual GEMM)",
        r2["k6"]["time_us"], r2["k6"]["config"], r2["k6"]["metadata"],
    )

    # ---- Comparison ----
    print(f"  {'─' * 60}")
    print(f"  Summary:")
    print(f"    LoRaQ.1 (K7):          {r1['time_us']:.1f} µs")
    print(f"    LoRaQ.2 (K8):          {r2_loraq['time_us']:.1f} µs")
    print(f"    SVDQuant K5+K6 (e2e):  {r2['e2e_time_us']:.1f} µs")

    ratio_v1 = r1["time_us"] / r2["e2e_time_us"] if r2["e2e_time_us"] > 0 else float("inf")
    ratio_v2 = r2_loraq["time_us"] / r2["e2e_time_us"] if r2["e2e_time_us"] > 0 else float("inf")
    ratio_12 = r2_loraq["time_us"] / r1["time_us"] if r1["time_us"] > 0 else float("inf")

    def speedup_label(r):
        return "(faster)" if r < 1 else ("(slower)" if r > 1 else "(equal)")

    print(f"    LoRaQ.1 / SVDQuant:   {ratio_v1:.2f}x {speedup_label(ratio_v1)}")
    print(f"    LoRaQ.2 / SVDQuant:   {ratio_v2:.2f}x {speedup_label(ratio_v2)}")
    print(f"    LoRaQ.2 / LoRaQ.1:    {ratio_12:.2f}x {speedup_label(ratio_12)}")

    # Spill / scratch warnings
    for name, meta in [("LoRaQ.1 K7", r1["metadata"]),
                        ("LoRaQ.2 K8", r2_loraq["metadata"]),
                        ("SVDQuant K5", r2["k5"]["metadata"]),
                        ("SVDQuant K6", r2["k6"]["metadata"])]:
        vgpr_sp = meta.get("vgpr_spill_count") or 0
        sgpr_sp = meta.get("sgpr_spill_count") or 0
        scratch = meta.get("scratch_size") or 0
        if vgpr_sp > 0 or sgpr_sp > 0 or scratch > 0:
            print(f"    ⚠ {name}: vgpr_spills={vgpr_sp}, sgpr_spills={sgpr_sp}, scratch={scratch} B")

    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Profile LoRaQ.1 vs SVDQuant kernel statistics"
    )
    parser.add_argument("--size", nargs=3, type=int, default=[4096, 1152, 4608],
                        metavar=("M", "K", "N"), help="Problem size")
    parser.add_argument("--all-sizes", action="store_true",
                        help="Profile multiple sizes")
    args = parser.parse_args()

    torch.manual_seed(42)

    if args.all_sizes:
        sizes = [
            (4096, 1152, 1152),
            (4096, 1152, 4608),
            (4096, 4608, 1152),
            (4096, 3072, 3072),
            (4096, 3072, 12288),
        ]
    else:
        sizes = [tuple(args.size)]

    for M, K, N in sizes:
        if K % 64 != 0 or N % 32 != 0:
            print(f"  Skipping ({M}, {K}, {N}): K must be % 64, N must be % 32")
            continue
        run_profile(M, K, N)


if __name__ == "__main__":
    main()
