"""
Autotune configurations for LoRaQ kernels.

Provides:
  - LORAQ_Q8_CONFIGS : list of triton.Config for loraq_fused_q8 kernels
  - AutotunedLoRaQ   : runtime autotuner that caches best config per (M,N,K)

The configs sweep BLOCK_M, BLOCK_N, BLOCK_K, GROUP_SIZE_M, num_warps,
and num_stages to find the optimal tiling for each problem size.

Usage with the runtime autotuner:
    from fast_loraq.autotune_configs import AutotunedLoRaQ, LORAQ_Q8_CONFIGS
    from fast_loraq.kernels import loraq_fused_q8_kernel

    autotuned = AutotunedLoRaQ(loraq_fused_q8_kernel, LORAQ_Q8_CONFIGS)
    # First call tunes; subsequent calls use cached best config:
    c_fp8, c_scale = autotuned(a_fp8, a_scale, r_fp8, r_scale, ...)
"""

import torch
import triton
import triton.testing as tt


# ---------------------------------------------------------------------------
# Config definitions  (same format as triton.Config for @triton.autotune)
# ---------------------------------------------------------------------------

LORAQ_Q8_CONFIGS = [
    # ---- Standard configs (good for M ≥ 128) ----
    triton.Config(
        {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_SIZE_M": 8},
        num_warps=8, num_stages=2,
    ),
    triton.Config(
        {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 128, "GROUP_SIZE_M": 8},
        num_warps=8, num_stages=2,
    ),
    triton.Config(
        {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_SIZE_M": 4},
        num_warps=4, num_stages=2,
    ),
    triton.Config(
        {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_SIZE_M": 1},
        num_warps=8, num_stages=1,
    ),
    # ---- Large tiles (good for large M, N) ----
    triton.Config(
        {"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_SIZE_M": 4},
        num_warps=8, num_stages=2,
    ),
    triton.Config(
        {"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 64, "GROUP_SIZE_M": 4},
        num_warps=8, num_stages=2,
    ),
    triton.Config(
        {"BLOCK_M": 256, "BLOCK_N": 256, "BLOCK_K": 64, "GROUP_SIZE_M": 1},
        num_warps=8, num_stages=1,
    ),
    triton.Config(
        {"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 128, "GROUP_SIZE_M": 8},
        num_warps=8, num_stages=1,
    ),
    triton.Config(
        {"BLOCK_M": 256, "BLOCK_N": 256, "BLOCK_K": 128, "GROUP_SIZE_M": 4},
        num_warps=8, num_stages=1,
    ),
    # ---- Small tiles (good for small M, decode) ----
    triton.Config(
        {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_SIZE_M": 8},
        num_warps=4, num_stages=2,
    ),
    triton.Config(
        {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 64, "GROUP_SIZE_M": 8},
        num_warps=4, num_stages=2,
    ),
    triton.Config(
        {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 128, "GROUP_SIZE_M": 8},
        num_warps=8, num_stages=2,
    ),
    triton.Config(
        {"BLOCK_M": 64, "BLOCK_N": 256, "BLOCK_K": 64, "GROUP_SIZE_M": 4},
        num_warps=8, num_stages=2,
    ),
    # ---- num_stages=1 variants ----
    triton.Config(
        {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_SIZE_M": 8},
        num_warps=8, num_stages=1,
    ),
    triton.Config(
        {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 128, "GROUP_SIZE_M": 4},
        num_warps=8, num_stages=1,
    ),
]


# ---------------------------------------------------------------------------
# Runtime autotuner (no kernel code duplication needed)
# ---------------------------------------------------------------------------

class AutotunedLoRaQ:
    """
    Runtime autotuner for LoRaQ kernels.

    Wraps an existing Triton JIT kernel and caches the best
    (BLOCK_M, BLOCK_N, BLOCK_K, GROUP_SIZE_M, num_warps, num_stages)
    configuration for each (M, N, K) problem size.

    This achieves the same effect as ``@triton.autotune`` without
    requiring kernel code duplication.

    Parameters
    ----------
    kernel_fn : triton JIT kernel
        The kernel to autotune (e.g., loraq_fused_q8_kernel).
    configs : list[triton.Config]
        Configurations to sweep.
    rank : int
        Fixed rank (default 64).
    warmup : int
        Warmup iterations for do_bench.
    rep : int
        Repetitions for do_bench.
    """

    def __init__(
        self,
        kernel_fn,
        configs=None,
        rank: int = 64,
        warmup: int = 10,
        rep: int = 50,
    ):
        self.kernel_fn = kernel_fn
        self.configs = configs or LORAQ_Q8_CONFIGS
        self.rank = rank
        self.warmup = warmup
        self.rep = rep
        self._cache: dict[tuple[int, int, int], dict] = {}

    def _make_launch_fn(self, cfg, a_fp8, a_scale, r_fp8, r_scale,
                        l_fp8, l_scale, w_fp4_t, w_scale, bias_ptr,
                        channel_scale,
                        c_fp8, c_scale, M, N, K, has_bias):
        """Create a lambda that launches the kernel with the given config."""
        bm = cfg.kwargs["BLOCK_M"]
        bn = cfg.kwargs["BLOCK_N"]
        bk = cfg.kwargs["BLOCK_K"]
        gm = cfg.kwargs["GROUP_SIZE_M"]
        nw = cfg.num_warps
        ns = cfg.num_stages

        grid = (triton.cdiv(M, bm) * triton.cdiv(N, bn),)

        def launch():
            self.kernel_fn[grid](
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
                HAS_BIAS=has_bias,
                RANK=self.rank,
                BLOCK_M=bm,
                BLOCK_N=bn,
                BLOCK_K=bk,
                GROUP_SIZE_M=gm,
                num_warps=nw,
                num_stages=ns,
            )
        return launch

    def _tune(self, a_fp8, a_scale, r_fp8, r_scale, l_fp8, l_scale,
              w_fp4_t, w_scale, M, N, K, bias, has_bias, channel_scale):
        """Run all configs and pick the fastest."""
        bias_ptr = bias if has_bias else a_fp8

        c_fp8 = torch.empty((M, N), dtype=torch.uint8, device=a_fp8.device)
        c_scale = torch.empty((M, N // 32), dtype=torch.uint8, device=a_fp8.device)

        best_ms = float("inf")
        best_cfg = None

        for cfg in self.configs:
            bm = cfg.kwargs["BLOCK_M"]
            bn = cfg.kwargs["BLOCK_N"]
            bk = cfg.kwargs["BLOCK_K"]

            # Skip invalid configs
            if bk > K:
                continue

            try:
                fn = self._make_launch_fn(
                    cfg, a_fp8, a_scale, r_fp8, r_scale,
                    l_fp8, l_scale, w_fp4_t, w_scale,
                    bias_ptr, channel_scale, c_fp8, c_scale, M, N, K, has_bias,
                )
                ms = tt.do_bench(fn, warmup=self.warmup, rep=self.rep)
                if ms < best_ms:
                    best_ms = ms
                    best_cfg = cfg
            except Exception:
                continue

        if best_cfg is None:
            # Fall back to default config
            best_cfg = triton.Config(
                {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_SIZE_M": 8},
                num_warps=8, num_stages=2,
            )

        return {
            "config": best_cfg,
            "time_ms": best_ms,
        }

    def __call__(self, a_fp8, a_scale, r_fp8, r_scale, l_fp8, l_scale,
                 w_fp4_t, w_scale, M, N, K, bias=None, channel_scale=None):
        """
        Launch the kernel with the best config for (M, N, K).
        First call for a given (M,N,K) triggers autotuning.

        Returns
        -------
        c_fp8   : (M, N) float8_e4m3fn
        c_scale : (M, N//32) uint8
        """
        key = (M, N, K)
        has_bias = bias is not None
        bias_ptr = bias if has_bias else a_fp8

        # Default channel_scale to ones if not provided
        if channel_scale is None:
            channel_scale = torch.ones(N, dtype=torch.float16, device=a_fp8.device)

        if key not in self._cache:
            self._cache[key] = self._tune(
                a_fp8, a_scale, r_fp8, r_scale,
                l_fp8, l_scale, w_fp4_t, w_scale,
                M, N, K, bias, has_bias, channel_scale,
            )

        cfg = self._cache[key]["config"]
        bm = cfg.kwargs["BLOCK_M"]
        bn = cfg.kwargs["BLOCK_N"]
        bk = cfg.kwargs["BLOCK_K"]
        gm = cfg.kwargs["GROUP_SIZE_M"]

        c_fp8 = torch.empty((M, N), dtype=torch.uint8, device=a_fp8.device)
        c_scale = torch.empty((M, N // 32), dtype=torch.uint8, device=a_fp8.device)

        grid = (triton.cdiv(M, bm) * triton.cdiv(N, bn),)

        self.kernel_fn[grid](
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
            HAS_BIAS=has_bias,
            RANK=self.rank,
            BLOCK_M=bm,
            BLOCK_N=bn,
            BLOCK_K=bk,
            GROUP_SIZE_M=gm,
            num_warps=cfg.num_warps,
            num_stages=cfg.num_stages,
        )

        c_fp8 = c_fp8.view(torch.float8_e4m3fn)
        return c_fp8, c_scale

    def get_best_config(self, M, N, K):
        """Return the cached best config for (M,N,K), or None if not tuned."""
        return self._cache.get((M, N, K))
