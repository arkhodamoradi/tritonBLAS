"""
Autotune configurations for LoRaQ kernels.

Provides:
  - LORAQ_Q8_CONFIGS   : list of triton.Config for loraq_fused_q8 kernels
  - AutotunedLoRaQ     : runtime autotuner that caches best config per (M,N,K)
  - AutotunedLoRaQ3    : runtime autotuner for loraq_fused_fp16io_kernel (kernel 9)
  - AutotunedDualGEMM  : runtime autotuner for loraq_dual_gemm_kernel (kernel 6)

The configs sweep BLOCK_M, BLOCK_N, BLOCK_K, GROUP_SIZE_M, num_warps,
and num_stages to find the optimal tiling for each problem size.

Usage with the runtime autotuner:
    from loraq.autotune_configs import AutotunedLoRaQ, LORAQ_Q8_CONFIGS
    from loraq.kernels import loraq_fused_q8_kernel

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
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_SIZE_M": 8}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_SIZE_M": 4}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_SIZE_M": 1}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 128, "GROUP_SIZE_M": 8}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 128, "GROUP_SIZE_M": 4}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 128, "GROUP_SIZE_M": 1}, num_warps=8, num_stages=2),
    # ---- Large tiles (good for large M, N) ----
    triton.Config({"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_SIZE_M": 4}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 128, "GROUP_SIZE_M": 4}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 128, "GROUP_SIZE_M": 8}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 64, "GROUP_SIZE_M": 4}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 64, "GROUP_SIZE_M": 8}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 128, "GROUP_SIZE_M": 4}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 128, "GROUP_SIZE_M": 8}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 128, "GROUP_SIZE_M": 1}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK_M": 256, "BLOCK_N": 256, "BLOCK_K": 64, "GROUP_SIZE_M": 4}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK_M": 256, "BLOCK_N": 256, "BLOCK_K": 64, "GROUP_SIZE_M": 8}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK_M": 256, "BLOCK_N": 256, "BLOCK_K": 64, "GROUP_SIZE_M": 1}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK_M": 256, "BLOCK_N": 256, "BLOCK_K": 128, "GROUP_SIZE_M": 4}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK_M": 256, "BLOCK_N": 256, "BLOCK_K": 128, "GROUP_SIZE_M": 8}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK_M": 256, "BLOCK_N": 256, "BLOCK_K": 128, "GROUP_SIZE_M": 1}, num_warps=8, num_stages=2),
    # ---- Small tiles (good for small M, decode) ----
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_SIZE_M": 8}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 64, "GROUP_SIZE_M": 8}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 128, "GROUP_SIZE_M": 8}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 256, "BLOCK_K": 64, "GROUP_SIZE_M": 4}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 256, "BLOCK_K": 128, "GROUP_SIZE_M": 4}, num_warps=8, num_stages=2),
    # ---- num_stages=1 variants ----
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_SIZE_M": 8}, num_warps=8, num_stages=1),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 128, "GROUP_SIZE_M": 4}, num_warps=8, num_stages=1),
    triton.Config({"BLOCK_M": 256, "BLOCK_N": 256, "BLOCK_K": 64, "GROUP_SIZE_M": 1}, num_warps=8, num_stages=1),
    triton.Config({"BLOCK_M": 256, "BLOCK_N": 256, "BLOCK_K": 128, "GROUP_SIZE_M": 1}, num_warps=8, num_stages=1),
    triton.Config({"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 128, "GROUP_SIZE_M": 8}, num_warps=8, num_stages=1),
    # ---- num_warps=4 variants ----
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_SIZE_M": 4}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 64, "GROUP_SIZE_M": 4}, num_warps=4, num_stages=2),
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
                matrix_instr_nonkdim=32,
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
            matrix_instr_nonkdim=32,
        )

        c_fp8 = c_fp8.view(torch.float8_e4m3fn)
        return c_fp8, c_scale

    def get_best_config(self, M, N, K):
        """Return the cached best config for (M,N,K), or None if not tuned."""
        return self._cache.get((M, N, K))


# ---------------------------------------------------------------------------
# Runtime autotuner for LoRaQ.3 fp16io kernel (kernel 9)
# ---------------------------------------------------------------------------

class AutotunedLoRaQ3:
    """
    Runtime autotuner for ``loraq_fused_fp16io_kernel`` (kernel 9).

    LoRaQ.3: fp16 input (in-register quant), fp16 output (no output quant).

    Parameters
    ----------
    kernel_fn : triton JIT kernel
        The fp16io kernel (loraq_fused_fp16io_kernel).
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

    def _make_launch_fn(self, cfg, A, r_fp8, r_scale, l_fp8, l_scale,
                        w_fp4_t, w_scale, bias_ptr, C, M, N, K, has_bias):
        bm = cfg.kwargs["BLOCK_M"]
        bn = cfg.kwargs["BLOCK_N"]
        bk = cfg.kwargs["BLOCK_K"]
        gm = cfg.kwargs["GROUP_SIZE_M"]
        nw = cfg.num_warps
        ns = cfg.num_stages

        grid = (triton.cdiv(M, bm) * triton.cdiv(N, bn),)

        def launch():
            self.kernel_fn[grid](
                A,
                r_fp8, r_scale,
                l_fp8, l_scale,
                w_fp4_t, w_scale,
                bias_ptr,
                C,
                M, N, K,
                A.stride(0), A.stride(1),
                r_fp8.stride(0), r_fp8.stride(1),
                r_scale.stride(0), r_scale.stride(1),
                l_fp8.stride(0), l_fp8.stride(1),
                l_scale.stride(0), l_scale.stride(1),
                w_fp4_t.stride(0), w_fp4_t.stride(1),
                w_scale.stride(0), w_scale.stride(1),
                C.stride(0), C.stride(1),
                HAS_BIAS=has_bias,
                RANK=self.rank,
                BLOCK_M=bm,
                BLOCK_N=bn,
                BLOCK_K=bk,
                GROUP_SIZE_M=gm,
                num_warps=nw,
                num_stages=ns,
                matrix_instr_nonkdim=32,
            )
        return launch

    def _tune(self, A, r_fp8, r_scale, l_fp8, l_scale,
              w_fp4_t, w_scale, M, N, K, bias, has_bias):
        bias_ptr = bias if has_bias else A
        C = torch.empty((M, N), dtype=torch.float16, device=A.device)

        best_ms = float("inf")
        best_cfg = None

        for cfg in self.configs:
            bk = cfg.kwargs["BLOCK_K"]
            if bk > K:
                continue
            try:
                fn = self._make_launch_fn(
                    cfg, A, r_fp8, r_scale, l_fp8, l_scale,
                    w_fp4_t, w_scale, bias_ptr, C, M, N, K, has_bias,
                )
                ms = tt.do_bench(fn, warmup=self.warmup, rep=self.rep)
                if ms < best_ms:
                    best_ms = ms
                    best_cfg = cfg
            except Exception:
                continue

        if best_cfg is None:
            best_cfg = triton.Config(
                {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_SIZE_M": 8},
                num_warps=8, num_stages=2,
            )

        return {"config": best_cfg, "time_ms": best_ms}

    def __call__(self, A, r_fp8, r_scale, l_fp8, l_scale,
                 w_fp4_t, w_scale, M, N, K, bias=None):
        """
        Launch kernel 9 with the best config for (M, N, K).
        First call triggers autotuning.

        Returns
        -------
        C : (M, N) fp16
        """
        key = (M, N, K)
        has_bias = bias is not None
        bias_ptr = bias if has_bias else A

        if key not in self._cache:
            self._cache[key] = self._tune(
                A, r_fp8, r_scale, l_fp8, l_scale,
                w_fp4_t, w_scale, M, N, K, bias, has_bias,
            )

        cfg = self._cache[key]["config"]
        bm = cfg.kwargs["BLOCK_M"]
        bn = cfg.kwargs["BLOCK_N"]
        bk = cfg.kwargs["BLOCK_K"]
        gm = cfg.kwargs["GROUP_SIZE_M"]

        C = torch.empty((M, N), dtype=torch.float16, device=A.device)
        grid = (triton.cdiv(M, bm) * triton.cdiv(N, bn),)

        self.kernel_fn[grid](
            A,
            r_fp8, r_scale,
            l_fp8, l_scale,
            w_fp4_t, w_scale,
            bias_ptr,
            C,
            M, N, K,
            A.stride(0), A.stride(1),
            r_fp8.stride(0), r_fp8.stride(1),
            r_scale.stride(0), r_scale.stride(1),
            l_fp8.stride(0), l_fp8.stride(1),
            l_scale.stride(0), l_scale.stride(1),
            w_fp4_t.stride(0), w_fp4_t.stride(1),
            w_scale.stride(0), w_scale.stride(1),
            C.stride(0), C.stride(1),
            HAS_BIAS=has_bias,
            RANK=self.rank,
            BLOCK_M=bm,
            BLOCK_N=bn,
            BLOCK_K=bk,
            GROUP_SIZE_M=gm,
            num_warps=cfg.num_warps,
            num_stages=cfg.num_stages,
            matrix_instr_nonkdim=32,
        )

        return C

    def get_best_config(self, M, N, K):
        return self._cache.get((M, N, K))


# ---------------------------------------------------------------------------
# Runtime autotuner for LoRaQ.6 (K13: fp16 L and R)
# ---------------------------------------------------------------------------

class AutotunedLoRaQFP16LR:
    """
    Runtime autotuner for ``loraq_fused_q8_fp16lr_kernel`` (K13).

    Like AutotunedLoRaQ but R and L are fp16 (no scales).
    """

    def __init__(self, kernel_fn, configs=None, rank=64, warmup=10, rep=50):
        self.kernel_fn = kernel_fn
        self.configs = configs or LORAQ_Q8_CONFIGS
        self.rank = rank
        self.warmup = warmup
        self.rep = rep
        self._cache = {}

    def _launch(self, cfg, a_fp8, a_scale, R, L, w_fp4_t, w_scale,
                bias_ptr, channel_scale, c_fp8, c_scale, M, N, K, has_bias):
        bm = cfg.kwargs["BLOCK_M"]
        bn = cfg.kwargs["BLOCK_N"]
        bk = cfg.kwargs["BLOCK_K"]
        gm = cfg.kwargs["GROUP_SIZE_M"]
        grid = (triton.cdiv(M, bm) * triton.cdiv(N, bn),)
        self.kernel_fn[grid](
            a_fp8, a_scale, R, L,
            w_fp4_t, w_scale, bias_ptr, channel_scale,
            c_fp8, c_scale, M, N, K,
            a_fp8.stride(0), a_fp8.stride(1),
            a_scale.stride(0), a_scale.stride(1),
            R.stride(0), R.stride(1),
            L.stride(0), L.stride(1),
            w_fp4_t.stride(0), w_fp4_t.stride(1),
            w_scale.stride(0), w_scale.stride(1),
            c_fp8.stride(0), c_fp8.stride(1),
            c_scale.stride(0), c_scale.stride(1),
            channel_scale.stride(0),
            HAS_BIAS=has_bias, RANK=self.rank,
            BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk, GROUP_SIZE_M=gm,
            num_warps=cfg.num_warps, num_stages=cfg.num_stages,
            matrix_instr_nonkdim=32,
        )

    def __call__(self, a_fp8, a_scale, R, L, w_fp4_t, w_scale,
                 M, N, K, bias=None, channel_scale=None):
        key = (M, N, K)
        has_bias = bias is not None
        bias_ptr = bias if has_bias else a_fp8
        if channel_scale is None:
            channel_scale = torch.ones(N, dtype=torch.float16, device=a_fp8.device)

        if key not in self._cache:
            c_fp8 = torch.empty((M, N), dtype=torch.uint8, device=a_fp8.device)
            c_scale = torch.empty((M, N // 32), dtype=torch.uint8, device=a_fp8.device)
            best_ms, best_cfg = float("inf"), None
            for cfg in self.configs:
                if cfg.kwargs["BLOCK_K"] > K:
                    continue
                try:
                    def fn(cfg=cfg):
                        self._launch(cfg, a_fp8, a_scale, R, L, w_fp4_t, w_scale,
                                     bias_ptr, channel_scale, c_fp8, c_scale,
                                     M, N, K, has_bias)
                    ms = tt.do_bench(fn, warmup=self.warmup, rep=self.rep)
                    if ms < best_ms:
                        best_ms, best_cfg = ms, cfg
                except Exception:
                    continue
            if best_cfg is None:
                best_cfg = triton.Config(
                    {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_SIZE_M": 8},
                    num_warps=8, num_stages=2)
            self._cache[key] = {"config": best_cfg, "time_ms": best_ms}

        cfg = self._cache[key]["config"]
        c_fp8 = torch.empty((M, N), dtype=torch.uint8, device=a_fp8.device)
        c_scale = torch.empty((M, N // 32), dtype=torch.uint8, device=a_fp8.device)
        self._launch(cfg, a_fp8, a_scale, R, L, w_fp4_t, w_scale,
                     bias_ptr, channel_scale, c_fp8, c_scale,
                     M, N, K, has_bias)
        c_fp8 = c_fp8.view(torch.float8_e4m3fn)
        return c_fp8, c_scale

    def get_best_config(self, M, N, K):
        return self._cache.get((M, N, K))


# ---------------------------------------------------------------------------
# Runtime autotuner for LoRaQ.4 split kernels (K10 proj + K11 main)
# ---------------------------------------------------------------------------

LORAQ_PROJ_CONFIGS = [
    (64, 64), (64, 128), (128, 64), (128, 128), (256, 64), (256, 128),
]


class AutotunedLoRaQProj:
    """Runtime autotuner for loraq_split_proj_kernel (K10)."""

    def __init__(self, kernel_fn, configs=None, rank=64, warmup=10, rep=50):
        self.kernel_fn = kernel_fn
        self.configs = configs or LORAQ_PROJ_CONFIGS
        self.rank = rank
        self.warmup = warmup
        self.rep = rep
        self._cache = {}

    def __call__(self, a_fp8, a_scale, r_fp8, r_scale, M, K):
        key = (M, K)
        if key not in self._cache:
            P = torch.empty((M, self.rank), dtype=torch.float16, device=a_fp8.device)
            best_ms, best_cfg = float("inf"), (128, 64)
            for bm, bk in self.configs:
                if bk > K:
                    continue
                try:
                    grid = (triton.cdiv(M, bm),)
                    def fn(bm=bm, bk=bk):
                        self.kernel_fn[grid](
                            a_fp8, a_scale, r_fp8, r_scale, P, M, K,
                            a_fp8.stride(0), a_fp8.stride(1),
                            a_scale.stride(0), a_scale.stride(1),
                            r_fp8.stride(0), r_fp8.stride(1),
                            r_scale.stride(0), r_scale.stride(1),
                            P.stride(0), P.stride(1),
                            RANK=self.rank, BLOCK_M=bm, BLOCK_K=bk,
                            num_warps=8, num_stages=2, matrix_instr_nonkdim=32,
                        )
                    ms = tt.do_bench(fn, warmup=self.warmup, rep=self.rep)
                    if ms < best_ms:
                        best_ms, best_cfg = ms, (bm, bk)
                except Exception:
                    continue
            self._cache[key] = {"config": best_cfg, "time_ms": best_ms}

        bm, bk = self._cache[key]["config"]
        P = torch.empty((M, self.rank), dtype=torch.float16, device=a_fp8.device)
        grid = (triton.cdiv(M, bm),)
        self.kernel_fn[grid](
            a_fp8, a_scale, r_fp8, r_scale, P, M, K,
            a_fp8.stride(0), a_fp8.stride(1),
            a_scale.stride(0), a_scale.stride(1),
            r_fp8.stride(0), r_fp8.stride(1),
            r_scale.stride(0), r_scale.stride(1),
            P.stride(0), P.stride(1),
            RANK=self.rank, BLOCK_M=bm, BLOCK_K=bk,
            num_warps=8, num_stages=2, matrix_instr_nonkdim=32,
        )
        return P

    def get_best_config(self, M, K):
        return self._cache.get((M, K))


class AutotunedLoRaQMain:
    """Runtime autotuner for loraq_split_main_kernel (K11)."""

    def __init__(self, kernel_fn, configs=None, rank=64, warmup=10, rep=50):
        self.kernel_fn = kernel_fn
        self.configs = configs or LORAQ_Q8_CONFIGS
        self.rank = rank
        self.warmup = warmup
        self.rep = rep
        self._cache = {}

    def _make_launch_fn(self, cfg, P, l_fp8, l_scale, a_fp8, a_scale,
                        w_fp4_t, w_scale, bias_ptr, channel_scale,
                        c_fp8, c_scale, M, N, K, has_bias):
        bm = cfg.kwargs["BLOCK_M"]
        bn = cfg.kwargs["BLOCK_N"]
        bk = cfg.kwargs["BLOCK_K"]
        gm = cfg.kwargs["GROUP_SIZE_M"]
        grid = (triton.cdiv(M, bm) * triton.cdiv(N, bn),)

        def launch():
            self.kernel_fn[grid](
                P, l_fp8, l_scale, a_fp8, a_scale,
                w_fp4_t, w_scale, bias_ptr, channel_scale,
                c_fp8, c_scale, M, N, K,
                P.stride(0), P.stride(1),
                l_fp8.stride(0), l_fp8.stride(1),
                l_scale.stride(0), l_scale.stride(1),
                a_fp8.stride(0), a_fp8.stride(1),
                a_scale.stride(0), a_scale.stride(1),
                w_fp4_t.stride(0), w_fp4_t.stride(1),
                w_scale.stride(0), w_scale.stride(1),
                c_fp8.stride(0), c_fp8.stride(1),
                c_scale.stride(0), c_scale.stride(1),
                channel_scale.stride(0),
                HAS_BIAS=has_bias, RANK=self.rank,
                BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk, GROUP_SIZE_M=gm,
                num_warps=cfg.num_warps, num_stages=cfg.num_stages,
                matrix_instr_nonkdim=32,
            )
        return launch

    def __call__(self, P, l_fp8, l_scale, a_fp8, a_scale,
                 w_fp4_t, w_scale, M, N, K, bias=None, channel_scale=None):
        key = (M, N, K)
        has_bias = bias is not None
        bias_ptr = bias if has_bias else a_fp8
        if channel_scale is None:
            channel_scale = torch.ones(N, dtype=torch.float16, device=a_fp8.device)

        if key not in self._cache:
            c_fp8 = torch.empty((M, N), dtype=torch.uint8, device=a_fp8.device)
            c_scale = torch.empty((M, N // 32), dtype=torch.uint8, device=a_fp8.device)
            best_ms, best_cfg = float("inf"), None
            for cfg in self.configs:
                if cfg.kwargs["BLOCK_K"] > K:
                    continue
                try:
                    fn = self._make_launch_fn(
                        cfg, P, l_fp8, l_scale, a_fp8, a_scale,
                        w_fp4_t, w_scale, bias_ptr, channel_scale,
                        c_fp8, c_scale, M, N, K, has_bias)
                    ms = tt.do_bench(fn, warmup=self.warmup, rep=self.rep)
                    if ms < best_ms:
                        best_ms, best_cfg = ms, cfg
                except Exception:
                    continue
            if best_cfg is None:
                best_cfg = triton.Config(
                    {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_SIZE_M": 8},
                    num_warps=8, num_stages=2)
            self._cache[key] = {"config": best_cfg, "time_ms": best_ms}

        cfg = self._cache[key]["config"]
        bm, bn = cfg.kwargs["BLOCK_M"], cfg.kwargs["BLOCK_N"]
        bk, gm = cfg.kwargs["BLOCK_K"], cfg.kwargs["GROUP_SIZE_M"]

        c_fp8 = torch.empty((M, N), dtype=torch.uint8, device=a_fp8.device)
        c_scale = torch.empty((M, N // 32), dtype=torch.uint8, device=a_fp8.device)
        grid = (triton.cdiv(M, bm) * triton.cdiv(N, bn),)

        self.kernel_fn[grid](
            P, l_fp8, l_scale, a_fp8, a_scale,
            w_fp4_t, w_scale, bias_ptr, channel_scale,
            c_fp8, c_scale, M, N, K,
            P.stride(0), P.stride(1),
            l_fp8.stride(0), l_fp8.stride(1),
            l_scale.stride(0), l_scale.stride(1),
            a_fp8.stride(0), a_fp8.stride(1),
            a_scale.stride(0), a_scale.stride(1),
            w_fp4_t.stride(0), w_fp4_t.stride(1),
            w_scale.stride(0), w_scale.stride(1),
            c_fp8.stride(0), c_fp8.stride(1),
            c_scale.stride(0), c_scale.stride(1),
            channel_scale.stride(0),
            HAS_BIAS=has_bias, RANK=self.rank,
            BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk, GROUP_SIZE_M=gm,
            num_warps=cfg.num_warps, num_stages=cfg.num_stages,
            matrix_instr_nonkdim=32,
        )
        c_fp8 = c_fp8.view(torch.float8_e4m3fn)
        return c_fp8, c_scale

    def get_best_config(self, M, N, K):
        return self._cache.get((M, N, K))


# ---------------------------------------------------------------------------
# Runtime autotuner for LoRA+Q project-and-quant kernel (kernel 5)
# ---------------------------------------------------------------------------

KERNEL5_BLOCK_MS = [64, 128, 256]


class AutotunedProjectAndQuant:
    """
    Runtime autotuner for ``loraq_project_and_quant_kernel`` (kernel 5).

    The only meaningful knob is BLOCK_M (rows per program).
    QUANT_GROUP is fixed at RANK (32).

    Parameters
    ----------
    kernel_fn : triton JIT kernel
        The project-and-quant kernel (loraq_project_and_quant_kernel).
    block_ms : list[int]
        BLOCK_M values to sweep (default [64, 128, 256]).
    rank : int
        Fixed rank (default 32, must equal QUANT_GROUP).
    warmup : int
        Warmup iterations for do_bench.
    rep : int
        Repetitions for do_bench.
    """

    def __init__(
        self,
        kernel_fn,
        block_ms=None,
        rank: int = 32,
        warmup: int = 10,
        rep: int = 50,
    ):
        self.kernel_fn = kernel_fn
        self.block_ms = block_ms or KERNEL5_BLOCK_MS
        self.rank = rank
        self.warmup = warmup
        self.rep = rep
        self._cache: dict[tuple[int, int], dict] = {}

    def _make_launch_fn(self, block_m, A, R, channel_scale, P, A_fp4, A_scale,
                        M, K):
        grid = (triton.cdiv(M, block_m),)

        def launch():
            self.kernel_fn[grid](
                A, R, channel_scale, P, A_fp4, A_scale,
                M, K,
                A.stride(0), A.stride(1),
                R.stride(0), R.stride(1),
                P.stride(0), P.stride(1),
                A_fp4.stride(0), A_fp4.stride(1),
                A_scale.stride(0), A_scale.stride(1),
                channel_scale.stride(0),
                RANK=self.rank,
                BLOCK_M=block_m,
                QUANT_GROUP=self.rank,
            )
        return launch

    def _tune(self, A, R, channel_scale, M, K):
        P = torch.empty((M, self.rank), device=A.device, dtype=A.dtype)
        A_fp4 = torch.empty((M, K // 2), device=A.device, dtype=torch.uint8)
        A_scale = torch.empty((M, K // 32), device=A.device, dtype=torch.uint8)

        best_ms = float("inf")
        best_bm = 128

        for bm in self.block_ms:
            try:
                fn = self._make_launch_fn(
                    bm, A, R, channel_scale, P, A_fp4, A_scale, M, K,
                )
                ms = tt.do_bench(fn, warmup=self.warmup, rep=self.rep)
                if ms < best_ms:
                    best_ms = ms
                    best_bm = bm
            except Exception:
                continue

        return {"block_m": best_bm, "time_ms": best_ms}

    def __call__(self, A, R, channel_scale):
        M, K = A.shape
        key = (M, K)

        if key not in self._cache:
            self._cache[key] = self._tune(A, R, channel_scale, M, K)

        block_m = self._cache[key]["block_m"]

        P = torch.empty((M, self.rank), device=A.device, dtype=A.dtype)
        A_fp4 = torch.empty((M, K // 2), device=A.device, dtype=torch.uint8)
        A_scale = torch.empty((M, K // 32), device=A.device, dtype=torch.uint8)

        grid = (triton.cdiv(M, block_m),)

        self.kernel_fn[grid](
            A, R, channel_scale, P, A_fp4, A_scale,
            M, K,
            A.stride(0), A.stride(1),
            R.stride(0), R.stride(1),
            P.stride(0), P.stride(1),
            A_fp4.stride(0), A_fp4.stride(1),
            A_scale.stride(0), A_scale.stride(1),
            channel_scale.stride(0),
            RANK=self.rank,
            BLOCK_M=block_m,
            QUANT_GROUP=self.rank,
        )

        return P, A_fp4, A_scale

    def get_best_config(self, M, K):
        return self._cache.get((M, K))


# ---------------------------------------------------------------------------
# Runtime autotuner for LoRA+Q dual GEMM kernel (kernel 6)
# ---------------------------------------------------------------------------

class AutotunedDualGEMM:
    """
    Runtime autotuner for the ``loraq_dual_gemm_kernel`` (kernel 6).

    Sweeps (BLOCK_M, BLOCK_N, BLOCK_K, GROUP_SIZE_M, num_warps, num_stages)
    and caches the best configuration per (M, N, K).

    Parameters
    ----------
    kernel_fn : triton JIT kernel
        The dual GEMM kernel (loraq_dual_gemm_kernel).
    configs : list[triton.Config]
        Configurations to sweep.
    rank : int
        Fixed rank (default 32).
    warmup : int
        Warmup iterations for do_bench.
    rep : int
        Repetitions for do_bench.
    """

    def __init__(
        self,
        kernel_fn,
        configs=None,
        rank: int = 32,
        warmup: int = 10,
        rep: int = 50,
    ):
        self.kernel_fn = kernel_fn
        self.configs = configs or LORAQ_Q8_CONFIGS
        self.rank = rank
        self.warmup = warmup
        self.rep = rep
        self._cache: dict[tuple[int, int, int], dict] = {}

    def _make_launch_fn(self, cfg, P, L, A_fp4, A_scale, W_fp4, W_scale,
                        C, M, N, K):
        """Create a callable that launches the kernel with the given config."""
        bm = cfg.kwargs["BLOCK_M"]
        bn = cfg.kwargs["BLOCK_N"]
        bk = cfg.kwargs["BLOCK_K"]
        gm = cfg.kwargs["GROUP_SIZE_M"]
        nw = cfg.num_warps
        ns = cfg.num_stages

        grid = (triton.cdiv(M, bm) * triton.cdiv(N, bn),)

        def launch():
            self.kernel_fn[grid](
                P, L,
                A_fp4, A_scale, W_fp4, W_scale,
                C,
                M, N, K,
                P.stride(0), P.stride(1),
                L.stride(0), L.stride(1),
                A_fp4.stride(0), A_fp4.stride(1),
                A_scale.stride(0), A_scale.stride(1),
                W_fp4.stride(0), W_fp4.stride(1),
                W_scale.stride(0), W_scale.stride(1),
                C.stride(0), C.stride(1),
                RANK=self.rank,
                BLOCK_M=bm,
                BLOCK_N=bn,
                BLOCK_K=bk,
                GROUP_SIZE_M=gm,
                num_warps=nw,
                num_stages=ns,
                matrix_instr_nonkdim=32,
            )
        return launch

    def _tune(self, P, L, A_fp4, A_scale, W_fp4, W_scale, M, N, K,
              out_dtype):
        """Run all configs and pick the fastest."""
        C = torch.empty((M, N), device=P.device, dtype=out_dtype)

        best_ms = float("inf")
        best_cfg = None

        for cfg in self.configs:
            bk = cfg.kwargs["BLOCK_K"]

            # Skip invalid configs
            if bk > K:
                continue

            try:
                fn = self._make_launch_fn(
                    cfg, P, L, A_fp4, A_scale, W_fp4, W_scale,
                    C, M, N, K,
                )
                ms = tt.do_bench(fn, warmup=self.warmup, rep=self.rep)
                if ms < best_ms:
                    best_ms = ms
                    best_cfg = cfg
            except Exception:
                continue

        if best_cfg is None:
            best_cfg = triton.Config(
                {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_SIZE_M": 8},
                num_warps=8, num_stages=2,
            )

        return {
            "config": best_cfg,
            "time_ms": best_ms,
        }

    def __call__(self, P, L, A_fp4, A_scale, W_fp4, W_scale,
                 M, N, K, out_dtype=torch.bfloat16):
        """
        Launch the kernel with the best config for (M, N, K).
        First call for a given (M,N,K) triggers autotuning.

        Returns
        -------
        C : (M, N) tensor of *out_dtype*
        """
        key = (M, N, K)

        if key not in self._cache:
            self._cache[key] = self._tune(
                P, L, A_fp4, A_scale, W_fp4, W_scale,
                M, N, K, out_dtype,
            )

        cfg = self._cache[key]["config"]
        bm = cfg.kwargs["BLOCK_M"]
        bn = cfg.kwargs["BLOCK_N"]
        bk = cfg.kwargs["BLOCK_K"]
        gm = cfg.kwargs["GROUP_SIZE_M"]

        C = torch.empty((M, N), device=P.device, dtype=out_dtype)

        grid = (triton.cdiv(M, bm) * triton.cdiv(N, bn),)

        self.kernel_fn[grid](
            P, L,
            A_fp4, A_scale, W_fp4, W_scale,
            C,
            M, N, K,
            P.stride(0), P.stride(1),
            L.stride(0), L.stride(1),
            A_fp4.stride(0), A_fp4.stride(1),
            A_scale.stride(0), A_scale.stride(1),
            W_fp4.stride(0), W_fp4.stride(1),
            W_scale.stride(0), W_scale.stride(1),
            C.stride(0), C.stride(1),
            RANK=self.rank,
            BLOCK_M=bm,
            BLOCK_N=bn,
            BLOCK_K=bk,
            GROUP_SIZE_M=gm,
            num_warps=cfg.num_warps,
            num_stages=cfg.num_stages,
            matrix_instr_nonkdim=32,
        )

        return C

    def get_best_config(self, M, N, K):
        """Return the cached best config for (M,N,K), or None if not tuned."""
        return self._cache.get((M, N, K))
