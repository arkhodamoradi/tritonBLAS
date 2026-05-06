"""
Autotune configurations for LoRaQ kernels.

Provides:
  - LORAQ_Q8_CONFIGS      : list of triton.Config for loraq_fused_q8 kernels
  - AutotunedLoRaQ        : runtime autotuner that caches best config per (M,N,K)
  - AutotunedLoRaQ_8_16   : variant for _8_16 kernel (FP8 in, FP16 out)
  - AutotunedLoRaQ_16_8   : variant for _16_8 kernel (FP16 in, FP8 out)
  - AutotunedLoRaQ_16_16  : variant for _16_16 kernel (FP16 in, FP16 out)
  - AutotunedLoRaQ4       : runtime autotuner for loraq_fused_q4_kernel (K13, MXFP4 in/out)
  - AutotunedLoRaQ4_16    : variant for loraq_fused_q4_kernel_4_16 (MXFP4 in, FP16 out)
  - AutotunedLoRaQ3       : runtime autotuner for loraq_fused_fp16io_kernel (kernel 9)
  - AutotunedDualGEMM     : runtime autotuner for loraq_dual_gemm_kernel (kernel 6)

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

# Base configs without waves_per_eu
_BASE_CONFIGS = [
    # ---- Standard configs (good for M ≥ 128) ----
    (128, 128, 64, 8, 8, 2),
    (128, 128, 64, 4, 8, 2),
    (128, 128, 128, 8, 8, 2),
    (128, 128, 128, 4, 8, 2),
    # ---- Large tiles ----
    (256, 128, 128, 4, 8, 2),
    (256, 128, 128, 8, 8, 2),
    (128, 256, 64, 4, 8, 2),
    (128, 256, 64, 8, 8, 2),
    (128, 256, 128, 4, 8, 2),
    (128, 256, 128, 8, 8, 2),
    (256, 256, 64, 4, 8, 2),
    (256, 256, 128, 4, 8, 2),
    # ---- Small tiles ----
    (64, 128, 64, 8, 4, 2),
    (64, 256, 64, 4, 8, 2),
    (64, 256, 128, 4, 8, 2),
    # ---- num_stages=1 ----
    (128, 128, 128, 4, 8, 1),
    (256, 128, 128, 8, 8, 1),
    # ---- num_warps=4 ----
    (128, 128, 64, 4, 4, 2),
    (128, 256, 64, 4, 4, 2),
    # ---- Small BLOCK_M for rank-128 kernels (lower register pressure) ----
    (64, 128, 128, 4, 4, 2),
    (64, 128, 128, 8, 4, 2),
    (64, 64, 64, 4, 4, 2),
    (64, 64, 128, 4, 4, 2),
    (32, 128, 64, 4, 4, 2),
    (32, 128, 128, 4, 4, 2),
    (32, 256, 64, 4, 4, 2),
    (32, 256, 128, 4, 4, 2),
]

# Expand with waves_per_eu sweep (0 = driver default, 1-4 = explicit)
LORAQ_Q8_CONFIGS = []
for bm, bn, bk, gm, nw, ns in _BASE_CONFIGS:
    for wpe in [0, 2, 4]:
        LORAQ_Q8_CONFIGS.append(
            triton.Config(
                {"BLOCK_M": bm, "BLOCK_N": bn, "BLOCK_K": bk, "GROUP_SIZE_M": gm},
                num_warps=nw, num_stages=ns,
                pre_hook=None,
            )
        )
# Store waves_per_eu in a parallel list (triton.Config doesn't support custom fields)
LORAQ_Q8_WPE = []
for bm, bn, bk, gm, nw, ns in _BASE_CONFIGS:
    for wpe in [0, 2, 4]:
        LORAQ_Q8_WPE.append(wpe)


# ---------------------------------------------------------------------------
# Runtime autotuner (no kernel code duplication needed)
# ---------------------------------------------------------------------------

class AutotunedLoRaQ:
    """
    Runtime autotuner for LoRaQ kernels.

    Sweeps (BLOCK_M, BLOCK_N, BLOCK_K, GROUP_SIZE_M, num_warps, num_stages,
    waves_per_eu) and caches the best combination per (M, N, K).

    Parameters
    ----------
    kernel_fn : triton JIT kernel
    configs : list[triton.Config]
    rank : int
    waves_per_eu_values : list[int]
        AMD occupancy hint. 0 = compiler default. Lower values give each
        wave more VGPRs at the cost of occupancy.
    warmup, rep : int
    """

    def __init__(
        self,
        kernel_fn,
        configs=None,
        rank: int = 64,
        waves_per_eu_values: list = None,
        warmup: int = 10,
        rep: int = 50,
    ):
        self.kernel_fn = kernel_fn
        self.configs = configs or LORAQ_Q8_CONFIGS
        self.rank = rank
        self.waves_per_eu_values = waves_per_eu_values if waves_per_eu_values is not None else [0, 1, 2]
        self.warmup = warmup
        self.rep = rep
        self._cache: dict[tuple[int, int, int], dict] = {}

    def _make_launch_fn(self, cfg, wpe, a_fp8, a_scale, r_fp8, r_scale,
                        l_fp8, l_scale, w_fp4_t, w_scale, bias_ptr,
                        channel_scale, c_fp8, c_scale, M, N, K, has_bias):
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
                waves_per_eu=wpe,
                matrix_instr_nonkdim=32,
            )
        return launch

    def _tune(self, a_fp8, a_scale, r_fp8, r_scale, l_fp8, l_scale,
              w_fp4_t, w_scale, M, N, K, bias, has_bias, channel_scale):
        bias_ptr = bias if has_bias else a_fp8
        c_fp8   = torch.empty((M, N),       dtype=torch.uint8, device=a_fp8.device)
        c_scale = torch.empty((M, N // 32), dtype=torch.uint8, device=a_fp8.device)

        best_ms, best_cfg, best_wpe = float("inf"), None, 0

        for cfg in self.configs:
            if cfg.kwargs["BLOCK_K"] > K:
                continue
            for wpe in self.waves_per_eu_values:
                try:
                    fn = self._make_launch_fn(
                        cfg, wpe, a_fp8, a_scale, r_fp8, r_scale,
                        l_fp8, l_scale, w_fp4_t, w_scale,
                        bias_ptr, channel_scale, c_fp8, c_scale, M, N, K, has_bias,
                    )
                    ms = tt.do_bench(fn, warmup=self.warmup, rep=self.rep)
                    if ms < best_ms:
                        best_ms, best_cfg, best_wpe = ms, cfg, wpe
                except Exception:
                    continue

        if best_cfg is None:
            best_cfg = triton.Config(
                {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_SIZE_M": 8},
                num_warps=8, num_stages=2,
            )
        return {"config": best_cfg, "waves_per_eu": best_wpe, "time_ms": best_ms}

    def __call__(self, a_fp8, a_scale, r_fp8, r_scale, l_fp8, l_scale,
                 w_fp4_t, w_scale, M, N, K, bias=None, channel_scale=None):
        key = (M, N, K)
        has_bias = bias is not None
        bias_ptr = bias if has_bias else a_fp8

        if channel_scale is None:
            channel_scale = torch.ones(N, dtype=torch.float16, device=a_fp8.device)

        if key not in self._cache:
            self._cache[key] = self._tune(
                a_fp8, a_scale, r_fp8, r_scale,
                l_fp8, l_scale, w_fp4_t, w_scale,
                M, N, K, bias, has_bias, channel_scale,
            )

        cfg = self._cache[key]["config"]
        wpe = self._cache[key]["waves_per_eu"]
        bm  = cfg.kwargs["BLOCK_M"]
        bn  = cfg.kwargs["BLOCK_N"]
        bk  = cfg.kwargs["BLOCK_K"]
        gm  = cfg.kwargs["GROUP_SIZE_M"]

        c_fp8   = torch.empty((M, N),       dtype=torch.uint8, device=a_fp8.device)
        c_scale = torch.empty((M, N // 32), dtype=torch.uint8, device=a_fp8.device)
        grid    = (triton.cdiv(M, bm) * triton.cdiv(N, bn),)

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
            waves_per_eu=wpe,
            matrix_instr_nonkdim=32,
        )

        c_fp8 = c_fp8.view(torch.float8_e4m3fn)
        return c_fp8, c_scale

    def get_best_config(self, M, N, K):
        """Return the cached best config for (M,N,K), or None if not tuned."""
        return self._cache.get((M, N, K))


# ---------------------------------------------------------------------------
# Runtime autotuner for _8_16 kernel (FP8 in, FP16 out)
# ---------------------------------------------------------------------------

class AutotunedLoRaQ_8_16:
    """
    Runtime autotuner for ``loraq_fused_q8_scaled_kernel_8_16``.

    Identical to AutotunedLoRaQ except the output is a single fp16 tensor C
    (no c_scale, no channel_scale); the kernel call omits those arguments.
    """

    def __init__(
        self,
        kernel_fn,
        configs=None,
        rank: int = 64,
        waves_per_eu_values: list = None,
        warmup: int = 10,
        rep: int = 50,
    ):
        self.kernel_fn = kernel_fn
        self.configs = configs or LORAQ_Q8_CONFIGS
        self.rank = rank
        self.waves_per_eu_values = waves_per_eu_values if waves_per_eu_values is not None else [0, 1, 2]
        self.warmup = warmup
        self.rep = rep
        self._cache: dict[tuple[int, int, int], dict] = {}

    def _make_launch_fn(self, cfg, wpe, a_fp8, a_scale, r_fp8, r_scale,
                        l_fp8, l_scale, w_fp4_t, w_scale, bias_ptr,
                        C, M, N, K, has_bias):
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
                C,
                M, N, K,
                a_fp8.stride(0), a_fp8.stride(1),
                a_scale.stride(0), a_scale.stride(1),
                r_fp8.stride(0), r_fp8.stride(1),
                r_scale.stride(0), r_scale.stride(1),
                l_fp8.stride(0), l_fp8.stride(1),
                l_scale.stride(0), l_scale.stride(1),
                w_fp4_t.stride(0), w_fp4_t.stride(1),
                w_scale.stride(0), w_scale.stride(1),
                C.stride(0), C.stride(1),
                HAS_BIAS=has_bias,
                RANK=self.rank,
                BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk, GROUP_SIZE_M=gm,
                num_warps=nw, num_stages=ns,
                waves_per_eu=wpe, matrix_instr_nonkdim=32,
            )
        return launch

    def _tune(self, a_fp8, a_scale, r_fp8, r_scale, l_fp8, l_scale,
              w_fp4_t, w_scale, M, N, K, bias, has_bias):
        bias_ptr = bias if has_bias else a_fp8
        C = torch.empty((M, N), dtype=torch.float16, device=a_fp8.device)

        best_ms, best_cfg, best_wpe = float("inf"), None, 0

        for cfg in self.configs:
            if cfg.kwargs["BLOCK_K"] > K:
                continue
            for wpe in self.waves_per_eu_values:
                try:
                    fn = self._make_launch_fn(
                        cfg, wpe, a_fp8, a_scale, r_fp8, r_scale,
                        l_fp8, l_scale, w_fp4_t, w_scale,
                        bias_ptr, C, M, N, K, has_bias,
                    )
                    ms = tt.do_bench(fn, warmup=self.warmup, rep=self.rep)
                    if ms < best_ms:
                        best_ms, best_cfg, best_wpe = ms, cfg, wpe
                except Exception:
                    continue

        if best_cfg is None:
            best_cfg = triton.Config(
                {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_SIZE_M": 8},
                num_warps=8, num_stages=2,
            )
        return {"config": best_cfg, "waves_per_eu": best_wpe, "time_ms": best_ms}

    def __call__(self, a_fp8, a_scale, r_fp8, r_scale, l_fp8, l_scale,
                 w_fp4_t, w_scale, M, N, K, bias=None):
        key = (M, N, K)
        has_bias = bias is not None
        bias_ptr = bias if has_bias else a_fp8

        if key not in self._cache:
            self._cache[key] = self._tune(
                a_fp8, a_scale, r_fp8, r_scale,
                l_fp8, l_scale, w_fp4_t, w_scale,
                M, N, K, bias, has_bias,
            )

        cfg = self._cache[key]["config"]
        wpe = self._cache[key]["waves_per_eu"]
        bm  = cfg.kwargs["BLOCK_M"]
        bn  = cfg.kwargs["BLOCK_N"]
        bk  = cfg.kwargs["BLOCK_K"]
        gm  = cfg.kwargs["GROUP_SIZE_M"]

        C    = torch.empty((M, N), dtype=torch.float16, device=a_fp8.device)
        grid = (triton.cdiv(M, bm) * triton.cdiv(N, bn),)

        self.kernel_fn[grid](
            a_fp8, a_scale,
            r_fp8, r_scale,
            l_fp8, l_scale,
            w_fp4_t, w_scale,
            bias_ptr,
            C,
            M, N, K,
            a_fp8.stride(0), a_fp8.stride(1),
            a_scale.stride(0), a_scale.stride(1),
            r_fp8.stride(0), r_fp8.stride(1),
            r_scale.stride(0), r_scale.stride(1),
            l_fp8.stride(0), l_fp8.stride(1),
            l_scale.stride(0), l_scale.stride(1),
            w_fp4_t.stride(0), w_fp4_t.stride(1),
            w_scale.stride(0), w_scale.stride(1),
            C.stride(0), C.stride(1),
            HAS_BIAS=has_bias,
            RANK=self.rank,
            BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk, GROUP_SIZE_M=gm,
            num_warps=cfg.num_warps, num_stages=cfg.num_stages,
            waves_per_eu=wpe, matrix_instr_nonkdim=32,
        )
        return C

    def get_best_config(self, M, N, K):
        return self._cache.get((M, N, K))


# ---------------------------------------------------------------------------
# Runtime autotuner for _16_8 kernel (FP16 in, FP8 out)
# ---------------------------------------------------------------------------

class AutotunedLoRaQ_16_8:
    """
    Runtime autotuner for ``loraq_fused_q8_scaled_kernel_16_8``.

    Takes fp16 A + per-column channel_scale (K,) applied before in-register
    MXFP8 quantization.  R and L are pre-quantized MXFP8.  W stays FP4.
    Output is FP8 + e8m0 block scales, same as AutotunedLoRaQ.
    """

    def __init__(
        self,
        kernel_fn,
        configs=None,
        rank: int = 64,
        waves_per_eu_values: list = None,
        warmup: int = 10,
        rep: int = 50,
    ):
        self.kernel_fn = kernel_fn
        self.configs = configs or LORAQ_Q8_CONFIGS
        self.rank = rank
        self.waves_per_eu_values = waves_per_eu_values if waves_per_eu_values is not None else [0, 1, 2]
        self.warmup = warmup
        self.rep = rep
        self._cache: dict[tuple[int, int, int], dict] = {}

    def _make_launch_fn(self, cfg, wpe, A, channel_scale, out_channel_scale,
                        r_fp8, r_scale, l_fp8, l_scale,
                        w_fp4_t, w_scale, bias_ptr,
                        c_fp8, c_scale, M, N, K, has_bias):
        bm = cfg.kwargs["BLOCK_M"]
        bn = cfg.kwargs["BLOCK_N"]
        bk = cfg.kwargs["BLOCK_K"]
        gm = cfg.kwargs["GROUP_SIZE_M"]
        nw = cfg.num_warps
        ns = cfg.num_stages
        grid = (triton.cdiv(M, bm) * triton.cdiv(N, bn),)

        def launch():
            self.kernel_fn[grid](
                A, channel_scale,
                r_fp8, r_scale,
                l_fp8, l_scale,
                w_fp4_t, w_scale,
                bias_ptr,
                out_channel_scale,
                c_fp8, c_scale,
                M, N, K,
                A.stride(0), A.stride(1),
                channel_scale.stride(0),
                r_fp8.stride(0), r_fp8.stride(1),
                r_scale.stride(0), r_scale.stride(1),
                l_fp8.stride(0), l_fp8.stride(1),
                l_scale.stride(0), l_scale.stride(1),
                w_fp4_t.stride(0), w_fp4_t.stride(1),
                w_scale.stride(0), w_scale.stride(1),
                c_fp8.stride(0), c_fp8.stride(1),
                c_scale.stride(0), c_scale.stride(1),
                out_channel_scale.stride(0),
                HAS_BIAS=has_bias,
                RANK=self.rank,
                BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk, GROUP_SIZE_M=gm,
                num_warps=nw, num_stages=ns,
                waves_per_eu=wpe, matrix_instr_nonkdim=32,
            )
        return launch

    def _tune(self, A, channel_scale, out_channel_scale,
              r_fp8, r_scale, l_fp8, l_scale,
              w_fp4_t, w_scale, M, N, K, bias, has_bias):
        bias_ptr = bias if has_bias else A
        c_fp8   = torch.empty((M, N),       dtype=torch.uint8, device=A.device)
        c_scale = torch.empty((M, N // 32), dtype=torch.uint8, device=A.device)

        best_ms, best_cfg, best_wpe = float("inf"), None, 0

        for cfg in self.configs:
            if cfg.kwargs["BLOCK_K"] > K:
                continue
            for wpe in self.waves_per_eu_values:
                try:
                    fn = self._make_launch_fn(
                        cfg, wpe, A, channel_scale, out_channel_scale,
                        r_fp8, r_scale, l_fp8, l_scale,
                        w_fp4_t, w_scale, bias_ptr,
                        c_fp8, c_scale, M, N, K, has_bias,
                    )
                    ms = tt.do_bench(fn, warmup=self.warmup, rep=self.rep)
                    if ms < best_ms:
                        best_ms, best_cfg, best_wpe = ms, cfg, wpe
                except Exception:
                    continue

        if best_cfg is None:
            best_cfg = triton.Config(
                {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_SIZE_M": 8},
                num_warps=8, num_stages=2,
            )
        return {"config": best_cfg, "waves_per_eu": best_wpe, "time_ms": best_ms}

    def __call__(self, A, channel_scale, out_channel_scale,
                 r_fp8, r_scale, l_fp8, l_scale,
                 w_fp4_t, w_scale, M, N, K, bias=None):
        key = (M, N, K)
        has_bias = bias is not None
        bias_ptr = bias if has_bias else A

        if key not in self._cache:
            self._cache[key] = self._tune(
                A, channel_scale, out_channel_scale,
                r_fp8, r_scale, l_fp8, l_scale,
                w_fp4_t, w_scale, M, N, K, bias, has_bias,
            )

        cfg = self._cache[key]["config"]
        wpe = self._cache[key]["waves_per_eu"]
        bm  = cfg.kwargs["BLOCK_M"]
        bn  = cfg.kwargs["BLOCK_N"]
        bk  = cfg.kwargs["BLOCK_K"]
        gm  = cfg.kwargs["GROUP_SIZE_M"]

        c_fp8   = torch.empty((M, N),       dtype=torch.uint8, device=A.device)
        c_scale = torch.empty((M, N // 32), dtype=torch.uint8, device=A.device)
        grid    = (triton.cdiv(M, bm) * triton.cdiv(N, bn),)

        self.kernel_fn[grid](
            A, channel_scale,
            r_fp8, r_scale,
            l_fp8, l_scale,
            w_fp4_t, w_scale,
            bias_ptr,
            out_channel_scale,
            c_fp8, c_scale,
            M, N, K,
            A.stride(0), A.stride(1),
            channel_scale.stride(0),
            r_fp8.stride(0), r_fp8.stride(1),
            r_scale.stride(0), r_scale.stride(1),
            l_fp8.stride(0), l_fp8.stride(1),
            l_scale.stride(0), l_scale.stride(1),
            w_fp4_t.stride(0), w_fp4_t.stride(1),
            w_scale.stride(0), w_scale.stride(1),
            c_fp8.stride(0), c_fp8.stride(1),
            c_scale.stride(0), c_scale.stride(1),
            out_channel_scale.stride(0),
            HAS_BIAS=has_bias,
            RANK=self.rank,
            BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk, GROUP_SIZE_M=gm,
            num_warps=cfg.num_warps, num_stages=cfg.num_stages,
            waves_per_eu=wpe, matrix_instr_nonkdim=32,
        )

        c_fp8 = c_fp8.view(torch.float8_e4m3fn)
        return c_fp8, c_scale

    def get_best_config(self, M, N, K):
        return self._cache.get((M, N, K))


# ---------------------------------------------------------------------------
# Runtime autotuner for _16_16 kernel (FP16 in, FP16 out)
# ---------------------------------------------------------------------------

class AutotunedLoRaQ_16_16:
    """
    Runtime autotuner for ``loraq_fused_q8_scaled_kernel_16_16``.

    Takes fp16 A + per-column input channel_scale (K,); R and L are
    pre-quantized MXFP8; W stays FP4.  Output is a single fp16 tensor C
    (no c_scale) — activation tensors between stacked layers stay FP16.
    """

    def __init__(
        self,
        kernel_fn,
        configs=None,
        rank: int = 64,
        waves_per_eu_values: list = None,
        warmup: int = 10,
        rep: int = 50,
    ):
        self.kernel_fn = kernel_fn
        self.configs = configs or LORAQ_Q8_CONFIGS
        self.rank = rank
        self.waves_per_eu_values = waves_per_eu_values if waves_per_eu_values is not None else [0, 1, 2]
        self.warmup = warmup
        self.rep = rep
        self._cache: dict[tuple[int, int, int], dict] = {}

    def _make_launch_fn(self, cfg, wpe, A, channel_scale,
                        r_fp8, r_scale, l_fp8, l_scale,
                        w_fp4_t, w_scale, bias_ptr,
                        C, M, N, K, has_bias):
        bm = cfg.kwargs["BLOCK_M"]
        bn = cfg.kwargs["BLOCK_N"]
        bk = cfg.kwargs["BLOCK_K"]
        gm = cfg.kwargs["GROUP_SIZE_M"]
        nw = cfg.num_warps
        ns = cfg.num_stages
        grid = (triton.cdiv(M, bm) * triton.cdiv(N, bn),)

        def launch():
            self.kernel_fn[grid](
                A, channel_scale,
                r_fp8, r_scale,
                l_fp8, l_scale,
                w_fp4_t, w_scale,
                bias_ptr,
                C,
                M, N, K,
                A.stride(0), A.stride(1),
                channel_scale.stride(0),
                r_fp8.stride(0), r_fp8.stride(1),
                r_scale.stride(0), r_scale.stride(1),
                l_fp8.stride(0), l_fp8.stride(1),
                l_scale.stride(0), l_scale.stride(1),
                w_fp4_t.stride(0), w_fp4_t.stride(1),
                w_scale.stride(0), w_scale.stride(1),
                C.stride(0), C.stride(1),
                HAS_BIAS=has_bias,
                RANK=self.rank,
                BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk, GROUP_SIZE_M=gm,
                num_warps=nw, num_stages=ns,
                waves_per_eu=wpe, matrix_instr_nonkdim=32,
            )
        return launch

    def _tune(self, A, channel_scale, r_fp8, r_scale, l_fp8, l_scale,
              w_fp4_t, w_scale, M, N, K, bias, has_bias):
        bias_ptr = bias if has_bias else A
        C = torch.empty((M, N), dtype=torch.float16, device=A.device)

        best_ms, best_cfg, best_wpe = float("inf"), None, 0

        for cfg in self.configs:
            if cfg.kwargs["BLOCK_K"] > K:
                continue
            for wpe in self.waves_per_eu_values:
                try:
                    fn = self._make_launch_fn(
                        cfg, wpe, A, channel_scale,
                        r_fp8, r_scale, l_fp8, l_scale,
                        w_fp4_t, w_scale, bias_ptr,
                        C, M, N, K, has_bias,
                    )
                    ms = tt.do_bench(fn, warmup=self.warmup, rep=self.rep)
                    if ms < best_ms:
                        best_ms, best_cfg, best_wpe = ms, cfg, wpe
                except Exception:
                    continue

        if best_cfg is None:
            best_cfg = triton.Config(
                {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_SIZE_M": 8},
                num_warps=8, num_stages=2,
            )
        return {"config": best_cfg, "waves_per_eu": best_wpe, "time_ms": best_ms}

    def __call__(self, A, channel_scale, r_fp8, r_scale, l_fp8, l_scale,
                 w_fp4_t, w_scale, M, N, K, bias=None):
        key = (M, N, K)
        has_bias = bias is not None
        bias_ptr = bias if has_bias else A

        if key not in self._cache:
            self._cache[key] = self._tune(
                A, channel_scale, r_fp8, r_scale, l_fp8, l_scale,
                w_fp4_t, w_scale, M, N, K, bias, has_bias,
            )

        cfg = self._cache[key]["config"]
        wpe = self._cache[key]["waves_per_eu"]
        bm  = cfg.kwargs["BLOCK_M"]
        bn  = cfg.kwargs["BLOCK_N"]
        bk  = cfg.kwargs["BLOCK_K"]
        gm  = cfg.kwargs["GROUP_SIZE_M"]

        C    = torch.empty((M, N), dtype=torch.float16, device=A.device)
        grid = (triton.cdiv(M, bm) * triton.cdiv(N, bn),)

        self.kernel_fn[grid](
            A, channel_scale,
            r_fp8, r_scale,
            l_fp8, l_scale,
            w_fp4_t, w_scale,
            bias_ptr,
            C,
            M, N, K,
            A.stride(0), A.stride(1),
            channel_scale.stride(0),
            r_fp8.stride(0), r_fp8.stride(1),
            r_scale.stride(0), r_scale.stride(1),
            l_fp8.stride(0), l_fp8.stride(1),
            l_scale.stride(0), l_scale.stride(1),
            w_fp4_t.stride(0), w_fp4_t.stride(1),
            w_scale.stride(0), w_scale.stride(1),
            C.stride(0), C.stride(1),
            HAS_BIAS=has_bias,
            RANK=self.rank,
            BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk, GROUP_SIZE_M=gm,
            num_warps=cfg.num_warps, num_stages=cfg.num_stages,
            waves_per_eu=wpe, matrix_instr_nonkdim=32,
        )
        return C

    def get_best_config(self, M, N, K):
        return self._cache.get((M, N, K))


# ---------------------------------------------------------------------------
# Runtime autotuner for loraq_fused_q4_kernel (K13 — MXFP4 in/out)
# ---------------------------------------------------------------------------

class AutotunedLoRaQ4:
    """
    Runtime autotuner for ``loraq_fused_q4_kernel`` (K13).

    Activation (A) is MXFP8 (float8_e4m3fn + e8m0).
    R, L, and W are MXFP4 packed e2m1 + e8m0 scales.
    Output is (c_fp8, c_scale) where c_fp8 is (M, N) uint8 (float8_e4m3fn).
    """

    def __init__(
        self,
        kernel_fn,
        configs=None,
        rank: int = 128,
        waves_per_eu_values: list = None,
        warmup: int = 10,
        rep: int = 50,
    ):
        self.kernel_fn = kernel_fn
        self.configs = configs or LORAQ_Q8_CONFIGS
        self.rank = rank
        self.waves_per_eu_values = waves_per_eu_values if waves_per_eu_values is not None else [0, 1, 2]
        self.warmup = warmup
        self.rep = rep
        self._cache: dict[tuple[int, int, int], dict] = {}

    def _make_launch_fn(self, cfg, wpe, a_fp8, a_scale, r_fp4, r_scale,
                        l_fp4, l_scale, w_fp4_t, w_scale, bias_ptr,
                        channel_scale, c_fp8, c_scale, M, N, K, has_bias):
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
                r_fp4, r_scale,
                l_fp4, l_scale,
                w_fp4_t, w_scale,
                bias_ptr,
                channel_scale,
                c_fp8, c_scale,
                M, N, K,
                a_fp8.stride(0), a_fp8.stride(1),
                a_scale.stride(0), a_scale.stride(1),
                r_fp4.stride(0), r_fp4.stride(1),
                r_scale.stride(0), r_scale.stride(1),
                l_fp4.stride(0), l_fp4.stride(1),
                l_scale.stride(0), l_scale.stride(1),
                w_fp4_t.stride(0), w_fp4_t.stride(1),
                w_scale.stride(0), w_scale.stride(1),
                c_fp8.stride(0), c_fp8.stride(1),
                c_scale.stride(0), c_scale.stride(1),
                channel_scale.stride(0),
                HAS_BIAS=has_bias,
                RANK=self.rank,
                BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk, GROUP_SIZE_M=gm,
                num_warps=nw, num_stages=ns,
                waves_per_eu=wpe, matrix_instr_nonkdim=32,
            )
        return launch

    def _tune(self, a_fp8, a_scale, r_fp4, r_scale, l_fp4, l_scale,
              w_fp4_t, w_scale, M, N, K, bias, has_bias, channel_scale):
        bias_ptr = bias if has_bias else a_fp8
        c_fp8   = torch.empty((M, N), dtype=torch.uint8, device=a_fp8.device)
        c_scale = torch.empty((M, N // 32), dtype=torch.uint8, device=a_fp8.device)

        best_ms, best_cfg, best_wpe = float("inf"), None, 0

        for cfg in self.configs:
            if cfg.kwargs["BLOCK_K"] > K:
                continue
            for wpe in self.waves_per_eu_values:
                try:
                    fn = self._make_launch_fn(
                        cfg, wpe, a_fp8, a_scale, r_fp4, r_scale,
                        l_fp4, l_scale, w_fp4_t, w_scale,
                        bias_ptr, channel_scale, c_fp8, c_scale, M, N, K, has_bias,
                    )
                    ms = tt.do_bench(fn, warmup=self.warmup, rep=self.rep)
                    if ms < best_ms:
                        best_ms, best_cfg, best_wpe = ms, cfg, wpe
                except Exception:
                    continue

        if best_cfg is None:
            best_cfg = triton.Config(
                {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_SIZE_M": 8},
                num_warps=8, num_stages=2,
            )
        return {"config": best_cfg, "waves_per_eu": best_wpe, "time_ms": best_ms}

    def __call__(self, a_fp8, a_scale, r_fp4, r_scale, l_fp4, l_scale,
                 w_fp4_t, w_scale, M, N, K, bias=None, channel_scale=None):
        key = (M, N, K)
        has_bias = bias is not None
        bias_ptr = bias if has_bias else a_fp8

        if channel_scale is None:
            channel_scale = torch.ones(N, dtype=torch.float16, device=a_fp8.device)

        if key not in self._cache:
            self._cache[key] = self._tune(
                a_fp8, a_scale, r_fp4, r_scale, l_fp4, l_scale,
                w_fp4_t, w_scale, M, N, K, bias, has_bias, channel_scale,
            )

        cfg = self._cache[key]["config"]
        wpe = self._cache[key]["waves_per_eu"]
        bm  = cfg.kwargs["BLOCK_M"]
        bn  = cfg.kwargs["BLOCK_N"]
        bk  = cfg.kwargs["BLOCK_K"]
        gm  = cfg.kwargs["GROUP_SIZE_M"]

        c_fp8   = torch.empty((M, N), dtype=torch.uint8, device=a_fp8.device)
        c_scale = torch.empty((M, N // 32), dtype=torch.uint8, device=a_fp8.device)
        grid    = (triton.cdiv(M, bm) * triton.cdiv(N, bn),)

        self.kernel_fn[grid](
            a_fp8, a_scale,
            r_fp4, r_scale,
            l_fp4, l_scale,
            w_fp4_t, w_scale,
            bias_ptr,
            channel_scale,
            c_fp8, c_scale,
            M, N, K,
            a_fp8.stride(0), a_fp8.stride(1),
            a_scale.stride(0), a_scale.stride(1),
            r_fp4.stride(0), r_fp4.stride(1),
            r_scale.stride(0), r_scale.stride(1),
            l_fp4.stride(0), l_fp4.stride(1),
            l_scale.stride(0), l_scale.stride(1),
            w_fp4_t.stride(0), w_fp4_t.stride(1),
            w_scale.stride(0), w_scale.stride(1),
            c_fp8.stride(0), c_fp8.stride(1),
            c_scale.stride(0), c_scale.stride(1),
            channel_scale.stride(0),
            HAS_BIAS=has_bias,
            RANK=self.rank,
            BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk, GROUP_SIZE_M=gm,
            num_warps=cfg.num_warps, num_stages=cfg.num_stages,
            waves_per_eu=wpe, matrix_instr_nonkdim=32,
        )

        return c_fp8, c_scale

    def get_best_config(self, M, N, K):
        return self._cache.get((M, N, K))


# ---------------------------------------------------------------------------
# Runtime autotuner for loraq_fused_q4_kernel_4_16 (K14 — MXFP8 A/R in, FP16 out)
# ---------------------------------------------------------------------------

class AutotunedLoRaQ4_16:
    """
    Runtime autotuner for ``loraq_fused_q4_kernel_4_16`` (K14).

    Activation (A) is MXFP8 (float8_e4m3fn + e8m0).
    R, L, and W are MXFP4.  Output is a single fp16 tensor C
    (no c_scale, no channel_scale).
    """

    def __init__(
        self,
        kernel_fn,
        configs=None,
        rank: int = 128,
        waves_per_eu_values: list = None,
        warmup: int = 10,
        rep: int = 50,
    ):
        self.kernel_fn = kernel_fn
        self.configs = configs or LORAQ_Q8_CONFIGS
        self.rank = rank
        self.waves_per_eu_values = waves_per_eu_values if waves_per_eu_values is not None else [0, 1, 2]
        self.warmup = warmup
        self.rep = rep
        self._cache: dict[tuple[int, int, int], dict] = {}

    def _make_launch_fn(self, cfg, wpe, a_fp8, a_scale, r_fp4, r_scale,
                        l_fp4, l_scale, w_fp4_t, w_scale, bias_ptr,
                        C, M, N, K, has_bias):
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
                r_fp4, r_scale,
                l_fp4, l_scale,
                w_fp4_t, w_scale,
                bias_ptr,
                C,
                M, N, K,
                a_fp8.stride(0), a_fp8.stride(1),
                a_scale.stride(0), a_scale.stride(1),
                r_fp4.stride(0), r_fp4.stride(1),
                r_scale.stride(0), r_scale.stride(1),
                l_fp4.stride(0), l_fp4.stride(1),
                l_scale.stride(0), l_scale.stride(1),
                w_fp4_t.stride(0), w_fp4_t.stride(1),
                w_scale.stride(0), w_scale.stride(1),
                C.stride(0), C.stride(1),
                HAS_BIAS=has_bias,
                RANK=self.rank,
                BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk, GROUP_SIZE_M=gm,
                num_warps=nw, num_stages=ns,
                waves_per_eu=wpe, matrix_instr_nonkdim=32,
            )
        return launch

    def _tune(self, a_fp8, a_scale, r_fp4, r_scale, l_fp4, l_scale,
              w_fp4_t, w_scale, M, N, K, bias, has_bias):
        bias_ptr = bias if has_bias else a_fp8
        C = torch.empty((M, N), dtype=torch.float16, device=a_fp8.device)

        best_ms, best_cfg, best_wpe = float("inf"), None, 0

        for cfg in self.configs:
            if cfg.kwargs["BLOCK_K"] > K:
                continue
            for wpe in self.waves_per_eu_values:
                try:
                    fn = self._make_launch_fn(
                        cfg, wpe, a_fp8, a_scale, r_fp4, r_scale,
                        l_fp4, l_scale, w_fp4_t, w_scale,
                        bias_ptr, C, M, N, K, has_bias,
                    )
                    ms = tt.do_bench(fn, warmup=self.warmup, rep=self.rep)
                    if ms < best_ms:
                        best_ms, best_cfg, best_wpe = ms, cfg, wpe
                except Exception:
                    continue

        if best_cfg is None:
            best_cfg = triton.Config(
                {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_SIZE_M": 8},
                num_warps=8, num_stages=2,
            )
        return {"config": best_cfg, "waves_per_eu": best_wpe, "time_ms": best_ms}

    def __call__(self, a_fp8, a_scale, r_fp4, r_scale, l_fp4, l_scale,
                 w_fp4_t, w_scale, M, N, K, bias=None):
        key = (M, N, K)
        has_bias = bias is not None
        bias_ptr = bias if has_bias else a_fp8

        if key not in self._cache:
            self._cache[key] = self._tune(
                a_fp8, a_scale, r_fp4, r_scale, l_fp4, l_scale,
                w_fp4_t, w_scale, M, N, K, bias, has_bias,
            )

        cfg = self._cache[key]["config"]
        wpe = self._cache[key]["waves_per_eu"]
        bm  = cfg.kwargs["BLOCK_M"]
        bn  = cfg.kwargs["BLOCK_N"]
        bk  = cfg.kwargs["BLOCK_K"]
        gm  = cfg.kwargs["GROUP_SIZE_M"]

        C    = torch.empty((M, N), dtype=torch.float16, device=a_fp8.device)
        grid = (triton.cdiv(M, bm) * triton.cdiv(N, bn),)

        self.kernel_fn[grid](
            a_fp8, a_scale,
            r_fp4, r_scale,
            l_fp4, l_scale,
            w_fp4_t, w_scale,
            bias_ptr,
            C,
            M, N, K,
            a_fp8.stride(0), a_fp8.stride(1),
            a_scale.stride(0), a_scale.stride(1),
            r_fp4.stride(0), r_fp4.stride(1),
            r_scale.stride(0), r_scale.stride(1),
            l_fp4.stride(0), l_fp4.stride(1),
            l_scale.stride(0), l_scale.stride(1),
            w_fp4_t.stride(0), w_fp4_t.stride(1),
            w_scale.stride(0), w_scale.stride(1),
            C.stride(0), C.stride(1),
            HAS_BIAS=has_bias,
            RANK=self.rank,
            BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk, GROUP_SIZE_M=gm,
            num_warps=cfg.num_warps, num_stages=cfg.num_stages,
            waves_per_eu=wpe, matrix_instr_nonkdim=32,
        )
        return C

    def get_best_config(self, M, N, K):
        return self._cache.get((M, N, K))


# ---------------------------------------------------------------------------
# Runtime autotuner for K13_2 (RL-restructured, FP16 L, MXFP8 output)
# ---------------------------------------------------------------------------

class AutotunedLoRaQ4_2:
    """
    Runtime autotuner for ``loraq_fused_q4_kernel_2`` (K13_2).

    Activation (A) is MXFP8.  R and W are MXFP4.  L is FP16 (no scale).
    Output is (c_fp8, c_scale) where c_fp8 is (M, N) uint8 (float8_e4m3fn).
    """

    def __init__(self, kernel_fn, configs=None, rank=128,
                 waves_per_eu_values=None, warmup=10, rep=50):
        self.kernel_fn = kernel_fn
        self.configs = configs or LORAQ_Q8_CONFIGS
        self.rank = rank
        self.waves_per_eu_values = waves_per_eu_values or [0, 1, 2]
        self.warmup = warmup
        self.rep = rep
        self._cache: dict[tuple[int, int, int], dict] = {}

    def _make_launch_fn(self, cfg, wpe, a_fp8, a_scale, r_fp4, r_scale,
                        l_fp16, w_fp4_t, w_scale, bias_ptr,
                        channel_scale, c_fp8, c_scale, M, N, K, has_bias):
        bm, bn, bk, gm = cfg.kwargs["BLOCK_M"], cfg.kwargs["BLOCK_N"], cfg.kwargs["BLOCK_K"], cfg.kwargs["GROUP_SIZE_M"]
        grid = (triton.cdiv(M, bm) * triton.cdiv(N, bn),)

        def launch():
            self.kernel_fn[grid](
                a_fp8, a_scale, r_fp4, r_scale,
                l_fp16,
                w_fp4_t, w_scale, bias_ptr, channel_scale,
                c_fp8, c_scale,
                M, N, K,
                a_fp8.stride(0), a_fp8.stride(1),
                a_scale.stride(0), a_scale.stride(1),
                r_fp4.stride(0), r_fp4.stride(1),
                r_scale.stride(0), r_scale.stride(1),
                l_fp16.stride(0), l_fp16.stride(1),
                w_fp4_t.stride(0), w_fp4_t.stride(1),
                w_scale.stride(0), w_scale.stride(1),
                c_fp8.stride(0), c_fp8.stride(1),
                c_scale.stride(0), c_scale.stride(1),
                channel_scale.stride(0),
                HAS_BIAS=has_bias, RANK=self.rank,
                BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk, GROUP_SIZE_M=gm,
                num_warps=cfg.num_warps, num_stages=cfg.num_stages,
                waves_per_eu=wpe, matrix_instr_nonkdim=32,
            )
        return launch

    def _tune(self, a_fp8, a_scale, r_fp4, r_scale, l_fp16,
              w_fp4_t, w_scale, M, N, K, bias, has_bias, channel_scale):
        bias_ptr = bias if has_bias else a_fp8
        c_fp8   = torch.empty((M, N), dtype=torch.uint8, device=a_fp8.device)
        c_scale = torch.empty((M, N // 32), dtype=torch.uint8, device=a_fp8.device)
        best_ms, best_cfg, best_wpe = float("inf"), None, 0
        for cfg in self.configs:
            if cfg.kwargs["BLOCK_K"] > K:
                continue
            for wpe in self.waves_per_eu_values:
                try:
                    fn = self._make_launch_fn(
                        cfg, wpe, a_fp8, a_scale, r_fp4, r_scale,
                        l_fp16, w_fp4_t, w_scale,
                        bias_ptr, channel_scale, c_fp8, c_scale, M, N, K, has_bias,
                    )
                    ms = tt.do_bench(fn, warmup=self.warmup, rep=self.rep)
                    if ms < best_ms:
                        best_ms, best_cfg, best_wpe = ms, cfg, wpe
                except Exception:
                    continue
        if best_cfg is None:
            best_cfg = triton.Config(
                {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_SIZE_M": 8},
                num_warps=8, num_stages=2,
            )
        return {"config": best_cfg, "waves_per_eu": best_wpe, "time_ms": best_ms}

    def __call__(self, a_fp8, a_scale, r_fp4, r_scale, l_fp16,
                 w_fp4_t, w_scale, M, N, K, bias=None, channel_scale=None):
        key = (M, N, K)
        has_bias = bias is not None
        bias_ptr = bias if has_bias else a_fp8
        if channel_scale is None:
            channel_scale = torch.ones(N, dtype=torch.float16, device=a_fp8.device)
        if key not in self._cache:
            self._cache[key] = self._tune(
                a_fp8, a_scale, r_fp4, r_scale, l_fp16,
                w_fp4_t, w_scale, M, N, K, bias, has_bias, channel_scale,
            )
        cfg = self._cache[key]["config"]
        wpe = self._cache[key]["waves_per_eu"]
        bm, bn, bk, gm = cfg.kwargs["BLOCK_M"], cfg.kwargs["BLOCK_N"], cfg.kwargs["BLOCK_K"], cfg.kwargs["GROUP_SIZE_M"]
        c_fp8   = torch.empty((M, N), dtype=torch.uint8, device=a_fp8.device)
        c_scale = torch.empty((M, N // 32), dtype=torch.uint8, device=a_fp8.device)
        grid    = (triton.cdiv(M, bm) * triton.cdiv(N, bn),)
        self.kernel_fn[grid](
            a_fp8, a_scale, r_fp4, r_scale,
            l_fp16,
            w_fp4_t, w_scale, bias_ptr, channel_scale,
            c_fp8, c_scale,
            M, N, K,
            a_fp8.stride(0), a_fp8.stride(1),
            a_scale.stride(0), a_scale.stride(1),
            r_fp4.stride(0), r_fp4.stride(1),
            r_scale.stride(0), r_scale.stride(1),
            l_fp16.stride(0), l_fp16.stride(1),
            w_fp4_t.stride(0), w_fp4_t.stride(1),
            w_scale.stride(0), w_scale.stride(1),
            c_fp8.stride(0), c_fp8.stride(1),
            c_scale.stride(0), c_scale.stride(1),
            channel_scale.stride(0),
            HAS_BIAS=has_bias, RANK=self.rank,
            BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk, GROUP_SIZE_M=gm,
            num_warps=cfg.num_warps, num_stages=cfg.num_stages,
            waves_per_eu=wpe, matrix_instr_nonkdim=32,
        )
        return c_fp8, c_scale


# ---------------------------------------------------------------------------
# Runtime autotuner for K14_2 (RL-restructured, FP16 L, FP16 output)
# ---------------------------------------------------------------------------

class AutotunedLoRaQ4_16_2:
    """
    Runtime autotuner for ``loraq_fused_q4_kernel_4_16_2`` (K14_2).

    Activation (A) is MXFP8.  R and W are MXFP4.  L is FP16 (no scale).
    Output is a single fp16 tensor C.
    """

    def __init__(self, kernel_fn, configs=None, rank=128,
                 waves_per_eu_values=None, warmup=10, rep=50):
        self.kernel_fn = kernel_fn
        self.configs = configs or LORAQ_Q8_CONFIGS
        self.rank = rank
        self.waves_per_eu_values = waves_per_eu_values or [0, 1, 2]
        self.warmup = warmup
        self.rep = rep
        self._cache: dict[tuple[int, int, int], dict] = {}

    def _make_launch_fn(self, cfg, wpe, a_fp8, a_scale, r_fp4, r_scale,
                        l_fp16, w_fp4_t, w_scale, bias_ptr,
                        C, M, N, K, has_bias):
        bm, bn, bk, gm = cfg.kwargs["BLOCK_M"], cfg.kwargs["BLOCK_N"], cfg.kwargs["BLOCK_K"], cfg.kwargs["GROUP_SIZE_M"]
        grid = (triton.cdiv(M, bm) * triton.cdiv(N, bn),)

        def launch():
            self.kernel_fn[grid](
                a_fp8, a_scale, r_fp4, r_scale,
                l_fp16,
                w_fp4_t, w_scale, bias_ptr,
                C,
                M, N, K,
                a_fp8.stride(0), a_fp8.stride(1),
                a_scale.stride(0), a_scale.stride(1),
                r_fp4.stride(0), r_fp4.stride(1),
                r_scale.stride(0), r_scale.stride(1),
                l_fp16.stride(0), l_fp16.stride(1),
                w_fp4_t.stride(0), w_fp4_t.stride(1),
                w_scale.stride(0), w_scale.stride(1),
                C.stride(0), C.stride(1),
                HAS_BIAS=has_bias, RANK=self.rank,
                BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk, GROUP_SIZE_M=gm,
                num_warps=cfg.num_warps, num_stages=cfg.num_stages,
                waves_per_eu=wpe, matrix_instr_nonkdim=32,
            )
        return launch

    def _tune(self, a_fp8, a_scale, r_fp4, r_scale, l_fp16,
              w_fp4_t, w_scale, M, N, K, bias, has_bias):
        bias_ptr = bias if has_bias else a_fp8
        C = torch.empty((M, N), dtype=torch.float16, device=a_fp8.device)
        best_ms, best_cfg, best_wpe = float("inf"), None, 0
        for cfg in self.configs:
            if cfg.kwargs["BLOCK_K"] > K:
                continue
            for wpe in self.waves_per_eu_values:
                try:
                    fn = self._make_launch_fn(
                        cfg, wpe, a_fp8, a_scale, r_fp4, r_scale,
                        l_fp16, w_fp4_t, w_scale,
                        bias_ptr, C, M, N, K, has_bias,
                    )
                    ms = tt.do_bench(fn, warmup=self.warmup, rep=self.rep)
                    if ms < best_ms:
                        best_ms, best_cfg, best_wpe = ms, cfg, wpe
                except Exception:
                    continue
        if best_cfg is None:
            best_cfg = triton.Config(
                {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_SIZE_M": 8},
                num_warps=8, num_stages=2,
            )
        return {"config": best_cfg, "waves_per_eu": best_wpe, "time_ms": best_ms}

    def __call__(self, a_fp8, a_scale, r_fp4, r_scale, l_fp16,
                 w_fp4_t, w_scale, M, N, K, bias=None):
        key = (M, N, K)
        has_bias = bias is not None
        bias_ptr = bias if has_bias else a_fp8
        if key not in self._cache:
            self._cache[key] = self._tune(
                a_fp8, a_scale, r_fp4, r_scale, l_fp16,
                w_fp4_t, w_scale, M, N, K, bias, has_bias,
            )
        cfg = self._cache[key]["config"]
        wpe = self._cache[key]["waves_per_eu"]
        bm, bn, bk, gm = cfg.kwargs["BLOCK_M"], cfg.kwargs["BLOCK_N"], cfg.kwargs["BLOCK_K"], cfg.kwargs["GROUP_SIZE_M"]
        C    = torch.empty((M, N), dtype=torch.float16, device=a_fp8.device)
        grid = (triton.cdiv(M, bm) * triton.cdiv(N, bn),)
        self.kernel_fn[grid](
            a_fp8, a_scale, r_fp4, r_scale,
            l_fp16,
            w_fp4_t, w_scale, bias_ptr,
            C,
            M, N, K,
            a_fp8.stride(0), a_fp8.stride(1),
            a_scale.stride(0), a_scale.stride(1),
            r_fp4.stride(0), r_fp4.stride(1),
            r_scale.stride(0), r_scale.stride(1),
            l_fp16.stride(0), l_fp16.stride(1),
            w_fp4_t.stride(0), w_fp4_t.stride(1),
            w_scale.stride(0), w_scale.stride(1),
            C.stride(0), C.stride(1),
            HAS_BIAS=has_bias, RANK=self.rank,
            BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk, GROUP_SIZE_M=gm,
            num_warps=cfg.num_warps, num_stages=cfg.num_stages,
            waves_per_eu=wpe, matrix_instr_nonkdim=32,
        )
        return C


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
