"""
Triton-accelerated linear layers.

Drop-in replacements for torch.nn.Linear that use custom Triton GEMM kernels
for the forward-pass matrix multiplication.

Classes
-------
TritonLinear      : fp16 / bf16 linear layer  (standard precision)
TritonLinearFP4   : MXFP4 e2m1 linear layer   (4-bit weight + online input quant)
TritonLinearLoRA  : LoRA+Q layer  (low-rank correction + MXFP4 weight)
TritonLinearLoRaQ    : Fused FP8/FP4 LoRA+Q layer, Phase 2 = tl.dot fp16
TritonLinearLoRaQFP8 : Fused FP8/FP4 LoRA+Q layer, Phase 2 = dot_scaled fp8
TritonLinearLoRaQ3   : LoRaQ.3 — fp16 in (fused quant), fp16 out
TritonLinearLoRaQ4   : LoRaQ.4 — split K_proj + K_main (lower register pressure)
"""

import torch
import torch.nn as nn
import triton

from loraq.kernels import (
    matmul_kernel,
    matmul_fp4_kernel,
    loraq_project_and_quant_kernel,
    loraq_dual_gemm_kernel,
    loraq_fused_q8_kernel,
    loraq_fused_q8_scaled_kernel
)
from loraq.quant import (
    dynamic_mxfp4_quant,
    dynamic_mxfp8_quant,
    mxfp4_to_f32,
    mxfp8_to_f32,
    e8m0_to_f32,
)
from loraq.autotune_configs import (
    AutotunedLoRaQ,
    AutotunedLoRaQProj,
    AutotunedLoRaQMain,
    AutotunedDualGEMM,
    AutotunedProjectAndQuant,
    LORAQ_Q8_CONFIGS,
)


# ---------------------------------------------------------------------------
# fp16 / bf16 matmul wrapper
# ---------------------------------------------------------------------------

# MI350 has 304 CUs.  We target enough concurrent tiles to keep most CUs
# busy.  The table below is calibrated so ceil(M/BLOCK_M)*ceil(N/BLOCK_N)
# stays ≥ ~128 while keeping each tile large enough for efficient MFMA.
#
# Key reference: hipBLAS selects MT32×128×128 (BLOCK_M=32) for M=128,
# giving 4×32=128 tiles at N=4096.  Our old fixed BLOCK_M=128 produced
# only 1×32=32 tiles, leaving ~90 % of the 304 CUs idle.
_TILE_CONFIGS = [
    # (max_M, BLOCK_M, BLOCK_N, BLOCK_K, num_warps, num_stages, nonkdim)
    (32,   16,  64,  64, 4, 2, 16),   # M ≤  32
    (64,   32,  64,  64, 4, 2, 32),   # M ≤  64
    (128,  32, 128, 128, 4, 2, 32),   # M ≤ 128  ← 128 tiles @ N=4096 (LDS 2×40KB=80KB)
    (256,  64, 128, 128, 8, 3, 32),   # M ≤ 256  ← 128 tiles, 3-stage (LDS 3×49KB=147KB)
    (None, 128, 128, 128, 8, 2, 32),  # M > 256  ← 256 tiles @ M=1024, 2-stage (LDS 2×64KB=128KB)
]


def _pick_tile_config(M: int, N: int) -> dict:
    """
    Return a kernel launch configuration dict for the given (M, N).

    Adapts BLOCK_M, BLOCK_N, BLOCK_K, num_warps, num_stages and
    matrix_instr_nonkdim to maximise MI350 CU utilisation.
    """
    for max_m, bm, bn, bk, nw, ns, nkd in _TILE_CONFIGS:
        if max_m is None or M <= max_m:
            return dict(
                BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk,
                num_warps=nw, num_stages=ns,
                matrix_instr_nonkdim=nkd,
            )
    return dict(BLOCK_M=128, BLOCK_N=128, BLOCK_K=128, num_warps=8, num_stages=2,
                matrix_instr_nonkdim=32)


def triton_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Compute ``a @ b`` using the basic Triton GEMM kernel.

    Parameters
    ----------
    a : (M, K) tensor
    b : (K, N) tensor

    Returns
    -------
    (M, N) tensor with the same dtype as *a*.
    """
    assert a.shape[1] == b.shape[0], (
        f"Incompatible dimensions: A is {a.shape}, B is {b.shape}"
    )
    assert a.is_cuda and b.is_cuda, "Both tensors must be on CUDA"

    M, K = a.shape
    K, N = b.shape

    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    cfg = _pick_tile_config(M, N)
    grid = (triton.cdiv(M, cfg["BLOCK_M"]) * triton.cdiv(N, cfg["BLOCK_N"]),)

    matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_M=cfg["BLOCK_M"],
        BLOCK_N=cfg["BLOCK_N"],
        BLOCK_K=cfg["BLOCK_K"],
        GROUP_SIZE_M=8,
        num_warps=cfg["num_warps"],
        num_stages=cfg["num_stages"],
        matrix_instr_nonkdim=cfg["matrix_instr_nonkdim"],
    )

    return c


def triton_matmul_nt(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Compute ``a @ b.T`` using the Triton GEMM kernel **without copying b**.

    The transpose is handled by swapping b's strides in the kernel call,
    so ``b`` may be passed in its natural ``(N, K)`` layout (e.g. the
    weight matrix of an ``nn.Linear``) with zero allocation overhead.

    Parameters
    ----------
    a : (M, K) tensor
    b : (N, K) tensor  — will be treated as (K, N) via transposed strides

    Returns
    -------
    (M, N) tensor with the same dtype as *a*.
    """
    assert a.ndim == 2 and b.ndim == 2, "Both inputs must be 2-D"
    assert a.shape[1] == b.shape[1], (
        f"Inner dimension mismatch: A is {a.shape}, B is {b.shape} "
        f"(expected a @ b.T, so A.shape[1] must equal B.shape[1])"
    )
    assert a.is_cuda and b.is_cuda, "Both tensors must be on CUDA"

    M, K = a.shape
    N = b.shape[0]

    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    cfg = _pick_tile_config(M, N)
    grid = (triton.cdiv(M, cfg["BLOCK_M"]) * triton.cdiv(N, cfg["BLOCK_N"]),)

    matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        # Swap strides so kernel reads b as (K, N) without a physical transpose:
        # b[k, n] = b_storage[n * stride(0) + k * stride(1)] with strides swapped
        b.stride(1), b.stride(0),  # stride_bk=b.stride(1), stride_bn=b.stride(0)
        c.stride(0), c.stride(1),
        BLOCK_M=cfg["BLOCK_M"],
        BLOCK_N=cfg["BLOCK_N"],
        BLOCK_K=cfg["BLOCK_K"],
        GROUP_SIZE_M=8,
        num_warps=cfg["num_warps"],
        num_stages=cfg["num_stages"],
        matrix_instr_nonkdim=cfg["matrix_instr_nonkdim"],
    )

    return c


# ---------------------------------------------------------------------------
# MXFP4 matmul wrapper
# ---------------------------------------------------------------------------

def triton_matmul_fp4(
    a_fp4: torch.Tensor,
    a_scale: torch.Tensor,
    b_fp4: torch.Tensor,
    b_scale: torch.Tensor,
    M: int,
    N: int,
    K: int,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """
    Compute an MXFP4 matrix multiplication  ``dequant(A) @ dequant(B)``
    using ``tl.dot_scaled`` inside the Triton kernel.

    Parameters
    ----------
    a_fp4   : (M, K // 2) uint8   – packed e2m1 for the activation
    a_scale : (M, K // 32) uint8  – e8m0 block scales for A
    b_fp4   : (K // 2, N) uint8   – packed e2m1 for the weight (col-major K)
    b_scale : (N, K // 32) uint8  – e8m0 block scales for B  **note row = N**
    M, N, K : logical dimensions (before packing)
    out_dtype : result dtype (bfloat16 recommended)

    Returns
    -------
    c : (M, N) tensor of *out_dtype*
    """
    c = torch.empty((M, N), device=a_fp4.device, dtype=out_dtype)

    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 64  # must be multiple of 64 for FP4
    GROUP_SIZE_M = 8

    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),)

    matmul_fp4_kernel[grid](
        a_fp4, b_fp4, c,
        a_scale, b_scale,
        M, N, K,
        a_fp4.stride(0), a_fp4.stride(1),
        b_fp4.stride(0), b_fp4.stride(1),
        c.stride(0), c.stride(1),
        a_scale.stride(0), a_scale.stride(1),
        b_scale.stride(0), b_scale.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        GROUP_SIZE_M=GROUP_SIZE_M,
    )

    return c


# ---------------------------------------------------------------------------
# TritonLinear  (fp16 / bf16)
# ---------------------------------------------------------------------------

class TritonLinear(nn.Module):
    """
    Linear layer backed by a Triton matmul kernel.

    Functionally equivalent to ``torch.nn.Linear`` but the matrix
    multiplication in the forward pass runs through a custom Triton kernel.

    Parameters
    ----------
    in_features : int
        Size of the input dimension.
    out_features : int
        Size of the output dimension.
    bias : bool
        If ``True``, adds a learnable bias.
    dtype : torch.dtype
        ``torch.float16`` or ``torch.bfloat16``.
    device : str | torch.device
        Target device (default ``"cuda"``).
    """

    SUPPORTED_DTYPES = {torch.float16, torch.bfloat16}

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        dtype: torch.dtype = torch.float16,
        device: str | torch.device = "cuda",
    ):
        super().__init__()

        if dtype not in self.SUPPORTED_DTYPES:
            raise ValueError(
                f"Unsupported dtype {dtype}. Choose from {self.SUPPORTED_DTYPES}"
            )

        self.in_features = in_features
        self.out_features = out_features

        self.register_buffer(
            "weight",
            torch.empty(out_features, in_features, device=device, dtype=dtype),
        )
        if bias:
            self.register_buffer(
                "bias",
                torch.zeros(out_features, device=device, dtype=dtype),
            )
        else:
            self.register_buffer("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``out = x @ W^T + bias``"""
        orig_shape = x.shape
        x_2d = x.reshape(-1, self.in_features)
        # triton_matmul_nt handles the transpose via swapped strides —
        # no .t().contiguous() allocation on every forward call.
        out = triton_matmul_nt(x_2d, self.weight)
        if self.bias is not None:
            out = out + self.bias
        return out.reshape(*orig_shape[:-1], self.out_features)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"bias={self.bias is not None}, "
            f"dtype={self.weight.dtype}"
        )


# ---------------------------------------------------------------------------
# TritonLinearFP4  (MXFP4 e2m1)
# ---------------------------------------------------------------------------

class TritonLinearFP4(nn.Module):
    """
    Linear layer that stores its weight in MXFP4 (e2m1) format and
    performs the forward-pass GEMM via ``tl.dot_scaled``.

    At construction time, the full-precision weight (fp16/bf16) is
    quantized to packed e2m1 uint8 + e8m0 block scales.  During the
    forward pass the activation is quantized on-the-fly before being
    fed into the FP4 GEMM kernel.

    Parameters
    ----------
    in_features  : int     – input dimension (must be divisible by 32)
    out_features : int     – output dimension (must be divisible by 32)
    bias         : bool    – learnable bias
    out_dtype    : torch.dtype – output dtype (default bf16)
    device       : str | torch.device
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        out_dtype: torch.dtype = torch.bfloat16,
        device: str | torch.device = "cuda",
    ):
        super().__init__()

        assert in_features % 32 == 0, (
            f"in_features={in_features} must be divisible by 32"
        )
        assert out_features % 32 == 0, (
            f"out_features={out_features} must be divisible by 32"
        )

        self.in_features = in_features
        self.out_features = out_features
        self.out_dtype = out_dtype

        # Packed weight: (out_features, in_features // 2) uint8
        self.register_buffer(
            "weight_fp4",
            torch.zeros(
                out_features, in_features // 2,
                dtype=torch.uint8, device=device,
            ),
        )
        # Weight scales: (out_features, in_features // 32) uint8  (e8m0)
        self.register_buffer(
            "weight_scale",
            torch.zeros(
                out_features, in_features // 32,
                dtype=torch.uint8, device=device,
            ),
        )

        if bias:
            self.register_buffer(
                "bias",
                torch.zeros(out_features, device=device, dtype=out_dtype),
            )
        else:
            self.register_buffer("bias", None)

    # ----- factory: from an existing full-precision weight ----- #

    @classmethod
    def from_float(
        cls,
        linear: nn.Linear,
        out_dtype: torch.dtype = torch.bfloat16,
    ) -> "TritonLinearFP4":
        """
        Create a ``TritonLinearFP4`` from an existing ``nn.Linear``
        by quantizing its weight to MXFP4.
        """
        device = linear.weight.device
        layer = cls(
            in_features=linear.in_features,
            out_features=linear.out_features,
            bias=linear.bias is not None,
            out_dtype=out_dtype,
            device=device,
        )

        # Quantize weight  (out_features, in_features)
        w_fp4, w_scale = dynamic_mxfp4_quant(
            linear.weight.to(dtype=torch.float16, device=device)
        )
        layer.weight_fp4.copy_(w_fp4)
        layer.weight_scale.copy_(w_scale)

        if linear.bias is not None:
            layer.bias.copy_(linear.bias.to(out_dtype))

        return layer

    @classmethod
    def from_weight(
        cls,
        weight: torch.Tensor,
        bias: torch.Tensor | None = None,
        out_dtype: torch.dtype = torch.bfloat16,
    ) -> "TritonLinearFP4":
        """
        Create from raw weight tensor ``(out_features, in_features)``.
        """
        assert weight.ndim == 2
        out_features, in_features = weight.shape
        device = weight.device

        layer = cls(
            in_features=in_features,
            out_features=out_features,
            bias=bias is not None,
            out_dtype=out_dtype,
            device=device,
        )

        w_fp4, w_scale = dynamic_mxfp4_quant(
            weight.to(dtype=torch.float16, device=device)
        )
        layer.weight_fp4.copy_(w_fp4)
        layer.weight_scale.copy_(w_scale)

        if bias is not None:
            layer.bias.copy_(bias.to(out_dtype))

        return layer

    # ----- forward ----- #

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass:  ``out = dequant(x_fp4) @ dequant(W_fp4)^T + bias``

        The multiplication is computed entirely in FP4 through
        ``tl.dot_scaled``.
        """
        orig_shape = x.shape
        x_2d = x.reshape(-1, self.in_features).contiguous()
        M = x_2d.shape[0]
        K = self.in_features
        N = self.out_features

        # Online quantization of the activation
        if x_2d.dtype not in (torch.float16, torch.bfloat16):
            x_2d = x_2d.to(torch.float16)
        a_fp4, a_scale = dynamic_mxfp4_quant(x_2d)

        # Weight is stored as (N, K//2) and scale as (N, K//32)
        # Kernel expects b_fp4 shape (K//2, N) so we transpose
        b_fp4 = self.weight_fp4.t().contiguous()   # (K//2, N)
        b_scale = self.weight_scale                 # (N, K//32) -- stays

        out = triton_matmul_fp4(
            a_fp4, a_scale,
            b_fp4, b_scale,
            M, N, K,
            out_dtype=self.out_dtype,
        )

        if self.bias is not None:
            out = out + self.bias

        return out.reshape(*orig_shape[:-1], self.out_features)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"bias={self.bias is not None}, "
            f"format=MXFP4_e2m1, "
            f"out_dtype={self.out_dtype}"
        )


# ---------------------------------------------------------------------------
# LoRA+Q wrappers
# ---------------------------------------------------------------------------

def triton_loraq_project_and_quant(
    A: torch.Tensor,
    R: torch.Tensor,
    channel_scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Fused projection + MXFP4 quantization in a single pass over A.

    Parameters
    ----------
    A : (M, K) fp16 activation tensor
    R : (rank, K) fp16 low-rank factor
    channel_scale : (K,) fp16/bf16 — per-column scale applied before quant

    Returns
    -------
    P       : (M, rank) fp16  — projection ``A @ R^T``
    A_fp4   : (M, K // 2) uint8 — packed e2m1 quantized A
    A_scale : (M, K // 32) uint8 — e8m0 block scales for A
    """
    assert A.ndim == 2 and R.ndim == 2
    M, K = A.shape
    rank = R.shape[0]
    assert R.shape[1] == K
    assert K % 32 == 0, f"K={K} must be divisible by 32"

    BLOCK_M = 128
    QUANT_GROUP = 32

    P = torch.empty((M, rank), device=A.device, dtype=A.dtype)
    A_fp4 = torch.empty((M, K // 2), device=A.device, dtype=torch.uint8)
    A_scale = torch.empty((M, K // 32), device=A.device, dtype=torch.uint8)

    grid = (triton.cdiv(M, BLOCK_M),)

    loraq_project_and_quant_kernel[grid](
        A, R, channel_scale, P, A_fp4, A_scale,
        M, K,
        A.stride(0), A.stride(1),
        R.stride(0), R.stride(1),
        P.stride(0), P.stride(1),
        A_fp4.stride(0), A_fp4.stride(1),
        A_scale.stride(0), A_scale.stride(1),
        channel_scale.stride(0),
        RANK=rank,
        BLOCK_M=BLOCK_M,
        QUANT_GROUP=QUANT_GROUP,
    )

    return P, A_fp4, A_scale


def triton_loraq_dual_gemm(
    P: torch.Tensor,
    L: torch.Tensor,
    A_fp4: torch.Tensor,
    A_scale: torch.Tensor,
    W_fp4: torch.Tensor,
    W_scale: torch.Tensor,
    M: int,
    N: int,
    K: int,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """
    Fused dual GEMM:  ``P @ L^T  +  Q(A) @ Q(W)^T``

    Parameters
    ----------
    P       : (M, rank) fp16
    L       : (N, rank) fp16
    A_fp4   : (M, K // 2) uint8  packed e2m1
    A_scale : (M, K // 32) uint8 e8m0
    W_fp4   : (K // 2, N) uint8  packed e2m1 (transposed weight)
    W_scale : (N, K // 32) uint8 e8m0
    M, N, K : logical dimensions

    Returns
    -------
    C : (M, N) tensor of *out_dtype*
    """
    rank = P.shape[1]
    C = torch.empty((M, N), device=P.device, dtype=out_dtype)

    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 64
    GROUP_SIZE_M = 8

    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),)

    loraq_dual_gemm_kernel[grid](
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
        RANK=rank,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        GROUP_SIZE_M=GROUP_SIZE_M,
        matrix_instr_nonkdim=16 if BLOCK_K % 128 == 0 else 32,
    )

    return C


# ---------------------------------------------------------------------------
# TritonLinearLoRA  (LoRA + MXFP4)
# ---------------------------------------------------------------------------

class TritonLinearLoRA(nn.Module):
    """
    Linear layer computing ``A @ R^T @ L^T + Q(A) @ Q(W)^T`` via two
    fused Triton kernels that minimise memory traffic.

    The weight W is stored in MXFP4 (e2m1) format.  L (out_features, rank)
    and R (rank, in_features) are fp16 low-rank correction factors that
    capture the quantization residual ``W - dequant(Q(W))``.

    Parameters
    ----------
    in_features  : int  – input dimension (must be divisible by 32)
    out_features : int  – output dimension (must be divisible by 32)
    rank         : int  – low-rank dimension (default 32, must equal 32)
    bias         : bool – learnable bias
    out_dtype    : torch.dtype – output dtype (default bfloat16)
    device       : str | torch.device
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 32,
        bias: bool = True,
        out_dtype: torch.dtype = torch.bfloat16,
        device: str | torch.device = "cuda",
    ):
        super().__init__()

        assert in_features % 32 == 0, (
            f"in_features={in_features} must be divisible by 32"
        )
        assert out_features % 32 == 0, (
            f"out_features={out_features} must be divisible by 32"
        )
        assert rank == 32, (
            f"rank must be 32 (== QUANT_GROUP) for fused kernel, got {rank}"
        )

        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.out_dtype = out_dtype

        # Quantized weight buffers (not trainable)
        self.register_buffer(
            "weight_fp4",
            torch.zeros(
                out_features, in_features // 2,
                dtype=torch.uint8, device=device,
            ),
        )
        self.register_buffer(
            "weight_scale",
            torch.zeros(
                out_features, in_features // 32,
                dtype=torch.uint8, device=device,
            ),
        )

        # Low-rank correction factors (non-trainable buffers)
        self.register_buffer(
            "L",
            torch.zeros(out_features, rank, device=device, dtype=torch.float16),
        )
        self.register_buffer(
            "R",
            torch.zeros(rank, in_features, device=device, dtype=torch.float16),
        )

        # Channel-wise quantization scale for activation (applied before MXFP4 quant)
        self.register_buffer(
            "channel_scale",
            torch.ones(in_features, device=device, dtype=torch.float16),
        )

        if bias:
            self.register_buffer(
                "bias",
                torch.zeros(out_features, device=device, dtype=out_dtype),
            )
        else:
            self.register_buffer("bias", None)

    # ----- factory from nn.Linear ----- #

    @classmethod
    def from_float(
        cls,
        linear: nn.Linear,
        rank: int = 32,
        out_dtype: torch.dtype = torch.bfloat16,
    ) -> "TritonLinearLoRA":
        """
        Create a ``TritonLinearLoRA`` from an existing ``nn.Linear``.

        1. Quantize weight to MXFP4.
        2. Compute the quantization residual ``W - dequant(Q(W))``.
        3. Initialise L, R via truncated SVD of the residual so that
           ``L @ R`` is the best rank-32 approximation.
        """
        device = linear.weight.device
        W = linear.weight.to(dtype=torch.float16, device=device)

        layer = cls(
            in_features=linear.in_features,
            out_features=linear.out_features,
            rank=rank,
            bias=linear.bias is not None,
            out_dtype=out_dtype,
            device=device,
        )

        # Quantize weight
        w_fp4, w_scale = dynamic_mxfp4_quant(W)
        layer.weight_fp4.copy_(w_fp4)
        layer.weight_scale.copy_(w_scale)

        # Dequantize for residual computation
        w_deq = mxfp4_to_f32(w_fp4)  # (out, in) normalised
        s_f32 = e8m0_to_f32(w_scale)  # (out, in//32)
        s_f32 = s_f32.repeat_interleave(32, dim=-1)  # (out, in)
        w_reconstructed = (w_deq * s_f32).to(torch.float16).to(device)

        # SVD of residual
        residual = (W - w_reconstructed).float()
        U, S, Vh = torch.linalg.svd(residual, full_matrices=False)
        sqrt_S = S[:rank].sqrt()
        L_init = U[:, :rank] * sqrt_S[None, :]      # (out, rank)
        R_init = sqrt_S[:, None] * Vh[:rank, :]      # (rank, in)

        layer.L.copy_(L_init.to(torch.float16))
        layer.R.copy_(R_init.to(torch.float16))

        if linear.bias is not None:
            layer.bias.copy_(linear.bias.to(out_dtype))

        return layer

    # ----- factory from raw weight tensor ----- #

    @classmethod
    def from_weight(
        cls,
        weight: torch.Tensor,
        bias: torch.Tensor | None = None,
        rank: int = 32,
        out_dtype: torch.dtype = torch.bfloat16,
    ) -> "TritonLinearLoRA":
        """
        Create from raw weight tensor ``(out_features, in_features)``.
        """
        assert weight.ndim == 2
        dummy = nn.Linear(
            weight.shape[1], weight.shape[0],
            bias=bias is not None,
            device=weight.device,
            dtype=weight.dtype,
        )
        with torch.no_grad():
            dummy.weight.copy_(weight)
            if bias is not None:
                dummy.bias.copy_(bias)
        return cls.from_float(dummy, rank=rank, out_dtype=out_dtype)

    # ----- autotuners (class-level, shared across instances) ----- #
    _at_pq: AutotunedProjectAndQuant | None = None
    _at_dg: AutotunedDualGEMM | None = None

    def _get_autotuners(self):
        """Lazy-init autotuners on first use."""
        if TritonLinearLoRA._at_pq is None:
            TritonLinearLoRA._at_pq = AutotunedProjectAndQuant(
                loraq_project_and_quant_kernel, rank=self.rank, warmup=5, rep=25,
            )
        if TritonLinearLoRA._at_dg is None:
            TritonLinearLoRA._at_dg = AutotunedDualGEMM(
                loraq_dual_gemm_kernel, LORAQ_Q8_CONFIGS, rank=self.rank, warmup=5, rep=25,
            )
        return TritonLinearLoRA._at_pq, TritonLinearLoRA._at_dg

    # ----- forward ----- #

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with autotuned kernels:

            out = A @ R^T @ L^T  +  Q(A) @ Q(W)^T  +  bias

        Kernels are autotuned on first call for each (M, K) / (M, N, K) shape.
        """
        orig_shape = x.shape
        x_2d = x.reshape(-1, self.in_features).contiguous()
        if x_2d.dtype != torch.float16:
            x_2d = x_2d.to(torch.float16)

        M = x_2d.shape[0]
        K = self.in_features
        N = self.out_features

        at_pq, at_dg = self._get_autotuners()

        # Kernel 1: autotuned fused projection + quantization
        P, a_fp4, a_scale = at_pq(x_2d, self.R, self.channel_scale)

        # Weight transpose for kernel 2
        w_fp4_t = self.weight_fp4.t().contiguous()

        # Kernel 2: autotuned fused dual GEMM
        out = at_dg(
            P, self.L,
            a_fp4, a_scale,
            w_fp4_t, self.weight_scale,
            M, N, K,
            out_dtype=self.out_dtype,
        )

        if self.bias is not None:
            out = out + self.bias

        return out.reshape(*orig_shape[:-1], self.out_features)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"rank={self.rank}, "
            f"bias={self.bias is not None}, "
            f"format=LoRA+MXFP4_e2m1, "
            f"out_dtype={self.out_dtype}"
        )


# ---------------------------------------------------------------------------
# LoRaQ fused FP8/FP4 wrapper
# ---------------------------------------------------------------------------

def triton_loraq_fused_q8(
    A_fp8: torch.Tensor,
    A_scale: torch.Tensor,
    R_fp8: torch.Tensor,
    R_scale: torch.Tensor,
    L_fp8: torch.Tensor,
    L_scale: torch.Tensor,
    W_fp4: torch.Tensor,
    W_scale: torch.Tensor,
    M: int,
    N: int,
    K: int,
    bias: torch.Tensor | None = None,
    channel_scale: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Fused LoRaQ kernel:

        C_fp8 = MXFP8_quant(
            channel_scale * (
                fp16(q8(A) @ q8(R)^T) @ dequant_fp16(q8(L))^T
                + q8(A) @ q4(W)^T  [+ bias]
            )
        )

    Parameters
    ----------
    A_fp8   : (M, K) float8_e4m3fn   – pre-quantized activation
    A_scale : (M, K//32) uint8       – e8m0 block scales for A
    R_fp8   : (rank, K) float8_e4m3fn – low-rank factor R
    R_scale : (rank, K//32) uint8    – e8m0 block scales for R
    L_fp8   : (N, rank) float8_e4m3fn – low-rank factor L
    L_scale : (N, rank//32) uint8    – e8m0 block scales for L
    W_fp4   : (K//2, N) uint8        – packed e2m1 weight (already transposed)
    W_scale : (N, K//32) uint8       – e8m0 block scales for W
    M, N, K : logical dimensions
    bias    : (N,) float32 or None
    channel_scale : (N,) fp16/bf16 or None — per-column scale before output quant

    Returns
    -------
    C_fp8   : (M, N) float8_e4m3fn
    C_scale : (M, N//32) uint8
    """
    rank = R_fp8.shape[0]

    # Default channel_scale to ones if not provided
    if channel_scale is None:
        channel_scale = torch.ones(N, dtype=torch.float16, device=A_fp8.device)

    C_fp8 = torch.empty((M, N), dtype=torch.uint8, device=A_fp8.device)
    C_scale = torch.empty((M, N // 32), dtype=torch.uint8, device=A_fp8.device)

    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 128
    GROUP_SIZE_M = 8

    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),)

    # Use a dummy pointer for bias when not present
    has_bias = bias is not None
    bias_ptr = bias if has_bias else A_fp8  # dummy, never dereferenced

    loraq_fused_q8_kernel[grid](
        A_fp8, A_scale,
        R_fp8, R_scale,
        L_fp8, L_scale,
        W_fp4, W_scale,
        bias_ptr,
        channel_scale,
        C_fp8, C_scale,
        M, N, K,
        # A_fp8 strides
        A_fp8.stride(0), A_fp8.stride(1),
        # A_scale strides
        A_scale.stride(0), A_scale.stride(1),
        # R_fp8 strides
        R_fp8.stride(0), R_fp8.stride(1),
        # R_scale strides
        R_scale.stride(0), R_scale.stride(1),
        # L_fp8 strides
        L_fp8.stride(0), L_fp8.stride(1),
        # L_scale strides
        L_scale.stride(0), L_scale.stride(1),
        # W_fp4 strides  (already transposed: K//2, N)
        W_fp4.stride(0), W_fp4.stride(1),
        # W_scale strides
        W_scale.stride(0), W_scale.stride(1),
        # C_fp8 strides
        C_fp8.stride(0), C_fp8.stride(1),
        # C_scale strides
        C_scale.stride(0), C_scale.stride(1),
        # channel_scale stride
        channel_scale.stride(0),
        # constexpr
        HAS_BIAS=has_bias,
        RANK=rank,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        GROUP_SIZE_M=GROUP_SIZE_M,
        matrix_instr_nonkdim=16 if BLOCK_K % 128 == 0 else 32,
    )

    # Reinterpret raw bytes as float8_e4m3fn
    C_fp8 = C_fp8.view(torch.float8_e4m3fn)
    return C_fp8, C_scale


# ---------------------------------------------------------------------------
# TritonLinearLoRaQ  (FP8/FP4 LoRA + Quantized output)
# ---------------------------------------------------------------------------

class TritonLinearLoRaQ(nn.Module):
    """
    Linear layer computing:

        C_fp8 = MXFP8_quant(
            fp16(q8(A) @ q8(R)^T) @ dequant_fp16(q8(L))^T
            + q8(A) @ q4(W)^T  [+ bias]
        )

    in a single fused Triton kernel.

    The activation arrives **pre-quantized** as MXFP8 (e4m3 + e8m0
    scales).  The weight W is stored in MXFP4 (e2m1), while the LoRA
    factors L and R are stored as MXFP8 buffers (non-trainable,
    quantized once at init from the SVD of the quantization residual).

    The output is an MXFP8 ``(float8_e4m3fn, uint8)`` tuple, suitable
    for chaining quantized layers.

    Parameters
    ----------
    in_features  : int  – input dimension (must be divisible by 64)
    out_features : int  – output dimension (must be divisible by 32)
    rank         : int  – low-rank dimension (default 64)
    bias         : bool – learnable bias (applied in fp32 before output quant)
    device       : str | torch.device
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 64,
        bias: bool = True,
        device: str | torch.device = "cuda",
    ):
        super().__init__()

        assert in_features % 64 == 0, (
            f"in_features={in_features} must be divisible by 64 "
            f"(BLOCK_K minimum for dot_scaled)"
        )
        assert out_features % 32 == 0, (
            f"out_features={out_features} must be divisible by 32"
        )
        assert rank == 64, (
            f"rank must be 64 for TritonLinearLoRaQ, got {rank}"
        )

        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank

        # ----- Quantized weight (MXFP4) -----
        self.register_buffer(
            "weight_fp4",
            torch.zeros(
                out_features, in_features // 2,
                dtype=torch.uint8, device=device,
            ),
        )
        self.register_buffer(
            "weight_scale",
            torch.zeros(
                out_features, in_features // 32,
                dtype=torch.uint8, device=device,
            ),
        )

        # ----- Low-rank factors (MXFP8, non-trainable) -----
        self.register_buffer(
            "R_fp8",
            torch.zeros(
                rank, in_features,
                dtype=torch.float8_e4m3fn, device=device,
            ),
        )
        self.register_buffer(
            "R_scale",
            torch.zeros(
                rank, in_features // 32,
                dtype=torch.uint8, device=device,
            ),
        )
        self.register_buffer(
            "L_fp8",
            torch.zeros(
                out_features, rank,
                dtype=torch.float8_e4m3fn, device=device,
            ),
        )
        self.register_buffer(
            "L_scale",
            torch.zeros(
                out_features, rank // 32,
                dtype=torch.uint8, device=device,
            ),
        )

        # ----- Channel-wise quantization scale (applied before output quant) -----
        self.register_buffer(
            "channel_scale",
            torch.ones(out_features, device=device, dtype=torch.float16),
        )

        # ----- Optional bias (fp32, applied before output quant) -----
        if bias:
            self.register_buffer(
                "bias",
                torch.zeros(out_features, device=device, dtype=torch.float32),
            )
        else:
            self.register_buffer("bias", None)

    # ----- factory from nn.Linear ----- #

    @classmethod
    def from_float(
        cls,
        linear: nn.Linear,
        rank: int = 64,
    ) -> "TritonLinearLoRaQ":
        """
        Create a ``TritonLinearLoRaQ`` from an existing ``nn.Linear``.

        1. Quantize weight to MXFP4.
        2. Compute the quantization residual ``W - dequant(Q4(W))``.
        3. Truncated SVD → L_init (N, rank), R_init (rank, K) in fp16.
        4. Quantize R_init to MXFP8 → R_fp8, R_scale.
        5. Quantize L_init to MXFP8 → L_fp8, L_scale.
        """
        device = linear.weight.device
        W = linear.weight.to(dtype=torch.float16, device=device)

        layer = cls(
            in_features=linear.in_features,
            out_features=linear.out_features,
            rank=rank,
            bias=linear.bias is not None,
            device=device,
        )

        # 1. Quantize weight to FP4
        w_fp4, w_scale = dynamic_mxfp4_quant(W)
        layer.weight_fp4.copy_(w_fp4)
        layer.weight_scale.copy_(w_scale)

        # 2. Dequantize for residual computation
        w_deq = mxfp4_to_f32(w_fp4)
        s_f32 = e8m0_to_f32(w_scale)
        s_f32 = s_f32.repeat_interleave(32, dim=-1)
        w_reconstructed = (w_deq * s_f32).to(torch.float16).to(device)

        # 3. SVD of residual
        residual = (W - w_reconstructed).float()
        U, S, Vh = torch.linalg.svd(residual, full_matrices=False)
        sqrt_S = S[:rank].sqrt()
        L_init = U[:, :rank] * sqrt_S[None, :]      # (out, rank)
        R_init = sqrt_S[:, None] * Vh[:rank, :]      # (rank, in)

        # 4. Quantize R to MXFP8
        r_fp8, r_scale = dynamic_mxfp8_quant(
            R_init.to(torch.float16).to(device)
        )
        layer.R_fp8.copy_(r_fp8)
        layer.R_scale.copy_(r_scale)

        # 5. Quantize L to MXFP8
        l_fp8, l_scale = dynamic_mxfp8_quant(
            L_init.to(torch.float16).to(device)
        )
        layer.L_fp8.copy_(l_fp8)
        layer.L_scale.copy_(l_scale)

        # Bias
        if linear.bias is not None:
            layer.bias.copy_(linear.bias.to(torch.float32))

        return layer

    # ----- factory from raw weight tensor ----- #

    @classmethod
    def from_weight(
        cls,
        weight: torch.Tensor,
        bias: torch.Tensor | None = None,
        rank: int = 64,
    ) -> "TritonLinearLoRaQ":
        """
        Create from raw weight tensor ``(out_features, in_features)``.
        """
        assert weight.ndim == 2
        dummy = nn.Linear(
            weight.shape[1], weight.shape[0],
            bias=bias is not None,
            device=weight.device,
            dtype=weight.dtype,
        )
        with torch.no_grad():
            dummy.weight.copy_(weight)
            if bias is not None:
                dummy.bias.copy_(bias)
        return cls.from_float(dummy, rank=rank)

    # ----- autotuner (class-level, shared across instances) ----- #
    _at_v1: AutotunedLoRaQ | None = None

    def _get_autotuner(self):
        """Lazy-init autotuner on first use."""
        if TritonLinearLoRaQ._at_v1 is None:
            TritonLinearLoRaQ._at_v1 = AutotunedLoRaQ(
                loraq_fused_q8_kernel, LORAQ_Q8_CONFIGS,
                rank=self.rank, warmup=5, rep=25,
            )
        return TritonLinearLoRaQ._at_v1

    # ----- forward ----- #

    def forward(
        self,
        a_fp8: torch.Tensor,
        a_scale: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with autotuned kernel — takes pre-quantized MXFP8 activation.

        Autotuned on first call for each (M, N, K) shape. Subsequent calls
        use the cached best config directly.

        Parameters
        ----------
        a_fp8   : (..., in_features) float8_e4m3fn
        a_scale : (..., in_features // 32) uint8  e8m0

        Returns
        -------
        c_fp8   : (..., out_features) float8_e4m3fn
        c_scale : (..., out_features // 32) uint8  e8m0
        """
        orig_shape = a_fp8.shape
        a_fp8_2d = a_fp8.reshape(-1, self.in_features).contiguous()
        a_scale_2d = a_scale.reshape(-1, self.in_features // 32).contiguous()

        M = a_fp8_2d.shape[0]
        K = self.in_features
        N = self.out_features

        w_fp4_t = self.weight_fp4.t().contiguous()

        at_v1 = self._get_autotuner()
        c_fp8, c_scale = at_v1(
            a_fp8_2d, a_scale_2d,
            self.R_fp8, self.R_scale,
            self.L_fp8, self.L_scale,
            w_fp4_t, self.weight_scale,
            M, N, K,
            bias=self.bias,
            channel_scale=self.channel_scale,
        )

        out_shape = (*orig_shape[:-1], self.out_features)
        scale_shape = (*orig_shape[:-1], self.out_features // 32)
        return c_fp8.reshape(out_shape), c_scale.reshape(scale_shape)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"rank={self.rank}, "
            f"bias={self.bias is not None}, "
            f"format=LoRaQ_FP8+FP4, "
            f"output=MXFP8"
        )


# ---------------------------------------------------------------------------
# LoRaQ fused FP8/FP4 wrapper (scaled variant — Phase 2 uses dot_scaled)
# ---------------------------------------------------------------------------

def triton_loraq_fused_q8_scaled(
    A_fp8: torch.Tensor,
    A_scale: torch.Tensor,
    R_fp8: torch.Tensor,
    R_scale: torch.Tensor,
    L_fp8: torch.Tensor,
    L_scale: torch.Tensor,
    W_fp4: torch.Tensor,
    W_scale: torch.Tensor,
    M: int,
    N: int,
    K: int,
    bias: torch.Tensor | None = None,
    channel_scale: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Fused LoRaQ kernel (fully-fp8 Phase 2 variant):

        C_fp8 = MXFP8_quant(
            channel_scale * (
                q8(q8(A) @ q8(R)^T) @ q8(L)^T
                + q8(A) @ q4(W)^T  [+ bias]
            )
        )

    Same interface as ``triton_loraq_fused_q8`` but Phase 2 quantizes P
    to MXFP8 in-register and uses ``dot_scaled("e4m3","e4m3")``.
    """
    rank = R_fp8.shape[0]

    # Default channel_scale to ones if not provided
    if channel_scale is None:
        channel_scale = torch.ones(N, dtype=torch.float16, device=A_fp8.device)

    C_fp8 = torch.empty((M, N), dtype=torch.uint8, device=A_fp8.device)
    C_scale = torch.empty((M, N // 32), dtype=torch.uint8, device=A_fp8.device)

    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 64
    GROUP_SIZE_M = 8

    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),)

    has_bias = bias is not None
    bias_ptr = bias if has_bias else A_fp8

    loraq_fused_q8_scaled_kernel[grid](
        A_fp8, A_scale,
        R_fp8, R_scale,
        L_fp8, L_scale,
        W_fp4, W_scale,
        bias_ptr,
        channel_scale,
        C_fp8, C_scale,
        M, N, K,
        A_fp8.stride(0), A_fp8.stride(1),
        A_scale.stride(0), A_scale.stride(1),
        R_fp8.stride(0), R_fp8.stride(1),
        R_scale.stride(0), R_scale.stride(1),
        L_fp8.stride(0), L_fp8.stride(1),
        L_scale.stride(0), L_scale.stride(1),
        W_fp4.stride(0), W_fp4.stride(1),
        W_scale.stride(0), W_scale.stride(1),
        C_fp8.stride(0), C_fp8.stride(1),
        C_scale.stride(0), C_scale.stride(1),
        channel_scale.stride(0),
        HAS_BIAS=has_bias,
        RANK=rank,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        GROUP_SIZE_M=GROUP_SIZE_M,
        matrix_instr_nonkdim=16 if BLOCK_K % 128 == 0 else 32,
    )

    C_fp8 = C_fp8.view(torch.float8_e4m3fn)
    return C_fp8, C_scale


# ---------------------------------------------------------------------------
# TritonLinearLoRaQFP8  (fully-fp8 Phase 2 variant)
# ---------------------------------------------------------------------------

class TritonLinearLoRaQFP8(TritonLinearLoRaQ):
    """
    Same computation as ``TritonLinearLoRaQ`` but Phase 2 quantizes P to
    MXFP8 in-register and uses ``dot_scaled("e4m3","e4m3")`` for the
    P×L^T multiplication instead of ``tl.dot`` with fp16.

    This trades some precision (extra P re-quantization to fp8) for
    potentially higher throughput via hardware-accelerated scaled-dot.

    Inherits all buffers, factories, and interface from
    ``TritonLinearLoRaQ``; only the forward pass kernel differs.
    """

    def forward(
        self,
        a_fp8: torch.Tensor,
        a_scale: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass — identical interface to ``TritonLinearLoRaQ.forward``
        but uses the scaled kernel variant (Kernel 8).
        """
        orig_shape = a_fp8.shape
        a_fp8_2d = a_fp8.reshape(-1, self.in_features).contiguous()
        a_scale_2d = a_scale.reshape(-1, self.in_features // 32).contiguous()

        M = a_fp8_2d.shape[0]
        K = self.in_features
        N = self.out_features

        w_fp4_t = self.weight_fp4.t().contiguous()

        c_fp8, c_scale = triton_loraq_fused_q8_scaled(
            a_fp8_2d, a_scale_2d,
            self.R_fp8, self.R_scale,
            self.L_fp8, self.L_scale,
            w_fp4_t, self.weight_scale,
            M, N, K,
            bias=self.bias,
            channel_scale=self.channel_scale,
        )

        out_shape = (*orig_shape[:-1], self.out_features)
        scale_shape = (*orig_shape[:-1], self.out_features // 32)
        return c_fp8.reshape(out_shape), c_scale.reshape(scale_shape)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"rank={self.rank}, "
            f"bias={self.bias is not None}, "
            f"format=LoRaQ_FP8+FP4_scaled, "
            f"output=MXFP8"
        )
