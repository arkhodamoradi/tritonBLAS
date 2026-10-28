import triton
import triton.language as tl
import torch

from k1 import k1
from mxfp4 import kron, mxfp4_store

CONFIGS = [triton.Config({"BLOCK_ROWS": r}, num_warps=w, num_stages=s) for r, w, s in [
    (16, 4, 2), (16, 4, 3), (32, 4, 2), (32, 4, 3), (32, 8, 2), (32, 8, 3), (64, 4, 2), (64, 8, 2), (64, 8, 3),
    (128, 8, 2), (128, 8, 3)]]


@triton.autotune(configs=CONFIGS, key=["m", "nb"])
@triton.jit
def k2_kernel(A, P, OP, OS, m: tl.constexpr, nb: tl.constexpr, BLOCK_ROWS: tl.constexpr):
    b = tl.program_id(1)
    k = kron(P + b * 15)
    rows = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    c = tl.arange(0, 32)
    a = tl.load(A + rows[:, None] * (nb * 32) + (b * 32 + c)[None, :], mask=(rows < m)[:, None], other=0.0)
    mxfp4_store(tl.dot(a, k.to(a.dtype), out_dtype=tl.float32), OP, OS, rows, b, m, nb, BLOCK_ROWS)


@triton.jit
def build_kernel(P, K):
    b = tl.program_id(0)
    k = kron(P + b * 15)
    c = tl.arange(0, 32)
    tl.store(K + b * 1024 + c[:, None] * 32 + c[None, :], k.to(K.dtype.element_ty))


def k2_fused(A, P):
    m, d = A.shape
    nb = d // 32
    fp4 = torch.empty(m, d // 2, dtype=torch.uint8, device=A.device)
    e8m0 = torch.empty(m, nb, dtype=torch.uint8, device=A.device)
    k2_kernel[lambda meta: (triton.cdiv(m, meta["BLOCK_ROWS"]), nb)](A, P, fp4, e8m0, m, nb)
    return fp4, e8m0


def build(P):
    K = torch.empty(P.shape[0], 32, 32, dtype=P.dtype, device=P.device)
    build_kernel[(P.shape[0],)](P, K)
    return K


def k2_split(A, P):
    return k1(A, build(P))
