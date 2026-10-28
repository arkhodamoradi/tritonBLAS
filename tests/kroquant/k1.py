import triton
import triton.language as tl
import torch

from mxfp4 import mxfp4_store

CONFIGS = [triton.Config({"BLOCK_ROWS": r}, num_warps=w, num_stages=s) for r, w, s in [
    (16, 4, 2), (16, 4, 3), (32, 4, 2), (32, 4, 3), (32, 8, 3), (64, 4, 2), (64, 4, 3), (64, 8, 2), (64, 8, 3),
    (64, 8, 4), (128, 8, 2), (128, 8, 3), (128, 8, 4), (128, 4, 3), (256, 8, 2), (256, 8, 3), (256, 8, 4)]]


@triton.autotune(configs=CONFIGS, key=["m", "nb"])
@triton.jit
def k1_kernel(A, K, OP, OS, m: tl.constexpr, nb: tl.constexpr, BLOCK_ROWS: tl.constexpr):
    b = tl.program_id(1)
    rows = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    ok = rows < m
    c = tl.arange(0, 32)
    k = tl.load(K + b * 1024 + c[:, None] * 32 + c[None, :])
    a = tl.load(A + rows[:, None] * (nb * 32) + (b * 32 + c)[None, :], mask=ok[:, None], other=0.0)
    mxfp4_store(tl.dot(a, k, out_dtype=tl.float32), OP, OS, rows, b, m, nb, BLOCK_ROWS)


def k1(A, K):
    m, d = A.shape
    nb = d // 32
    fp4 = torch.empty(m, d // 2, dtype=torch.uint8, device=A.device)
    e8m0 = torch.empty(m, nb, dtype=torch.uint8, device=A.device)
    k1_kernel[lambda meta: (triton.cdiv(m, meta["BLOCK_ROWS"]), nb)](A, K, fp4, e8m0, m, nb)
    return fp4, e8m0
