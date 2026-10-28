import triton
import triton.language as tl


@triton.jit
def _params(P, s: tl.constexpr):
    return tl.load(P + 3 * s).to(tl.float32), tl.load(P + 3 * s + 1).to(tl.float32), tl.load(P + 3 * s + 2).to(tl.float32)


@triton.jit
def _lu(p):
    return p[0], p[1], p[2] * p[0], p[2] * p[1] - 1.0 / p[0]


@triton.jit
def _factor(g, i, j, s: tl.constexpr):
    ir = (i >> s) & 1
    jc = (j >> s) & 1
    return tl.where(ir == 0, tl.where(jc == 0, g[0], g[1]), tl.where(jc == 0, g[2], g[3]))


@triton.jit
def kron(P):
    p1 = _params(P, 0)
    p2 = _params(P, 1)
    p3 = _params(P, 2)
    p4 = _params(P, 3)
    p5 = _params(P, 4)
    g1 = _lu(p1)
    g2 = _lu(p2)
    g3 = _lu(p3)
    g4 = _lu(p4)
    g5 = _lu(p5)
    i = tl.arange(0, 32)[:, None]
    j = tl.arange(0, 32)[None, :]
    k = _factor(g1, i, j, 4)
    k = k * _factor(g2, i, j, 3)
    k = k * _factor(g3, i, j, 2)
    k = k * _factor(g4, i, j, 1)
    k = k * _factor(g5, i, j, 0)
    return k


@triton.jit
def mxfp4_store(x, OP, OS, rows, b, m, nb, BLOCK_ROWS: tl.constexpr):
    amax = tl.max(tl.abs(x), axis=1, keep_dims=True)
    amax_i = amax.to(tl.int32, bitcast=True)
    amax_i = (amax_i + 0x200000).to(tl.uint32, bitcast=True) & 0xFF800000
    amax = amax_i.to(tl.float32, bitcast=True)
    se = tl.log2(amax).floor() - 2
    se = tl.clamp(se, min=-127, max=127)
    q = x * tl.exp2(-se)
    e8m0 = se.to(tl.uint8) + 127
    q = q.to(tl.uint32, bitcast=True)
    s = q & 0x80000000
    e = (q >> 23) & 0xFF
    mb = q & 0x7FFFFF
    adj = tl.core.sub(127, e + 1, sanitize_overflow=False)
    mb = tl.where(e < 127, (0x400000 | (mb >> 1)) >> adj, mb)
    e = tl.maximum(e, 126) - 126
    t = tl.minimum((((e << 2) | (mb >> 21)) + 1) >> 1, 0x7)
    n = ((s >> 28) | t).to(tl.uint8)
    lo, hi = tl.split(tl.reshape(n, [BLOCK_ROWS, 16, 2]))
    ok = rows < m
    tl.store(OP + rows[:, None] * (nb * 16) + (b * 16 + tl.arange(0, 16))[None, :], lo | (hi << 4), mask=ok[:, None])
    tl.store(OS + rows * nb + b, tl.reshape(e8m0, [BLOCK_ROWS]), mask=ok)
