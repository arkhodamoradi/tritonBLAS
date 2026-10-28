# Requires the gfx950 Gluon build of Triton (https://github.com/ROCm/gfx950-gluon-tutorials) on
# PYTHONPATH, imported before torch. Upstream Triton rejects this kernel's layouts.
import triton
import triton.testing
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
import torch

CONFIGS = [dict(tc=t, nw=w, rows_mult=r) for t in (4, 16, 32, 64) for w in (4, 8) for r in (1, 4, 16)]


@gluon.jit
def _coef(P, c0, nb, K: gl.constexpr, B: gl.constexpr, BC: gl.constexpr, TR: gl.constexpr, TC: gl.constexpr,
          NW: gl.constexpr, MASK: gl.constexpr):
    G: gl.constexpr = 16 >> B
    H: gl.constexpr = 1 << B
    L5: gl.constexpr = gl.BlockedLayout([1, 1, G, 2, H], [TR, TC, 1, 1, 1], [NW, 1, 1, 1, 1], [4, 3, 2, 1, 0])
    S4: gl.constexpr = gl.SliceLayout(3, L5)
    c = c0 + gl.arange(0, BC, layout=gl.SliceLayout(0, gl.SliceLayout(2, gl.SliceLayout(3, S4))))
    if MASK:
        v = gl.load(P + c * 12 + K, mask=c < nb, other=0.0)
    else:
        v = gl.load(P + c * 12 + K)
    return v[None, :, None, None]


@gluon.jit
def _stage(x, ca, cb, cc, cd, DIRECT: gl.constexpr, B: gl.constexpr, BM: gl.constexpr, BC: gl.constexpr,
           TR: gl.constexpr, TC: gl.constexpr, NW: gl.constexpr, L3: gl.constexpr):
    G: gl.constexpr = 16 >> B
    H: gl.constexpr = 1 << B
    L5: gl.constexpr = gl.BlockedLayout([1, 1, G, 2, H], [TR, TC, 1, 1, 1], [NW, 1, 1, 1, 1], [4, 3, 2, 1, 0])
    y = gl.convert_layout(gl.reshape(x, [BM, BC, G, 2, H]), L5, assert_trivial=True)
    lo, hi = gl.split(gl.permute(y, [0, 1, 2, 4, 3]))
    ka = gl.convert_layout(ca, lo.type.layout)
    kb = gl.convert_layout(cb, lo.type.layout)
    kc = gl.convert_layout(cc, lo.type.layout)
    kd = gl.convert_layout(cd, lo.type.layout)
    if DIRECT:
        nlo = lo * ka + hi * kc
        nhi = lo * kb + hi * kd
    else:
        nlo = hi * ka + lo
        nhi = nlo * kb + hi
    y = gl.permute(gl.join(nlo, nhi), [0, 1, 2, 4, 3])
    return gl.convert_layout(gl.reshape(y, [BM, BC, 32]), L3, assert_trivial=True)


@gluon.jit
def _quant_store(x, OP, OS, rows, cols, m, nb, BM: gl.constexpr, BC: gl.constexpr, TR: gl.constexpr,
                 TC: gl.constexpr, NW: gl.constexpr, L3: gl.constexpr, MASK: gl.constexpr, ksh):
    L4: gl.constexpr = gl.BlockedLayout([1, 1, 16, 2], [TR, TC, 1, 1], [NW, 1, 1, 1], [3, 2, 1, 0])
    LP3: gl.constexpr = gl.BlockedLayout([1, 1, 16], [TR, TC, 1], [NW, 1, 1], [2, 1, 0])
    xi = x.to(gl.int16, bitcast=True).to(gl.int32) & 0x7FFF
    amb = gl.max(xi, axis=2)
    amax = amb.to(gl.int16).to(gl.float16, bitcast=True).to(gl.float32)
    ai = amax.to(gl.int32, bitcast=True)
    ai = (ai + 0x200000) & -8388608
    e = ((ai >> 23) & 0xFF) - 127
    se = gl.minimum(gl.maximum(e - 2, -127), 127)
    sc = gl.convert_layout(gl.exp2(se.to(gl.float32)), gl.SliceLayout(2, gl.SliceLayout(3, L4)))
    a, b = gl.split(gl.convert_layout(gl.reshape(x, [BM, BC, 16, 2]), L4))
    w = (a.to(gl.int16, bitcast=True).to(gl.int32) & 0xFFFF) | (b.to(gl.int16, bitcast=True).to(gl.int32) << 16)
    sc = sc[:, :, None] + gl.zeros_like(w).to(gl.float32)
    r = gl.inline_asm_elementwise("v_cvt_scalef32_pk_fp4_f16 $0, $1, $2", "=v,v,v", [w, sc], dtype=gl.int32,
                                  is_pure=True, pack=1)
    byte = gl.convert_layout((r & 0xFF).to(gl.uint8), LP3)
    rp = gl.convert_layout(rows, gl.SliceLayout(1, gl.SliceLayout(2, LP3)))
    cp = gl.convert_layout(cols, gl.SliceLayout(0, gl.SliceLayout(2, LP3)))
    k = gl.arange(0, 16, layout=gl.SliceLayout(0, gl.SliceLayout(1, LP3)))
    off = rp[:, None, None] * (nb * 16) + cp[None, :, None] * 16 + k[None, None, :]
    if MASK:
        gl.store(OP + off, byte, mask=(rp < m)[:, None, None] & (cp < nb)[None, :, None])
    else:
        gl.store(OP + off, byte)
    S2: gl.constexpr = gl.SliceLayout(2, L3)
    se = gl.minimum(gl.maximum(gl.convert_layout(se, S2) + ksh[None, :], -127), 127)
    rs = gl.convert_layout(rows, gl.SliceLayout(1, S2))
    cs = gl.convert_layout(cols, gl.SliceLayout(0, S2))
    so = rs[:, None] * nb + cs[None, :]
    if MASK:
        gl.store(OS + so, gl.convert_layout((se + 127).to(gl.uint8), S2), mask=(rs < m)[:, None] & (cs < nb)[None, :])
    else:
        gl.store(OS + so, gl.convert_layout((se + 127).to(gl.uint8), S2))


@gluon.jit
def k3_kernel(A, DP, CF, OP, OS, m, nb, ROWS, KS, TR: gl.constexpr, TC: gl.constexpr, NW: gl.constexpr,
              MASK: gl.constexpr):
    BC: gl.constexpr = TC
    BM: gl.constexpr = TR * NW
    L3: gl.constexpr = gl.BlockedLayout([1, 1, 32], [TR, TC, 1], [NW, 1, 1], [2, 1, 0])
    c0 = gl.program_id(0) * BC
    r_lo = gl.program_id(1) * ROWS
    cols = c0 + gl.arange(0, BC, layout=gl.SliceLayout(0, gl.SliceLayout(2, L3)))
    j = gl.arange(0, 32, layout=gl.SliceLayout(0, gl.SliceLayout(1, L3)))
    cmask = cols < nb
    LD: gl.constexpr = gl.SliceLayout(0, L3)
    cd = c0 + gl.arange(0, BC, layout=gl.SliceLayout(1, LD))
    jd = gl.arange(0, 32, layout=gl.SliceLayout(0, LD))
    if MASK:
        dfull = gl.load(DP + cd[:, None] * 32 + jd[None, :], mask=(cd < nb)[:, None], other=0.0)
    else:
        dfull = gl.load(DP + cd[:, None] * 32 + jd[None, :])
    dfull = dfull[None, :, :]
    S2K: gl.constexpr = gl.SliceLayout(2, L3)
    ck = c0 + gl.arange(0, BC, layout=gl.SliceLayout(0, S2K))
    if MASK:
        ksh = gl.load(KS + ck, mask=ck < nb, other=0)
    else:
        ksh = gl.load(KS + ck)
    g00 = _coef(CF, c0, nb, 0, 0, BC, TR, TC, NW, MASK)
    g01 = _coef(CF, c0, nb, 1, 0, BC, TR, TC, NW, MASK)
    g10 = _coef(CF, c0, nb, 2, 0, BC, TR, TC, NW, MASK)
    g11 = _coef(CF, c0, nb, 3, 0, BC, TR, TC, NW, MASK)
    lt1 = _coef(CF, c0, nb, 4, 4, BC, TR, TC, NW, MASK)
    u1 = _coef(CF, c0, nb, 5, 4, BC, TR, TC, NW, MASK)
    lt2 = _coef(CF, c0, nb, 6, 3, BC, TR, TC, NW, MASK)
    u2 = _coef(CF, c0, nb, 7, 3, BC, TR, TC, NW, MASK)
    lt3 = _coef(CF, c0, nb, 8, 2, BC, TR, TC, NW, MASK)
    u3 = _coef(CF, c0, nb, 9, 2, BC, TR, TC, NW, MASK)
    lt4 = _coef(CF, c0, nb, 10, 1, BC, TR, TC, NW, MASK)
    u4 = _coef(CF, c0, nb, 11, 1, BC, TR, TC, NW, MASK)
    r_hi = gl.minimum(r_lo + ROWS, m)
    for r0 in range(r_lo, r_hi, BM):
        rows = r0 + gl.arange(0, BM, layout=gl.SliceLayout(1, gl.SliceLayout(2, L3)))
        off = rows[:, None, None] * (nb * 32) + cols[None, :, None] * 32 + j[None, None, :]
        msk = (rows < m)[:, None, None] & cmask[None, :, None]
        if MASK:
            x = gl.load(A + off, mask=msk, other=0.0)
        else:
            x = gl.load(A + off)
        x = x * dfull
        x = _stage(x, g00, g01, g10, g11, True, 0, BM, BC, TR, TC, NW, L3)
        x = _stage(x, lt1, u1, lt1, u1, False, 4, BM, BC, TR, TC, NW, L3)
        x = _stage(x, lt2, u2, lt2, u2, False, 3, BM, BC, TR, TC, NW, L3)
        x = _stage(x, lt3, u3, lt3, u3, False, 2, BM, BC, TR, TC, NW, L3)
        x = _stage(x, lt4, u4, lt4, u4, False, 1, BM, BC, TR, TC, NW, L3)
        _quant_store(x, OP, OS, rows, cols, m, nb, BM, BC, TR, TC, NW, L3, MASK, ksh)


def _apply(M, B, f):
    y = M.view(M.shape[0], 32, 16 >> B, 2, 1 << B)
    lo, hi = f(y[:, :, :, 0], y[:, :, :, 1])
    return torch.stack([lo, hi], 3).reshape(M.shape)


def prep(P):
    p = P.float()
    a, b, c = p[:, 0::3], p[:, 1::3], p[:, 2::3]
    g = torch.stack([a, b, c * a, c * b - 1.0 / a], -1).double()
    g00, g01, g10, g11 = g.unbind(-1)
    det = g00 * g11 - g01 * g10
    d0, d1, lt, u = g00, det / g00, g10 * g00 / det, g01 / g00
    dp = torch.ones_like(d0[:, :1])
    for s in range(4):
        dp = (dp[:, :, None] * torch.stack([d0[:, s], d1[:, s]], -1)[:, None, :]).flatten(1)
    v = lambda t: t[:, None, None, None]
    M = torch.diag_embed(dp.repeat_interleave(2, 1))
    bound = M.abs().sum(1).amax(1)
    M = _apply(M, 0, lambda lo, hi: (lo * v(g[:, 4, 0]) + hi * v(g[:, 4, 2]), lo * v(g[:, 4, 1]) + hi * v(g[:, 4, 3])))
    bound = torch.maximum(bound, M.abs().sum(1).amax(1))
    for s, B in enumerate((4, 3, 2, 1)):
        M = _apply(M, B, lambda lo, hi: (lo + hi * v(lt[:, s]), hi))
        bound = torch.maximum(bound, M.abs().sum(1).amax(1))
        M = _apply(M, B, lambda lo, hi: (lo, hi + lo * v(u[:, s])))
        bound = torch.maximum(bound, M.abs().sum(1).amax(1))
    ks = torch.clamp(torch.ceil(torch.log2(bound.clamp(min=1e-30))) - 8, min=0)
    dp = dp * torch.exp2(-ks)[:, None]
    cf = torch.stack([g[:, 4, 0], g[:, 4, 1], g[:, 4, 2], g[:, 4, 3],
                      lt[:, 0], u[:, 0], lt[:, 1], u[:, 1], lt[:, 2], u[:, 2], lt[:, 3], u[:, 3]], 1)
    return dp.repeat_interleave(2, 1).half().contiguous(), cf.half().contiguous(), ks.int().contiguous()


def k3(A, coeffs, tc=64, nw=4, rows_mult=1):
    m, d = A.shape
    nb = d // 32
    dp, cf, ks = coeffs
    bm = (64 // tc) * nw
    fp4 = torch.empty(m, d // 2, dtype=torch.uint8, device=A.device)
    e8m0 = torch.empty(m, nb, dtype=torch.uint8, device=A.device)
    k3_kernel[(triton.cdiv(nb, tc), triton.cdiv(m, rows_mult * bm))](
        A, dp, cf, fp4, e8m0, m, nb, rows_mult * bm, ks, TR=64 // tc, TC=tc, NW=nw,
        MASK=m % bm != 0 or nb % tc != 0, num_warps=nw)
    return fp4, e8m0


def tune(A, coeffs):
    return min(CONFIGS, key=lambda c: triton.testing.do_bench(lambda: k3(A, coeffs, **c), warmup=25, rep=100))
