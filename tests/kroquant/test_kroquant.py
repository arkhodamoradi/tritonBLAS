import triton
import torch
import pytest

from k1 import k1
from k2 import k2_fused, k2_split
from k3 import k3, prep

LEVELS = [0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6]


def decode(fp4, e8m0):
    n = torch.stack([fp4 & 0xF, fp4 >> 4], -1).view(fp4.shape[0], -1).long()
    lv = torch.tensor(LEVELS, dtype=torch.float64, device=fp4.device)
    return lv[n] * torch.exp2(e8m0.double() - 127).repeat_interleave(32, 1)


def kron(P):
    p = P.float()
    a, b, c = p[:, 0::3], p[:, 1::3], p[:, 2::3]
    g = torch.stack([a, b, c * a, c * b - 1 / a], -1).double().view(-1, 5, 2, 2)
    K = g[:, 0]
    for s in range(1, 5):
        K = torch.einsum("nij,nkl->nikjl", K, g[:, s]).reshape(K.shape[0], 2 * K.shape[1], -1)
    return K


def transform(A, K):
    m, d = A.shape
    return torch.einsum("mbi,bij->mbj", A.double().view(m, -1, 32), K.double()).reshape(m, d)


def sqnr(ref, out):
    return (10 * torch.log10(ref.pow(2).sum() / (ref - out).pow(2).sum())).item()


@pytest.mark.parametrize("m, d", [(4096, 1024), (4096, 4096), (512, 4096), (512, 16384), (100, 1152)])
def test_kroquant(m, d):
    torch.manual_seed(42)
    A = torch.randn(m, d, dtype=torch.float16, device="cuda")
    P = torch.randn(d // 32, 15, dtype=torch.float16, device="cuda")
    P[:, ::3] = P[:, ::3].sign() * P[:, ::3].abs().clamp(min=0.1)
    Kb = torch.randn(d // 32, 32, 32, dtype=torch.float16, device="cuda")
    ref = transform(A, kron(P))
    assert sqnr(transform(A, Kb), decode(*k1(A, Kb))) > 17.5
    assert sqnr(ref, decode(*k2_fused(A, P))) > 17.5
    assert sqnr(ref, decode(*k2_split(A, P))) > 17.5
    assert sqnr(ref, decode(*k3(A, prep(P)))) > 17.5
