"""Inference-only fused operations preserving Qwen's BF16 rounding points."""
import types
import torch
import triton
import triton.language as tl


@triton.jit
def _rms(X, W, Y, D: tl.constexpr, EPS: tl.constexpr, B: tl.constexpr):
    row = tl.program_id(0)
    d = tl.arange(0, B)
    x = tl.load(X + row * D + d, d < D, 0).to(tl.float32)
    inv = tl.rsqrt(tl.sum(x * x, 0) / D + EPS)
    normalized = (x * inv).to(Y.dtype.element_ty).to(tl.float32)
    w = tl.load(W + d, d < D, 0).to(tl.float32)
    tl.store(Y + row * D + d, normalized * w, d < D)


def rms(x, weight, eps):
    assert x.is_contiguous()
    y = torch.empty_like(x)
    d = x.shape[-1]
    _rms[(x.numel() // d,)](x, weight, y, d, eps, triton.next_power_of_2(d), enable_fp_fusion=False)
    return y


def install_rms(model):
    for module in model.modules():
        if type(module).__name__ == 'Qwen3RMSNorm':
            def forward(self, x):
                return rms(x, self.weight, self.variance_epsilon)
            module.forward = types.MethodType(forward, module)


@triton.jit
def _rope(Q, K, COS, SIN, OQ, OK, NH: tl.constexpr, NK: tl.constexpr, D: tl.constexpr):
    h = tl.program_id(0)
    d = tl.arange(0, D)
    other = (d + D // 2) % D
    cos = tl.load(COS + d).to(tl.float32)
    sin = tl.load(SIN + d).to(tl.float32)
    if h < NH:
        x = tl.load(Q + h * D + d).to(tl.float32)
        rot = tl.load(Q + h * D + other).to(tl.float32)
        rot = tl.where(d < D // 2, -rot, rot)
        a = (x * cos).to(OQ.dtype.element_ty).to(tl.float32)
        b = (rot * sin).to(OQ.dtype.element_ty).to(tl.float32)
        tl.store(OQ + h * D + d, a + b)
    else:
        kh = h - NH
        x = tl.load(K + kh * D + d).to(tl.float32)
        rot = tl.load(K + kh * D + other).to(tl.float32)
        rot = tl.where(d < D // 2, -rot, rot)
        a = (x * cos).to(OK.dtype.element_ty).to(tl.float32)
        b = (rot * sin).to(OK.dtype.element_ty).to(tl.float32)
        tl.store(OK + kh * D + d, a + b)


def rope(q, k, cos, sin):
    assert q.shape[0] == k.shape[0] == q.shape[2] == k.shape[2] == 1
    oq, ok = torch.empty_like(q), torch.empty_like(k)
    _rope[(q.shape[1] + k.shape[1],)](q, k, cos, sin, oq, ok,
                                    q.shape[1], k.shape[1], q.shape[-1], enable_fp_fusion=False)
    return oq, ok
