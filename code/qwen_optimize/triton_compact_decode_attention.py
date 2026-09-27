#!/usr/bin/env python3
"""Fixed-shape Triton diagnostic kernel for compact weighted-coalescing decode.

This file is intentionally independent from the validated reference evaluator.
It supports batch=1, H_KV=8, GQA=4, D=128, P=32 and FP32 compact storage.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _reconstruct_k_kernel(c_ptr, coeff_ptr, owner_ptr, out_ptr,
                          n_pages, sc_page, sc_head, sc_p, sc_m,
                          sx_page, sx_head, so_page, so_head, so_d,
                          so_head_out, so_tok_out, so_d_out,
                          BLOCK_D: tl.constexpr):
    pid = tl.program_id(0)
    d = tl.arange(0, BLOCK_D)
    mask = d < BLOCK_D
    page = pid // 8
    head = pid % 8
    valid = page < n_pages
    own = tl.load(owner_ptr + page * so_page + head * so_head + d,
                  mask=mask & valid, other=0).to(tl.int32)
    coef = tl.load(coeff_ptr + page * sx_page + head * sx_head + d,
                   mask=mask & valid, other=0.0).to(tl.float32)
    for tok in tl.static_range(0, 32):
        c = tl.load(c_ptr + page * sc_page + head * sc_head + tok * sc_p + own * sc_m,
                    mask=mask & valid, other=0.0).to(tl.float32)
        tl.store(out_ptr + head * so_head_out + (page * 32 + tok) * so_tok_out + d * so_d_out,
                 c * coef, mask=mask & valid)


def reconstruct_k_triton(c, coeff, owner, out):
    """Exact compact reconstruction directly into [8, N*32, 128] workspace."""
    n, h, p, m = c.shape
    assert h == 8 and p == 32 and out.shape == (8, n * 32, 128)
    _reconstruct_k_kernel[(n * 8,)](
        c, coeff, owner, out, n,
        c.stride(0), c.stride(1), c.stride(2), c.stride(3),
        coeff.stride(0), coeff.stride(1), owner.stride(0), owner.stride(1), owner.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        BLOCK_D=128, num_warps=4, num_stages=2)
    return out


@triton.jit
def _compact_attention_kernel(
    q_ptr, c_ptr, coeff_ptr, owner_ptr, hist_v_ptr, active_k_ptr, active_v_ptr,
    out_ptr, num_pages, active_len,
    sq_h, sc_page, sc_kv, sc_p, sc_m,
    sco_page, sco_kv, sv_page, sv_kv, sv_p, sv_d,
    sak_kv, sak_t, sak_d, sav_kv, sav_t, sav_d,
    so_h, so_d, scale,
    BLOCK_D: tl.constexpr, BLOCK_P: tl.constexpr,
):
    qh = tl.program_id(0)
    offs_d = tl.arange(0, BLOCK_D)
    offs_p = tl.arange(0, BLOCK_P)
    mask_d = offs_d < BLOCK_D
    kv = qh // 4

    q = tl.load(q_ptr + qh * sq_h + offs_d, mask=mask_d, other=0.0).to(tl.float32)
    running_m = tl.full((), -float("inf"), tl.float32)
    running_l = tl.zeros((), tl.float32)
    acc = tl.zeros((BLOCK_D,), tl.float32)

    # Historical pages.  Direct compact score:
    # score[p] = sum_j q[j] * coeff[j] * C[p, owner[j]].
    # Short diagnostic shape: up to 16 sealed pages (512 historical tokens).
    # A separate larger-context specialization will be added only after this
    # version is validated; keeping compilation bounded is important here.
    for page in tl.static_range(0, 16):
        valid_page = page < num_pages
        owner = tl.load(owner_ptr + page * sco_page + kv * sco_kv + offs_d,
                        mask=mask_d & valid_page, other=0).to(tl.int32)
        coeff = tl.load(coeff_ptr + page * sco_page + kv * sco_kv + offs_d,
                        mask=mask_d & valid_page, other=0.0).to(tl.float32)
        qcoeff = q * coeff
        c = tl.load(
            c_ptr + page * sc_page + kv * sc_kv
            + offs_p[:, None] * sc_p + owner[None, :] * sc_m,
            mask=(offs_p[:, None] < BLOCK_P) & mask_d[None, :] & valid_page, other=0.0,
        ).to(tl.float32)
        scores = tl.sum(c * qcoeff[None, :], axis=1) * scale
        v = tl.load(
            hist_v_ptr + page * sv_page + kv * sv_kv
            + offs_p[:, None] * sv_p + offs_d[None, :] * sv_d,
            mask=(offs_p[:, None] < BLOCK_P) & mask_d[None, :] & valid_page, other=0.0,
        ).to(tl.float32)
        scores = tl.where(valid_page, scores, -float("inf"))
        block_m = tl.max(scores, axis=0)
        new_m = tl.maximum(running_m, block_m)
        old_scale = tl.exp(running_m - new_m)
        block_w = tl.exp(scores - new_m)
        acc = acc * old_scale + tl.sum(block_w[:, None] * v, axis=0)
        running_l = running_l * old_scale + tl.sum(block_w, axis=0)
        running_m = new_m

    # Current active page, which can be shorter than P.
    for start in tl.static_range(0, 32, BLOCK_P):
        tok = start + offs_p
        mask_t = tok < active_len
        scores = tl.sum(
            tl.load(active_k_ptr + kv * sak_kv + tok[:, None] * sak_t
                    + offs_d[None, :] * sak_d,
                    mask=mask_t[:, None] & mask_d[None, :], other=0.0).to(tl.float32)
            * q[None, :], axis=1) * scale
        v = tl.load(active_v_ptr + kv * sav_kv + tok[:, None] * sav_t
                    + offs_d[None, :] * sav_d,
                    mask=mask_t[:, None] & mask_d[None, :], other=0.0).to(tl.float32)
        block_m = tl.max(tl.where(mask_t, scores, -float("inf")), axis=0)
        new_m = tl.maximum(running_m, block_m)
        old_scale = tl.exp(running_m - new_m)
        block_w = tl.where(mask_t, tl.exp(scores - new_m), 0.0)
        acc = acc * old_scale + tl.sum(block_w[:, None] * v, axis=0)
        running_l = running_l * old_scale + tl.sum(block_w, axis=0)
        running_m = new_m

    out = acc / running_l
    tl.store(out_ptr + qh * so_h + offs_d * so_d, out, mask=mask_d)


def compact_attention(q, c, coeff, owner, hist_v, active_k, active_v, scale):
    """Run the fixed-shape kernel.

    Shapes: q [32,128], c [N,8,32,80], coeff/owner [N,8,128],
    hist_v [N,8,32,128], active_k/v [8,T,128].
    """
    assert q.ndim == 2 and q.shape == (32, 128)
    assert c.ndim == 4 and c.shape[1:] == (8, 32, 80)
    assert coeff.shape == owner.shape == (c.shape[0], 8, 128)
    assert hist_v.shape == (c.shape[0], 8, 32, 128)
    assert active_k.shape == active_v.shape and active_k.shape[0] == 8
    out = torch.empty_like(q, dtype=torch.float32)
    grid = (32,)
    _compact_attention_kernel[grid](
        q, c, coeff, owner, hist_v, active_k, active_v, out,
        c.shape[0], active_k.shape[1],
        q.stride(0), c.stride(0), c.stride(1), c.stride(2), c.stride(3),
        coeff.stride(0), coeff.stride(1), hist_v.stride(0), hist_v.stride(1),
        hist_v.stride(2), hist_v.stride(3), active_k.stride(0), active_k.stride(1),
        active_k.stride(2), active_v.stride(0), active_v.stride(1), active_v.stride(2),
        out.stride(0), out.stride(1), scale,
        BLOCK_D=128, BLOCK_P=32, num_warps=4, num_stages=2,
    )
    return out


def reference_attention(q, c, coeff, owner, hist_v, active_k, active_v, scale):
    owner = owner.long()
    k = c.gather(-1, owner[:, :, None, :].expand(-1, -1, c.shape[2], -1))
    k = k * coeff[:, :, None, :]
    k = k.permute(1, 0, 2, 3).reshape(8, -1, 128)
    v = hist_v.permute(1, 0, 2, 3).reshape(8, -1, 128)
    k = torch.cat((k, active_k), dim=1).repeat_interleave(4, dim=0)
    v = torch.cat((v, active_v), dim=1).repeat_interleave(4, dim=0)
    scores = torch.einsum("hd,hnd->hn", q, k) * scale
    return torch.softmax(scores, dim=-1).unsqueeze(1).bmm(v).squeeze(1)


if __name__ == "__main__":
    torch.manual_seed(0)
    dev = "cuda"
    n = 4
    q = torch.randn((32, 128), device=dev)
    c = torch.randn((n, 8, 32, 80), device=dev)
    coeff = torch.randn((n, 8, 128), device=dev)
    owner = torch.randint(0, 80, (n, 8, 128), device=dev, dtype=torch.int64)
    hist_v = torch.randn((n, 8, 32, 128), device=dev)
    active_k = torch.randn((8, 7, 128), device=dev)
    active_v = torch.randn((8, 7, 128), device=dev)
    scale = 128 ** -0.5
    ref = reference_attention(q, c, coeff, owner, hist_v, active_k, active_v, scale)
    got = compact_attention(q, c, coeff, owner, hist_v, active_k, active_v, scale)
    torch.cuda.synchronize()
    print({"max_abs": float((ref - got).abs().max()),
           "relative": float(torch.linalg.norm(ref - got) / torch.linalg.norm(ref)),
           "finite": bool(torch.isfinite(got).all())})
