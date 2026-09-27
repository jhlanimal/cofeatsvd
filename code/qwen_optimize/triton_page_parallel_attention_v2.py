#!/usr/bin/env python3
"""Exact page-parallel compact attention prototype.

This is an isolated Kernel A + Kernel B implementation.  It never reconstructs
historical K.  It is intentionally independent of the validated model runner
until correctness and microbenchmark gates pass.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _page_partial_kernel(
    q_ptr, c_ptr, coeff_ptr, owner_ptr, v_ptr, active_k_ptr, active_v_ptr,
    max_ptr, sum_ptr, out_ptr, n_pages, active_len, scale,
    sq_h, sc_n, sc_h, sc_p, sc_m, sco_n, sco_h, sco_d,
    sv_n, sv_h, sv_p, sv_d, sak_h, sak_t, sak_d, sav_h, sav_t, sav_d,
    sm_b, sm_h, sm_g, ss_b, ss_h, ss_g, so_b, so_h, so_g, so_d,
    BLOCK_D: tl.constexpr, BLOCK_P: tl.constexpr, M: tl.constexpr,
):
    pid = tl.program_id(0)
    blocks_per_page = 8 * 4
    block = pid // blocks_per_page
    rem = pid % blocks_per_page
    kv = rem // 4
    g = rem % 4
    qh = kv * 4 + g
    d = tl.arange(0, BLOCK_D)
    p = tl.arange(0, BLOCK_P)
    md = d < BLOCK_D
    mp = p < BLOCK_P
    q = tl.load(q_ptr + qh * sq_h + d, mask=md, other=0.0).to(tl.float32)
    is_active = block == n_pages
    valid_block = block <= n_pages
    tok_mask = mp & (p < active_len)

    if is_active:
        k = tl.load(active_k_ptr + kv * sak_h + p[:, None] * sak_t + d[None, :] * sak_d,
                    mask=mp[:, None] & md[None, :] & (p[:, None] < active_len), other=0.0).to(tl.float32)
        v = tl.load(active_v_ptr + kv * sav_h + p[:, None] * sav_t + d[None, :] * sav_d,
                    mask=mp[:, None] & md[None, :] & (p[:, None] < active_len), other=0.0).to(tl.float32)
        scores = tl.sum(q[None, :] * k, axis=1) * scale
    else:
        own = tl.load(owner_ptr + block * sco_n + kv * sco_h + d * sco_d,
                      mask=md & valid_block, other=0).to(tl.int32)
        coef = tl.load(coeff_ptr + block * sco_n + kv * sco_h + d * sco_d,
                       mask=md & valid_block, other=0.0).to(tl.float32)
        group = tl.arange(0, BLOCK_D)
        projected = tl.sum(tl.where(group[:, None] == own[None, :],
                                   (q * coef)[None, :], 0.0), axis=1)
        c = tl.load(c_ptr + block * sc_n + kv * sc_h + p[:, None] * sc_p + group[None, :] * sc_m,
                    mask=mp[:, None] & (group[None, :] < M) & valid_block, other=0.0).to(tl.float32)
        v = tl.load(v_ptr + block * sv_n + kv * sv_h + p[:, None] * sv_p + d[None, :] * sv_d,
                    mask=mp[:, None] & md[None, :] & valid_block, other=0.0).to(tl.float32)
        tok_mask = mp & valid_block
        scores = tl.sum(c * projected[None, :], axis=1) * scale
    scores = tl.where(tok_mask, scores, -float('inf'))
    local_max = tl.max(scores, axis=0)
    weights = tl.where(tok_mask, tl.exp(scores - local_max), 0.0)
    local_sum = tl.sum(weights, axis=0)
    local_out = tl.sum(weights[:, None] * v, axis=0)
    tl.store(max_ptr + block * sm_b + kv * sm_h + g * sm_g, local_max, mask=valid_block)
    tl.store(sum_ptr + block * ss_b + kv * ss_h + g * ss_g, local_sum, mask=valid_block)
    tl.store(out_ptr + block * so_b + kv * so_h + g * so_g + d * so_d,
             local_out, mask=md & valid_block)


@triton.jit
def _reduce_kernel(max_ptr, sum_ptr, out_ptr, final_ptr, n_blocks,
                   sm_b, sm_h, sm_g, ss_b, ss_h, ss_g, so_b, so_h, so_g, so_d,
                   sf_h, sf_d, BLOCK_D: tl.constexpr):
    qh = tl.program_id(0)
    kv = qh // 4
    g = qh % 4
    d = tl.arange(0, BLOCK_D)
    md = d < BLOCK_D
    m = -float('inf')
    for b in range(0, n_blocks):
        valid = b < n_blocks
        mb = tl.load(max_ptr + b * sm_b + kv * sm_h + g * sm_g, mask=valid, other=-float('inf'))
        m = tl.maximum(m, mb)
    den = 0.0
    acc = tl.zeros((BLOCK_D,), tl.float32)
    for b in range(0, n_blocks):
        valid = b < n_blocks
        mb = tl.load(max_ptr + b * sm_b + kv * sm_h + g * sm_g, mask=valid, other=-float('inf'))
        sb = tl.load(sum_ptr + b * ss_b + kv * ss_h + g * ss_g, mask=valid, other=0.0)
        ob = tl.load(out_ptr + b * so_b + kv * so_h + g * so_g + d * so_d,
                     mask=md & valid, other=0.0)
        w = tl.exp(mb - m)
        den += w * sb
        acc += w * ob
    tl.store(final_ptr + qh * sf_h + d * sf_d, acc / den, mask=md)


def compact_page_attention_v2(q, c, coeff, owner, hist_v, active_k, active_v, scale,
                              num_warps=4):
    """Exact attention over compact historical pages plus dense active page.

    Shapes: q [32,D], c [N,8,32,m], coeff/owner [N,8,D], hist_v [N,8,32,D],
    active_k/v [8,T,D]. Returns [32,D] FP32.
    """
    assert q.ndim == 2
    n, h, p, m = c.shape
    d = q.shape[1]
    assert (h, p, d) == (8, 32, 128)
    assert q.shape[0] == 32
    assert coeff.shape == owner.shape == (n, 8, d)
    assert hist_v.shape == (n, 8, 32, d)
    assert active_k.shape == active_v.shape and active_k.shape[0] == 8
    blocks = n + 1
    local_max = torch.empty((blocks, 8, 4), device=q.device, dtype=torch.float32)
    local_sum = torch.empty_like(local_max)
    local_out = torch.empty((blocks, 8, 4, d), device=q.device, dtype=torch.float32)
    out = torch.empty((32, d), device=q.device, dtype=torch.float32)
    _page_partial_kernel[(blocks * 8 * 4,)](
        q, c, coeff, owner, hist_v, active_k, active_v,
        local_max, local_sum, local_out, n, active_k.shape[1], scale,
        q.stride(0), c.stride(0), c.stride(1), c.stride(2), c.stride(3),
        coeff.stride(0), coeff.stride(1), coeff.stride(2),
        hist_v.stride(0), hist_v.stride(1), hist_v.stride(2), hist_v.stride(3),
        active_k.stride(0), active_k.stride(1), active_k.stride(2),
        active_v.stride(0), active_v.stride(1), active_v.stride(2),
        local_max.stride(0), local_max.stride(1), local_max.stride(2),
        local_sum.stride(0), local_sum.stride(1), local_sum.stride(2),
        local_out.stride(0), local_out.stride(1), local_out.stride(2), local_out.stride(3),
        BLOCK_D=128, BLOCK_P=32, M=m, num_warps=num_warps, num_stages=2)
    _reduce_kernel[(32,)](
        local_max, local_sum, local_out, out, blocks,
        local_max.stride(0), local_max.stride(1), local_max.stride(2),
        local_sum.stride(0), local_sum.stride(1), local_sum.stride(2),
        local_out.stride(0), local_out.stride(1), local_out.stride(2), local_out.stride(3),
        out.stride(0), out.stride(1), BLOCK_D=128, num_warps=num_warps, num_stages=2)
    return out


def reference_attention_v2(q, c, coeff, owner, hist_v, active_k, active_v, scale):
    owner = owner.long()
    kh = c.gather(-1, owner[:, :, None, :].expand(-1, -1, c.shape[2], -1)) * coeff[:, :, None, :]
    kh = kh.permute(1, 0, 2, 3).reshape(8, -1, 128)
    vv = hist_v.permute(1, 0, 2, 3).reshape(8, -1, 128)
    kh = torch.cat((kh, active_k), dim=1).repeat_interleave(4, dim=0)
    vv = torch.cat((vv, active_v), dim=1).repeat_interleave(4, dim=0)
    scores = torch.einsum('hd,hnd->hn', q, kh) * scale
    return torch.softmax(scores, dim=-1).unsqueeze(1).bmm(vv).squeeze(1)


@triton.jit
def _grouped_page_kernel(q_ptr, c_ptr, idx_ptr, coeff_ptr, v_ptr,
                         active_k_ptr, active_v_ptr, max_ptr, sum_ptr, out_ptr,
                         n_pages, active_len, scale, sq_h, sc_n, sc_h, sc_p, sc_m,
                         si_n, si_h, si_d, sf_n, sf_h, sf_d,
                         sv_n, sv_h, sv_p, sv_d, sak_h, sak_t, sak_d, sav_h, sav_t, sav_d,
                         sm_b, sm_h, sm_g, ss_b, ss_h, ss_g, so_b, so_h, so_g, so_d,
                         BLOCK_D: tl.constexpr, BLOCK_P: tl.constexpr, M: tl.constexpr):
    pid = tl.program_id(0); blocks_per_page = 32; block = pid // blocks_per_page
    rem = pid % blocks_per_page; kv = rem // 4; g = rem % 4; qh = kv * 4 + g
    d = tl.arange(0, BLOCK_D); p = tl.arange(0, BLOCK_P)
    md = d < BLOCK_D; mp = p < BLOCK_P; valid_block = block <= n_pages
    q = tl.load(q_ptr + qh * sq_h + d, mask=md, other=0.0).to(tl.float32)
    scores = tl.zeros((BLOCK_P,), tl.float32)
    k = tl.zeros((BLOCK_P, BLOCK_D), tl.float32)
    vv = tl.zeros((BLOCK_P, BLOCK_D), tl.float32)
    tok_mask = mp & (p < active_len)
    if block == n_pages:
        k = tl.load(active_k_ptr + kv * sak_h + p[:,None] * sak_t + d[None,:] * sak_d,
                    mask=mp[:,None] & md[None,:] & (p[:,None] < active_len), other=0.0).to(tl.float32)
        vv = tl.load(active_v_ptr + kv * sav_h + p[:,None] * sav_t + d[None,:] * sav_d,
                     mask=mp[:,None] & md[None,:] & (p[:,None] < active_len), other=0.0).to(tl.float32)
        tok_mask = mp & (p < active_len)
    else:
        idx = tl.load(idx_ptr + block * si_n + kv * si_h + d * si_d, mask=md, other=0).to(tl.int32)
        cf = tl.load(coeff_ptr + block * sf_n + kv * sf_h + d * sf_d, mask=md, other=0.0).to(tl.float32)
        y = tl.zeros((128,), tl.float32)
        group_ids = tl.arange(0, 128)
        for group in tl.static_range(0, 128):
            y_group = tl.sum(tl.where((idx == group) & md, q * cf, 0.0), axis=0)
            y = tl.where(group_ids == group, y_group, y)
        gm = tl.arange(0, 128)
        cvals = tl.load(c_ptr + block * sc_n + kv * sc_h + p[:,None] * sc_p + gm[None,:] * sc_m,
                        mask=mp[:,None] & (gm[None,:] < M), other=0.0).to(tl.float32)
        scores = tl.sum(cvals * y[None,:], axis=1) * scale
        vv = tl.load(v_ptr + block * sv_n + kv * sv_h + p[:,None] * sv_p + d[None,:] * sv_d,
                     mask=mp[:,None] & md[None,:], other=0.0).to(tl.float32)
        tok_mask = mp
    if block == n_pages:
        scores = tl.sum(q[None,:] * k, axis=1) * scale
    scores = tl.where(tok_mask & valid_block, scores, -float('inf'))
    lm = tl.max(scores, axis=0); ww = tl.where(tok_mask & valid_block, tl.exp(scores-lm), 0.0)
    ls = tl.sum(ww, axis=0); lo = tl.sum(ww[:,None] * vv, axis=0)
    tl.store(max_ptr + block * sm_b + kv * sm_h + g * sm_g, lm, mask=valid_block)
    tl.store(sum_ptr + block * ss_b + kv * ss_h + g * ss_g, ls, mask=valid_block)
    tl.store(out_ptr + block * so_b + kv * so_h + g * so_g + d * so_d, lo, mask=md & valid_block)


def compact_page_attention_grouped_v2(q, c, coeff, owner, hist_v, active_k, active_v, scale, num_warps=4):
    n, h, p, m = c.shape; d = q.shape[1]; assert (h,p,d)==(8,32,128)
    order = torch.argsort(owner, dim=-1)
    idx = torch.gather(owner, -1, order).contiguous()
    grouped_coeff = torch.gather(coeff, -1, order).contiguous()
    blocks=n+1; lm=torch.empty((blocks,8,4),device=q.device,dtype=torch.float32); ls=torch.empty_like(lm); lo=torch.empty((blocks,8,4,d),device=q.device,dtype=torch.float32); out=torch.empty((32,d),device=q.device,dtype=torch.float32)
    _grouped_page_kernel[(blocks*32,)](q,c,idx,grouped_coeff,hist_v,active_k,active_v,lm,ls,lo,n,active_k.shape[1],scale,q.stride(0),c.stride(0),c.stride(1),c.stride(2),c.stride(3),idx.stride(0),idx.stride(1),idx.stride(2),grouped_coeff.stride(0),grouped_coeff.stride(1),grouped_coeff.stride(2),hist_v.stride(0),hist_v.stride(1),hist_v.stride(2),hist_v.stride(3),active_k.stride(0),active_k.stride(1),active_k.stride(2),active_v.stride(0),active_v.stride(1),active_v.stride(2),lm.stride(0),lm.stride(1),lm.stride(2),ls.stride(0),ls.stride(1),ls.stride(2),lo.stride(0),lo.stride(1),lo.stride(2),lo.stride(3),BLOCK_D=128,BLOCK_P=32,M=80,num_warps=num_warps,num_stages=2)
    _reduce_kernel[(32,)](lm,ls,lo,out,blocks,lm.stride(0),lm.stride(1),lm.stride(2),ls.stride(0),ls.stride(1),ls.stride(2),lo.stride(0),lo.stride(1),lo.stride(2),lo.stride(3),out.stride(0),out.stride(1),BLOCK_D=128,num_warps=num_warps,num_stages=2)
    return out
