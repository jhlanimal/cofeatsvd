#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, time
from pathlib import Path
import sys
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))
from eval_qwen3_multiseq_ppl_diagnostics import weighted_coalescing_page
from triton_compact_decode_attention import reconstruct_k_triton


def reconstruct_batched(c, coeff, owner, workspace):
    """Exact compact reconstruction into [H,N*P,D] workspace."""
    n, h, p, m = c.shape
    d = coeff.shape[-1]
    khat = c.gather(-1, owner.long()[:, :, None, :].expand(n, h, p, d))
    khat = khat.mul(coeff[:, :, None, :])
    workspace[:, : n * p].copy_(khat.permute(1, 0, 2, 3).reshape(h, n * p, d))
    return workspace[:, : n * p]


def reconstruct_loop(c, coeff, owner):
    parts = []
    for kh in range(c.shape[1]):
        pages = []
        for page in range(c.shape[0]):
            pages.append(c[page, kh][:, owner[page, kh].long()] * coeff[page, kh][None, :])
        parts.append(torch.cat(pages, dim=0))
    return torch.stack(parts, dim=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--m', type=int, default=80)
    ap.add_argument('--repeats', type=int, default=100)
    ap.add_argument('--output', default='qwen/optimize/results/reconstruct_optimization/reconstruction_microbench.csv')
    a = ap.parse_args()
    rows = []
    for tokens in (128, 256, 384, 512, 1024):
        n = tokens // 32
        c = torch.randn((n, 8, 32, a.m), device=a.device, dtype=torch.float32)
        coeff = torch.randn((n, 8, 128), device=a.device, dtype=torch.float32)
        owner = torch.randint(0, a.m, (n, 8, 128), device=a.device, dtype=torch.int64)
        workspace = torch.empty((8, tokens, 128), device=a.device, dtype=torch.float32)
        ref = reconstruct_loop(c, coeff, owner)
        got = reconstruct_batched(c, coeff, owner, workspace).clone()
        err = float((ref - got).abs().max().item())
        triton_workspace = torch.empty_like(workspace)
        got_triton = reconstruct_k_triton(c, coeff, owner, triton_workspace).clone()
        torch.cuda.synchronize()
        triton_err = float((ref - got_triton).abs().max().item())
        for _ in range(10): reconstruct_loop(c, coeff, owner)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(a.repeats): reconstruct_loop(c, coeff, owner)
        torch.cuda.synchronize(); loop_us = (time.perf_counter() - t0) * 1e6 / a.repeats
        for _ in range(10): reconstruct_batched(c, coeff, owner, workspace)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(a.repeats): reconstruct_batched(c, coeff, owner, workspace)
        torch.cuda.synchronize(); batch_us = (time.perf_counter() - t0) * 1e6 / a.repeats
        for _ in range(10): reconstruct_k_triton(c, coeff, owner, triton_workspace)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(a.repeats): reconstruct_k_triton(c, coeff, owner, triton_workspace)
        torch.cuda.synchronize(); triton_us = (time.perf_counter() - t0) * 1e6 / a.repeats
        rows.append({'historical_tokens': tokens, 'loop_us_per_layer': loop_us,
                     'batched_us_per_layer': batch_us, 'speedup': loop_us / batch_us,
                     'triton_us_per_layer': triton_us, 'triton_vs_batched_speedup': batch_us / triton_us,
                     'max_abs_error': err, 'triton_max_abs_error': triton_err, 'dtype_c': str(c.dtype),
                     'dtype_coeff': str(coeff.dtype), 'dtype_owner': str(owner.dtype)})
        print(rows[-1], flush=True)
    out = Path(a.output); out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)


if __name__ == '__main__': main()
