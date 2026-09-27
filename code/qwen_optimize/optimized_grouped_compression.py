"""Exact grouped low-rank compression without dense one-hot/covariance tensors."""
from __future__ import annotations

import torch

EPS = 1e-8


def weighted_pair_cost_vectorized(k_high, k_low, w_high, w_low, eps=EPS):
    nh = (k_high * k_high).sum(0)
    nl = (k_low * k_low).sum(0)
    dot = k_high.transpose(0, 1) @ k_low
    a = w_high * nh
    d = w_low * nl
    b2 = w_high[:, None] * w_low[None, :] * dot.square()
    return 0.5 * (a[:, None] + d[None, :] - torch.sqrt((a[:, None] - d[None, :]).square() + 4 * b2 + eps))


def assign_groups_vectorized(k, q_energy, m, eps=EPS):
    importance = (k.square().sum(0)) * q_energy
    order = torch.argsort(importance, descending=True)
    high, low = order[:m], order[m:]
    owner_low = (torch.argmin(weighted_pair_cost_vectorized(
        k[:, high], k[:, low], q_energy[high], q_energy[low], eps), dim=0)
                 if low.numel() else torch.empty(0, dtype=torch.long, device=k.device))
    owner = torch.empty(k.shape[1], dtype=torch.long, device=k.device)
    owner[high] = torch.arange(m, device=k.device)
    owner[low] = owner_low
    return high, low, owner


def _solve_group(k_group, weights, eps=EPS):
    """Return the same principal left vector as weighted covariance EVD."""
    x = k_group * torch.sqrt(weights.clamp_min(eps)).unsqueeze(0)
    n = x.shape[1]
    if n == 1:
        c = x[:, 0] / torch.linalg.vector_norm(x[:, 0]).clamp_min(eps)
    else:
        gram = x.transpose(0, 1) @ x
        values, vectors = torch.linalg.eigh(gram)
        lam = values[-1].clamp_min(eps)
        c = (x @ vectors[:, -1]) / torch.sqrt(lam)
        c = c / torch.linalg.vector_norm(c).clamp_min(eps)
    return c, c @ k_group


def compress_optimized_exact(k, q_page, m, return_compact=False):
    k = k.float(); q_page = q_page.float()
    q_energy = q_page.square().mean(dim=(0, 1)) if q_page.ndim == 3 else q_page.square().mean(0)
    high, low, owner = assign_groups_vectorized(k, q_energy, m)
    # Sort members once, then solve all groups of the same size as a batch.
    counts = torch.bincount(owner, minlength=m)
    sorted_members = torch.argsort(owner, stable=True)
    offsets = torch.cat((torch.zeros(1, dtype=torch.long, device=k.device), counts.cumsum(0)))
    c = torch.empty((k.shape[0], m), device=k.device, dtype=k.dtype)
    coeff = torch.empty(k.shape[1], device=k.device, dtype=k.dtype)
    for size in torch.unique(counts).tolist():
        groups = torch.where(counts == size)[0]
        if size == 1:
            members = sorted_members[offsets[groups]]
            x = k[:, members] * torch.sqrt(q_energy[members].clamp_min(EPS))[None, :]
            c_batch = x / torch.linalg.vector_norm(x, dim=0).clamp_min(EPS)
            coeff_batch = (c_batch * k[:, members]).sum(0)
            c[:, groups] = c_batch
            coeff[members] = coeff_batch
            continue
        starts = offsets[groups]
        positions = starts[:, None] + torch.arange(size, device=k.device)[None, :]
        members = sorted_members[positions]
        kg = k[:, members].permute(1, 0, 2)  # [B,P,size]
        wg = q_energy[members]
        x = kg * torch.sqrt(wg.clamp_min(EPS))[:, None, :]
        gram = x.transpose(1, 2) @ x
        values, vectors = torch.linalg.eigh(gram)
        c_batch = (x @ vectors[:, :, -1].unsqueeze(-1)).squeeze(-1) / torch.sqrt(values[:, -1].clamp_min(EPS))[:, None]
        c_batch = c_batch / torch.linalg.vector_norm(c_batch, dim=1).clamp_min(EPS)[:, None]
        coeff_batch = (c_batch[:, :, None] * kg).sum(1)
        c[:, groups] = c_batch.transpose(0, 1)
        coeff[members] = coeff_batch
    if return_compact:
        return c, coeff, owner
    a = torch.zeros((k.shape[1], m), device=k.device)
    a[torch.arange(k.shape[1], device=k.device), owner] = coeff
    return c, a


def compress_optimized_exact_batched(k_heads, q_heads, m):
    return [compress_optimized_exact(k, q, m, True) for k, q in zip(k_heads, q_heads)]
