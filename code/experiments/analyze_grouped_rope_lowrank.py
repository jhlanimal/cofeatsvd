#!/usr/bin/env python
"""Measure real CoFeatKV weighted grouping followed by SVD on Post-RoPE K."""
import argparse
import csv
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", default="/mnt/data_sda1/share/Qwen3-8B")
    p.add_argument("--data_path", default="/mnt/data1/jhl1/llama_data/llm_data/wikitext-2-raw-v1")
    p.add_argument("--output_dir", default="results/rope_svd_qwen8b_wikitext/grouped_m80")
    p.add_argument("--seq_len", type=int, default=4096)
    p.add_argument("--num_sequences", type=int, default=32)
    p.add_argument("--page_size", type=int, default=128)
    p.add_argument("--m", type=int, default=80)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda:0")
    return p.parse_args()


def ci95(x):
    x = np.asarray(x, dtype=np.float64)
    return float(1.96 * x.std(ddof=1) / math.sqrt(len(x))) if len(x) > 1 else 0.0


def main():
    a = parse_args()
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    out = Path(a.output_dir); out.mkdir(parents=True, exist_ok=True)
    device = torch.device(a.device)
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ruler_resume_20260919_1652" / "code"))
    from transformers import AutoModelForCausalLM, AutoTokenizer, __version__ as tf_version
    from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb
    from analyze_rope_lowrank import load_tokens
    from eval_qwen3_multiseq_ppl_diagnostics import weighted_coalescing_page

    tokenizer = AutoTokenizer.from_pretrained(a.model_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(a.model_path, torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True, local_files_only=True, attn_implementation="eager").to(device).eval()
    base = model.model; cfg = model.config
    layers = int(getattr(cfg, "num_hidden_layers", getattr(cfg, "num_layers", len(base.layers))))
    selected = [layers // 4, layers // 2, 3 * layers // 4]
    middle = selected[1]; attn = base.layers[selected[0]].self_attn
    nk = int(getattr(attn, "num_key_value_heads", cfg.num_key_value_heads))
    nh = int(getattr(attn, "num_heads", cfg.num_attention_heads))
    hd = int(getattr(attn, "head_dim", cfg.hidden_size // cfg.num_attention_heads))
    if a.m > hd or a.seq_len % a.page_size:
        raise ValueError("Require m <= head_dim and seq_len divisible by page_size")
    tokens = load_tokens(tokenizer, a.data_path, a.seq_len * a.num_sequences)
    windows = tokens.view(a.num_sequences, a.seq_len)
    pages = a.seq_len // a.page_size
    page_ids = np.unique(np.rint(np.linspace(1, pages - 2, 8)).astype(int)).tolist()
    ranks = [r for r in [8, 16, 24, 32, 48, 64, 80] if r <= a.m]
    captures = {}

    def hook_factory(layer_id):
        def hook(module, args_, kwargs):
            h = kwargs.get("hidden_states", args_[0] if args_ else None)
            pos = kwargs.get("position_embeddings")
            if pos is None:
                pos = module.rotary_emb(h, torch.arange(h.shape[1], device=h.device)[None, :])
            q = module.q_norm(module.q_proj(h).view(h.shape[0], h.shape[1], nh, hd)).transpose(1, 2)
            k = module.k_norm(module.k_proj(h).view(h.shape[0], h.shape[1], nk, hd)).transpose(1, 2)
            q, k = apply_rotary_pos_emb(q, k, pos[0], pos[1])
            captures[layer_id] = (q.detach().float().cpu(), k.detach().float().cpu())
        return hook

    hooks = [base.layers[i].self_attn.register_forward_pre_hook(hook_factory(i), with_kwargs=True) for i in selected]
    errors = {layer: {r: [] for r in ranks} for layer in selected}
    base_group_errors = {layer: [] for layer in selected}
    r95 = {layer: [] for layer in selected}
    r99 = {layer: [] for layer in selected}
    rows = []
    with torch.inference_mode():
        for si in range(a.num_sequences):
            captures.clear()
            ids = windows[si:si + 1].to(device)
            base(input_ids=ids, use_cache=False, output_hidden_states=False, output_attentions=False, return_dict=True)
            for layer in selected:
                q, k = captures[layer]
                page_errors = {r: [] for r in ranks}; full_errors = []; r95_page = []; r99_page = []
                for pid in page_ids:
                    sl = slice(pid * a.page_size, (pid + 1) * a.page_size)
                    for kh in range(nk):
                        qh = q[0, kh * (nh // nk):(kh + 1) * (nh // nk), sl, :]
                        kp = k[0, kh, sl, :]
                        c, coeff, owner = weighted_coalescing_page(kp, qh, a.m, return_compact=True)
                        sv = torch.linalg.svdvals(c.float())
                        energy = sv.square(); total = energy.sum().clamp_min(1e-30)
                        r95_page.append(int((torch.cumsum(energy, 0) >= 0.95 * total).nonzero()[0]) + 1)
                        r99_page.append(int((torch.cumsum(energy, 0) >= 0.99 * total).nonzero()[0]) + 1)
                        full_hat = c[:, owner] * coeff[None, :]
                        denom = kp.float().square().sum().clamp_min(1e-30)
                        full_err = (kp.float() - full_hat.float()).square().sum() / denom
                        full_errors.append(float(full_err))
                        u, s, vh = torch.linalg.svd(c.float(), full_matrices=False)
                        for r in ranks:
                            cr = (u[:, :r] * s[:r]) @ vh[:r]
                            khat = cr[:, owner] * coeff[None, :]
                            page_errors[r].append(float((kp.float() - khat).square().sum() / denom))
                base_group_errors[layer].append(float(np.mean(full_errors)))
                r95[layer].append(float(np.mean(r95_page))); r99[layer].append(float(np.mean(r99_page)))
                for r in ranks:
                    errors[layer][r].append(float(np.mean(page_errors[r])))
                    rows.append([si, layer, r, errors[layer][r][-1]])
            del ids; captures.clear(); torch.cuda.empty_cache()
            print(f"sequence {si + 1}/{a.num_sequences} complete", flush=True)
    for h in hooks: h.remove()

    with (out / "grouped_svd_error_per_sequence.csv").open("w", newline="") as f:
        w = csv.writer(f); w.writerow(["sequence", "layer", "rank", "normalized_error"]); w.writerows(rows)
    with (out / "grouped_svd_summary.csv").open("w", newline="") as f:
        w = csv.writer(f); w.writerow(["layer", "rank", "mean_error", "ci95", "full_grouping_error", "r95_mean", "r99_mean"])
        for layer in selected:
            for r in ranks:
                w.writerow([layer, r, np.mean(errors[layer][r]), ci95(errors[layer][r]),
                            np.mean(base_group_errors[layer]), np.mean(r95[layer]), np.mean(r99[layer])])
    summary = {"model_path": a.model_path, "data_path": a.data_path, "transformers": tf_version,
               "num_layers": layers, "selected_layers": selected, "num_attention_heads": nh,
               "num_key_value_heads": nk, "head_dim": hd, "m": a.m, "seq_len": a.seq_len,
               "page_size": a.page_size, "num_sequences": a.num_sequences, "sampled_pages": page_ids,
               "ranks": ranks, "grouping": "weighted_coalescing_page from eval_qwen3_multiseq_ppl_diagnostics.py",
               "full_grouping_error": {str(l): float(np.mean(base_group_errors[l])) for l in selected},
               "r95_mean": {str(l): float(np.mean(r95[l])) for l in selected},
               "r99_mean": {str(l): float(np.mean(r99[l])) for l in selected},
               "error_mean": {str(l): {str(r): float(np.mean(errors[l][r])) for r in ranks} for l in selected}}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    lines = ["Grouped Post-RoPE K + SVD summary", f"m={a.m}, layers={selected}, sequences={a.num_sequences}, pages={page_ids}"]
    for r in (32, 64, 80):
        if r in ranks:
            vals = [np.mean(errors[l][r]) for l in selected]
            lines.append(f"rank {r}: mean error across layers={np.mean(vals):.6f}; per-layer=" + ", ".join(f"{v:.6f}" for v in vals))
    lines.append("full grouping error (before SVD): " + ", ".join(f"layer {l}={np.mean(base_group_errors[l]):.6f}" for l in selected))
    lines.append("95% energy rank of grouped C: " + ", ".join(f"layer {l}={np.mean(r95[l]):.3f}" for l in selected) + f"; aggregate={np.mean([np.mean(r95[l]) for l in selected]):.3f}")
    lines.append("99% energy rank of grouped C: " + ", ".join(f"layer {l}={np.mean(r99[l]):.3f}" for l in selected) + f"; aggregate={np.mean([np.mean(r99[l]) for l in selected]):.3f}")
    (out / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
