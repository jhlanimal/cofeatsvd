#!/usr/bin/env python
"""Independent Qwen pre/post-RoPE Key low-rank analysis."""
import argparse
import csv
import json
import math
import random
from pathlib import Path

import numpy as np
import torch


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", default="/mnt/data_sda1/share/Qwen3-8B")
    p.add_argument("--data_path", default="/mnt/data1/jhl1/llama_data/llm_data/wikitext-2-raw-v1")
    p.add_argument("--output_dir", default="results/rope_svd_qwen8b_wikitext")
    p.add_argument("--seq_len", type=int, default=4096)
    p.add_argument("--num_sequences", type=int, default=32)
    p.add_argument("--page_size", type=int, default=128)
    p.add_argument("--num_sampled_pages", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda:0")
    return p.parse_args()


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_tokens(tokenizer, data_path, needed):
    import pyarrow.parquet as pq
    path = Path(data_path)
    files = sorted(path.glob("*.parquet")) if path.is_dir() else [path]
    if not files:
        raise FileNotFoundError(f"No parquet files found under {data_path}")
    chunks, count = [], 0
    for file in files:
        table = pq.read_table(file, columns=["text"])
        for text in table.column("text").to_pylist():
            if text is None:
                continue
            ids = tokenizer(str(text), add_special_tokens=False, return_tensors="pt")["input_ids"][0]
            if ids.numel():
                chunks.append(ids)
                count += ids.numel()
            if count >= needed:
                return torch.cat(chunks)[:needed]
    raise RuntimeError(f"WikiText has only {count} tokens, need {needed}")


def pearson_abs(x):
    x = x.float()
    x = x - x.mean(dim=0, keepdim=True)
    norm = torch.linalg.vector_norm(x, dim=0)
    return ((x.T @ x) / (norm[:, None] * norm[None, :] + 1e-12)).abs()


def ci95(values):
    values = np.asarray(values, dtype=np.float64)
    return float(1.96 * values.std(ddof=1) / math.sqrt(len(values))) if len(values) > 1 else 0.0


def main():
    a = parse_args()
    seed_all(a.seed)
    out = Path(a.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; run with CUDA_VISIBLE_DEVICES=0")
    device = torch.device(a.device)

    from transformers import AutoModelForCausalLM, AutoTokenizer, __version__ as tf_version
    from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb

    tokenizer = AutoTokenizer.from_pretrained(a.model_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        a.model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        local_files_only=True,
        attn_implementation="eager",
    ).to(device).eval()
    base = model.model
    cfg = model.config
    num_layers = int(getattr(cfg, "num_hidden_layers", getattr(cfg, "num_layers", len(base.layers))))
    selected = [num_layers // 4, num_layers // 2, 3 * num_layers // 4]
    middle = selected[1]
    attn0 = base.layers[selected[0]].self_attn
    kv_heads = int(getattr(attn0, "num_key_value_heads", cfg.num_key_value_heads))
    head_dim = int(getattr(attn0, "head_dim", cfg.hidden_size // cfg.num_attention_heads))
    if a.seq_len % a.page_size:
        raise ValueError("seq_len must be divisible by page_size")

    tokens = load_tokens(tokenizer, a.data_path, a.seq_len * a.num_sequences)
    windows = tokens.view(a.num_sequences, a.seq_len)
    num_pages = a.seq_len // a.page_size
    page_ids = np.unique(np.rint(np.linspace(1, num_pages - 2, a.num_sampled_pages)).astype(int)).tolist()
    ranks = [r for r in [8, 16, 24, 32, 48, 64, 80, 96, 112, 128] if r <= min(a.page_size, head_dim)]
    print(json.dumps({"model": a.model_path, "transformers": tf_version, "num_layers": num_layers,
                      "num_kv_heads": kv_heads, "head_dim": head_dim, "selected_layers": selected,
                      "seq_len": a.seq_len, "page_size": a.page_size, "sampled_pages": page_ids}, indent=2))

    captures = {}

    def make_hook(layer_id):
        def hook(module, args_, kwargs):
            hidden = kwargs.get("hidden_states", args_[0] if args_ else None)
            position_embeddings = kwargs.get("position_embeddings")
            if position_embeddings is None:
                position_ids = torch.arange(hidden.shape[1], device=hidden.device)[None, :]
                position_embeddings = module.rotary_emb(hidden, position_ids)
            # Exact Qwen3 order: k_proj -> KV reshape -> k_norm -> RoPE.
            key = module.k_proj(hidden).view(hidden.shape[0], hidden.shape[1], kv_heads, head_dim).transpose(1, 2)
            key = module.k_norm(key)
            pre = key.detach().float().cpu()
            post = apply_rotary_pos_emb(key, key, position_embeddings[0], position_embeddings[1])[1].detach().float().cpu()
            captures[layer_id] = (pre, post)
        return hook

    hooks = [base.layers[i].self_attn.register_forward_pre_hook(make_hook(i), with_kwargs=True) for i in selected]
    per_seq = {l: {s: np.zeros((a.num_sequences, len(ranks))) for s in ("pre", "post")} for l in selected}
    effective = {l: {s: np.zeros((a.num_sequences, 3)) for s in ("pre", "post")} for l in selected}
    sv_store = {l: {s: [] for s in ("pre", "post")} for l in selected}
    norm_values = []
    corr_sum = torch.zeros((head_dim, head_dim), dtype=torch.float64)
    corr_count = 0

    with torch.inference_mode():
        for seq_id in range(a.num_sequences):
            captures.clear()
            input_ids = windows[seq_id:seq_id + 1].to(device)
            base(input_ids=input_ids, use_cache=False, output_hidden_states=False,
                 output_attentions=False, return_dict=True)
            if set(captures) != set(selected):
                raise RuntimeError(f"Missing captures: {set(selected) - set(captures)}")
            for layer_id in selected:
                pre, post = captures[layer_id]
                pre_norm = torch.linalg.vector_norm(pre, dim=-1)
                post_norm = torch.linalg.vector_norm(post, dim=-1)
                norm_values.extend(((pre_norm - post_norm).abs() / (pre_norm + 1e-12)).reshape(-1).numpy().tolist())
                for state, tensor in (("pre", pre), ("post", post)):
                    errors, ranks_eff = [], []
                    for page_id in page_ids:
                        page = slice(page_id * a.page_size, (page_id + 1) * a.page_size)
                        for head_id in range(kv_heads):
                            sv = torch.linalg.svdvals(tensor[0, head_id, page, :].contiguous())
                            energy = sv.square()
                            total_energy = energy.sum().clamp_min(1e-30)
                            errors.append([(energy[r:].sum() / total_energy).item() for r in ranks])
                            cumulative = torch.cumsum(energy, dim=0)
                            ranks_eff.append([int((cumulative >= threshold * total_energy).nonzero()[0]) + 1
                                              for threshold in (0.90, 0.95, 0.99)])
                            sv_store[layer_id][state].append(sv.numpy())
                    per_seq[layer_id][state][seq_id] = np.asarray(errors).mean(axis=0)
                    effective[layer_id][state][seq_id] = np.asarray(ranks_eff, dtype=np.float64).mean(axis=0)
                if layer_id == middle:
                    for page_id in page_ids:
                        page = slice(page_id * a.page_size, (page_id + 1) * a.page_size)
                        for head_id in range(kv_heads):
                            corr_sum += pearson_abs(post[0, head_id, page, :]).double()
                            corr_count += 1
            del input_ids
            captures.clear()
            torch.cuda.empty_cache()
            print(f"sequence {seq_id + 1}/{a.num_sequences} complete", flush=True)
    for hook in hooks:
        hook.remove()

    norm_values = np.asarray(norm_values)
    norm = {"mean": float(norm_values.mean()), "median": float(np.median(norm_values)), "max": float(norm_values.max())}
    # BF16 rotary arithmetic introduces small relative norm round-off; values above
    # 1% indicate a capture/order bug rather than normal numerical noise.
    if norm["max"] > 1e-2:
        raise RuntimeError(f"RoPE norm preservation failed: {norm}")

    corr = (corr_sum / max(corr_count, 1)).numpy().astype(np.float32)
    np.fill_diagonal(corr, 0.0)
    off_diag = corr[~np.eye(head_dim, dtype=bool)]
    max_similarity = corr.max(axis=1)
    feature_summary = {
        "metric": "absolute Pearson correlation", "middle_layer": middle, "num_matrices": corr_count,
        "mean_off_diagonal": float(off_diag.mean()), "median_off_diagonal": float(np.median(off_diag)),
        "max_similarity_mean": float(max_similarity.mean()), "max_similarity_median": float(np.median(max_similarity)),
        "max_similarity_p25": float(np.percentile(max_similarity, 25)),
        "max_similarity_p75": float(np.percentile(max_similarity, 75)),
        "max_similarity_p90": float(np.percentile(max_similarity, 90)),
    }
    np.save(out / "feature_similarity.npy", corr)
    (out / "feature_similarity_summary.json").write_text(json.dumps(feature_summary, indent=2))

    write_csvs(out, selected, ranks, per_seq, effective)
    np.savez_compressed(out / "singular_value_summary.npz",
                        **{f"layer{l}_{s}": np.asarray(sv_store[l][s]) for l in selected for s in ("pre", "post")})
    make_figures(out, selected, ranks, per_seq, corr, middle, head_dim)
    write_summary(out, a, tf_version, num_layers, selected, kv_heads, head_dim, page_ids,
                  ranks, norm, feature_summary, per_seq, effective)


def write_csvs(out, selected, ranks, per_seq, effective):
    with (out / "svd_error_per_sequence.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sequence", "layer", "rope_state", "rank", "mean_error"])
        for layer in selected:
            for state in ("pre", "post"):
                for seq_id in range(per_seq[layer][state].shape[0]):
                    for rank_id, rank in enumerate(ranks):
                        writer.writerow([seq_id, layer, state, rank, per_seq[layer][state][seq_id, rank_id]])
    with (out / "svd_error_by_rank.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["layer", "rank", "pre_mean", "pre_ci95", "post_mean", "post_ci95"])
        for layer in selected:
            for rank_id, rank in enumerate(ranks):
                pre = per_seq[layer]["pre"][:, rank_id]
                post = per_seq[layer]["post"][:, rank_id]
                writer.writerow([layer, rank, pre.mean(), ci95(pre), post.mean(), ci95(post)])
    with (out / "effective_rank.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["layer", "rope_state", "r90_mean", "r95_mean", "r99_mean"])
        for layer in selected:
            for state in ("pre", "post"):
                writer.writerow([layer, state, *effective[layer][state].mean(axis=0)])


def make_figures(out, selected, ranks, per_seq, corr, middle, head_dim):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    aggregate = {state: np.stack([per_seq[layer][state] for layer in selected], axis=1).mean(axis=1)
                 for state in ("pre", "post")}
    plt.rcParams.update({"font.size": 9, "axes.linewidth": 0.8, "pdf.fonttype": 42})
    fig, ax = plt.subplots(figsize=(3.35, 2.45))
    for state, color, label in (("pre", "#2166ac", "Pre-RoPE"), ("post", "#b2182b", "Post-RoPE")):
        mean = aggregate[state].mean(axis=0)
        band = np.asarray([ci95(aggregate[state][:, i]) for i in range(len(ranks))])
        ax.plot(ranks, mean, marker="o", ms=3, lw=1.5, color=color, label=label)
        ax.fill_between(ranks, np.maximum(0, mean - band), np.minimum(1, mean + band), color=color, alpha=0.16, linewidth=0)
    ax.set(xlabel="SVD Rank", ylabel="Normalized Reconstruction Error", ylim=(0, 1))
    ax.grid(alpha=0.2); ax.legend(frameon=False); fig.tight_layout()
    fig.savefig(out / "fig1a_pre_post_rope_svd.pdf", bbox_inches="tight")
    fig.savefig(out / "fig1a_pre_post_rope_svd.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(3.0, 2.65))
    image = ax.imshow(np.where(np.eye(head_dim, dtype=bool), np.nan, corr), vmin=0, vmax=1,
                      cmap="magma", interpolation="nearest")
    ax.set(xlabel="Feature dimension", ylabel="Feature dimension", title=f"Post-RoPE, layer {middle}")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="Absolute Pearson Correlation")
    fig.tight_layout()
    fig.savefig(out / "fig1b_post_rope_feature_correlation.pdf", bbox_inches="tight")
    fig.savefig(out / "fig1b_post_rope_feature_correlation.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_summary(out, args_, tf_version, num_layers, selected, kv_heads, head_dim, page_ids,
                  ranks, norm, feature_summary, per_seq, effective):
    aggregate = {state: np.stack([per_seq[layer][state] for layer in selected], axis=1).mean(axis=1).mean(axis=0)
                 for state in ("pre", "post")}
    lines = ["RoPE low-rank analysis summary", f"model: {args_.model_path}", f"transformers: {tf_version}",
             f"layers={num_layers}, selected={selected}, kv_heads={kv_heads}, head_dim={head_dim}",
             f"sequences={args_.num_sequences}, seq_len={args_.seq_len}, page_size={args_.page_size}, sampled_pages={page_ids}",
             f"norm_relative_error: {json.dumps(norm)}", "", "Question 1: Post-RoPE vs Pre-RoPE reconstruction error"]
    for rank in (32, 64, 96):
        if rank in ranks:
            i = ranks.index(rank)
            pre, post = aggregate["pre"][i], aggregate["post"][i]
            lines.append(f"rank {rank}: Pre={pre:.6f}, Post={post:.6f}, relative increase={(post-pre)/max(abs(pre), 1e-12):.4f}")
    r95_pre = np.mean([effective[layer]["pre"][:, 1].mean() for layer in selected])
    r95_post = np.mean([effective[layer]["post"][:, 1].mean() for layer in selected])
    lines += [f"95% explained energy: Pre-RoPE mean rank={r95_pre:.3f}, Post-RoPE mean rank={r95_post:.3f}",
              "", "Question 2: Post-RoPE feature similarity", json.dumps(feature_summary, indent=2)]
    if 64 in ranks:
        i = ranks.index(64)
        lines.append("Layer trends (Post-Pre error at rank 64): " + "; ".join(
            f"layer {layer}: {per_seq[layer]['post'][:, i].mean() - per_seq[layer]['pre'][:, i].mean():.6f}" for layer in selected))
    counter = [(layer, rank) for layer in selected for i, rank in enumerate(ranks)
               if per_seq[layer]["post"][:, i].mean() <= per_seq[layer]["pre"][:, i].mean()]
    lines.append(f"Counterexample layer/rank pairs where Post <= Pre: {counter if counter else 'none'}")
    better = int(np.sum(aggregate["post"] > aggregate["pre"]))
    lines.append(f"Observation 1: supported for {better}/{len(ranks)} tested ranks based on aggregate error.")
    lines.append(f"Observation 2: aggregate mean off-diagonal absolute Pearson similarity={feature_summary['mean_off_diagonal']:.6f}; the result is reported without selecting a page/head.")
    lines.append("Experiment C skipped: no standalone final CoFeatKV matched-budget grouping analysis API was used; no substitute grouping was invented.")
    (out / "summary.txt").write_text("\n".join(lines) + "\n")
    config = {"model_path": args_.model_path, "data_path": args_.data_path, "transformers": tf_version,
              "torch": torch.__version__, "device": str(args_.device), "dtype": "BF16 inference / FP32 statistics",
              "num_layers": num_layers, "num_key_value_heads": kv_heads, "head_dim": head_dim,
              "selected_layers": selected, "sequence_length": args_.seq_len, "page_size": args_.page_size,
              "num_sequences": args_.num_sequences, "sampled_page_ids": page_ids, "ranks": ranks,
              "seed": args_.seed, "norm_relative_error": norm, "experiment_c": "skipped"}
    (out / "config.json").write_text(json.dumps(config, indent=2))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
