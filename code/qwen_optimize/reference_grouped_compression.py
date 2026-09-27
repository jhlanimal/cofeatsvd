#!/usr/bin/env python3
"""Same-page comparison of PuzzleKV ranks and weighted feature coalescing."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from analyze_attention_scale_similarity import capture_qk
from analyze_post_rope_k_pages import DEFAULT_DATA, DEFAULT_MODEL, load_tokens


M_VALUES = (112, 96, 80, 64)
PUZZLE_RANKS = (4, 8, 12, 16, 20, 24)
SEED = 42
EPS = 1e-8


def weighted_pair_cost(k_high: torch.Tensor, k_low: torch.Tensor,
                       w_high: torch.Tensor, w_low: torch.Tensor,
                       eps: float = EPS) -> torch.Tensor:
    nh = (k_high * k_high).sum(dim=0)
    nl = (k_low * k_low).sum(dim=0)
    dot = k_high.T @ k_low
    a = w_high * nh
    d = w_low * nl
    b2 = w_high[:, None] * w_low[None, :] * dot.pow(2)
    return 0.5 * (a[:, None] + d[None, :] - torch.sqrt(
        (a[:, None] - d[None, :]).pow(2) + 4.0 * b2 + eps
    ))


def build_weighted_representative(k_group: torch.Tensor,
                                  weights: torch.Tensor,
                                  eps: float = EPS):
    x = k_group * torch.sqrt(weights.clamp_min(eps)).unsqueeze(0)
    u, _, _ = torch.linalg.svd(x, full_matrices=False)
    c = u[:, 0]
    a = c @ k_group
    return c, a


def assign_groups(k: torch.Tensor, importance: torch.Tensor,
                  q_energy: torch.Tensor, m: int, eps: float = EPS):
    order = torch.argsort(importance, descending=True)
    high, low = order[:m], order[m:]
    weights_high = q_energy[high]
    weights_low = q_energy[low]
    costs = weighted_pair_cost(k[:, high], k[:, low], weights_high, weights_low, eps)
    return high, low, torch.argmin(costs, dim=0)


def coalesced_scores(k: torch.Tensor, q_future: torch.Tensor,
                     high: torch.Tensor, low: torch.Tensor,
                     owners: torch.Tensor, q_energy: torch.Tensor):
    m = high.numel()
    q_agg = torch.zeros((q_future.shape[0], m), device=k.device, dtype=k.dtype)
    c_cols = torch.zeros((k.shape[0], m), device=k.device, dtype=k.dtype)
    for group_index in range(m):
        members = torch.cat((high[group_index:group_index + 1],
                             low[owners == group_index]))
        c, coeff = build_weighted_representative(k[:, members], q_energy[members])
        c_cols[:, group_index] = c
        q_agg[:, group_index] = q_future[:, members] @ coeff
    return q_agg @ c_cols.T


def puzzle_scores(k: torch.Tensor, q_future: torch.Tensor,
                  vectors: torch.Tensor, rank: int):
    use_rank = min(rank, k.shape[0], k.shape[1])
    left = vectors[:, -use_rank:]
    r = left.T @ k
    return (q_future @ r.T) @ left.T


def relative_error(full: torch.Tensor, approx: torch.Tensor) -> float:
    return float((torch.linalg.norm(full - approx) /
                  torch.linalg.norm(full).clamp_min(EPS)).item())


def factorization(k: torch.Tensor):
    gram = k @ k.T
    _, vectors = torch.linalg.eigh(gram)
    return vectors


def page_points(k_np: np.ndarray, q_page_np: np.ndarray,
                q_future_np: np.ndarray, ranks: tuple[int, ...],
                sanity: bool, device: str) -> list[dict[str, Any]]:
    k = torch.as_tensor(k_np, device=device, dtype=torch.float32)
    q_page = torch.as_tensor(q_page_np, device=device, dtype=torch.float32)
    q_future = torch.as_tensor(q_future_np, device=device, dtype=torch.float32)
    p, d = k.shape
    full = q_future @ k.T
    vectors = factorization(k)
    q_energy = (q_page * q_page).mean(dim=0)
    importance = (k * k).sum(dim=0) * q_energy
    rows: list[dict[str, Any]] = []

    for rank in ranks:
        error = relative_error(full, puzzle_scores(k, q_future, vectors, rank))
        storage = p * rank + rank * d
        compute = rank * (p + d)
        rows.append({"method": "PuzzleKV", "rank": rank, "m": "",
                     "storage_elements": storage, "storage_ratio": storage / (p * d),
                     "theoretical_compute": compute, "compute_ratio": compute / (p * d),
                     "attention_error": error})

    for m in M_VALUES:
        high, low, owners = assign_groups(k, importance, q_energy, m)
        score = coalesced_scores(k, q_future, high, low, owners, q_energy)
        error = relative_error(full, score)
        storage = p * m + d
        compute = d + p * m
        rows.append({"method": "WeightedCoalescing", "rank": "", "m": m,
                     "storage_elements": storage, "storage_ratio": storage / (p * d),
                     "storage_vs_puzzlekv_r16": storage / (p * 16 + 16 * d),
                     "theoretical_compute": compute, "compute_ratio": compute / (p * d),
                     "compute_vs_puzzlekv_r16": compute / (16 * (p + d)),
                     "attention_error": error})

        if sanity:
            with torch.no_grad():
                q_agg = torch.zeros((q_future.shape[0], m), device=device)
                c_cols = torch.zeros((p, m), device=device)
                a_map = torch.zeros((d, m), device=device)
                for group_index in range(m):
                    members = torch.cat((high[group_index:group_index + 1],
                                         low[owners == group_index]))
                    c, coeff = build_weighted_representative(k[:, members], q_energy[members])
                    c_cols[:, group_index] = c
                    q_agg[:, group_index] = q_future[:, members] @ coeff
                    a_map[members, group_index] = coeff
                explicit = q_future @ (c_cols @ a_map.T).T
                reconstruction_free = q_agg @ c_cols.T
                diff = float(torch.max(torch.abs(explicit - reconstruction_free)).item())
                if diff >= 1e-4:
                    raise RuntimeError(f"coalescing sanity check failed: max_abs_diff={diff}")
    return rows


def stats(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    keys = ("storage_elements", "storage_ratio", "theoretical_compute", "compute_ratio",
            "attention_error")
    return {key: {name: float(fn([row[key] for row in rows]))
                  for name, fn in (("mean", np.mean), ("median", np.median),
                                   ("q25", lambda x: np.percentile(x, 25)),
                                   ("q75", lambda x: np.percentile(x, 75)))}
            for key in keys}


def closest_rank(ours_storage: float, p: int, d: int) -> tuple[float, list[int]]:
    req = ours_storage / (p + d)
    candidates = sorted(set([max(1, int(np.floor(req))), max(1, int(np.ceil(req)))]))
    return req, candidates


def main() -> None:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model-path", default=DEFAULT_MODEL)
    parser.add_argument("--data-path", default=DEFAULT_DATA)
    parser.add_argument("--layers", default="0,8,16,24,30")
    parser.add_argument("--tokens", type=int, default=2048)
    parser.add_argument("--page-size", type=int, default=32)
    parser.add_argument("--pages-per-layer", type=int, default=200)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", default="results_weighted_vs_puzzlekv")
    args = parser.parse_args()
    layers = [int(x) for x in args.layers.split(",") if x.strip()]
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True, use_fast=True)
    ids = load_tokens(tokenizer, args.data_path, args.tokens)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, local_files_only=True, dtype=torch.float16,
        low_cpu_mem_usage=True, use_safetensors=True,
    ).eval().to(args.device)
    queries, keys = capture_qk(model, ids, layers, args.device)

    rank_set = set(PUZZLE_RANKS)
    for m in M_VALUES:
        storage = args.page_size * m + 128
        _, eq = closest_rank(storage, args.page_size, 128)
        rank_set.update(eq)
    ranks = tuple(sorted(rank_set))
    all_rows: list[dict[str, Any]] = []
    layer_summaries: list[dict[str, Any]] = []
    sanity_left = 20

    for layer in layers:
        k_all, q_all = keys[layer], queries[layer]
        kv_heads, token_count, _ = k_all.shape
        q_per_kv = q_all.shape[0] // kv_heads
        population = np.asarray([(h, p) for h in range(kv_heads)
                                 for p in range(max(0, token_count // args.page_size - 1))], dtype=np.int64)
        chosen = population[rng.choice(len(population), size=min(args.pages_per_layer, len(population)), replace=False)]
        page_records: list[dict[str, Any]] = []
        for head, page in chosen:
            start = int(page) * args.page_size
            end = start + args.page_size
            q_head = q_all[int(head) * q_per_kv]
            future = q_head[end:min(token_count, end + 128)]
            if future.shape[0] == 0:
                continue
            points = page_points(k_all[int(head), start:end], q_head[start:end], future,
                                 ranks, sanity_left > 0, args.device)
            sanity_left -= 1
            for point in points:
                point.update(layer=layer, head=int(head), page=int(page))
                all_rows.append(point)
                page_records.append(point)
        # Full K is the zero-error reference point on every sampled page.
        for head, page in chosen:
            page_records.append({"method": "Full", "rank": "", "m": "",
                                 "storage_elements": args.page_size * 128,
                                 "storage_ratio": 1.0,
                                 "theoretical_compute": args.page_size * 128,
                                 "compute_ratio": 1.0,
                                 "attention_error": 0.0,
                                 "layer": layer, "head": int(head), "page": int(page)})
            all_rows.append(page_records[-1])
        summaries = {}
        for method in ("Full", "PuzzleKV", "WeightedCoalescing"):
            values = [r for r in page_records if r["method"] == method]
            keys_for_method = sorted({r.get("rank") if method == "PuzzleKV" else r.get("m") for r in values})
            summaries[method] = {str(key): stats([r for r in values if (r.get("rank") if method == "PuzzleKV" else r.get("m")) == key])
                                 for key in keys_for_method}
        layer_summaries.append({"layer": layer, "sampled_pages": len(chosen),
                                "summary": summaries})
        print(f"completed layer {layer}: {len(chosen)} pages")

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in all_rows for key in row})
    with (out / "pareto_per_page.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(row for row in all_rows)

    layer_csv_rows: list[dict[str, Any]] = []
    for item in layer_summaries:
        layer = item["layer"]
        for method, method_values in item["summary"].items():
            for config, value in method_values.items():
                row = {"layer": layer, "method": method, "rank_or_m": config}
                for metric, aggregate in value.items():
                    for statistic, number in aggregate.items():
                        row[f"{metric}_{statistic}"] = number
                layer_csv_rows.append(row)
    with (out / "pareto_layer_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({key for row in layer_csv_rows for key in row}))
        writer.writeheader()
        writer.writerows(layer_csv_rows)

    by_layer: dict[int, dict[tuple[str, int], dict[str, float]]] = {}
    for layer in layers:
        by_layer[layer] = {}
        layer_rows = [r for r in all_rows if r["layer"] == layer]
        for method in ("PuzzleKV", "WeightedCoalescing"):
            configs = sorted({int(r["rank"] if method == "PuzzleKV" else r["m"])
                              for r in layer_rows if r["method"] == method})
            for config in configs:
                subset = [r for r in layer_rows if r["method"] == method and
                          int(r["rank"] if method == "PuzzleKV" else r["m"]) == config]
                by_layer[layer][(method, config)] = {
                    "mean_error": float(np.mean([r["attention_error"] for r in subset])),
                    "median_error": float(np.median([r["attention_error"] for r in subset])),
                }

    same_storage: list[dict[str, Any]] = []
    for layer in layers:
        layer_rows = [r for r in all_rows if r["layer"] == layer]
        for m in M_VALUES:
            ours = [r for r in layer_rows if r["method"] == "WeightedCoalescing" and r["m"] == m]
            storage = args.page_size * m + 128
            req, candidates = closest_rank(storage, args.page_size, 128)
            candidate_rows = []
            for rank in candidates:
                point = [r for r in layer_rows if r["method"] == "PuzzleKV" and r["rank"] == rank]
                if point:
                    candidate_rows.append({"rank": rank, "storage_elements": rank * (args.page_size + 128),
                                           "mean_error": float(np.mean([r["attention_error"] for r in point]))})
            ours_error = float(np.mean([r["attention_error"] for r in ours]))
            best = min(candidate_rows, key=lambda x: abs(x["storage_elements"] - storage)) if candidate_rows else None
            puzzle_all = []
            for rank in sorted({int(r["rank"]) for r in layer_rows if r["method"] == "PuzzleKV"}):
                point = [r for r in layer_rows if r["method"] == "PuzzleKV" and r["rank"] == rank]
                puzzle_all.append({"rank": rank, "storage_elements": rank * (args.page_size + 128),
                                   "mean_error": float(np.mean([r["attention_error"] for r in point]))})
            same_error = [x for x in puzzle_all if x["mean_error"] <= ours_error]
            best_same_error = min(same_error, key=lambda x: x["storage_elements"]) if same_error else None
            same_storage.append({"layer": layer, "m": m, "ours_storage": storage,
                                 "ours_error": ours_error, "r_equiv_float": req,
                                 "puzzle_candidates": candidate_rows,
                                 "closest_rank": best["rank"] if best else None,
                                 "closest_error": best["mean_error"] if best else None,
                                 "error_difference_ours_minus_puzzle": (ours_error - best["mean_error"] if best else None),
                                 "same_storage_error_improvement": (best["mean_error"] - ours_error if best else None),
                                 "same_error_puzzle_rank": best_same_error["rank"] if best_same_error else None,
                                 "same_error_puzzle_storage": best_same_error["storage_elements"] if best_same_error else None,
                                 "same_error_storage_saving": ((best_same_error["storage_elements"] - storage) / storage
                                                                if best_same_error else None),
                                 "ours_wins": bool(best and ours_error < best["mean_error"])})

    overall = {}
    full_rows = [r for r in all_rows if r["method"] == "Full"]
    overall["Full"] = {"full": {
        "mean_error": 0.0, "median_error": 0.0,
        "storage_ratio": 1.0, "compute_ratio": 1.0,
    }}
    for method in ("PuzzleKV", "WeightedCoalescing"):
        overall[method] = {}
        configs = sorted({int(r["rank"] if method == "PuzzleKV" else r["m"])
                          for r in all_rows if r["method"] == method})
        for config in configs:
            subset = [r for r in all_rows if r["method"] == method and
                      int(r["rank"] if method == "PuzzleKV" else r["m"]) == config]
            overall[method][str(config)] = {
                "mean_error": float(np.mean([r["attention_error"] for r in subset])),
                "median_error": float(np.median([r["attention_error"] for r in subset])),
                "storage_ratio": subset[0]["storage_ratio"],
                "compute_ratio": subset[0]["compute_ratio"],
            }

    summary = {"layers": layers, "seed": args.seed, "pages_per_layer": args.pages_per_layer,
               "page_size": args.page_size, "tokens": args.tokens, "puzzle_ranks": ranks,
               "m_values": M_VALUES, "layer_summaries": layer_summaries,
               "overall": overall, "same_storage": same_storage}
    (out / "pareto_summary.json").write_text(json.dumps(summary, indent=2))

    try:
        import matplotlib.pyplot as plt
        for x_metric, x_label, filename in (
            ("storage_ratio", "Storage / Full K", "pareto_storage_vs_error.png"),
            ("compute_ratio", "Theoretical MACs / Full attention", "pareto_compute_vs_error.png"),
        ):
            fig, axes = plt.subplots(2, 3, figsize=(15, 8), squeeze=False)
            for index, layer in enumerate(layers):
                ax = axes.flat[index]
                subset = [r for r in all_rows if r["layer"] == layer]
                for method, marker in (("PuzzleKV", "o"), ("WeightedCoalescing", "s"), ("Full", "*")):
                    points = [r for r in subset if r["method"] == method]
                    points.sort(key=lambda r: r[x_metric])
                    ax.plot([r[x_metric] for r in points], [r["attention_error"] for r in points],
                            marker=marker, label=method)
                ax.set_title(f"Layer {layer}")
                ax.set_xlabel(x_label)
                ax.set_ylabel("Attention relative error")
                ax.grid(alpha=0.25)
                ax.legend(fontsize=8)
            ax = axes.flat[-1]
            for method, marker in (("PuzzleKV", "o"), ("WeightedCoalescing", "s"), ("Full", "*")):
                points = []
                for config in sorted(overall.get(method, {}), key=lambda x: float(overall[method][x][x_metric])):
                    point = overall[method][config]
                    points.append((point[x_metric], point["mean_error"]))
                ax.plot([p[0] for p in points], [p[1] for p in points], marker=marker, label=method)
            ax.set_title("Overall mean")
            ax.set_xlabel(x_label)
            ax.set_ylabel("Attention relative error")
            ax.grid(alpha=0.25)
            ax.legend(fontsize=8)
            axes.flat[-2].axis("off")
            fig.tight_layout()
            fig.savefig(out / filename, dpi=160)
            plt.close(fig)
    except ImportError:
        print("matplotlib unavailable; Pareto figures were not generated")

    for item in layer_summaries:
        print(f"\n===== Layer {item['layer']} =====\nPuzzleKV:")
        for rank in ranks:
            s = item["summary"]["PuzzleKV"][str(rank)]
            print(f"r={rank:<2} storage={100*s['storage_ratio']['mean']:.2f}% error={s['attention_error']['mean']:.6f}")
        print("Weighted Coalescing:")
        for m in M_VALUES:
            s = item["summary"]["WeightedCoalescing"][str(m)]
            print(f"m={m:<3} storage={100*s['storage_ratio']['mean']:.2f}% error={s['attention_error']['mean']:.6f}")
        print("Same-storage comparisons:")
        for match in [x for x in same_storage if x["layer"] == item["layer"]]:
            print(f"m={match['m']}: ours error={match['ours_error']:.6f}, "
                  f"equiv rank={match['r_equiv_float']:.2f}, candidates={match['puzzle_candidates']}, "
                  f"winner={'Ours' if match['ours_wins'] else 'PuzzleKV'}")
            if match["same_error_puzzle_rank"] is not None:
                print(f"  same-error PuzzleKV: r={match['same_error_puzzle_rank']}, "
                      f"storage saving={(100*match['same_error_storage_saving']):.2f}%")
    wins = sum(x["ours_wins"] for x in same_storage)
    print(f"\n===== Overall =====\nOurs wins same-storage mean error: {wins}/{len(same_storage)} points")
    print(f"Pareto per-page: {out/'pareto_per_page.csv'}")
    print(f"Pareto layer summary: {out/'pareto_layer_summary.csv'}")
    print(f"Pareto summary: {out/'pareto_summary.json'}")


if __name__ == "__main__":
    main()
