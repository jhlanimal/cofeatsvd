#!/usr/bin/env python
"""Make Figure 1 from previously completed Qwen3 PPL evaluations."""
import json
import math
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/rope_svd_qwen8b_wikitext"
SOURCES = [
    OUT / "ppl_sources" / "blocks0_2_full.json",
    OUT / "ppl_sources" / "blocks0_2_ours64.json",
    OUT / "ppl_sources" / "blocks0_2_ours80.json",
    OUT / "ppl_sources" / "blocks0_2_ours96.json",
    OUT / "ppl_sources" / "blocks3_12.json",
    OUT / "ppl_sources" / "blocks3_12_ours80.json",
    OUT / "ppl_sources" / "blocks3_12_ours96.json",
]


def collect():
    merged = {}
    for path in SOURCES:
        data = json.loads(path.read_text())
        for method, rows in data["methods"].items():
            for row in rows:
                merged.setdefault(method, {})[int(row["block_id"])] = row
    return {m: [rows[k] for k in sorted(rows)] for m, rows in merged.items()}


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    methods = collect()
    names = ["full", "ours64", "ours80", "ours96"]
    labels = {"full": "Full", "ours64": "Ours m=64", "ours80": "Ours m=80", "ours96": "Ours m=96"}
    colors = {"full": "#222222", "ours64": "#4c78a8", "ours80": "#59a14f", "ours96": "#f28e2b"}
    blocks = sorted({int(r["block_id"]) for m in names for r in methods[m]})
    by = {(m, int(r["block_id"])): float(r["ppl"]) for m in names for r in methods[m]}
    aggregate = {}
    for m in names:
        rows = methods[m]
        nll = sum(float(r["nll"]) for r in rows)
        tokens = sum(int(r["num_pred_tokens"]) for r in rows)
        aggregate[m] = math.exp(nll / tokens)
    OUT.joinpath("ppl_aggregate.json").write_text(json.dumps({"blocks": blocks, "aggregate_ppl": aggregate}, indent=2))

    plt.rcParams.update({"font.size": 9, "axes.linewidth": 0.8, "pdf.fonttype": 42})
    fig = plt.figure(figsize=(7.1, 4.8))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.35, 1], hspace=0.43, wspace=0.30)
    ax = fig.add_subplot(gs[0, :]); axb = fig.add_subplot(gs[1, 0]); axc = fig.add_subplot(gs[1, 1])

    for m in names:
        vals = [by[(m, b)] for b in blocks]
        ax.plot(blocks, vals, marker="o", ms=3, lw=1.25, color=colors[m], alpha=0.75, label=labels[m])
        ax.axhline(aggregate[m], color=colors[m], lw=2.2 if m != "full" else 1.8, alpha=0.95)
    ax.set_title("Qwen3-8B WikiText perplexity across 13 evaluation windows")
    ax.set_xlabel("4096-token WikiText window")
    ax.set_ylabel("Perplexity")
    ax.grid(alpha=0.2); ax.legend(frameon=False, ncol=4, fontsize=8)
    ax.text(-0.02, 1.03, "(a)", transform=ax.transAxes, fontweight="bold")

    x = np.arange(len(names))
    means = [np.mean([by[(m, b)] for b in blocks]) for m in names]
    cis = [1.96 * np.std([by[(m, b)] for b in blocks], ddof=1) / np.sqrt(len(blocks)) for m in names]
    bars = axb.bar(x, means, yerr=cis, capsize=3, color=[colors[m] for m in names], alpha=0.82)
    for bar, m in zip(bars, names):
        axb.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.08, f"{aggregate[m]:.3f}", ha="center", fontsize=7)
    axb.set_xticks(x, ["Full", "m64", "m80", "m96"])
    axb.set_ylabel("PPL"); axb.set_title("Mean PPL ± 95% CI")
    axb.grid(axis="y", alpha=0.2); axb.text(-0.08, 1.03, "(b)", transform=axb.transAxes, fontweight="bold")

    full = np.array([by[("full", b)] for b in blocks])
    for m in names[1:]:
        delta = np.array([by[(m, b)] for b in blocks]) - full
        axc.plot(blocks, delta, marker="o", ms=3, lw=1.25, color=colors[m], label=labels[m])
    axc.axhline(0, color="#222222", lw=0.8)
    axc.set_xlabel("WikiText window"); axc.set_ylabel("PPL - Full PPL")
    axc.set_title("Per-window PPL change")
    axc.grid(alpha=0.2); axc.legend(frameon=False, fontsize=7)
    axc.text(-0.08, 1.03, "(c)", transform=axc.transAxes, fontweight="bold")

    fig.text(0.01, 0.005, "Lines show block-level PPL; horizontal lines in (a) show aggregate PPL computed from total NLL. "
             "Results are from the previously completed Qwen3-8B WikiText evaluation.", fontsize=7.5)
    fig.savefig(OUT / "fig1_ppl_motivation.pdf", bbox_inches="tight")
    fig.savefig(OUT / "fig1_ppl_motivation.png", dpi=300, bbox_inches="tight")
    print(json.dumps(aggregate, indent=2))
    print(OUT / "fig1_ppl_motivation.pdf")
    print(OUT / "fig1_ppl_motivation.png")


if __name__ == "__main__":
    main()
