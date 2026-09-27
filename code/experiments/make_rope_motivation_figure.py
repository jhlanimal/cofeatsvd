#!/usr/bin/env python
"""Build one richer, paper-ready motivation figure from completed diagnostics."""
import csv
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results/rope_svd_qwen8b_wikitext"
GROUP = BASE / "grouped_m80"
OUT = BASE


def load_csv(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def ci(x):
    x = np.asarray(x, dtype=float)
    return 1.96 * x.std(ddof=1) / np.sqrt(len(x))


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    rows = load_csv(BASE / "svd_error_per_sequence.csv")
    grouped = load_csv(GROUP / "grouped_svd_error_per_sequence.csv")
    layers = sorted({int(r["layer"]) for r in rows})
    ranks = sorted({int(r["rank"]) for r in rows})
    colors = {layers[0]: "#4c78a8", layers[1]: "#59a14f", layers[2]: "#f28e2b"}

    direct = {(int(r["sequence"]), int(r["layer"]), r["rope_state"], int(r["rank"])): float(r["mean_error"])
              for r in rows}
    grouped_map = {(int(r["sequence"]), int(r["layer"]), int(r["rank"])): float(r["normalized_error"])
                   for r in grouped}

    plt.rcParams.update({"font.size": 9, "axes.linewidth": 0.8, "pdf.fonttype": 42,
                         "font.family": "DejaVu Sans"})
    fig = plt.figure(figsize=(7.1, 4.8))
    grid = fig.add_gridspec(2, 2, height_ratios=[1.35, 1], hspace=0.42, wspace=0.30)
    ax = fig.add_subplot(grid[0, :])
    axb = fig.add_subplot(grid[1, 0])
    axc = fig.add_subplot(grid[1, 1])

    # (a) All selected layers plus the sequence-level aggregate mean and CI.
    for layer in layers:
        for state, ls, alpha in (("pre", "--", 0.55), ("post", "-", 0.55)):
            vals = np.array([[direct[(s, layer, state, r)] for r in ranks] for s in range(32)])
            ax.plot(ranks, vals.mean(0), color=colors[layer], ls=ls, lw=1.1, alpha=alpha)
    for state, color, label in (("pre", "#2166ac", "Pre-RoPE"), ("post", "#b2182b", "Post-RoPE")):
        vals = np.array([[np.mean([direct[(s, layer, state, r)] for layer in layers]) for r in ranks] for s in range(32)])
        mean = vals.mean(0); band = np.array([ci(vals[:, i]) for i in range(len(ranks))])
        ax.plot(ranks, mean, color=color, lw=2.5, marker="o", ms=3.2, label=f"{label} (3-layer mean)")
        ax.fill_between(ranks, np.maximum(0, mean-band), mean+band, color=color, alpha=0.12, linewidth=0)
    # The real grouping point is intentionally marked separately; it is not a direct-SVD curve.
    gm80 = float(np.mean([grouped_map[(s, layer, 80)] for layer in layers for s in range(32)]))
    ax.scatter([80], [gm80], marker="D", s=42, color="#7b3294", zorder=6, label="CoFeatKV grouping (m=80)")
    ax.annotate("actual grouping\n(m=80)", (80, gm80), xytext=(8, 12), textcoords="offset points", fontsize=8,
                arrowprops={"arrowstyle": "-", "lw": 0.7, "color": "#7b3294"})
    ax.set_title("RoPE weakens direct low-rank structure, while CoFeatKV uses feature grouping")
    ax.set_xlabel("SVD rank / grouped feature budget")
    ax.set_ylabel("Normalized K reconstruction error")
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.2); ax.legend(frameon=False, ncol=3, loc="upper right", fontsize=8)
    ax.text(-0.02, 1.03, "(a)", transform=ax.transAxes, fontweight="bold")

    # (b) Rank-64 direct comparison by layer; grouped point is shown at its actual m=80 budget.
    x = np.arange(len(layers)); width = 0.18
    for offset, state, color, label in ((-width, "pre", "#2166ac", "Pre"), (0, "post", "#b2182b", "Post")):
        means = []; cis = []
        for layer in layers:
            v = [direct[(s, layer, state, 64)] for s in range(32)]
            means.append(np.mean(v)); cis.append(ci(v))
        axb.bar(x + offset, means, width, yerr=cis, capsize=2, color=color, alpha=0.82, label=label)
    gmeans = [np.mean([grouped_map[(s, layer, 80)] for s in range(32)]) for layer in layers]
    gcis = [ci([grouped_map[(s, layer, 80)] for s in range(32)]) for layer in layers]
    axb.errorbar(x + width, gmeans, yerr=gcis, fmt="D", ms=4.5, color="#7b3294", capsize=2, label="CoFeatKV m=80")
    axb.set_xticks(x, [f"L{l}" for l in layers]); axb.set_ylabel("Error"); axb.set_title("Layer-wise mean ± 95% CI")
    axb.grid(axis="y", alpha=0.2); axb.legend(frameon=False, fontsize=7)
    axb.text(-0.08, 1.03, "(b)", transform=axb.transAxes, fontweight="bold")

    # (c) Across-sequence distributions at the same diagnostic rank/budget.
    data = []
    labels = []
    for state, label in (("pre", "Pre"), ("post", "Post")):
        data.append([np.mean([direct[(s, layer, state, 64)] for layer in layers]) for s in range(32)])
        labels.append(label + " rank64")
    data.append([np.mean([grouped_map[(s, layer, 80)] for layer in layers]) for s in range(32)])
    labels.append("CoFeatKV m80")
    violin = axc.violinplot(data, showmeans=True, showmedians=True, widths=0.78)
    for body, color in zip(violin["bodies"], ["#2166ac", "#b2182b", "#7b3294"]):
        body.set_facecolor(color); body.set_edgecolor("white"); body.set_alpha(0.72)
    for key in ("cmeans", "cmedians", "cbars", "cmins", "cmaxes"):
        if key in violin: violin[key].set_color("#333333"); violin[key].set_linewidth(0.8)
    axc.set_xticks([1, 2, 3], labels, rotation=18, ha="right")
    axc.set_ylabel("Error"); axc.set_title("32 WikiText windows")
    axc.grid(axis="y", alpha=0.2); axc.text(-0.08, 1.03, "(c)", transform=axc.transAxes, fontweight="bold")

    fig.text(0.01, 0.005, "Thin lines: selected layers. Thick lines/bands: mean ± 95% CI over 32 sequences. "
             "CoFeatKV marker reports the actual weighted grouping diagnostic, not a direct-SVD curve.", fontsize=7.5)
    fig.savefig(OUT / "fig1_rope_motivation_composite.pdf", bbox_inches="tight")
    fig.savefig(OUT / "fig1_rope_motivation_composite.png", dpi=300, bbox_inches="tight")
    print(OUT / "fig1_rope_motivation_composite.pdf")
    print(OUT / "fig1_rope_motivation_composite.png")


if __name__ == "__main__":
    main()
