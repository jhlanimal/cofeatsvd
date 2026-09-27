#!/usr/bin/env python
"""Plot relative aggregate-PPL change for the completed Qwen3 experiment."""
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "results/rope_svd_qwen8b_wikitext/ppl_aggregate.json"
OUT = ROOT / "results/rope_svd_qwen8b_wikitext"


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    aggregate = json.loads(DATA.read_text())["aggregate_ppl"]
    methods = ["full", "ours64", "ours80", "ours96"]
    labels = ["Full", "Ours\nm=64", "Ours\nm=80", "Ours\nm=96"]
    full = float(aggregate["full"])
    relative = np.array([(float(aggregate[m]) - full) / full * 100.0 for m in methods])
    colors = ["#4d4d4d", "#4c78a8", "#59a14f", "#f28e2b"]

    result = {m: {"aggregate_ppl": float(aggregate[m]),
                  "relative_ppl_change_percent": float(relative[i])}
              for i, m in enumerate(methods)}
    (OUT / "ours_relative_ppl.json").write_text(json.dumps({"full_ppl": full, "methods": result}, indent=2))

    plt.rcParams.update({"font.size": 10, "axes.linewidth": 0.8, "pdf.fonttype": 42})
    fig, ax = plt.subplots(figsize=(4.5, 3.0))
    bars = ax.bar(np.arange(len(methods)), relative, color=colors, width=0.62, alpha=0.88)
    ax.axhline(0.0, color="#222222", lw=0.9)
    for bar, value in zip(bars, relative):
        y = value + (0.003 if value >= 0 else -0.003)
        va = "bottom" if value >= 0 else "top"
        ax.text(bar.get_x() + bar.get_width() / 2, y, f"{value:+.3f}%", ha="center", va=va, fontsize=9)
    ax.set_xticks(np.arange(len(methods)), labels)
    ax.set_ylabel("Relative Aggregate PPL Change (%)")
    ax.set_title("PPL change relative to Full")
    ax.grid(axis="y", alpha=0.2)
    ax.text(0.01, -0.18, "Lower is better; values are computed from 13 completed WikiText windows.",
            transform=ax.transAxes, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "fig1_ours_relative_ppl.pdf", bbox_inches="tight")
    fig.savefig(OUT / "fig1_ours_relative_ppl.png", dpi=300, bbox_inches="tight")
    print(json.dumps(result, indent=2))
    print(OUT / "fig1_ours_relative_ppl.pdf")
    print(OUT / "fig1_ours_relative_ppl.png")


if __name__ == "__main__":
    main()
