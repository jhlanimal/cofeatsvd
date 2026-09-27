"""Plot layer-wise attention-output fidelity for the Qwen3-8B diagnostic."""
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "results/rope_svd_qwen8b_wikitext/attention_o_one_sequence"
OUT = ROOT / "iclr_latex/iclr2027/figures/feature_similarity.pdf"


def main():
    import csv
    rows = list(csv.DictReader((DATA / "metrics.csv").open()))
    ranks = [64, 80, 96]
    layers = [9, 18, 27]
    methods = [("pre", "Pre-RoPE SVD", "#356f9f", "o"),
               ("post", "Post-RoPE SVD", "#b54f55", "s"),
               ("ours", "CoFeatKV", "#2f8069", "D")]
    plt.rcParams.update({
        # Nimbus Roman is the metrically compatible local fallback; the
        # editable PPTX uses the requested Times New Roman font name.
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Nimbus Roman", "Liberation Serif"],
        "font.size": 10.5,
        "mathtext.fontset": "stix",
        "pdf.fonttype": 42,
        "axes.spines.top": False, "axes.spines.right": False,
    })
    fig, axes = plt.subplots(1, 3, figsize=(6.25, 2.38), sharex=True)
    fig.subplots_adjust(left=.09, right=.985, bottom=.26, top=.77,
                        wspace=.20)

    def series(layer, method, field):
        return [float(next(r[field] for r in rows
                           if int(r["layer"]) == layer
                           and r["method"] == method
                           and int(r["rank_or_m"]) == rank))
                for rank in ranks]
    for col_idx, layer in enumerate(layers):
        ax = axes[col_idx]
        for method, label, color, marker in methods:
            ax.plot(ranks, series(layer, method, "cosine_mean"), color=color,
                    marker=marker, lw=1.35, ms=3.7, label=label)
        ax.set_title(f"Layer {layer}", pad=5, weight="normal", fontsize=12)
        ax.set_ylabel("Mean output cosine", fontsize=10.5, labelpad=4)
        ax.set_yticks([.7, .8, .9, 1.0])
        ax.set_ylim(.65, 1.01)
        ax.set_xticks(ranks)
        ax.set_xlabel("Rank / group budget", fontsize=10.5, labelpad=3)
        ax.tick_params(axis="both", labelsize=9.5, pad=2)
        ax.grid(axis="y", color="#e0e3e5", linewidth=.55)
        ax.set_axisbelow(True)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False,
               fontsize=10.5, bbox_to_anchor=(.52, .975), handlelength=2.0,
               columnspacing=1.1)
    fig.savefig(OUT, bbox_inches="tight")
    fig.savefig(OUT.with_suffix(".png"), dpi=240, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
