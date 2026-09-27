"""Plot separate Qwen3-8B diagnostics from their recorded results."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PAPER = Path(__file__).resolve().parents[1]
SOURCE = PAPER.parents[1] / "results/rope_svd_qwen8b_wikitext"


def main():
    svd = json.loads((SOURCE / "global_one_sequence/results.json").read_text())["results"]
    ours = json.loads((SOURCE / "ours_relative_ppl.json").read_text())
    ranks = [64, 80, 96]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "pdf.fonttype": 42, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.labelcolor": "#303030"})
    fig, (left, right) = plt.subplots(1, 2, figsize=(7.1, 3.15))
    fig.subplots_adjust(left=.09, right=.98, bottom=.19, top=.73, wspace=.38)
    for ax in (left, right):
        ax.set_xticks(ranks)
        ax.set_xlim(60, 100)
        ax.grid(axis="y", color="#e4e7e9", linewidth=.7)
        ax.set_axisbelow(True)
        ax.tick_params(length=3, color="#888888")
    left.plot(ranks, [svd[f"post{r}"]["ppl"] for r in ranks],
              color="#c44e52", marker="s", lw=1.8, label="Post-RoPE SVD")
    left.plot(ranks, [svd[f"pre_exact{r}"]["ppl"] for r in ranks],
              color="#3274a1", marker="o", lw=1.8, label="Pre-SVD, original RoPE")
    left.axhline(svd["full"]["ppl"], color="#555555", ls="--", lw=1.2,
                 label=f'Full ({svd["full"]["ppl"]:.3f})')
    left.set_ylim(4.5, 25)
    left.set_yticks([5, 10, 15, 20, 25])
    left.set_xlabel("SVD rank $r$")
    left.set_ylabel("Perplexity (lower is better)")
    left.set_title("(a) Whole-sequence SVD", loc="left", weight="bold", pad=43)
    left.legend(frameon=False, fontsize=7.5, loc="lower left",
                bbox_to_anchor=(-.03, 1.01), ncol=1, borderaxespad=0, labelspacing=.3)
    for r in ranks:
        left.annotate(f'{svd[f"post{r}"]["ppl"]:.2f}',
                      (r, svd[f"post{r}"]["ppl"]), xytext=(0, 7),
                      textcoords="offset points", ha="center", fontsize=8, color="#a5393e")
    values = [ours["methods"][f"ours{r}"]["relative_ppl_change_percent"] for r in ranks]
    right.plot(ranks, values, color="#27836b", marker="D", lw=1.8, label="CoFeatKV")
    right.axhline(0, color="#555555", ls="--", lw=1.2, label="Full reference")
    for r, v in zip(ranks, values):
        ppl = ours["methods"][f"ours{r}"]["aggregate_ppl"]
        right.annotate(f"{ppl:.4f}", (r, v), xytext=(0, 9),
                       textcoords="offset points", ha="center", fontsize=8, color="#216b58")
    right.set_ylim(-.06, .55)
    right.set_yticks([0, .1, .2, .3, .4, .5])
    right.set_xlabel("Feature-group budget $m$")
    right.set_ylabel("PPL increase over Full (%)")
    right.set_title("(b) CoFeatKV (13 windows)", loc="left", weight="bold", pad=43)
    right.legend(frameon=False, fontsize=7.5, loc="lower left", bbox_to_anchor=(-.03, 1.01),
                 borderaxespad=0, title=f'Full PPL = {ours["full_ppl"]:.4f}', title_fontsize=8)
    fig.savefig(PAPER / "figures/rope_same_rank.pdf")
    fig.savefig(PAPER / "figures/rope_same_rank.png", dpi=220)
    plt.close(fig)


if __name__ == "__main__":
    main()
