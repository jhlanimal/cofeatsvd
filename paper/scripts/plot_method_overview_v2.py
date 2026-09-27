from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle, Polygon


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "iclr_latex/iclr2027/figures/method_overview_v2.pdf"

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Nimbus Roman", "Times New Roman", "Liberation Serif"],
    "font.size": 8.0,
    "mathtext.fontset": "stix",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


COLORS = {
    "ink": "#243447",
    "muted": "#64727c",
    "blue": "#3c8fc5",
    "blue_fill": "#eef7fc",
    "orange": "#e8a329",
    "orange_fill": "#fff7e5",
    "green": "#4b9b68",
    "green_fill": "#eef9f0",
    "teal": "#20a38a",
    "teal_fill": "#eefaf7",
    "purple": "#8960b5",
    "purple_fill": "#f7f1fc",
    "red": "#e45a61",
    "red_fill": "#fff1f2",
    "gray": "#a0adb5",
}


def rounded(ax, x, y, w, h, edge, fill="white", radius=0.06, lw=1.0,
            linestyle="-"):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.015,rounding_size={radius}",
        linewidth=lw, edgecolor=edge, facecolor=fill, linestyle=linestyle,
        zorder=2,
    )
    ax.add_patch(patch)
    return patch


def arrow(ax, x1, y1, x2, y2, color=COLORS["ink"], lw=1.1, ls="-",
          mutation=9, rad=0.0):
    ax.add_patch(FancyArrowPatch(
        (x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=mutation,
        linewidth=lw, linestyle=ls, color=color, shrinkA=2, shrinkB=3,
        connectionstyle=f"arc3,rad={rad}", zorder=8,
    ))


def box(ax, x, y, w, h, text, edge, fill="white", fs=7.3, weight="normal",
        lw=0.9):
    rounded(ax, x, y, w, h, edge, fill, radius=0.05, lw=lw)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, color=COLORS["ink"], fontweight=weight,
            linespacing=0.95, zorder=5)


def dashed_region(ax, x, y, w, h, edge, fill, title):
    rounded(ax, x, y, w, h, edge, fill, radius=0.04, lw=0.85,
            linestyle=(0, (3, 2)))
    ax.text(x + 0.12, y + h - 0.16, title, ha="left", va="top",
            fontsize=8.2, color=COLORS["ink"], fontweight="bold", zorder=5)


def matrix(ax, x, y, rows, cols, cw, ch, edge, fill, gap=0.018):
    for r in range(rows):
        for c in range(cols):
            ax.add_patch(Rectangle(
                (x + c * (cw + gap), y + (rows - 1 - r) * (ch + gap)),
                cw, ch, linewidth=0.42, edgecolor=edge, facecolor=fill,
                zorder=4,
            ))


def draw():
    fig, ax = plt.subplots(figsize=(14.0, 5.25))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 5.25)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    # The left side follows one prompt through Q/K/V projection and page
    # sealing. The right side is the steady-state decoding path.
    dashed_region(ax, 0.18, 0.28, 6.62, 4.66, COLORS["orange"], "#fffdf7",
                  "Page sealing (once per full page)")
    dashed_region(ax, 7.05, 0.28, 6.77, 4.66, COLORS["blue"], "#fbfdff",
                  "Reconstruction-free decoding")

    # Input and the three projection branches.
    box(ax, 0.34, 2.33, 0.66, 0.42, "Input", COLORS["gray"], "white", fs=7.3)
    box(ax, 1.28, 4.10, 0.96, 0.55, r"$W^q$", COLORS["purple"],
        COLORS["purple_fill"], fs=9.0, weight="bold")
    box(ax, 1.28, 2.55, 0.96, 0.55, r"$W^k$", COLORS["purple"],
        COLORS["purple_fill"], fs=9.0, weight="bold")
    box(ax, 1.28, 1.00, 0.96, 0.55, r"$W^v$", COLORS["purple"],
        COLORS["purple_fill"], fs=9.0, weight="bold")
    arrow(ax, 1.00, 2.54, 1.25, 4.35, COLORS["ink"], 1.0, rad=0.10)
    arrow(ax, 1.00, 2.54, 1.25, 2.83, COLORS["ink"], 1.0)
    arrow(ax, 1.00, 2.54, 1.25, 1.27, COLORS["ink"], 1.0, rad=-0.10)

    # Query path.
    box(ax, 2.52, 4.10, 0.62, 0.55, r"$q_t$", COLORS["blue"],
        COLORS["blue_fill"], fs=8.6, weight="bold")
    box(ax, 3.42, 4.10, 0.72, 0.55, "RoPE", COLORS["gray"], "#eef1f3", fs=7.1,
        weight="bold")
    box(ax, 4.42, 4.10, 0.86, 0.55, r"current $q$", COLORS["orange"],
        COLORS["orange_fill"], fs=7.0, weight="bold")
    arrow(ax, 2.25, 4.38, 2.49, 4.38, COLORS["ink"])
    arrow(ax, 3.16, 4.38, 3.39, 4.38, COLORS["ink"])
    arrow(ax, 4.16, 4.38, 4.39, 4.38, COLORS["ink"])

    # Key path. A completed page is the object that is compressed.
    box(ax, 2.52, 2.55, 0.62, 0.55, r"$K_t$", COLORS["blue"],
        COLORS["blue_fill"], fs=8.6, weight="bold")
    box(ax, 3.42, 2.55, 0.72, 0.55, "RoPE", COLORS["gray"], "#eef1f3", fs=7.1,
        weight="bold")
    rounded(ax, 4.38, 2.42, 1.12, 0.81, COLORS["blue"], COLORS["blue_fill"],
            radius=0.05, lw=0.9)
    ax.text(4.94, 3.10, "post-RoPE K page", ha="center", va="center",
            fontsize=6.8, color=COLORS["ink"], fontweight="bold", zorder=5)
    matrix(ax, 4.52, 2.55, 3, 5, 0.12, 0.075, COLORS["blue"], "#cce9f6")
    ax.text(4.94, 2.33, r"$K\in\mathbb{R}^{P\times d}$", ha="center", va="top",
            fontsize=6.2, color=COLORS["muted"])
    arrow(ax, 2.25, 2.83, 2.49, 2.83, COLORS["ink"])
    arrow(ax, 3.16, 2.83, 3.39, 2.83, COLORS["ink"])
    arrow(ax, 4.16, 2.83, 4.35, 2.83, COLORS["ink"])

    # Value path. Values remain a separate cache component.
    box(ax, 2.52, 1.00, 0.62, 0.55, r"$V_t$", COLORS["blue"],
        COLORS["blue_fill"], fs=8.6, weight="bold")
    box(ax, 3.42, 1.00, 1.08, 0.55, "value page", COLORS["green"],
        COLORS["green_fill"], fs=7.0, weight="bold")
    arrow(ax, 2.25, 1.27, 2.49, 1.27, COLORS["ink"])
    arrow(ax, 3.16, 1.27, 3.39, 1.27, COLORS["ink"])

    # CoFeatKV page-local grouping branch.
    dashed_region(ax, 4.14, 0.66, 2.38, 1.44, COLORS["teal"], "#f2fbf8",
                  "CoFeatKV page seal")
    box(ax, 4.32, 1.18, 0.78, 0.43, r"$s_j$", COLORS["orange"],
        COLORS["orange_fill"], fs=8.2, weight="bold")
    box(ax, 5.23, 1.18, 1.10, 0.43, "top-m +\nmerge cost", COLORS["orange"],
        COLORS["orange_fill"], fs=6.2, weight="bold")
    arrow(ax, 4.95, 2.40, 4.70, 1.66, COLORS["teal"], 1.0, ls=(0, (3, 2)))
    box(ax, 4.32, 0.78, 1.96, 0.25,
        r"$C[P\times m],\;\alpha[d],\;owner[d]$", COLORS["teal"],
        COLORS["teal_fill"], fs=5.75, weight="bold")
    arrow(ax, 5.23, 1.16, 5.23, 1.06, COLORS["teal"], 0.8)
    arrow(ax, 4.95, 2.42, 5.23, 2.12, COLORS["teal"], 1.0, ls=(0, (3, 2)))
    ax.text(5.33, 0.52, "one coefficient + one owner per feature",
            ha="center", va="center", fontsize=5.9, color=COLORS["teal"],
            fontweight="bold")

    # Cache outputs from the page seal.
    box(ax, 5.44, 2.42, 1.05, 0.40, "Key cache\n(C, α, owner)", COLORS["teal"],
        COLORS["teal_fill"], fs=6.0, weight="bold")
    box(ax, 3.08, 0.36, 1.05, 0.25, "V cache (V + scale)", COLORS["green"],
        COLORS["green_fill"], fs=5.55, weight="bold")
    arrow(ax, 5.50, 2.42, 5.70, 2.04, COLORS["teal"], 0.85, ls=(0, (3, 2)))
    arrow(ax, 3.96, 1.00, 3.62, 0.64, COLORS["green"], 0.9)

    # Decode path: q -> qA, cache C -> scores, active dense page is separate.
    box(ax, 7.35, 4.10, 0.98, 0.55, r"current $q$", COLORS["orange"],
        COLORS["orange_fill"], fs=7.0, weight="bold")
    box(ax, 8.60, 4.10, 0.85, 0.55, r"$qA$", COLORS["blue"],
        COLORS["blue_fill"], fs=8.8, weight="bold")
    box(ax, 9.72, 4.10, 1.56, 0.55, r"scores $=(qA)C^\top$", COLORS["teal"],
        COLORS["teal_fill"], fs=6.6, weight="bold")
    arrow(ax, 8.36, 4.38, 8.57, 4.38, COLORS["ink"])
    arrow(ax, 9.48, 4.38, 9.69, 4.38, COLORS["ink"])
    ax.text(8.98, 3.88, "owner map selects one group per feature",
            ha="center", va="center", fontsize=5.9, color=COLORS["muted"])

    box(ax, 7.35, 2.92, 1.30, 0.44, "sealed pages\nC", COLORS["blue"],
        COLORS["blue_fill"], fs=6.5, weight="bold")
    box(ax, 8.95, 2.92, 1.42, 0.44, "active page\ndense K", COLORS["green"],
        COLORS["green_fill"], fs=6.3, weight="bold")
    arrow(ax, 8.00, 2.42, 10.10, 4.05, COLORS["teal"], 1.0, ls=(0, (3, 2)), rad=0.14)
    arrow(ax, 9.66, 3.38, 9.86, 4.05, COLORS["green"], 0.95, rad=-0.12)

    # Online softmax and values.
    box(ax, 10.95, 2.20, 1.10, 0.55, "online\nsoftmax", COLORS["red"],
        COLORS["red_fill"], fs=6.8, weight="bold")
    box(ax, 12.25, 2.20, 0.98, 0.55, "weighted V", COLORS["green"],
        COLORS["green_fill"], fs=6.8, weight="bold")
    box(ax, 13.34, 2.20, 0.38, 0.55, r"$O$", COLORS["blue"],
        COLORS["blue_fill"], fs=8.5, weight="bold")
    arrow(ax, 10.52, 4.10, 11.47, 2.80, COLORS["ink"], 1.0, rad=-0.14)
    arrow(ax, 12.08, 2.48, 12.22, 2.48, COLORS["ink"])
    arrow(ax, 13.25, 2.48, 13.30, 2.48, COLORS["ink"])
    box(ax, 10.95, 1.45, 0.94, 0.34, "V cache", COLORS["green"],
        COLORS["green_fill"], fs=6.2, weight="bold")
    arrow(ax, 11.89, 1.62, 12.55, 2.16, COLORS["green"], 0.95, ls=(0, (3, 2)), rad=-0.08)
    ax.text(11.67, 1.38, "original token positions", ha="center", va="top",
            fontsize=5.9, color=COLORS["muted"])

    # Explicit invariants: these are the claims the figure should communicate.
    rounded(ax, 7.35, 0.74, 5.95, 0.55, COLORS["red"], "#fffafa", radius=0.05,
            lw=0.9)
    ax.text(10.32, 1.015, "post-RoPE attention: no historical RoPE and no full-Key reconstruction",
            ha="center", va="center", fontsize=7.0, color=COLORS["red"],
            fontweight="bold")

    # Main construction arrows make the causal order explicit.
    arrow(ax, 5.50, 2.84, 5.50, 2.45, COLORS["teal"], 0.9, ls=(0, (3, 2)))
    arrow(ax, 6.53, 2.62, 7.30, 2.62, COLORS["red"], 1.05)

    fig.subplots_adjust(left=0.005, right=0.995, bottom=0.012, top=0.95)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(OUT.with_suffix(".png"), dpi=320, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


if __name__ == "__main__":
    draw()
