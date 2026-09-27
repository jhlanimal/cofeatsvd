from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, Polygon, FancyArrowPatch


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "iclr_latex/iclr2027/figures/method_overview.pdf"

mpl.rcParams.update({
    # Nimbus Roman is the local Times-compatible fallback for PDF rendering.
    "font.family": "serif",
    "font.serif": ["Nimbus Roman", "Times New Roman", "Liberation Serif"],
    "font.size": 8.2,
    "mathtext.fontset": "stix",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


COLORS = {
    "ink": "#243447",
    "muted": "#62717c",
    "stage": "#9aa8b2",
    "blue": "#3c8fc5",
    "blue_fill": "#eef7fc",
    "teal": "#20a38a",
    "teal_fill": "#eefaf7",
    "orange": "#e9a62a",
    "orange_fill": "#fff8e8",
    "purple": "#8960b5",
    "purple_fill": "#f7f1fc",
    "red": "#e45a61",
    "red_fill": "#fff1f2",
    "green": "#4b9b68",
    "green_fill": "#eef9f0",
    "yellow": "#f2bb46",
}


def rounded(ax, x, y, w, h, edge, fill="white", radius=0.10, lw=1.15,
            linestyle="-"):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.015,rounding_size={radius}",
        linewidth=lw, edgecolor=edge, facecolor=fill, linestyle=linestyle,
        zorder=2,
    )
    ax.add_patch(patch)
    return patch


def stage(ax, x, y, w, h, title):
    rounded(ax, x, y, w, h, COLORS["stage"], "#ffffff", radius=0.13,
            lw=1.05, linestyle=(0, (3, 2)))
    ax.text(x + w / 2, y + h + 0.13, title, ha="center", va="bottom",
            color=COLORS["ink"], fontsize=10.5, fontweight="bold",
            style="italic", zorder=5)


def module(ax, x, y, w, h, number, title, edge, fill):
    rounded(ax, x, y, w, h, edge, fill, radius=0.09, lw=1.0)
    ax.text(x + 0.13, y + h - 0.20, f"{number}  {title}", ha="left",
            va="top", color=COLORS["ink"], fontsize=8.1, fontweight="bold",
            zorder=5)


def arrow(ax, x1, y1, x2, y2, color=COLORS["red"], lw=1.25, ls="-",
          mutation=10, connectionstyle="arc3,rad=0"):
    ax.add_patch(FancyArrowPatch(
        (x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=mutation,
        linewidth=lw, linestyle=ls, color=color, shrinkA=2, shrinkB=3,
        connectionstyle=connectionstyle, zorder=7,
    ))


def mini(ax, x, y, w, h, text, edge, fill="white", fs=7.0, weight="normal"):
    rounded(ax, x, y, w, h, edge, fill, radius=0.06, lw=0.75)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            color=COLORS["ink"], fontsize=fs, fontweight=weight,
            linespacing=0.95, zorder=5)


def matrix(ax, x, y, rows, cols, cw, ch, edge, fill, gap=0.018, alpha=1.0):
    for r in range(rows):
        for c in range(cols):
            ax.add_patch(Rectangle(
                (x + c * (cw + gap), y + (rows - 1 - r) * (ch + gap)),
                cw, ch, linewidth=0.42, edgecolor=edge, facecolor=fill,
                alpha=alpha, zorder=4,
            ))


def bars(ax, x, y, values, bw, gap, color):
    for i, value in enumerate(values):
        ax.add_patch(Rectangle((x + i * (bw + gap), y), bw, value,
                               linewidth=0, facecolor=color, zorder=4))


def diamond(ax, cx, cy, w, h, edge, fill, text):
    ax.add_patch(Polygon(
        [(cx, cy + h / 2), (cx + w / 2, cy), (cx, cy - h / 2),
         (cx - w / 2, cy)], closed=True, edgecolor=edge, facecolor=fill,
        linewidth=0.9, zorder=3,
    ))
    ax.text(cx, cy, text, ha="center", va="center", fontsize=6.8,
            color=COLORS["ink"], fontweight="bold", zorder=5)


def draw():
    fig, ax = plt.subplots(figsize=(14.0, 5.45))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 5.70)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    # The diagram is organized around the actual runtime boundary: sealing a
    # completed post-RoPE page, then decoding directly from its stored tuple.
    stage(ax, 0.16, 0.38, 7.70, 4.78, "A. Seal a completed post-RoPE page")
    stage(ax, 8.02, 0.38, 5.82, 4.78, "B. Decode without Key reconstruction")

    # ------------------------------------------------------------------
    # A1: page formation
    # ------------------------------------------------------------------
    module(ax, 0.36, 2.74, 2.08, 2.03, "1", "Form a page",
           COLORS["blue"], COLORS["blue_fill"])
    ax.text(1.40, 4.38, "hidden states", ha="center", va="center",
            fontsize=6.9, color=COLORS["muted"])
    for i in range(6):
        ax.add_patch(Rectangle((0.66 + i * 0.23, 4.04), 0.17, 0.16,
                               edgecolor=COLORS["orange"], facecolor="#fff2c8",
                               linewidth=0.62, zorder=4))
    ax.text(2.08, 4.12, "...", fontsize=7.5, color=COLORS["muted"],
            ha="center", va="center")
    mini(ax, 0.70, 3.62, 1.40, 0.30, "Q / K / V projections",
         COLORS["purple"], COLORS["purple_fill"], fs=6.7, weight="bold")
    arrow(ax, 1.40, 3.98, 1.40, 3.92, COLORS["purple"], 0.82, mutation=7)
    mini(ax, 0.61, 3.22, 0.76, 0.26, "RoPE", COLORS["orange"],
         COLORS["orange_fill"], fs=6.8, weight="bold")
    mini(ax, 1.49, 3.22, 0.76, 0.26, "page P", COLORS["blue"],
         COLORS["blue_fill"], fs=6.8, weight="bold")
    arrow(ax, 1.38, 3.60, 1.00, 3.49, COLORS["orange"], 0.72, mutation=7)
    arrow(ax, 1.42, 3.60, 1.82, 3.49, COLORS["blue"], 0.72, mutation=7)
    matrix(ax, 0.62, 2.82, 3, 7, 0.17, 0.10, COLORS["blue"], "#cce9f6")
    ax.text(1.26, 2.67, r"post-RoPE $K\in\mathbb{R}^{P\times d}$",
            ha="center", va="top", fontsize=6.4, color=COLORS["ink"])
    ax.text(1.26, 2.51, r"and $V$ for the same page", ha="center", va="top",
            fontsize=6.2, color=COLORS["muted"])

    # ------------------------------------------------------------------
    # A2: importance calibration and grouping
    # ------------------------------------------------------------------
    module(ax, 2.64, 2.74, 2.18, 2.03, "2", "Calibrate features",
           COLORS["orange"], COLORS["orange_fill"])
    ax.text(3.73, 4.36, r"$s_j=w_j\,\|k_j\|_2^2$", ha="center", va="center",
            fontsize=7.4, color=COLORS["ink"], fontweight="bold")
    bars(ax, 3.07, 3.43, [0.34, 0.64, 0.23, 0.51, 0.28, 0.43],
         0.15, 0.055, COLORS["orange"])
    ax.text(3.73, 3.34, "query-calibrated importance", ha="center", va="top",
            fontsize=6.1, color=COLORS["muted"])
    mini(ax, 2.89, 2.88, 0.78, 0.27, "top-m", COLORS["blue"],
         COLORS["blue_fill"], fs=6.6, weight="bold")
    mini(ax, 3.80, 2.88, 0.80, 0.27, "d-m", COLORS["orange"],
         COLORS["orange_fill"], fs=6.6, weight="bold")
    arrow(ax, 3.70, 3.12, 3.29, 3.12, COLORS["blue"], 0.75, mutation=7)
    arrow(ax, 3.76, 3.12, 4.17, 3.12, COLORS["orange"], 0.75, mutation=7)
    ax.text(3.73, 2.75, r"pairwise merge cost $\Delta(h,\ell)$",
            ha="center", va="center", fontsize=6.1, color=COLORS["muted"])

    # ------------------------------------------------------------------
    # A3: shared directions and sparse feature-to-group map
    # ------------------------------------------------------------------
    module(ax, 5.02, 2.74, 2.56, 2.03, "3", "Build shared groups",
           COLORS["teal"], COLORS["teal_fill"])
    ax.text(6.30, 4.39, "feature columns in one page", ha="center", va="center",
            fontsize=6.1, color=COLORS["muted"])
    # Three actual groups are shown as colored columns; the pattern represents
    # a one-owner assignment, not a dense projection matrix.
    group_cols = [COLORS["blue"], COLORS["orange"], COLORS["teal"]]
    group_fills = ["#cce9f6", "#ffe6ad", "#c9eee5"]
    owners = [0, 0, 1, 2, 1, 0, 2, 1]
    for j, owner in enumerate(owners):
        xx = 5.38 + j * 0.20
        ax.add_patch(Rectangle((xx, 3.70), 0.14, 0.22,
                               edgecolor=group_cols[owner], facecolor=group_fills[owner],
                               linewidth=0.7, zorder=4))
    ax.text(6.18, 3.57, "each feature has one owner", ha="center", va="top",
            fontsize=5.9, color=COLORS["muted"])
    for i, (label, color, fill) in enumerate(zip(["G1", "G2", "G3"],
                                                  group_cols, group_fills)):
        mini(ax, 5.37 + i * 0.65, 3.08, 0.48, 0.28, label, color, fill,
             fs=6.3, weight="bold")
    ax.text(6.23, 2.87, r"$C=[c_1,\ldots,c_m]$;  $\alpha\in\mathbb{R}^d$;  owner$\in\{1,\ldots,m\}^d$",
            ha="center", va="center", fontsize=5.55, color=COLORS["ink"])
    ax.text(6.23, 2.58, "dense projection A is never stored", ha="center", va="center",
            fontsize=6.0, color=COLORS["teal"], fontweight="bold")

    # Sealed-cache strip under page construction.
    rounded(ax, 0.36, 0.69, 7.18, 1.52, COLORS["teal"], "#f7fcfb",
            radius=0.10, lw=1.0)
    ax.text(0.57, 1.95, "sealed KV cache", ha="left", va="center",
            fontsize=8.0, color=COLORS["ink"], fontweight="bold")
    for i, (x, label) in enumerate([(0.62, "P1"), (1.72, "P2"), (2.82, "P3")]):
        rounded(ax, x, 1.14, 0.88, 0.52, COLORS["blue"], COLORS["blue_fill"],
                radius=0.04, lw=0.75)
        matrix(ax, x + 0.10, 1.28, 2, 5, 0.10, 0.07, COLORS["blue"], "#cce9f6")
        ax.text(x + 0.44, 1.00, label + " : C, α, owner", ha="center", va="top",
                fontsize=5.6, color=COLORS["muted"])
    ax.text(4.02, 1.42, "+", fontsize=12, color=COLORS["muted"], ha="center", va="center")
    rounded(ax, 4.28, 1.14, 1.02, 0.52, COLORS["green"], COLORS["green_fill"],
            radius=0.04, lw=0.75)
    matrix(ax, 4.43, 1.28, 2, 4, 0.11, 0.07, COLORS["green"], "#ccefd5")
    ax.text(4.79, 1.00, "active page: dense", ha="center", va="top",
            fontsize=5.6, color=COLORS["muted"])
    mini(ax, 5.58, 1.20, 1.45, 0.38, "V: INT8 + scale", COLORS["purple"],
         COLORS["purple_fill"], fs=6.0, weight="bold")
    arrow(ax, 1.40, 2.70, 1.40, 2.23, COLORS["teal"], 0.95, mutation=8)
    arrow(ax, 6.25, 2.70, 6.25, 2.23, COLORS["teal"], 0.95, mutation=8)
    ax.text(3.77, 0.78, "seal page → store (C, α, owner, V) → release dense historical K",
            ha="center", va="center", fontsize=6.3, color=COLORS["teal"],
            fontweight="bold")

    # ------------------------------------------------------------------
    # B1: direct grouped attention
    # ------------------------------------------------------------------
    module(ax, 8.22, 2.74, 3.34, 2.03, "4", "Direct grouped attention",
           COLORS["purple"], COLORS["purple_fill"])
    mini(ax, 8.52, 4.02, 0.84, 0.36, "current $q$", COLORS["orange"],
         COLORS["orange_fill"], fs=7.0, weight="bold")
    mini(ax, 9.45, 4.02, 0.96, 0.36, r"$qA$", COLORS["blue"],
         COLORS["blue_fill"], fs=8.0, weight="bold")
    mini(ax, 10.55, 4.02, 0.93, 0.36, r"scores $=(qA)C^\top$", COLORS["teal"],
         COLORS["teal_fill"], fs=5.0, weight="bold")
    arrow(ax, 9.38, 4.20, 9.50, 4.20, COLORS["orange"], 0.85, mutation=8)
    arrow(ax, 10.43, 4.20, 10.51, 4.20, COLORS["blue"], 0.85, mutation=8)
    ax.text(9.51, 3.77, "one-sparse owner map", ha="center", va="top",
            fontsize=5.9, color=COLORS["muted"])
    mini(ax, 8.52, 3.22, 1.30, 0.34, "sealed pages: C", COLORS["blue"],
         COLORS["blue_fill"], fs=6.4, weight="bold")
    mini(ax, 9.98, 3.22, 1.28, 0.34, "active page: dense", COLORS["green"],
         COLORS["green_fill"], fs=6.2, weight="bold")
    arrow(ax, 9.82, 3.39, 10.82, 4.00, COLORS["teal"], 0.82, ls=(0, (3, 2)),
          mutation=8, connectionstyle="arc3,rad=0.16")
    arrow(ax, 11.26, 3.39, 11.26, 4.00, COLORS["green"], 0.82, mutation=8)
    mini(ax, 8.44, 2.81, 2.72, 0.30, "post-RoPE: no historical RoPE, no K reconstruction",
         COLORS["red"], COLORS["red_fill"], fs=6.15, weight="bold")

    # ------------------------------------------------------------------
    # B2: page update and output
    # ------------------------------------------------------------------
    module(ax, 11.62, 2.74, 1.95, 2.03, "5", "Update",
           COLORS["red"], COLORS["red_fill"])
    mini(ax, 11.93, 4.02, 0.68, 0.34, "append", COLORS["orange"],
         COLORS["orange_fill"], fs=6.5, weight="bold")
    diamond(ax, 12.60, 3.52, 0.66, 0.47, COLORS["ink"], "white", "full?")
    arrow(ax, 12.28, 4.00, 12.48, 3.78, COLORS["orange"], 0.76, mutation=7)
    mini(ax, 11.88, 3.02, 0.62, 0.28, "No: dense", COLORS["green"],
         COLORS["green_fill"], fs=5.8)
    mini(ax, 12.70, 3.02, 0.48, 0.28, "Yes", COLORS["teal"],
         COLORS["teal_fill"], fs=5.9, weight="bold")
    arrow(ax, 12.55, 3.29, 12.22, 3.04, COLORS["green"], 0.7, mutation=7)
    arrow(ax, 12.72, 3.29, 12.94, 3.04, COLORS["teal"], 0.7, mutation=7)
    mini(ax, 12.00, 2.61, 1.06, 0.25, "seal + compress", COLORS["teal"],
         COLORS["teal_fill"], fs=5.6, weight="bold")
    arrow(ax, 13.06, 2.74, 7.60, 1.66, COLORS["teal"], 0.92, ls=(0, (3, 2)),
          mutation=8, connectionstyle="arc3,rad=0.18")

    # Final attention output band.
    rounded(ax, 8.22, 0.69, 5.15, 1.52, COLORS["red"], "#fffafa",
            radius=0.10, lw=1.0)
    ax.text(8.44, 1.95, "attention output", ha="left", va="center",
            fontsize=8.0, color=COLORS["ink"], fontweight="bold")
    mini(ax, 8.52, 1.22, 1.06, 0.40, "scores", COLORS["purple"],
         COLORS["purple_fill"], fs=6.8, weight="bold")
    arrow(ax, 9.68, 1.42, 9.93, 1.42, COLORS["red"], 0.85, mutation=8)
    mini(ax, 10.00, 1.22, 1.04, 0.40, "online\nsoftmax", COLORS["red"],
         COLORS["red_fill"], fs=6.4, weight="bold")
    arrow(ax, 11.14, 1.42, 11.39, 1.42, COLORS["red"], 0.85, mutation=8)
    mini(ax, 11.46, 1.22, 0.90, 0.40, "weighted V", COLORS["green"],
         COLORS["green_fill"], fs=6.4, weight="bold")
    arrow(ax, 12.47, 1.42, 12.72, 1.42, COLORS["red"], 0.85, mutation=8)
    mini(ax, 12.78, 1.22, 0.48, 0.40, "O", COLORS["blue"],
         COLORS["blue_fill"], fs=7.0, weight="bold")
    arrow(ax, 12.22, 2.70, 12.22, 2.25, COLORS["red"], 0.9, mutation=8)
    ax.text(10.80, 0.86, "compressed scores + original value positions",
            ha="center", va="center", fontsize=6.0, color=COLORS["muted"])

    # Cross-stage arrows and a compact storage callout.
    arrow(ax, 2.48, 3.76, 2.58, 3.76, COLORS["red"], 1.0, mutation=9)
    arrow(ax, 4.87, 3.76, 4.97, 3.76, COLORS["red"], 1.0, mutation=9)
    arrow(ax, 7.68, 3.76, 8.16, 3.76, COLORS["red"], 1.0, mutation=9)
    rounded(ax, 4.87, 0.26, 4.30, 0.28, COLORS["stage"], "white", radius=0.05,
            lw=0.65)
    ax.text(7.02, 0.40, r"per sealed page: $C[P\times m]$ + $\alpha[d]$ + owner$[d]$ + $V$",
            ha="center", va="center", fontsize=6.4, color=COLORS["ink"],
            fontweight="bold")

    fig.subplots_adjust(left=0.005, right=0.995, bottom=0.015, top=0.94)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(OUT.with_suffix(".png"), dpi=320, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


if __name__ == "__main__":
    draw()
