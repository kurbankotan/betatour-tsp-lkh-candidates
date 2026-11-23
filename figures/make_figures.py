"""The five figures of the BetaTour manuscript in the journal style: Times-like serif text, inward ticks, purple and
dark mustard (a pair that passes the colour-vision checks), no titles inside the figures (the captions carry them).

    python figures/make_figures.py [OUT_DIR]

writes figure1_architecture ... figure5_self_correction as PNG (300 dpi) and PDF. The numbers are those of the paper:

  Figure 2  results/v3/RAPOR_v3.md, coverage table (per-set means of results/v3/results/coverage_v3.csv)
  Figure 3  results/v3/results/lkh_eval_v3.csv (W/T/L over the 51 instances at 25, 50 and 100% of the budget)
  Figure 4  analysis/v4_size_split_and_spread.py on results/v4 (mean single-run gap, population SD of the six final
            tours, per-run W/T/L)
  Figure 5  analysis/v12_self_correction.py on results/v12 (Table 10 and the 75% row of the same script)

Figure 1 is the architecture diagram of Section 3.
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import FancyArrowPatch, Patch, Rectangle  # noqa: E402

PURPLE, MUSTARD = "#5E3C99", "#B8860B"
INK, MID, LIGHT = "#111111", "#595959", "#BFBFBF"
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Liberation Serif", "TeX Gyre Termes", "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
    "legend.fontsize": 8.5, "legend.frameon": False,
    "axes.linewidth": 0.8, "axes.edgecolor": INK, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK, "ytick.color": INK, "xtick.direction": "in", "ytick.direction": "in",
    "xtick.top": True, "ytick.right": True, "xtick.major.size": 3.5, "ytick.major.size": 3.5,
    "xtick.minor.size": 2.0, "ytick.minor.size": 2.0, "xtick.major.width": 0.8, "ytick.major.width": 0.8,
    "savefig.dpi": 300, "pdf.fonttype": 42,
})


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{name}.{ext}", bbox_inches="tight", pad_inches=0.03, facecolor="white")
    plt.close(fig)


# ------------------------------------------------------------------------------------------------------- Figure 1
def box(ax, x, y, w, h, title, lines, band, columns=1, band_h=0.24):
    """Box with a coloured header band: title in the band, body lines below (one or two columns)."""
    ax.add_patch(Rectangle((x, y), w, h, facecolor="white", edgecolor="none", zorder=1))
    ax.add_patch(Rectangle((x, y + h - band_h), w, band_h, facecolor=band, edgecolor="none", zorder=2))
    ax.add_patch(Rectangle((x, y), w, h, facecolor="none", edgecolor=INK, linewidth=0.8, zorder=3))
    ax.text(x + 0.09, y + h - band_h / 2, title, ha="left", va="center", fontsize=8.6, fontweight="bold",
            color="white" if band == PURPLE else INK, zorder=4)
    per_col = int(np.ceil(len(lines) / columns))
    for k, line in enumerate(lines):
        col, row = divmod(k, per_col)
        ax.text(x + 0.09 + col * w / columns, y + h - band_h - 0.13 - row * 0.175, line, ha="left", va="center",
                fontsize=8, color=INK, zorder=4)


def arrow(ax, x0, y0, x1, y1, label=None):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=8, linewidth=0.8,
                                 color=MID, shrinkA=0, shrinkB=0, zorder=0))
    if label:
        ax.text(x0 + 0.06, (y0 + y1) / 2, label, ha="left", va="center", fontsize=7.6, style="italic", color=MID)


def figure1():
    heights = [0.62, 0.45, 0.82, 0.82, 0.45, 0.62]
    gaps = [0.30, 0.30, 0.36, 0.30, 0.30]
    W, H = 5.4, 0.04 + sum(heights) + sum(gaps) + 0.04
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")
    L, R, gap = 0.04, W - 0.04, 0.12
    # rows from the top: (y of the bottom edge, height)
    ya, ha = H - 0.04 - 0.62, 0.62
    yb, hb = ya - 0.30 - 0.45, 0.45
    yc, hc = yb - 0.30 - 0.82, 0.82
    yd, hd = yc - 0.36 - 0.82, 0.82
    ye, he = yd - 0.30 - 0.45, 0.45
    yf, hf = ye - 0.30 - 0.62, 0.62
    half = (R - L - gap) / 2
    third = (R - L - 2 * gap) / 3
    xa = [L, L + half + gap]
    xb = [L, L + third + gap, L + 2 * (third + gap)]
    box(ax, xa[0], ya, half, ha, "Instance",
        ["n points, 16 nearest neighbours each", "+ up to 4 edges per node that the tours use"], MUSTARD)
    box(ax, xa[1], ya, half, ha, "Tour ensemble, M tours",
        ["one LKH-3 trial per tour, different seeds", "POPMUSIC candidates, no ascent"], MUSTARD)
    box(ax, xb[0], yb, third, hb, "Nodes", ["n × 128"], PURPLE)
    box(ax, xb[1], yb, third, hb, "Edges", ["n × 16 directed edges × 64"], PURPLE)
    box(ax, xb[2], yb, third, hb, "Tour alignment", ["M tours × n nodes × 32"], PURPLE)
    box(ax, L, yc, R - L, hc, "Alignment block × 2",
        ["1  Column attention across the M tours", "2  Tour convolution: successor + predecessor",
         "3  Row attention over neighbours, edge bias", "4  Transition, two-layer perceptron",
         "5  Outer-product mean: tours → edges", "6  One pair block, as below"], PURPLE, columns=2)
    box(ax, L, yd, R - L, hd, "Pair block × 6",
        ["Node attention over neighbours, edge bias", "Edge update from its two end nodes",
         "Mixing of the two directions of an edge", "Triangle updates, outgoing and incoming",
         "k among the 12 nearest of both i and j", "Node and edge transitions"], PURPLE, columns=2)
    box(ax, xb[0], ye, third, he, "Edge scores", ["symmetric for mutual edges"], PURPLE)
    box(ax, xb[1], ye, third, he, "Penalties π", ["trained, not used in LKH runs"], PURPLE)
    box(ax, xb[2], ye, third, he, "Confidence", ["are the top 2 both tour edges?"], PURPLE)
    box(ax, L, yf, R - L, hf, "LKH-3 candidate lists",
        ["rank the 16 neighbours by score and write α = 0, 100, 200, …",
         "top 5, or top 3 then POPMUSIC; zero penalties, no ascent"], MUSTARD)
    cb = [x + third / 2 for x in xb]
    arrow(ax, cb[0], ya, cb[0], yb + hb)
    arrow(ax, cb[1] - 0.25, ya, cb[1] - 0.25, yb + hb)
    arrow(ax, cb[1] + 0.25, ya, cb[1] + 0.25, yb + hb, "edge-use fractions")
    arrow(ax, cb[2], ya, cb[2], yb + hb)
    for c in cb:
        arrow(ax, c, yb, c, yc + hc)
        arrow(ax, c, yd, c, ye + he)
    arrow(ax, cb[1], yc, cb[1], yd + hd, "node and edge states")
    arrow(ax, cb[0], ye, cb[0], yf + hf)
    save(fig, "figure1_architecture")


# ------------------------------------------------------------------------------------------------------- Figure 2
def figure2():
    sets = ["Validation", "Uniform 1k", "Clustered 1k", "Uniform 10k", "Clustered 10k", "TSPLIB"]
    series = [("POPMUSIC", [3.95, 5.23, 5.62, 5.60, 6.18, 7.25], "white"),
              ("α-nearness after ascent", [4.56, 2.27, 3.89, 2.79, 2.46, 4.69], LIGHT),
              ("Counting, M = 8", [1.32, 1.59, 1.86, 2.08, 2.69, 3.93], MUSTARD),
              ("BetaTour, M = 8", [0.48, 0.31, 0.61, 0.42, 0.52, 2.55], PURPLE)]
    fig, ax = plt.subplots(figsize=(6.5, 3.0))
    x = np.arange(len(sets))
    width = 0.19
    for k, (label, values, color) in enumerate(series):
        pos = x + (k - 1.5) * width
        ax.bar(pos, values, width * 0.92, color=color, edgecolor=INK, linewidth=0.6, label=label, zorder=2)
        if k == 3:
            for p, v in zip(pos, values):
                ax.text(p - width * 0.46, v + 0.1, f"{v:.2f}", ha="left", va="bottom", fontsize=7.8)
    ax.set_xticks(x, sets)
    ax.set_xlim(-0.55, len(sets) - 0.45)
    ax.set_ylim(0, 8)
    ax.set_ylabel("Missing reference-tour edges\nper 1,000 nodes")
    ax.legend(ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.0), handlelength=1.4, columnspacing=1.4)
    save(fig, "figure2_candidate_coverage")


# ------------------------------------------------------------------------------------------------------- Figure 3
def figure3():
    groups = [("BetaTour top 5 vs LKH-3 POPMUSIC", [(12, 0, 39), (15, 4, 32), (24, 4, 23)]),
              ("Control (ensemble start, POPMUSIC lists) vs LKH-3 POPMUSIC", [(10, 1, 40), (13, 4, 34), (16, 6, 29)]),
              ("BetaTour top 5 vs control", [(16, 26, 9), (27, 11, 13), (31, 6, 14)])]
    points = ["25% of budget", "50% of budget", "100% of budget"]
    fig, axes = plt.subplots(3, 1, figsize=(6.5, 3.9), sharex=True)
    for ax, (title, rows) in zip(axes, groups):
        y = np.arange(3)
        won = [w for w, _, _ in rows]
        lost = [l for _, _, l in rows]
        ax.barh(y, won, 0.62, color=PURPLE, edgecolor=INK, linewidth=0.5, zorder=2)
        ax.barh(y, [-v for v in lost], 0.62, color=MUSTARD, edgecolor=INK, linewidth=0.5, zorder=2)
        for yy, (w, t, l) in zip(y, rows):
            ax.text(w + 0.8, yy, str(w), ha="left", va="center", fontsize=8)
            ax.text(-l - 0.8, yy, str(l), ha="right", va="center", fontsize=8)
            ax.text(47, yy, f"{t} tie" + ("" if t == 1 else "s"), ha="right", va="center", fontsize=8, color=MID)
        ax.axvline(0, color=INK, linewidth=0.8, zorder=3)
        ax.set_yticks(y, points)
        ax.tick_params(axis="y", right=False)
        ax.set_ylim(2.6, -0.6)
        ax.set_title(title, loc="left", fontsize=8.8, fontweight="bold", pad=3)
    axes[-1].set_xlim(-48, 48)
    ticks = [-40, -20, 0, 20, 40]
    axes[-1].set_xticks(ticks, [str(abs(t)) for t in ticks])
    axes[-1].set_xlabel("Instances lost (left) and won (right) by the first configuration (51 instances)")
    handles = [Patch(facecolor=PURPLE, edgecolor=INK, linewidth=0.5, label="won"),
               Patch(facecolor=MUSTARD, edgecolor=INK, linewidth=0.5, label="lost")]
    fig.legend(handles=handles, ncol=2, loc="lower center", bbox_to_anchor=(0.5, 0.985))
    fig.tight_layout(h_pad=0.6)
    save(fig, "figure3_single_core_wtl")


# ------------------------------------------------------------------------------------------------------- Figure 4
def figure4():
    classes = [("1,000 nodes", "27/59/10"), ("10,000 nodes", "59/0/37"), ("TSPLIB, 3,038–18,512", "33/5/28"),
               ("20,000 nodes", "15/0/33")]
    gap = {"pop": [0.0122, 0.0252, 0.0277, 0.0173], "tf": [0.0055, 0.0199, 0.0194, 0.0323]}
    spread = {"pop": [0.0153, 0.0111, 0.0165, 0.0112], "tf": [0.0054, 0.0081, 0.0078, 0.0124]}
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 3.05), sharey=True, gridspec_kw={"wspace": 0.08})
    y = np.arange(len(classes))
    h = 0.36
    for ax, data, label, xmax in ((axes[0], gap, "Mean gap of a single run (%)", 0.038),
                                  (axes[1], spread, "Spread of the six final tours (%)", 0.0205)):
        ax.barh(y - h / 2, data["pop"], h * 0.92, color=MUSTARD, edgecolor=INK, linewidth=0.5, zorder=2)
        ax.barh(y + h / 2, data["tf"], h * 0.92, color=PURPLE, edgecolor=INK, linewidth=0.5, zorder=2)
        for yy, a, b in zip(y, data["pop"], data["tf"]):
            ax.text(a + xmax * 0.012, yy - h / 2, f"{a:.4f}", ha="left", va="center", fontsize=7.8)
            ax.text(b + xmax * 0.012, yy + h / 2, f"{b:.4f}", ha="left", va="center", fontsize=7.8)
        ax.set_xlim(0, xmax)
        ax.set_xlabel(label)
    axes[0].set_yticks(y, [f"{name}\nper run {wtl}" for name, wtl in classes])
    axes[0].set_ylim(len(classes) - 0.45, -0.55)
    handles = [Patch(facecolor=MUSTARD, edgecolor=INK, linewidth=0.5, label="LKH-3 POPMUSIC"),
               Patch(facecolor=PURPLE, edgecolor=INK, linewidth=0.5, label="BetaTour")]
    axes[0].legend(handles=handles, loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=2, handlelength=1.4,
                   borderaxespad=0.2)
    save(fig, "figure4_multi_core")


# ------------------------------------------------------------------------------------------------------- Figure 5
def figure5():
    names = ["Uniform 1k", "Clustered 1k", "Uniform 10k", "Clustered 10k", "TSPLIB"]
    share = [0, 25, 50, 75, 100]
    dynamic = [[3.83, 1.19, 0.85, 0.65, 0.58], [6.21, 2.10, 1.33, 1.23, 1.10], [5.58, 1.91, 1.68, 1.58, 1.47],
               [6.01, 2.21, 1.73, 1.51, 1.39], [5.81, 2.40, 2.16, 2.05, 2.00]]
    beta = [0.62, 0.88, 0.46, 0.47, 1.65]
    count = [2.25, 2.25, 2.33, 2.89, 3.32]
    fig, axes = plt.subplots(1, 5, figsize=(6.5, 2.75), sharey=True, gridspec_kw={"wspace": 0.16})
    for ax, name, d, b, c in zip(axes, names, dynamic, beta, count):
        ax.plot([0, 100], [c, c], color=MUSTARD, linewidth=1.4, linestyle=(0, (4, 2)), zorder=2)
        ax.plot([0, 100], [b, b], color=PURPLE, linewidth=1.6, zorder=2)
        ax.plot(share, d, color=INK, linewidth=1.0, marker="o", markersize=3.2, zorder=3)
        ax.set_title(name, fontsize=8.8, pad=3)
        ax.set_xlim(-12, 112)
        ax.set_xticks([0, 50, 100])
        ax.set_xticks([25, 75], minor=True)
    axes[0].set_ylim(0, 7)
    axes[0].set_ylabel("Missing reference-tour edges\nper 1,000 nodes")
    fig.supxlabel("Share of the time budget elapsed (%)", fontsize=9, y=0.0)
    handles = [Line2D([], [], color=INK, linewidth=1.0, marker="o", markersize=3.2,
                      label="LKH-3 candidate set (0% = initial POPMUSIC-5 lists)"),
               Line2D([], [], color=PURPLE, linewidth=1.6, label="BetaTour top 5, M = 6"),
               Line2D([], [], color=MUSTARD, linewidth=1.4, linestyle=(0, (4, 2)), label="Counting top 5, M = 6")]
    fig.legend(handles=handles, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 0.97), handlelength=2.2,
               columnspacing=1.2)
    save(fig, "figure5_self_correction")


if __name__ == "__main__":
    figure1()
    figure2()
    figure3()
    figure4()
    figure5()
    print("written to", OUT)
