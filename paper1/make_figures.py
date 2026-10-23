"""Figures for paper 1 (2026-10-23). Reads only recorded results -- no new runs.
  Figure 3  a moment in Helix Gold's own world vs one exposure record of the sealed exam (schematic)
  Figure 2  the v1 score table as a heat map (exam/v1/RESULTS.md: accuracy per judged line, pass marks)
  Figure 1  the learning-power curves (Helix Gold repository: d72_10.out, d72b_10.out, d72c_10.out, d72d_10.out)
    python3 paper1/make_figures.py [HELIX_GOLD_DIR]   (default: ../helix_gold in the release, else the working repository)"""
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyBboxPatch

HERE = os.path.dirname(os.path.abspath(__file__))
SHARED = os.path.dirname(HERE)
GOLD = (sys.argv[1] if len(sys.argv) > 1 else os.path.join(SHARED, "helix_gold")
        if os.path.isdir(os.path.join(SHARED, "helix_gold")) else os.path.join(os.path.dirname(SHARED), "recursive-improvement-experiment"))
OUT = os.path.join(HERE, "figures")
INK, INK2, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df", "#fcfcfb"
SEQ = ["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "figure.facecolor": "white", "axes.facecolor": "white"})


def box(ax, x, y, w, h, text, fc="#f0efec", ec=MUTED, size=8.5, color=INK, weight="normal"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.01,rounding_size=0.015", fc=fc, ec=ec, lw=0.8))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size, color=color, weight=weight)


def f1():
    fig, ax = plt.subplots(figsize=(7.2, 3.1))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.02, 0.95, "Helix Gold's own world: one moment", fontsize=9, weight="bold", color=INK)
    box(ax, 0.02, 0.50, 0.40, 0.36, "", fc="#eef4fc", ec="#86b6ef")
    for i, (t, sub) in enumerate((("CUP", "name"), ("glyph of\nthe cup", "picture"), ("cell\n(2, 3)", "position"),
                                  ("other\nglyphs ×1–3", "pictures"))):
        box(ax, 0.035 + i * 0.097, 0.62, 0.09, 0.16, t, fc="white", ec="#86b6ef", size=7)
        ax.text(0.035 + i * 0.097 + 0.045, 0.56, sub, ha="center", fontsize=7, color=INK2)
    ax.text(0.22, 0.42, "every fact of the moment arrives together;\nthe body pairs what arrives together",
            ha="center", va="top", fontsize=8, color=INK2)
    ax.text(0.53, 0.95, "Sealed exam v1: one exposure record", fontsize=9, weight="bold", color=INK)
    box(ax, 0.52, 0.74, 0.46, 0.12, "sentence “there is a cup” + scene rows (picture, cell) …", fc="#fdf0ea",
        ec="#ec835a", size=7.5)
    ax.annotate("", xy=(0.75, 0.58), xytext=(0.75, 0.73), arrowprops=dict(arrowstyle="->", color=INK2, lw=1))
    ax.text(0.765, 0.655, "format converter (conv-v1)", fontsize=7.5, color=INK2)
    for i, t in enumerate(("moment 1:\nthe words", "moment 2:\nrow 1", "moment 3:\nrow 2", "…")):
        box(ax, 0.52 + i * 0.118, 0.40, 0.108, 0.16, t, fc="white", ec="#ec835a", size=7)
    ax.text(0.75, 0.32, "the record is split across moments; names are asked\nthrough a sentence frame (“there is ___”)",
            ha="center", va="top", fontsize=8, color=INK2)
    ax.text(0.5, 0.04, "Rejoining a record into one moment did not rescue names (§7.4); asked through a frame, the "
            "frame's statistics win over the scene (§8).", ha="center", fontsize=7.5, color=MUTED)
    fig.savefig(os.path.join(OUT, "fig3_moment_vs_record.png"), dpi=220, bbox_inches="tight")
    fig.savefig(os.path.join(OUT, "fig3_moment_vs_record.pdf"), bbox_inches="tight")
    plt.close(fig)


def f2():
    lines = open(os.path.join(SHARED, "exam", "v1", "RESULTS.md")).read().splitlines()
    i0 = next(i for i, l in enumerate(lines) if l.startswith("| 기준 | 줄 |"))
    head = [c.strip() for c in lines[i0].strip("|").split("|")]
    rows, labels, passed = [], [], []
    for l in lines[i0 + 2:]:
        if not l.startswith("|"):
            break
        c = [x.strip() for x in l.strip("|").split("|")]
        if c[1].startswith("**"):
            continue
        name = re.sub(r"\s*\(선 [0-9.]+\)", "", c[1])
        line = re.search(r"\(선 ([0-9.]+)\)", c[1]).group(1)
        labels.append(f"{c[0]} {name}  [{line}]")
        vals, ok = [], []
        for x in c[2:]:
            m = re.match(r"([0-9.]+)", x)
            vals.append(float(m.group(1)) if m else float("nan"))
            ok.append("✓" in x)
        rows.append(vals)
        passed.append(ok)
    names = {"dami": "Helix\nGold", "sia": "Helix\nBlue", "ara_whole": "Helix\nTeal\nwhole", "ara_alone": "Helix\nTeal\nalone"}
    cols = [names.get(h, h) for h in head[2:]]
    import numpy as np
    A = np.array(rows)
    cmap = LinearSegmentedColormap.from_list("seq", SEQ)
    fig, ax = plt.subplots(figsize=(8.6, 9.2))
    im = ax.imshow(np.clip(A, 0, 1), cmap=cmap, vmin=0, vmax=1, aspect="auto")
    for r in range(A.shape[0]):
        for c in range(A.shape[1]):
            v = A[r, c]
            if v != v:
                ax.text(c, r, "–", ha="center", va="center", fontsize=6, color=MUTED)
                continue
            ax.text(c, r, f"{v:.2f}" + (" ✓" if passed[r][c] else ""), ha="center", va="center", fontsize=5.6,
                    color="white" if min(v, 1) > 0.55 else INK, weight="bold" if passed[r][c] else "normal")
    ax.set_xticks(range(len(cols)), cols, fontsize=7)
    ax.xaxis.tick_top()
    ax.set_yticks(range(len(labels)), labels, fontsize=6.5)
    ax.axvline(3.5, color=INK, lw=1)
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
    cb.set_label("accuracy (C2 kept: end / right after, capped at 1 for colour)", fontsize=7)
    cb.outline.set_visible(False)
    ax.set_title("Sealed exam v1: every judged line, 30 lives each; [line] = pass line, ✓ = Wilson lower bound clears it.\n"
                 "Left of the rule: the three learners; right: the ten baselines (M C N B L T simple, P V MLP TF learned).",
                 fontsize=7.5, color=INK2, loc="left", pad=28)
    fig.savefig(os.path.join(OUT, "fig2_v1_scores.png"), dpi=220, bbox_inches="tight")
    fig.savefig(os.path.join(OUT, "fig2_v1_scores.pdf"), bbox_inches="tight")
    plt.close(fig)


def f3():
    want = (("d72_10.out", "R1 CAT cat:", "new category word (162)"), ("d72b_10.out", "R1 NEG:", "negation marker (164)"),
            ("d72c_10.out", "R1 ICE:", "slippery floor (165)"), ("d72d_10.out", "R1 NEXT:", "next moment after a bell (173)"),
            ("d72b_10.out", "R1 CHAIN:", "two new verbs chained (in design)"))
    ks = ("0", "1", "2", "3-4", "5-8", "9-16", "17+")
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    for i, (f, tag, name) in enumerate(want):
        line = next(l for l in open(os.path.join(GOLD, f)) if l.startswith(tag))
        ys = [float(v) if v != "nan" else float("nan") for v in re.findall(r"k [0-9+\-]+: ([0-9.]+|nan)", line)]
        c = CAT[0] if "in design" in name else CAT[i + 1]
        ax.plot(range(len(ys)), ys, color=c, lw=2, marker="o", ms=4, label=name)
    ax.axhline(0.25, color=MUTED, lw=1, ls=(0, (3, 3)))
    ax.text(2.2, 0.262, "chance (4 candidates)", fontsize=7, color=MUTED)
    ax.set_xticks(range(len(ks)), ks)
    ax.set_xlabel("times the rule was heard before the item (k)")
    ax.set_ylabel("accuracy (10 seeds pooled)")
    ax.set_ylim(0, 1.05)
    ax.set_xlim(-0.2, len(ks) - 0.8)
    ax.grid(axis="y", color=GRID, lw=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(loc="center right", fontsize=7, frameon=False, bbox_to_anchor=(1.0, 0.62))
    ax.set_title("Learning-power test: in-design control vs rules outside the design", fontsize=8.5, color=INK, loc="left")
    fig.savefig(os.path.join(OUT, "fig1_learning_power.png"), dpi=220, bbox_inches="tight")
    fig.savefig(os.path.join(OUT, "fig1_learning_power.pdf"), bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    f1()
    f2()
    f3()
    print("figures written to", OUT)
