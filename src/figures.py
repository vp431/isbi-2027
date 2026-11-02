import json

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from pipeline import FEATURES, RESULTS, ROOT

OUT = ROOT / "figures"
BLUE, ORANGE, GREEN, PURPLE, GREY = "#3b6fb6", "#e08a2c", "#2e9e5b", "#7d5ba6", "#444444"
TEAL, NAVY = "#2a8c8c", "#3d4f7a"
STEPS = ["#4e79a7", "#59a14f", "#b07aa1", "#e15759"]


def tint(color, amount=0.85):
    rgb = np.array(matplotlib.colors.to_rgb(color))
    return tuple(rgb + (1 - rgb) * amount)


STYLE = {"font.family": "serif", "font.serif": ["Liberation Serif", "Times New Roman", "DejaVu Serif"],
         "mathtext.fontset": "stix"}


def pipeline_figure():
    plt.rcParams.update({**STYLE, "font.size": 6})
    fig, ax = plt.subplots(figsize=(3.45, 2.05))
    ax.set_xlim(0, 100)
    ax.set_ylim(2, 62)
    ax.axis("off")

    def box(x, y, w, h, fc, ec, lw=0.7, r=1.2):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0.3,rounding_size={r}", fc=fc, ec=ec, lw=lw))

    def arrow(x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", lw=0.7, color=GREY, mutation_scale=6, shrinkA=0, shrinkB=0))

    def tag(x, y, text):
        ax.text(x, y, text, fontsize=6.2, fontweight="bold", ha="left", va="center", color="#222")

    rng = np.random.default_rng(3)
    t = np.linspace(0, 1, 400)
    tag(0, 59.5, "(a) Paired recordings")
    for y, color, label, photic in [(43, BLUE, "D1: rest, eyes closed", False), (28, ORANGE, "D2: photic, eyes open", True)]:
        box(1, y, 26, 12, tint(color, 0.9), color, lw=0.9)
        ax.text(14, y + 10.2, label, color=color, fontsize=5.4, fontweight="bold", va="center", ha="center")
        for k in range(3):
            trace = 0.5 * np.sin(2 * np.pi * (6 if photic else 9) * t + k) + 0.3 * rng.standard_normal(400).cumsum() / 12
            if photic:
                trace += 0.5 * np.sin(2 * np.pi * 12 * t) * (np.sin(2 * np.pi * 1.5 * t) > 0)
            ax.plot(2.5 + 23 * t, y + 7.4 - k * 2.2 + trace * 0.5, color=color, lw=0.45)
        if photic:
            for x in np.arange(3, 26, 2.0):
                ax.plot([x, x], [y + 0.3, y + 1.0], color="#c9a227", lw=0.6)

    tag(33, 59.5, "(b) Same extractor")
    box(34, 28, 24, 27, tint(TEAL, 0.88), TEAL, lw=0.9)
    ax.text(46, 51, r"$\phi(\cdot)$: 539 features", ha="center", fontsize=6, fontweight="bold")
    for i, s in enumerate(["band/sub-band power", "band ratios", "asymmetry", "entropy, Hjorth", "alpha peak", "PLV network"]):
        ax.text(46, 46.3 - i * 3.3, s, ha="center", fontsize=5.0, color=GREY)
    arrow(27.6, 49, 33.4, 45)
    arrow(27.6, 34, 33.4, 38)

    tag(65, 59.5, "(c) Inputs compared")
    chips = [(BLUE, r"D1: $\mathbf{f}_1$ (539)"), (ORANGE, r"D2: $\mathbf{f}_2$ (539)"),
             (GREEN, r"Concat: $[\mathbf{f}_1;\mathbf{f}_2]$ (1078)"), (PURPLE, r"Mean: $(\mathbf{f}_1{+}\mathbf{f}_2)/2$ (539)")]
    for i, (color, text) in enumerate(chips):
        y = 50 - i * 6.6
        box(65, y, 34, 4.8, tint(color, 0.82), color, lw=0.9, r=0.8)
        ax.text(82, y + 2.4, text, ha="center", va="center", fontsize=5.6, color="#222")
    arrow(58.6, 41.5, 64.4, 41.5)

    tag(0, 23.5, "(d) Per training fold, fitted on training subjects only")
    ax.plot([82, 82], [29.5, 20.5], color=GREY, lw=0.7)
    ax.plot([82, 7], [20.5, 20.5], color=GREY, lw=0.7)
    arrow(7, 20.5, 7, 17.6)
    for i, s in enumerate(["Robust\nscaling", "ANOVA\ntop-$k$", "PCA\n(20)", "Extra\nTrees"]):
        x = 1 + i * 16.5
        box(x, 6, 12, 11, tint(STEPS[i], 0.8), STEPS[i], lw=0.9)
        ax.text(x + 6, 11.5, s, ha="center", va="center", fontsize=5.8)
        if i:
            arrow(x - 4.2, 11.5, x - 0.6, 11.5)
    box(70, 6, 29, 11, tint("#c9a227", 0.82), "#c9a227", lw=0.9)
    ax.text(84.5, 11.5, "10 seeds × 5-fold CV\nidentical splits\nfor every input", ha="center", va="center", fontsize=5.5)
    arrow(63.8, 11.5, 69.4, 11.5)
    fig.savefig(OUT / "fig_pipeline.pdf", bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)


def results_figure():
    plt.rcParams.update({**STYLE, "font.size": 7.5, "axes.linewidth": 0.6,
                         "xtick.major.width": 0.6, "ytick.major.width": 0.6})
    clf = json.loads((RESULTS / "classifiers.json").read_text())
    pairing = json.loads((RESULTS / "pairing.json").read_text())["control"]
    df = pd.read_csv(FEATURES)
    fig = plt.figure(figsize=(7.1, 2.05))
    grid = fig.add_gridspec(1, 4, width_ratios=[1.2, 1.2, 1.05, 1.25], wspace=0.45)
    labels = {"SVM-C1": "SVM$_{C=1}$", "SVM-C10": "SVM$_{C=10}$"}

    for i, task in enumerate(["AD vs. CN", "FTD vs. CN"]):
        ax = fig.add_subplot(grid[i])
        models = sorted({k.split("|")[1] for k in clf if k.startswith(task)}, key=lambda m: clf[f"{task}|{m}|Concat"][0])
        for j, m in enumerate(models):
            a, b, c = (clf[f"{task}|{m}|{n}"][0] for n in ["D1", "D2", "Concat"])
            ax.plot([min(a, b, c), max(a, b, c)], [j, j], color="#bbb", lw=0.8, zorder=1)
            ax.scatter(a, j, s=14, color=BLUE, marker="o", zorder=3, label="D1 (rest)" if j == 0 else None)
            ax.scatter(b, j, s=14, color=ORANGE, marker="s", zorder=3, label="D2 (photic)" if j == 0 else None)
            ax.scatter(c, j, s=18, color=GREEN, marker="D", zorder=4, label="Concat (D1+D2)" if j == 0 else None)
        ax.set_yticks(range(len(models)))
        ax.set_yticklabels([labels.get(m, m) for m in models], fontsize=6.8)
        ax.set_xlabel("Accuracy (%)")
        ax.set_title(f"({'ab'[i]}) {task.replace('.', '')}", fontsize=7.5)
        ax.grid(axis="x", alpha=0.3, lw=0.4)
        ax.spines[["top", "right"]].set_visible(False)
        if i == 0:
            handles = ax.get_legend_handles_labels()
    fig.legend(*handles, loc="upper left", bbox_to_anchor=(0.06, 1.08), ncol=3, fontsize=6.5,
               frameon=False, handletextpad=0.1, columnspacing=0.8)

    ax = fig.add_subplot(grid[2])
    ticks = []
    for gi, g in enumerate(["CN", "AD", "FTD"]):
        for ci, (cond, color) in enumerate([("D1", BLUE), ("D2", ORANGE)]):
            v = np.log10(df.loc[df.group == g, f"{cond}_theta_alpha_ratio_mean_O"].to_numpy())
            pos = gi * 2.6 + ci
            bp = ax.boxplot(v, positions=[pos], widths=0.8, patch_artist=True, showfliers=False,
                            medianprops=dict(color="k", lw=0.8), whiskerprops=dict(lw=0.6), capprops=dict(lw=0.6))
            bp["boxes"][0].set(facecolor=color, alpha=0.75, lw=0.6)
            jitter = np.random.default_rng(gi * 2 + ci).uniform(-0.18, 0.18, len(v))
            ax.scatter(pos + jitter, v, s=2, color="k", alpha=0.45, zorder=3, lw=0)
        ticks.append(gi * 2.6 + 0.5)
    ax.set_xticks(ticks)
    ax.set_xticklabels(["CN", "AD", "FTD"])
    ax.set_ylabel(r"$\log_{10}$ occipital $\theta/\alpha$")
    ax.set_title(r"(c) $\theta/\alpha$ in D1 (blue), D2 (orange)", fontsize=7.5)
    ax.spines[["top", "right"]].set_visible(False)

    ax = fig.add_subplot(grid[3])
    tasks = [("AD vs. CN", "AD\nvs CN"), ("FTD vs. CN", "FTD\nvs CN"), ("AD vs. FTD", "AD\nvs FTD"), ("3-class", "3-\nclass")]
    x, w = np.arange(len(tasks)), 0.2
    for oi, (name, color) in enumerate([("Concat", GREEN), ("Mean", PURPLE)]):
        true = [pairing[f"true|{t}|{name}"]["auc"] for t, _ in tasks]
        shuffled = np.array([[pairing[f"shuffled{r}|{t}|{name}"]["auc"] for r in range(3)] for t, _ in tasks])
        mean = shuffled.mean(1)
        off = (oi * 2 - 1.5) * w
        ax.bar(x + off, true, w, color=color, alpha=0.9, label=f"{name}, true", lw=0)
        ax.bar(x + off + w, mean, w, color="white", edgecolor=color, hatch="////", lw=0.6, label=f"{name}, shuffled")
        ax.errorbar(x + off + w, mean, yerr=[mean - shuffled.min(1), shuffled.max(1) - mean],
                    fmt="none", ecolor="k", elinewidth=0.6, capsize=1.5)
    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label in tasks], fontsize=6.2, linespacing=0.9)
    ax.set_ylim(40, 113)
    ax.set_yticks(range(40, 101, 10))
    ax.set_ylabel("AUC")
    ax.axhline(50, color="#999", lw=0.5, ls=":")
    ax.set_title("(d) pairing control", fontsize=7.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=5.3, frameon=False, loc="upper left", ncol=2, handlelength=1.2, columnspacing=0.6, borderaxespad=0.1)
    fig.savefig(OUT / "fig_results.pdf", bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    pipeline_figure()
    results_figure()
    print(f"figures written to {OUT}")


if __name__ == "__main__":
    main()
