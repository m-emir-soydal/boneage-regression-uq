"""Generate age distribution visualization for the presentation."""

import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_CSV = ROOT_DIR / "data/rsna_training/train.csv"
OUT_DIR = ROOT_DIR / "table_outputs"
OUT_DIR.mkdir(exist_ok=True)


def load_data(path: Path):
    ages_male, ages_female = [], []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            age = float(row["boneage"])
            if row["male"].strip() == "True":
                ages_male.append(age)
            else:
                ages_female.append(age)
    return ages_male, ages_female


def main() -> None:
    ages_male, ages_female = load_data(DATA_CSV)
    all_ages = ages_male + ages_female

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.patch.set_facecolor("#f8f9fa")

    # ── Left: overlapping histogram by sex ──────────────────────────────────
    ax = axes[0]
    ax.set_facecolor("#f8f9fa")
    bins = np.arange(0, 240, 12)  # 1-year bins

    ax.hist(
        ages_male,
        bins=bins,
        color="#2563eb",
        alpha=0.65,
        label=f"Male  (n={len(ages_male):,})",
        edgecolor="white",
        linewidth=0.4,
    )
    ax.hist(
        ages_female,
        bins=bins,
        color="#f97316",
        alpha=0.65,
        label=f"Female (n={len(ages_female):,})",
        edgecolor="white",
        linewidth=0.4,
    )

    ax.set_xlabel("Bone Age (months)", fontsize=12, labelpad=8)
    ax.set_ylabel("Number of Images", fontsize=12, labelpad=8)
    ax.set_title("Age Distribution by Sex", fontsize=14, fontweight="bold", pad=14)
    ax.legend(fontsize=11, framealpha=0.9)
    ax.set_xlim(0, 228)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(24))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(12))

    # secondary x-axis in years
    ax2 = ax.twiny()
    ax2.set_xlim(0, 228)
    ax2.set_xticks([0, 24, 48, 72, 96, 120, 144, 168, 192, 216])
    ax2.set_xticklabels(["0", "2", "4", "6", "8", "10", "12", "14", "16", "18"])
    ax2.set_xlabel("Bone Age (years)", fontsize=11, labelpad=8)

    ax.spines[["top", "right"]].set_visible(False)

    # ── Right: box + strip showing summary stats ─────────────────────────────
    ax = axes[1]
    ax.set_facecolor("#f8f9fa")

    data = [ages_male, ages_female]
    colors = ["#2563eb", "#f97316"]
    labels = ["Male", "Female"]
    positions = [1, 2]

    bp = ax.boxplot(
        data,
        positions=positions,
        patch_artist=True,
        widths=0.45,
        medianprops=dict(color="white", linewidth=2.5),
        whiskerprops=dict(linewidth=1.5),
        capprops=dict(linewidth=1.5),
        flierprops=dict(marker="o", markersize=2.5, alpha=0.3),
    )
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
    for whisker in bp["whiskers"]:
        whisker.set_color("#555555")
    for cap in bp["caps"]:
        cap.set_color("#555555")

    rng = np.random.default_rng(42)
    for pts, pos, color in zip(data, positions, colors):
        jitter = rng.uniform(-0.12, 0.12, size=len(pts))
        ax.scatter(
            [pos + j for j in jitter],
            pts,
            alpha=0.07,
            s=5,
            color=color,
            zorder=2,
        )

    # annotate medians
    for pts, pos in zip(data, positions):
        med = float(np.median(pts))
        ax.annotate(
            f"Median\n{med:.0f} mo",
            xy=(pos, med),
            xytext=(pos + 0.28, med),
            fontsize=9,
            va="center",
            color="#333333",
            arrowprops=dict(arrowstyle="-", color="#aaaaaa", lw=0.8),
        )

    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylabel("Bone Age (months)", fontsize=12, labelpad=8)
    ax.set_title("Distribution by Sex (Boxplot)", fontsize=14, fontweight="bold", pad=14)
    ax.set_ylim(-5, 240)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(24))
    ax.spines[["top", "right"]].set_visible(False)

    # ── summary stats text ────────────────────────────────────────────────────
    stats_lines = [
        f"Total images: {len(all_ages):,}",
        f"Age range: 1 – 228 months",
        f"Median: {float(np.median(all_ages)):.0f} months",
        f"Mean: {float(np.mean(all_ages)):.1f} months",
        f"Std dev: {float(np.std(all_ages, ddof=1)):.1f} months",
    ]
    fig.text(
        0.5,
        -0.03,
        "   |   ".join(stats_lines),
        ha="center",
        va="top",
        fontsize=10,
        color="#555555",
        style="italic",
    )

    fig.suptitle(
        "RSNA Pediatric Bone Age Dataset",
        fontsize=16,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout(rect=[0, 0.02, 1, 1])

    out = OUT_DIR / "age_distribution.png"
    fig.savefig(out, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"Saved → {out}")


if __name__ == "__main__":
    main()
