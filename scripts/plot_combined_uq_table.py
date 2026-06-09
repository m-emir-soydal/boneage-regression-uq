"""Plot combined UQ results from table_outputs/combined_uq_table.csv."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = ROOT / "table_outputs" / "combined_uq_table.csv"
OUT_PATH = ROOT / "table_outputs" / "combined_uq_charts.png"

C_BASELINE = "#2563eb"
C_MULTI = "#f97316"
C_TARGET = "#64748b"
C_IDEAL = "#94a3b8"
C_NAVY = "#1e3a5f"
C_BLACK = "#111827"


def load() -> pd.DataFrame:
    df = pd.read_csv(CSV_PATH)
    df["Alpha"] = df["Alpha"].astype(float)
    df["label"] = df["Method"] + " · " + df["Variant"]
    return df


def plot_calibration(ax: plt.Axes, df: pd.DataFrame) -> None:
    alphas = np.linspace(0.5, 1.0, 50)
    ax.plot(alphas, alphas, "--", color=C_IDEAL, lw=1.5, label="Ideal (PICP = α)")

    styles = {
        ("Conformal", "Baseline"): (C_BASELINE, "o", "-"),
        ("Conformal", "Multi"): (C_MULTI, "o", "-"),
        ("MC Dropout", "Baseline"): (C_BASELINE, "s", "--"),
        ("MC Dropout", "Multi"): (C_MULTI, "s", "--"),
    }
    for (method, variant), (color, marker, ls) in styles.items():
        sub = df[(df["Method"] == method) & (df["Variant"] == variant)].sort_values("Alpha")
        ax.plot(
            sub["Alpha"],
            sub["PICP"],
            marker=marker,
            linestyle=ls,
            color=color,
            lw=2,
            ms=9,
            label=f"{method} ({variant})",
        )

    ax.axhline(0.90, color=C_TARGET, ls=":", lw=1, alpha=0.6)
    ax.axvline(0.90, color=C_TARGET, ls=":", lw=1, alpha=0.6)
    ax.set_xlim(0.85, 0.97)
    ax.set_ylim(0.65, 1.0)
    ax.set_xlabel("Target confidence (α)")
    ax.set_ylabel("Observed coverage (PICP)")
    ax.set_title("Calibration: PICP vs nominal α")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, alpha=0.35)


def plot_mpiw(ax: plt.Axes, df: pd.DataFrame) -> None:
    x_labels = []
    baseline_vals = []
    multi_vals = []
    for method in ["MC Dropout", "Conformal"]:
        for alpha in [0.90, 0.95]:
            x_labels.append(f"{method}\nα={alpha:g}")
            b = df[(df["Method"] == method) & (df["Variant"] == "Baseline") & (df["Alpha"] == alpha)]
            m = df[(df["Method"] == method) & (df["Variant"] == "Multi") & (df["Alpha"] == alpha)]
            baseline_vals.append(float(b["MPIW"].iloc[0]))
            multi_vals.append(float(m["MPIW"].iloc[0]))

    x = np.arange(len(x_labels))
    w = 0.35
    ax.bar(x - w / 2, baseline_vals, w, label="Baseline", color=C_BASELINE, edgecolor="white")
    ax.bar(x + w / 2, multi_vals, w, label="Multi", color=C_MULTI, edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=8)
    ax.set_ylabel("MPIW (months)")
    ax.set_title("Mean prediction interval width")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, axis="y", alpha=0.35)


def plot_point_errors(ax: plt.Axes, df: pd.DataFrame, metric: str, title: str) -> None:
    """MAE/RMSE are identical across α within method×variant — use α=0.90 rows."""
    sub = df[df["Alpha"] == 0.90]
    methods = ["MC Dropout", "Conformal"]
    x = np.arange(len(methods))
    w = 0.35
    baseline = [
        float(sub[(sub["Method"] == m) & (sub["Variant"] == "Baseline")][metric].iloc[0])
        for m in methods
    ]
    multi = [
        float(sub[(sub["Method"] == m) & (sub["Variant"] == "Multi")][metric].iloc[0])
        for m in methods
    ]
    ax.bar(x - w / 2, baseline, w, label="Baseline", color=C_BASELINE, edgecolor="white")
    ax.bar(x + w / 2, multi, w, label="Multi", color=C_MULTI, edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=9)
    ax.set_ylabel(f"{metric} (months)")
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.35)


def plot_coverage_gap(ax: plt.Axes, df: pd.DataFrame) -> None:
    """PICP − α: negative gap = under-coverage (overconfident)."""
    df = df.copy()
    df["gap"] = df["PICP"] - df["Alpha"]
    x_labels = []
    gaps = []
    colors = []
    method_short = {"MC Dropout": "MC", "Conformal": "Conf"}
    for _, row in df.iterrows():
        m = method_short.get(row["Method"], row["Method"][:4])
        x_labels.append(f"{m}\n{row['Variant']}\nα={row['Alpha']:g}")
        gaps.append(row["gap"])
        ok = row["gap"] >= -0.01
        colors.append("#16a34a" if ok else "#dc2626")

    x = np.arange(len(x_labels))
    ax.bar(x, gaps, color=colors, edgecolor="white", width=0.7)
    ax.axhline(0, color=C_BLACK, lw=1.2)
    ax.axhline(-0.05, color=C_TARGET, ls=":", lw=1, alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=7)
    ax.set_ylabel("PICP − α")
    ax.set_title("Coverage gap (below 0 = under-covers)")
    ax.grid(True, axis="y", alpha=0.35)


def main() -> None:
    plt.rcParams.update({"font.size": 11, "figure.dpi": 150, "font.family": "sans-serif"})
    df = load()

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.patch.set_facecolor("#f8fafc")
    fig.suptitle(
        "UQ Method Comparison (test set, n = 1262)",
        fontsize=15,
        fontweight="bold",
        color=C_NAVY,
        y=0.98,
    )

    plot_calibration(axes[0, 0], df)
    plot_mpiw(axes[0, 1], df)
    plot_point_errors(axes[1, 0], df, "MAE", "Point prediction MAE (test)")
    plot_coverage_gap(axes[1, 1], df)

    for ax in axes.flat:
        ax.set_facecolor("#ffffff")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(OUT_PATH, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"Saved → {OUT_PATH}")


if __name__ == "__main__":
    main()
