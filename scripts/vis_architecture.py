"""Generate EfficientNet-B3 model architecture diagram for the presentation.

Draws two side-by-side architecture diagrams (Baseline and Multi-Input) as
clean block diagrams using matplotlib patches — no graphviz dependency needed.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch

ROOT_DIR = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT_DIR / "table_outputs"
OUT_DIR.mkdir(exist_ok=True)


# ── colour palette ────────────────────────────────────────────────────────────
C_INPUT = "#dbeafe"      # light blue
C_BACKBONE = "#1d4ed8"   # dark blue
C_POOL = "#7c3aed"       # purple
C_SEX = "#fed7aa"        # orange
C_CONCAT = "#f97316"     # orange
C_DENSE = "#065f46"      # dark green
C_DROPOUT = "#6ee7b7"    # mint
C_OUTPUT = "#dc2626"     # red

BORDER = "#374151"
TEXT_ON_DARK = "white"
TEXT_ON_LIGHT = "#1f2937"
ARROW_COLOR = "#6b7280"

BOX_W = 2.2
BOX_H = 0.5
GAP = 0.28      # vertical gap between boxes
LEFT_X = 0.4   # x-left edge of a column
SHARED_X = 0.25
SEX_X = 3.05    # x for sex branch in multi diagram


def _box(ax, x, y, w, h, label, color, text_color=TEXT_ON_LIGHT,
         fontsize=9.5, bold=False):
    rect = mpatches.FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.04",
        facecolor=color,
        edgecolor=BORDER,
        linewidth=0.9,
        zorder=3,
    )
    ax.add_patch(rect)
    weight = "bold" if bold else "normal"
    ax.text(
        x + w / 2, y + h / 2,
        label,
        ha="center", va="center",
        fontsize=fontsize, fontweight=weight,
        color=text_color, zorder=4,
    )


def _arrow(ax, x, y_top, y_bot, color=ARROW_COLOR):
    ax.annotate(
        "",
        xy=(x, y_bot),
        xytext=(x, y_top),
        arrowprops=dict(
            arrowstyle="-|>",
            color=color,
            lw=1.4,
            mutation_scale=11,
        ),
        zorder=2,
    )


def _arrow_h(ax, x_start, x_end, y, color=ARROW_COLOR):
    ax.annotate(
        "",
        xy=(x_end, y),
        xytext=(x_start, y),
        arrowprops=dict(
            arrowstyle="-|>",
            color=color,
            lw=1.4,
            mutation_scale=11,
        ),
        zorder=2,
    )


def draw_baseline(ax):
    ax.set_xlim(0, 3.2)
    ax.set_ylim(-0.15, 7.0)
    ax.axis("off")
    ax.set_title("Baseline Model\n(Image only)", fontsize=12,
                 fontweight="bold", pad=10, color="#1f2937")

    step = BOX_H + GAP
    cx = LEFT_X + BOX_W / 2
    y = 6.5

    layers = [
        ("X-ray Image\n300 × 300 × 3",           C_INPUT,    TEXT_ON_LIGHT, False),
        ("EfficientNet-B3\n(ImageNet pretrained)", C_BACKBONE, TEXT_ON_DARK,  True),
        ("Global Average Pooling",                 C_POOL,     TEXT_ON_DARK,  False),
        ("Dense (256)  +  ReLU",                   C_DENSE,    TEXT_ON_DARK,  False),
        ("Dropout",                                C_DROPOUT,  TEXT_ON_LIGHT, False),
        ("Dense (1)  —  Linear",                   C_DENSE,    TEXT_ON_DARK,  False),
        ("Bone Age prediction\n(months)",          C_OUTPUT,   TEXT_ON_DARK,  True),
    ]

    tops = []
    for i, (label, color, tc, bold) in enumerate(layers):
        yb = y - i * step
        _box(ax, LEFT_X, yb - BOX_H, BOX_W, BOX_H, label, color, tc,
             fontsize=9, bold=bold)
        tops.append((cx, yb - BOX_H, yb))

    for i in range(len(tops) - 1):
        _, _, y_top = tops[i]
        _, y_bot, _ = tops[i + 1]
        _arrow(ax, cx, y_top, y_bot + BOX_H)

    # small annotation on EfficientNet box
    ax.text(LEFT_X + BOX_W + 0.07, tops[1][1] + BOX_H / 2,
            "~12 M params\nfeature extractor",
            va="center", fontsize=7.5, color="#555555", style="italic")


def draw_multi(ax):
    ax.set_xlim(0, 5.5)
    ax.set_ylim(-0.15, 7.0)
    ax.axis("off")
    ax.set_title("Multi-Input Model\n(Image + Sex)", fontsize=12,
                 fontweight="bold", pad=10, color="#1f2937")

    step = BOX_H + GAP
    cx = LEFT_X + BOX_W / 2
    y = 6.5

    # shared backbone layers (same x as baseline)
    backbone_layers = [
        ("X-ray Image\n300 × 300 × 3",            C_INPUT,    TEXT_ON_LIGHT, False),
        ("EfficientNet-B3\n(ImageNet pretrained)", C_BACKBONE, TEXT_ON_DARK,  True),
        ("Global Average Pooling",                 C_POOL,     TEXT_ON_DARK,  False),
    ]

    tops_main = []
    for i, (label, color, tc, bold) in enumerate(backbone_layers):
        yb = y - i * step
        _box(ax, LEFT_X, yb - BOX_H, BOX_W, BOX_H, label, color, tc,
             fontsize=9, bold=bold)
        tops_main.append((cx, yb - BOX_H, yb))

    for i in range(len(tops_main) - 1):
        _, _, y_top = tops_main[i]
        _, y_bot, _ = tops_main[i + 1]
        _arrow(ax, cx, y_top, y_bot + BOX_H)

    # concatenation y level = just below GlobalAvgPool box
    concat_y_top = tops_main[-1][1]   # bottom of GlobalAvgPool box
    concat_y = concat_y_top - GAP * 1.5

    # ── sex branch ────────────────────────────────────────────────────────────
    sex_cx = SEX_X + BOX_W / 2
    sex_box_y = tops_main[2][1] + (BOX_H - BOX_H) / 2   # same height as GlobalAvgPool

    _box(ax, SEX_X, sex_box_y, BOX_W, BOX_H,
         "Patient Sex\n(0 = Female, 1 = Male)", C_SEX, TEXT_ON_LIGHT,
         fontsize=9, bold=False)

    # ── concat box ────────────────────────────────────────────────────────────
    concat_cx = (cx + sex_cx) / 2
    concat_bx = concat_cx - BOX_W / 2
    _box(ax, concat_bx, concat_y, BOX_W, BOX_H,
         "Concatenate", C_CONCAT, TEXT_ON_DARK, fontsize=9, bold=True)
    concat_box_top = concat_y + BOX_H

    # arrows into concat
    # from backbone (vertical then right)
    _arrow(ax, cx, concat_y_top, concat_y + BOX_H)
    # from sex (horizontal then down)
    sex_mid_y = sex_box_y + BOX_H / 2
    # draw line from sex box rightward then curve down to concat
    ax.annotate(
        "",
        xy=(concat_bx + BOX_W, concat_y + BOX_H / 2),
        xytext=(SEX_X, sex_mid_y),
        arrowprops=dict(
            arrowstyle="-|>",
            color=C_CONCAT,
            lw=1.4,
            connectionstyle="arc3,rad=-0.3",
            mutation_scale=11,
        ),
        zorder=2,
    )

    # ── post-concat layers ────────────────────────────────────────────────────
    post_layers = [
        ("Dense (256)  +  ReLU", C_DENSE,   TEXT_ON_DARK,  False),
        ("Dropout",              C_DROPOUT,  TEXT_ON_LIGHT, False),
        ("Dense (1)  —  Linear", C_DENSE,   TEXT_ON_DARK,  False),
        ("Bone Age prediction\n(months)", C_OUTPUT, TEXT_ON_DARK, True),
    ]

    prev_y = concat_y
    prev_cx = concat_cx
    for i, (label, color, tc, bold) in enumerate(post_layers):
        yb = prev_y - step
        _box(ax, concat_bx, yb, BOX_W, BOX_H, label, color, tc,
             fontsize=9, bold=bold)
        _arrow(ax, concat_cx, prev_y, yb + BOX_H)
        prev_y = yb

    # label sex contribution
    ax.text(SEX_X + BOX_W + 0.08, sex_box_y + BOX_H / 2,
            "sex-specific\ngrowth patterns",
            va="center", fontsize=7.5, color="#555555", style="italic")


def main() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 8))
    fig.patch.set_facecolor("#f8f9fa")

    draw_baseline(axes[0])
    draw_multi(axes[1])

    fig.suptitle(
        "EfficientNet-B3 Architecture: Baseline vs Multi-Input",
        fontsize=15, fontweight="bold", y=1.01,
    )

    # shared legend
    legend_patches = [
        mpatches.Patch(facecolor=C_INPUT,    edgecolor=BORDER, label="Input"),
        mpatches.Patch(facecolor=C_BACKBONE, edgecolor=BORDER, label="EfficientNet-B3 backbone"),
        mpatches.Patch(facecolor=C_POOL,     edgecolor=BORDER, label="Global Average Pooling"),
        mpatches.Patch(facecolor=C_SEX,      edgecolor=BORDER, label="Sex metadata input"),
        mpatches.Patch(facecolor=C_CONCAT,   edgecolor=BORDER, label="Concatenation"),
        mpatches.Patch(facecolor=C_DENSE,    edgecolor=BORDER, label="Dense / ReLU"),
        mpatches.Patch(facecolor=C_DROPOUT,  edgecolor=BORDER, label="Dropout"),
        mpatches.Patch(facecolor=C_OUTPUT,   edgecolor=BORDER, label="Regression output"),
    ]
    fig.legend(
        handles=legend_patches,
        loc="lower center",
        ncol=4,
        fontsize=9,
        framealpha=0.9,
        bbox_to_anchor=(0.5, -0.07),
    )

    fig.tight_layout(rect=[0, 0.06, 1, 1])
    out = OUT_DIR / "architecture_diagram.png"
    fig.savefig(out, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"Saved → {out}")


if __name__ == "__main__":
    main()
