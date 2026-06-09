"""Build presentation.pptx from presentation_draft.md content.

Produces a 14-slide widescreen (16:9) PowerPoint with academic blue theme,
embedded images from table_outputs/, native pptx tables, and speaker notes.

Run with:
    python scripts/build_pptx.py
"""

from __future__ import annotations

from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.util import Cm, Pt

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
IMGS = ROOT / "table_outputs"
OUT = ROOT / "presentation.pptx"

# ── Theme colours ─────────────────────────────────────────────────────────────
C_NAVY    = RGBColor(0x1E, 0x3A, 0x5F)   # dark navy – title bar bg
C_BLUE    = RGBColor(0x1D, 0x4E, 0xD8)   # accent blue
C_LBLUE   = RGBColor(0xDB, 0xEA, 0xFE)   # light blue – table header bg
C_WHITE   = RGBColor(0xFF, 0xFF, 0xFF)
C_BLACK   = RGBColor(0x11, 0x18, 0x27)   # near-black body text
C_GREY    = RGBColor(0x47, 0x55, 0x69)   # secondary text
C_LGREY   = RGBColor(0xF1, 0xF5, 0xF9)   # light grey – table alt rows
C_GREEN   = RGBColor(0x16, 0x6B, 0x40)   # positive highlight
C_RED     = RGBColor(0xDC, 0x26, 0x26)   # negative highlight
C_YELLOW  = RGBColor(0xF5, 0x9E, 0x0B)   # warning / note

FONT_FACE = "Calibri"

# Slide dimensions (widescreen 16:9)
SW = Cm(33.87)
SH = Cm(19.05)

# Standard zones
TITLE_H    = Cm(2.2)
TITLE_TOP  = Cm(0.55)
BODY_TOP   = Cm(3.1)
BODY_LEFT  = Cm(1.2)
BODY_W     = SW - Cm(2.4)
BODY_H     = SH - BODY_TOP - Cm(0.6)


# ── Low-level helpers ─────────────────────────────────────────────────────────

def blank_slide(prs: Presentation) -> object:
    """Add a blank slide (layout 6) with white background."""
    layout = prs.slide_layouts[6]
    slide = prs.slides.add_slide(layout)
    # White background fill
    bg = slide.background
    fill = bg.fill
    fill.solid()
    fill.fore_color.rgb = C_WHITE
    return slide


def title_bar(slide, text: str) -> None:
    """Navy title bar spanning full width at top of slide."""
    from pptx.util import Cm
    tb = slide.shapes.add_shape(
        1,  # MSO_SHAPE_TYPE.RECTANGLE
        Cm(0), TITLE_TOP, SW, TITLE_H,
    )
    tb.fill.solid()
    tb.fill.fore_color.rgb = C_NAVY
    tb.line.fill.background()

    tf = tb.text_frame
    tf.word_wrap = False
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.LEFT
    run = p.add_run()
    run.text = text
    run.font.name = FONT_FACE
    run.font.size = Pt(24)
    run.font.bold = True
    run.font.color.rgb = C_WHITE


def textbox(
    slide,
    text: str,
    left: float,
    top: float,
    width: float,
    height: float,
    size: int = 16,
    bold: bool = False,
    color: RGBColor = C_BLACK,
    align: PP_ALIGN = PP_ALIGN.LEFT,
    wrap: bool = True,
) -> object:
    """Add a single-run textbox."""
    box = slide.shapes.add_textbox(left, top, width, height)
    box.text_frame.word_wrap = wrap
    p = box.text_frame.paragraphs[0]
    p.alignment = align
    run = p.add_run()
    run.text = text
    run.font.name = FONT_FACE
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = color
    return box


def bullets(
    slide,
    items: list[tuple[str, int, bool, RGBColor | None]],
    left: float,
    top: float,
    width: float,
    height: float,
    default_size: int = 16,
) -> None:
    """
    Add a bulleted text box.
    items = list of (text, indent_level, bold, color_or_None)
    indent_level 0 = top-level bullet, 1 = sub-bullet
    """
    box = slide.shapes.add_textbox(left, top, width, height)
    tf = box.text_frame
    tf.word_wrap = True

    for i, (text, level, bold, color) in enumerate(items):
        if i == 0:
            p = tf.paragraphs[0]
        else:
            p = tf.add_paragraph()
        p.level = level
        indent = Pt(14 + level * 12)
        p.space_before = Pt(3 if level == 0 else 1)

        # Bullet character
        from pptx.oxml.ns import qn
        import lxml.etree as etree
        buChar = etree.SubElement(p._pPr if hasattr(p, '_pPr') else p._p.get_or_add_pPr(), qn('a:buChar'))
        buChar.set('char', '•' if level == 0 else '–')

        run = p.add_run()
        run.text = text
        run.font.name = FONT_FACE
        run.font.size = Pt(default_size - level * 1)
        run.font.bold = bold
        run.font.color.rgb = color if color else C_BLACK


def add_image(slide, path: Path, left: float, top: float,
              width: float, height: float | None = None) -> None:
    if height is None:
        slide.shapes.add_picture(str(path), left, top, width=width)
    else:
        slide.shapes.add_picture(str(path), left, top, width=width, height=height)


def set_notes(slide, note: str) -> None:
    notes_tf = slide.notes_slide.notes_text_frame
    notes_tf.text = note


def add_table(
    slide,
    headers: list[str],
    rows: list[list[str]],
    left: float,
    top: float,
    width: float,
    col_widths: list[float] | None = None,
    hdr_size: int = 13,
    row_size: int = 13,
    alt_rows: bool = True,
    highlight_col: int | None = None,
) -> None:
    """Build a native pptx table."""
    n_rows = len(rows) + 1  # +1 for header
    n_cols = len(headers)

    # estimate row height
    row_h = Cm(0.72)
    total_h = row_h * n_rows

    tbl = slide.shapes.add_table(n_rows, n_cols, left, top, width, total_h)
    table = tbl.table

    # Set column widths
    if col_widths:
        for ci, cw in enumerate(col_widths):
            table.columns[ci].width = cw
    else:
        even_w = width // n_cols
        for ci in range(n_cols):
            table.columns[ci].width = even_w

    def _style_cell(cell, text, bg: RGBColor, fg: RGBColor,
                    sz: int, bold: bool, align=PP_ALIGN.CENTER):
        cell.fill.solid()
        cell.fill.fore_color.rgb = bg
        tf = cell.text_frame
        tf.word_wrap = False
        p = tf.paragraphs[0]
        p.alignment = align
        run = p.add_run()
        run.text = text
        run.font.name = FONT_FACE
        run.font.size = Pt(sz)
        run.font.bold = bold
        run.font.color.rgb = fg
        # Vertical centering
        from pptx.enum.text import MSO_ANCHOR
        tf.auto_size = None
        cell.vertical_anchor = MSO_ANCHOR.MIDDLE

    # Header row
    for ci, hdr in enumerate(headers):
        _style_cell(table.cell(0, ci), hdr, C_NAVY, C_WHITE,
                    hdr_size, True)

    # Data rows
    for ri, row in enumerate(rows):
        bg = C_LGREY if (alt_rows and ri % 2 == 0) else C_WHITE
        for ci, val in enumerate(row):
            cell_bg = bg
            cell_fg = C_BLACK
            bold_cell = False
            if highlight_col is not None and ci == highlight_col:
                bold_cell = True
            _style_cell(table.cell(ri + 1, ci), val, cell_bg, cell_fg,
                        row_size, bold_cell)


def accent_line(slide, left, top, width, color=C_BLUE, height=Cm(0.07)):
    """Thin coloured horizontal rule."""
    line = slide.shapes.add_shape(1, left, top, width, height)
    line.fill.solid()
    line.fill.fore_color.rgb = color
    line.line.fill.background()


def section_label(slide, text, left, top, width=Cm(6), size=11):
    """Small blue ALL-CAPS label above a section."""
    box = slide.shapes.add_textbox(left, top, width, Cm(0.5))
    tf = box.text_frame
    p = tf.paragraphs[0]
    run = p.add_run()
    run.text = text.upper()
    run.font.name = FONT_FACE
    run.font.size = Pt(size)
    run.font.bold = True
    run.font.color.rgb = C_BLUE


def callout_box(slide, text, left, top, width, height,
                bg=C_LBLUE, fg=C_NAVY, size=13, bold=False):
    """Rounded-rectangle callout box."""
    box = slide.shapes.add_shape(
        5,  # ROUNDED_RECTANGLE
        left, top, width, height,
    )
    box.fill.solid()
    box.fill.fore_color.rgb = bg
    box.line.color.rgb = C_BLUE
    box.line.width = Pt(1)

    tf = box.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.LEFT
    run = p.add_run()
    run.text = text
    run.font.name = FONT_FACE
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = fg


# ── Slide builders ────────────────────────────────────────────────────────────

def slide_01_title(prs):
    slide = blank_slide(prs)

    # Navy full background
    bg_rect = slide.shapes.add_shape(1, Cm(0), Cm(0), SW, SH)
    bg_rect.fill.solid()
    bg_rect.fill.fore_color.rgb = C_NAVY
    bg_rect.line.fill.background()

    # Blue accent bar (lower 30%)
    bar = slide.shapes.add_shape(1, Cm(0), Cm(13.3), SW, Cm(5.75))
    bar.fill.solid()
    bar.fill.fore_color.rgb = C_BLUE
    bar.line.fill.background()

    # Main title
    textbox(slide, "Pediatric Bone Age Estimation",
            Cm(2), Cm(4.5), Cm(29.87), Cm(2.8),
            size=38, bold=True, color=C_WHITE, align=PP_ALIGN.CENTER)

    textbox(slide, "with Uncertainty Quantification",
            Cm(2), Cm(7.0), Cm(29.87), Cm(2.0),
            size=30, bold=False, color=RGBColor(0x93, 0xC5, 0xFD), align=PP_ALIGN.CENTER)

    # Subtitle
    textbox(slide, "Deep learning regression  ·  Post-hoc prediction intervals",
            Cm(2), Cm(14.0), Cm(29.87), Cm(1.2),
            size=17, bold=False, color=C_WHITE, align=PP_ALIGN.CENTER)

    # Stack: dataset / model / UQ labels
    textbox(slide, "RSNA Bone Age Dataset  ·  EfficientNet-B3  ·  MC Dropout  ·  Conformal Prediction",
            Cm(2), Cm(15.5), Cm(29.87), Cm(1.0),
            size=13, bold=False,
            color=RGBColor(0xBA, 0xD8, 0xF8), align=PP_ALIGN.CENTER)

    set_notes(slide, "Set the scene: this project builds a model that not only predicts bone age but also tells you how much to trust the prediction.")


def slide_02_motivation(prs):
    slide = blank_slide(prs)
    title_bar(slide, "Why Automate Bone Age Assessment?")

    section_label(slide, "Clinical Context", BODY_LEFT, BODY_TOP)

    items = [
        ("Pediatric bone age is assessed from hand X-rays to diagnose growth disorders", 0, False, None),
        ("Traditional method: radiologist compares X-ray to a reference atlas (Greulich-Pyle)", 0, False, None),
        ("Limitations of the manual approach:", 0, True, C_NAVY),
        ("Subjective — significant inter-observer variability documented in literature", 1, False, None),
        ("Time-consuming — 10–15 minutes per image", 1, False, None),
        ("Inconsistent across institutions and experience levels", 1, False, None),
        ("Goal of this project:", 0, True, C_NAVY),
        ("Automate prediction with deep learning (EfficientNet-B3)", 1, False, None),
        ("Add calibrated uncertainty intervals — clinicians need to know when not to trust the model", 1, False, None),
    ]
    bullets(slide, items, BODY_LEFT, Cm(3.6), BODY_W, Cm(13.0), default_size=17)

    accent_line(slide, BODY_LEFT, Cm(16.5), BODY_W * 0.7)
    textbox(slide, "Both methods are post-hoc — applied to an already-trained model without retraining.",
            BODY_LEFT, Cm(16.7), BODY_W, Cm(0.8),
            size=13, color=C_GREY)

    set_notes(slide, "Emphasize the clinical stakes. An incorrect bone age estimate without any uncertainty indicator is harder to catch and act on than one that signals low confidence.")


def slide_03_dataset(prs):
    slide = blank_slide(prs)
    title_bar(slide, "RSNA Pediatric Bone Age Dataset")

    # Left column — bullets
    col_w = Cm(15.5)
    items = [
        ("Source: RSNA 2017 / Kaggle — publicly available", 0, False, None),
        ("12,611 hand X-ray images", 0, True, C_NAVY),
        ("Labels per image:", 0, False, None),
        ("boneage — true bone age in months (continuous regression target)", 1, False, None),
        ("male — binary sex indicator (True / False)", 1, False, None),
        ("Sex split: 54.2% male / 45.8% female", 0, False, None),
        ("Pre-processing:", 0, False, None),
        ("Resize to 300×300, EfficientNet normalization", 1, False, None),
        ("Augmentation: rotation ±20°, zoom ±10%, brightness", 1, False, None),
    ]
    bullets(slide, items, BODY_LEFT, Cm(3.1), col_w, Cm(7.5), default_size=15)

    # Right column — stats table
    add_table(
        slide,
        headers=["Statistic", "Value"],
        rows=[
            ["Min", "1 month"],
            ["Max", "228 months (19 yr)"],
            ["Median", "132 months (11 yr)"],
            ["Mean", "127.3 months"],
            ["Std Dev", "41.2 months"],
            ["5th–95th pct", "50 – 186 months"],
        ],
        left=Cm(17.5), top=Cm(3.1),
        width=Cm(15.3),
        col_widths=[Cm(7.5), Cm(7.8)],
        hdr_size=13, row_size=13,
    )

    # Image below — age distribution
    add_image(slide, IMGS / "age_distribution.png",
              left=Cm(0.5), top=Cm(10.0),
              width=SW - Cm(1.0))

    set_notes(slide, "The age distribution is roughly Gaussian centered around 11 years. Males have a higher median (150 months) than females (120 months) — a direct consequence of sex-specific growth timing, which motivates the multi-input model design.")


def slide_04_architecture(prs):
    slide = blank_slide(prs)
    title_bar(slide, "EfficientNet-B3 Regression Models")

    # Brief description bullets — left side
    items = [
        ("ImageNet-pretrained EfficientNet-B3 · Global Avg Pool → Dense(256)+ReLU → Dropout → Linear output", 0, False, None),
        ("Output normalized by max training age; rescaled to months at inference", 0, False, None),
        ("Split: 80% train · 10% val (also calibration) · 10% test  (seed=42, stratified)", 0, False, None),
        ("Training: loss=MAE, Adam, batch=32, early stopping + LR scheduling", 0, False, None),
    ]
    bullets(slide, items, BODY_LEFT, Cm(3.1), BODY_W, Cm(3.0), default_size=14)

    # Architecture diagram image
    add_image(slide, IMGS / "architecture_diagram.png",
              left=Cm(0.5), top=Cm(6.0),
              width=SW - Cm(1.0), height=Cm(11.5))

    set_notes(slide, "The Dropout layer in the model head is the same one repurposed for MC Dropout inference later — no architectural change is needed. The key insight: sex concatenation lets the network learn sex-specific growth patterns alongside the visual features.")


def slide_05_regression(prs):
    slide = blank_slide(prs)
    title_bar(slide, "Point Prediction Performance (Regression)")

    add_table(
        slide,
        headers=["Model", "MAE (months)", "RMSE (months)", "R²"],
        rows=[
            ["Baseline — image only", "4.47", "5.78", "0.90"],
            ["Multi-Input — image + sex", "3.82 ↓15%", "4.98", "0.917"],
        ],
        left=Cm(2.5), top=Cm(3.2),
        width=Cm(28.87),
        col_widths=[Cm(10.0), Cm(6.3), Cm(6.3), Cm(6.3)],
        hdr_size=15, row_size=15,
        highlight_col=1,
    )

    items = [
        ("Adding sex reduces MAE by ~15%  (4.47 → 3.82 months)", 0, True, C_GREEN),
        ("~80% of predictions fall within ±6 months of the true age", 0, False, None),
        ("~98% of predictions fall within ±12 months of the true age", 0, False, None),
        ("Error distribution is approximately Gaussian, centred near zero", 0, False, None),
        ("Higher errors at extremes: very young ages and older adolescents (sparse data)", 0, False, C_GREY),
    ]
    bullets(slide, items, BODY_LEFT, Cm(6.5), BODY_W, Cm(9.5), default_size=16)

    callout_box(slide,
                "These are point predictions — they tell us HOW ACCURATE the model is, "
                "but not HOW CONFIDENT it is. That is what UQ addresses next.",
                BODY_LEFT, Cm(15.6), BODY_W, Cm(2.0),
                bg=C_LBLUE, fg=C_NAVY, size=14, bold=False)

    set_notes(slide, "These are training evaluation numbers. The UQ evaluation in later slides uses the same test split and will show slightly different MAE/RMSE because the inference pipeline re-runs the model. The direction is consistent: Multi-Input is always better on point accuracy.")


def slide_06_why_uq(prs):
    slide = blank_slide(prs)
    title_bar(slide, "From Point Predictions to Trustworthy Intervals")

    # Left panel
    lw = Cm(15.5)
    textbox(slide, "A point prediction answers:", BODY_LEFT, Cm(3.2), lw, Cm(0.7),
            size=16, bold=True, color=C_NAVY)
    callout_box(slide, '"What is the estimated bone age?"',
                BODY_LEFT, Cm(4.0), lw, Cm(1.1),
                bg=C_LGREY, fg=C_GREY, size=15)

    textbox(slide, "But a clinician also needs to know:", BODY_LEFT, Cm(5.4), lw, Cm(0.7),
            size=16, bold=True, color=C_NAVY)
    callout_box(slide, '"How confident should I be in this estimate?"',
                BODY_LEFT, Cm(6.2), lw, Cm(1.1),
                bg=C_LBLUE, fg=C_NAVY, size=15, bold=True)

    textbox(slide, "UQ answers:", BODY_LEFT, Cm(7.6), lw, Cm(0.6),
            size=15, bold=True, color=C_NAVY)
    items_uq = [
        ("Which predictions are reliable vs uncertain?", 0, False, None),
        ("What range covers the true age at a given probability?", 0, False, None),
    ]
    bullets(slide, items_uq, BODY_LEFT, Cm(8.2), lw, Cm(2.2), default_size=15)

    textbox(slide, "Two methods applied:", BODY_LEFT, Cm(10.6), lw, Cm(0.6),
            size=15, bold=True, color=C_NAVY)
    items_m = [
        ("Monte Carlo Dropout — model-based, stochastic inference", 0, False, C_BLUE),
        ("Split Conformal Prediction — data-driven, calibration residuals", 0, False, C_BLUE),
    ]
    bullets(slide, items_m, BODY_LEFT, Cm(11.3), lw, Cm(2.0), default_size=15)

    # Right panel — comparison table
    add_table(
        slide,
        headers=["Prediction", "What it tells the clinician"],
        rows=[
            ["132 months", "Model says 11 years — but how reliable is this?"],
            ["132 ± 16 months\n(90% CI)", "True age is 116–148 months\nwith 90% confidence"],
        ],
        left=Cm(17.5), top=Cm(3.2),
        width=Cm(15.3),
        col_widths=[Cm(5.0), Cm(10.3)],
        hdr_size=13, row_size=13,
    )

    set_notes(slide, "Both methods are post-hoc — applied to the already-trained model without retraining.")


def slide_07_mc_dropout(prs):
    slide = blank_slide(prs)
    title_bar(slide, "UQ Method 1: Monte Carlo Dropout")

    section_label(slide, "Gal & Ghahramani, 2016", BODY_LEFT, Cm(3.0))

    # Two column layout
    lw = Cm(18.0)
    textbox(slide, "Standard Dropout: active during training, OFF at inference",
            BODY_LEFT, Cm(3.6), lw, Cm(0.65), size=14, color=C_GREY)
    textbox(slide, "MC Dropout: keep dropout ON at inference — T = 50 forward passes",
            BODY_LEFT, Cm(4.25), lw, Cm(0.65), size=14, bold=True, color=C_NAVY)

    items = [
        ("For each test image run the model T = 50 times with dropout active", 0, False, None),
        ("Each pass randomly zeros different neurons → slightly different prediction", 0, False, None),
        ("Collect 50 predictions:  [ŷ₁, ŷ₂, ..., ŷ₅₀]", 0, False, None),
        ("Mean of 50 passes = final bone age estimate", 0, True, C_GREEN),
        ("Standard deviation of 50 passes = uncertainty estimate", 0, True, C_BLUE),
    ]
    bullets(slide, items, BODY_LEFT, Cm(5.0), lw, Cm(5.5), default_size=15)

    # Formula callout
    callout_box(slide,
                "Interval at confidence level α:\n"
                "   lower = mean − z(α) × std\n"
                "   upper = mean + z(α) × std\n"
                "   z = 1.645 for 90%,   z = 1.960 for 95%",
                BODY_LEFT, Cm(10.7), Cm(17.0), Cm(3.5),
                bg=RGBColor(0xF0, 0xF9, 0xFF), fg=C_NAVY, size=14)

    textbox(slide, "Key property:", Cm(19.0), Cm(10.7), Cm(14.0), Cm(0.6),
            size=14, bold=True, color=C_GREEN)
    textbox(slide, "No separate calibration step needed — uncertainty comes from the model's dropout structure.",
            Cm(19.0), Cm(11.4), Cm(14.0), Cm(1.2), size=14, color=C_BLACK)

    textbox(slide, "Limitation:", Cm(19.0), Cm(12.7), Cm(14.0), Cm(0.6),
            size=14, bold=True, color=C_RED)
    textbox(slide, "No guarantee that interval width reflects actual prediction errors — the model can be overconfident.",
            Cm(19.0), Cm(13.4), Cm(14.0), Cm(1.5), size=14, color=C_BLACK)

    set_notes(slide, "The model architecture already has a Dropout layer in the head. The MC Dropout script calls model(inputs, training=True) to keep it stochastic, then batches the 50 passes efficiently using tensor tiling.")


def slide_08_conformal(prs):
    slide = blank_slide(prs)
    title_bar(slide, "UQ Method 2: Split Conformal Prediction")

    section_label(slide, "Post-hoc · Distribution-free · Finite-sample coverage guarantee",
                  BODY_LEFT, Cm(3.0), width=Cm(25))

    step_top = Cm(3.7)
    step_gap = Cm(4.0)

    # Step 1
    callout_box(slide, "Step 1 — Calibrate on validation set  (n = 1,261 images)\n\n"
                "Compute residuals:   residuals = | y_true − y_pred |  for each image",
                BODY_LEFT, step_top, BODY_W, Cm(2.6),
                bg=C_LGREY, fg=C_NAVY, size=14)

    # Step 2
    callout_box(slide, "Step 2 — Compute the conformal quantile\n\n"
                "k = ceil((n + 1) × α)        →   q̂ = k-th smallest residual\n"
                "Example: α=0.90, baseline model  →  q̂ = 20.09 months",
                BODY_LEFT, step_top + step_gap, BODY_W, Cm(2.8),
                bg=C_LGREY, fg=C_NAVY, size=14)

    # Step 3
    callout_box(slide, "Step 3 — Build test-set intervals\n\n"
                "lower = ŷ_pred − q̂          upper = ŷ_pred + q̂\n"
                "Interval width = 2 × q̂   (same for every image at a given α)",
                BODY_LEFT, step_top + 2 * step_gap, BODY_W, Cm(2.8),
                bg=C_LGREY, fg=C_NAVY, size=14)

    # Coverage guarantee highlight
    callout_box(slide,
                "Coverage guarantee: at least α fraction of test samples will have "
                "y_true inside the interval — under the exchangeability assumption, "
                "regardless of model or data distribution.",
                BODY_LEFT, Cm(16.2), BODY_W, Cm(1.9),
                bg=C_LBLUE, fg=C_NAVY, size=14, bold=False)

    set_notes(slide, "Because q_hat is a single scalar from the calibration set, every test image gets the same interval width for a given model and alpha. This is the main limitation compared to adaptive conformal methods.")


def slide_09_metrics(prs):
    slide = blank_slide(prs)
    title_bar(slide, "UQ Metrics: PICP and MPIW")

    # PICP panel (left)
    lw = Cm(15.8)
    textbox(slide, "PICP — Prediction Interval Coverage Probability",
            BODY_LEFT, Cm(3.2), lw, Cm(0.7), size=17, bold=True, color=C_NAVY)
    callout_box(slide,
                "PICP = fraction of test samples where:   lower ≤ y_true ≤ upper",
                BODY_LEFT, Cm(4.0), lw, Cm(1.0),
                bg=C_LGREY, fg=C_BLACK, size=14)

    items_picp = [
        ("Target: PICP ≥ α  (e.g., ≥ 0.90 for a 90% interval)", 0, False, None),
        ("PICP < α  → intervals too narrow → overconfident", 0, False, C_RED),
        ("PICP >> α → intervals too wide  → over-conservative", 0, False, C_GREY),
    ]
    bullets(slide, items_picp, BODY_LEFT, Cm(5.2), lw, Cm(3.0), default_size=15)

    # MPIW panel (right)
    rx = Cm(17.8)
    rw = SW - rx - Cm(0.7)
    textbox(slide, "MPIW — Mean Prediction Interval Width",
            rx, Cm(3.2), rw, Cm(0.7), size=17, bold=True, color=C_NAVY)
    callout_box(slide,
                "MPIW = mean( upper − lower )   across all test images  (months)",
                rx, Cm(4.0), rw, Cm(1.0),
                bg=C_LGREY, fg=C_BLACK, size=14)

    items_mpiw = [
        ("Want this as SMALL as possible while still achieving target PICP", 0, True, C_GREEN),
        ("A narrow interval that covers the true value is more informative clinically", 0, False, None),
    ]
    bullets(slide, items_mpiw, rx, Cm(5.2), rw, Cm(2.0), default_size=15)

    accent_line(slide, BODY_LEFT, Cm(8.6), BODY_W)

    # Tension table
    textbox(slide, "The coverage–width tradeoff:", BODY_LEFT, Cm(8.9), BODY_W, Cm(0.6),
            size=15, bold=True, color=C_NAVY)
    add_table(
        slide,
        headers=["MPIW", "PICP", "Interpretation"],
        rows=[
            ["Very wide", "High", "Safe but uninformative"],
            ["Very narrow", "Low", "Precise but unreliable"],
            ["Narrow", "≥ α  ✓", "Ideal — calibrated and tight"],
        ],
        left=Cm(4.0), top=Cm(9.7),
        width=Cm(25.87),
        col_widths=[Cm(6.0), Cm(5.0), Cm(14.87)],
        hdr_size=14, row_size=14,
    )

    set_notes(slide, "Think of it as a precision-recall tradeoff. PICP is reliability; MPIW is usefulness. A good UQ method achieves target PICP with the smallest possible MPIW.")


def slide_10_mc_results(prs):
    slide = blank_slide(prs)
    title_bar(slide, "MC Dropout Results — Overconfident Intervals")

    add_image(slide, IMGS / "mc_dropout_table.png",
              left=Cm(1.0), top=Cm(2.8), width=SW - Cm(2.0), height=Cm(5.2))

    items = [
        ("Baseline severely overconfident: nominal 90% → actual coverage only 71.0%", 0, True, C_RED),
        ("At nominal 95%, baseline coverage is only 75.4% — far below target", 0, False, C_RED),
        ("Multi-input improves: 86.1% at α=0.90; 90.4% at α=0.95 (barely meets one target)", 0, False, C_GREY),
        ("Widths (29–43 months) are narrower than conformal — but coverage targets are missed", 0, False, None),
        ("Multi has wider MC Dropout intervals than Baseline despite better accuracy:", 0, False, None),
        ("Higher predictive std (10.9 vs 8.9 months) → wider Gaussian intervals", 1, False, C_GREY),
    ]
    bullets(slide, items, BODY_LEFT, Cm(8.3), BODY_W, Cm(8.8), default_size=15)

    callout_box(slide,
                "MC Dropout uncertainty reflects the model's internal stochasticity, "
                "not its actual errors — it cannot self-correct for systematic mis-calibration.",
                BODY_LEFT, Cm(17.0), BODY_W, Cm(1.5),
                bg=RGBColor(0xFF, 0xF1, 0xF0), fg=C_RED, size=13)

    set_notes(slide, "Overconfidence in MC Dropout typically occurs because dropout-based uncertainty reflects the model's internal variability, not its actual prediction errors on the data. It is a relative signal, not an absolutely calibrated one.")


def slide_11_conformal_results(prs):
    slide = blank_slide(prs)
    title_bar(slide, "Conformal Prediction Results — Coverage Targets Met")

    add_image(slide, IMGS / "conformal_table.png",
              left=Cm(1.0), top=Cm(2.8), width=SW - Cm(2.0), height=Cm(5.2))

    items = [
        ("Both variants meet their nominal coverage targets at α=0.90 and α=0.95", 0, True, C_GREEN),
        ("This is the conformal guarantee working in practice", 1, False, C_GREY),
        ("Multi-input is strictly better on both dimensions:", 0, True, C_NAVY),
        ("Higher PICP  AND  narrower intervals than baseline at every alpha level", 1, False, None),
        ("Narrower multi intervals explained by calibration: better point predictions →", 0, False, None),
        ("smaller validation residuals → smaller q̂  (20.09 → 16.54 months at α=0.90) → tighter intervals", 1, False, C_GREY),
        ("Width increases with α  (40.18 → 49.57 for baseline) — expected behaviour", 0, False, None),
    ]
    bullets(slide, items, BODY_LEFT, Cm(8.3), BODY_W, Cm(8.5), default_size=15)

    callout_box(slide,
                "q̂ values  (conformal threshold from validation residuals):\n"
                "Baseline α=0.90: 20.09 mo  ·  α=0.95: 24.78 mo       "
                "Multi α=0.90: 16.54 mo  ·  α=0.95: 21.86 mo     "
                "(MPIW = 2 × q̂, symmetric intervals)",
                BODY_LEFT, Cm(17.0), BODY_W, Cm(1.6),
                bg=C_LBLUE, fg=C_NAVY, size=12)

    set_notes(slide, "The q_hat values are the conformal threshold computed from the validation residuals. MPIW = 2 × q_hat since the intervals are symmetric around the point prediction.")


def slide_12_comparison(prs):
    slide = blank_slide(prs)
    title_bar(slide, "Combined Comparison: Method × Variant × Coverage Level")

    # Two images side by side
    img_top = Cm(2.8)
    img_h = Cm(7.5)
    half_w = (SW - Cm(2.2)) / 2

    add_image(slide, IMGS / "combined_uq_table.png",
              left=Cm(0.5), top=img_top, width=half_w, height=img_h)
    add_image(slide, IMGS / "mpiw_comparison.png",
              left=Cm(0.8) + half_w, top=img_top, width=half_w, height=img_h)

    accent_line(slide, BODY_LEFT, Cm(10.6), BODY_W)

    items = [
        ("MC Dropout (left chart): Multi has WIDER intervals than Baseline", 0, True, C_NAVY),
        ("Counter-intuitive — but multi has higher predictive std (10.9 vs 8.9 mo) → wider Gaussian intervals", 1, False, C_GREY),
        ("Conformal (right chart): Multi has NARROWER intervals than Baseline", 0, True, C_NAVY),
        ("Better predictions → smaller residuals → smaller q̂ → tighter intervals", 1, False, C_GREY),
    ]
    bullets(slide, items, BODY_LEFT, Cm(10.9), BODY_W, Cm(4.5), default_size=15)

    # Summary table
    add_table(
        slide,
        headers=["Method", "Variant", "PICP @ 0.90", "MPIW @ 0.90", "Verdict"],
        rows=[
            ["MC Dropout", "Baseline", "0.710", "29.23", "❌ Under-covers"],
            ["MC Dropout", "Multi",    "0.861", "35.89", "❌ Under-covers"],
            ["Conformal",  "Baseline", "0.894", "40.18", "✓ Meets target"],
            ["Conformal",  "Multi",    "0.903", "33.08", "✓ Best overall"],
        ],
        left=Cm(1.2), top=Cm(15.5),
        width=SW - Cm(2.4),
        col_widths=[Cm(5.8), Cm(5.0), Cm(5.3), Cm(5.3), Cm(12.0)],
        hdr_size=12, row_size=12,
    )

    set_notes(slide, "The key difference: MC Dropout width is driven by model-internal stochasticity; conformal width is driven by actual errors on held-out data. Only conformal + multi meets the coverage target with the smallest interval.")


def slide_13_takeaways(prs):
    slide = blank_slide(prs)
    title_bar(slide, "Key Takeaways")

    takeaways = [
        ("Adding sex metadata consistently improves the model.",
         "MAE drops ~15%  and  conformal intervals narrow by ~7 months at α=0.90."),
        ("Conformal prediction reliably achieves its stated coverage.",
         "Both variants meet targets at α=0.90 and α=0.95 — provable under exchangeability."),
        ("MC Dropout is overconfident, especially for the baseline model.",
         "At nominal 95%, actual baseline coverage is only 75.4%. Intervals are narrower but miss the target."),
        ("Best combination: Conformal + Multi-input.",
         "90.3% coverage with 33.1-month intervals at α=0.90 — reliably calibrated and tight."),
        ("MC Dropout is still useful as a relative uncertainty signal.",
         "Even if absolute coverage is off, high std → flag uncertain images for clinical review."),
        ("Conformal intervals are fixed-width per alpha.",
         "Every image gets the same interval — not adaptive to image difficulty. Main limitation."),
    ]

    top = Cm(3.2)
    for i, (main, detail) in enumerate(takeaways):
        y = top + i * Cm(2.42)
        # Number badge
        badge = slide.shapes.add_shape(1, BODY_LEFT, y, Cm(0.8), Cm(0.65))
        badge.fill.solid()
        badge.fill.fore_color.rgb = C_BLUE
        badge.line.fill.background()
        tf = badge.text_frame
        p = tf.paragraphs[0]
        p.alignment = PP_ALIGN.CENTER
        run = p.add_run()
        run.text = str(i + 1)
        run.font.name = FONT_FACE
        run.font.size = Pt(13)
        run.font.bold = True
        run.font.color.rgb = C_WHITE

        textbox(slide, main, Cm(2.3), y, BODY_W - Cm(1.1), Cm(0.65),
                size=15, bold=True, color=C_NAVY)
        textbox(slide, detail, Cm(2.3), y + Cm(0.68), BODY_W - Cm(1.1), Cm(0.8),
                size=13, color=C_GREY)

    set_notes(slide, "Summarise the narrative: the project shows both that adding metadata helps and that the choice of UQ method matters — conformal is better calibrated than MC Dropout for this task.")


def slide_14_limitations(prs):
    slide = blank_slide(prs)
    title_bar(slide, "Limitations and Future Work")

    half = (BODY_W - Cm(0.6)) / 2

    # Limitations column
    section_label(slide, "Current Limitations", BODY_LEFT, Cm(3.2))
    lim_items = [
        ("No hand segmentation — background noise and cropping artifacts can degrade predictions", 0, False, None),
        ("Only sex used as metadata — height, weight, ethnicity, pubertal stage could help", 0, False, None),
        ("Conformal intervals are symmetric and non-adaptive — same width for every image at a given α", 0, False, None),
        ("Higher errors at extreme ages (infants, older adolescents) are not specifically addressed", 0, False, None),
    ]
    bullets(slide, lim_items, BODY_LEFT, Cm(3.9), half, Cm(10.0), default_size=15)

    # Future work column
    rx = BODY_LEFT + half + Cm(0.6)
    section_label(slide, "Future Directions", rx, Cm(3.2))
    fut_items = [
        ("Segmentation pre-processing: U-Net to isolate hand region before EfficientNet", 0, False, None),
        ("Richer metadata: height, weight, clinical variables as additional model inputs", 0, False, None),
        ("Adaptive conformal prediction: Mondrian / locally-weighted methods for image-specific interval widths", 0, False, None),
        ("Hybrid architectures: CNN + Transformer (ViT) for richer feature extraction", 0, False, None),
        ("Deployment: FastAPI REST endpoint for real-world clinical integration", 0, False, None),
    ]
    bullets(slide, fut_items, rx, Cm(3.9), half, Cm(10.5), default_size=15)

    # Divider
    div = slide.shapes.add_shape(1, BODY_LEFT + half + Cm(0.25), Cm(3.2),
                                 Cm(0.05), Cm(13.5))
    div.fill.solid()
    div.fill.fore_color.rgb = C_BLUE
    div.line.fill.background()

    # Footer
    accent_line(slide, BODY_LEFT, Cm(17.2), BODY_W)
    textbox(slide,
            "Dataset: RSNA Bone Age (Kaggle)  ·  Models: EfficientNet-B3 (TensorFlow/Keras)  ·  "
            "UQ: MC Dropout (Gal & Ghahramani, 2016)  ·  Split Conformal Prediction",
            BODY_LEFT, Cm(17.4), BODY_W, Cm(0.8),
            size=11, color=C_GREY, align=PP_ALIGN.CENTER)

    set_notes(slide, "Future adaptive conformal methods (Mondrian conformal, RAPS) would give each image its own interval width based on difficulty, making the uncertainty more actionable for individual clinical decisions.")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    prs = Presentation()
    prs.slide_width  = SW
    prs.slide_height = SH

    builders = [
        slide_01_title,
        slide_02_motivation,
        slide_03_dataset,
        slide_04_architecture,
        slide_05_regression,
        slide_06_why_uq,
        slide_07_mc_dropout,
        slide_08_conformal,
        slide_09_metrics,
        slide_10_mc_results,
        slide_11_conformal_results,
        slide_12_comparison,
        slide_13_takeaways,
        slide_14_limitations,
    ]

    for i, fn in enumerate(builders, 1):
        print(f"  Building slide {i:02d}: {fn.__name__} ...", end=" ")
        fn(prs)
        print("ok")

    prs.save(str(OUT))
    print(f"\nSaved → {OUT}")


if __name__ == "__main__":
    main()
