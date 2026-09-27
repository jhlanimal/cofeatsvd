from math import atan2, degrees
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.dml import MSO_LINE_DASH_STYLE
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "iclr_latex/iclr2027/figures/method_overview_editable.pptx"


def rgb(value):
    value = value.lstrip("#")
    return RGBColor(int(value[0:2], 16), int(value[2:4], 16), int(value[4:6], 16))


C = {
    "ink": rgb("243447"), "muted": rgb("62717c"), "stage": rgb("9aa8b2"),
    "blue": rgb("3c8fc5"), "blue_fill": rgb("eef7fc"),
    "teal": rgb("20a38a"), "teal_fill": rgb("eefaf7"),
    "orange": rgb("e9a62a"), "orange_fill": rgb("fff8e8"),
    "purple": rgb("8960b5"), "purple_fill": rgb("f7f1fc"),
    "red": rgb("e45a61"), "red_fill": rgb("fff1f2"),
    "green": rgb("4b9b68"), "green_fill": rgb("eef9f0"),
    "white": rgb("ffffff"),
}


def set_text(shape, text, size=8, color="ink", bold=False, italic=False,
             align=PP_ALIGN.CENTER, font="Times New Roman"):
    tf = shape.text_frame
    tf.clear()
    tf.word_wrap = True
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = tf.paragraphs[0]
    p.alignment = align
    run = p.add_run()
    run.text = text
    run.font.name = font
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.italic = italic
    run.font.color.rgb = C[color] if isinstance(color, str) else color
    return shape


def box(slide, x, y, w, h, line, fill="white", radius=True, lw=1.1,
        text=None, size=8, bold=False, color="ink", align=PP_ALIGN.CENTER):
    shape_type = MSO_SHAPE.ROUNDED_RECTANGLE if radius else MSO_SHAPE.RECTANGLE
    s = slide.shapes.add_shape(shape_type, Inches(x), Inches(y), Inches(w), Inches(h))
    s.fill.solid()
    s.fill.fore_color.rgb = C[fill] if isinstance(fill, str) else fill
    s.line.color.rgb = C[line] if isinstance(line, str) else line
    s.line.width = Pt(lw)
    if text is not None:
        set_text(s, text, size=size, color=color, bold=bold, align=align)
    return s


def text(slide, x, y, w, h, value, size=8, color="ink", bold=False,
         italic=False, align=PP_ALIGN.CENTER):
    s = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    s.fill.background()
    s.line.fill.background()
    return set_text(s, value, size=size, color=color, bold=bold,
                    italic=italic, align=align)


def stage(slide, x, y, w, h, title):
    s = box(slide, x, y, w, h, "stage", "white", radius=True, lw=1.0)
    s.line.dash_style = MSO_LINE_DASH_STYLE.DASH
    text(slide, x, y - 0.34, w, 0.25, title, size=13, color="ink",
         bold=True, italic=True)


def module(slide, x, y, w, h, title, number, edge, fill):
    box(slide, x, y, w, h, edge, fill, radius=True, lw=1.0)
    text(slide, x + 0.12, y + 0.10, w - 0.24, 0.25,
         f"{number}  {title}", size=9.0, color="ink", bold=True,
         align=PP_ALIGN.LEFT)


def mini(slide, x, y, w, h, value, edge, fill="white", size=7.2, bold=False):
    return box(slide, x, y, w, h, edge, fill, radius=True, lw=0.75,
               text=value, size=size, bold=bold)


def cells(slide, x, y, rows, cols, cw, ch, edge, fill, gap=0.018):
    for r in range(rows):
        for c in range(cols):
            box(slide, x + c * (cw + gap), y + (rows - 1 - r) * (ch + gap),
                cw, ch, edge, fill, radius=False, lw=0.45)


def bars(slide, x, y, values, bw=0.14, gap=0.08, color="orange"):
    for i, height in enumerate(values):
        box(slide, x + i * (bw + gap), y, bw, height, color, color,
            radius=False, lw=0)


def arrow(slide, x1, y1, x2, y2, color="red", lw=1.1, dashed=False):
    line = slide.shapes.add_connector(
        MSO_CONNECTOR.STRAIGHT, Inches(x1), Inches(y1), Inches(x2), Inches(y2)
    )
    line.line.color.rgb = C[color]
    line.line.width = Pt(lw)
    if dashed:
        line.line.dash_style = MSO_LINE_DASH_STYLE.DASH
    angle = degrees(atan2(y2 - y1, x2 - x1))
    size = 0.12
    head = slide.shapes.add_shape(
        MSO_SHAPE.ISOSCELES_TRIANGLE,
        Inches(x2 - size / 2), Inches(y2 - size / 2), Inches(size), Inches(size),
    )
    head.fill.solid()
    head.fill.fore_color.rgb = C[color]
    head.line.color.rgb = C[color]
    head.rotation = angle + 90
    return line


def diamond(slide, x, y, w, h, value):
    s = slide.shapes.add_shape(MSO_SHAPE.DIAMOND, Inches(x), Inches(y), Inches(w), Inches(h))
    s.fill.solid()
    s.fill.fore_color.rgb = C["white"]
    s.line.color.rgb = C["ink"]
    s.line.width = Pt(0.9)
    return set_text(s, value, size=7.0, bold=True)


def build():
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    slide.background.fill.solid()
    slide.background.fill.fore_color.rgb = C["white"]

    # Stage containers.
    stage(slide, 0.18, 0.65, 7.18, 6.10,
          "Page Construction and Sealing")
    stage(slide, 7.62, 0.65, 5.55, 6.10, "Direct Grouped Decode")

    # 1. Post-RoPE page.
    module(slide, 0.38, 1.05, 2.12, 2.42, "Post-RoPE K/V page", "1",
           "blue", "blue_fill")
    for i in range(6):
        box(slide, 0.64 + i * 0.24, 1.72, 0.18, 0.20, "orange", "orange_fill",
            radius=False, lw=0.6, text=str(i + 1), size=5.8)
    text(slide, 2.00, 1.72, 0.28, 0.20, "...", size=8, color="muted")
    mini(slide, 0.76, 2.12, 1.36, 0.38, "Q / K / V projection", "purple",
         "purple_fill", size=7.2, bold=True)
    arrow(slide, 1.35, 1.95, 1.43, 2.10, "purple", 0.8)
    cells(slide, 0.62, 2.67, 3, 7, 0.17, 0.13, "blue", "blue_fill")
    text(slide, 0.62, 3.10, 1.85, 0.19, "post-RoPE K and V", size=6.6, color="muted")
    text(slide, 0.56, 3.30, 1.95, 0.19, "active page is dense", size=6.3, color="muted")

    # 2. Page-local calibration.
    module(slide, 2.68, 1.05, 2.12, 2.42, "Page-local calibration", "2",
           "orange", "orange_fill")
    text(slide, 2.92, 1.55, 1.64, 0.25, "s_j = ||k_j||^2 w_j", size=8.0,
         color="ink", bold=True)
    bars(slide, 3.15, 2.02, [0.42, 0.72, 0.30, 0.56, 0.22, 0.49])
    text(slide, 3.05, 2.75, 1.70, 0.18, "query-calibrated importance", size=6.2, color="muted")
    mini(slide, 2.92, 3.02, 0.78, 0.28, "top-m anchors", "blue", "blue_fill", 6.2, True)
    mini(slide, 3.91, 3.02, 0.76, 0.28, "remaining", "orange", "orange_fill", 6.4, True)
    arrow(slide, 3.73, 3.16, 3.72, 3.16, "orange", 0.8)

    # 3. Anchor-guided grouping.
    module(slide, 4.98, 1.05, 2.18, 2.42, "Anchor-guided grouping", "3",
           "teal", "teal_fill")
    mini(slide, 5.18, 1.55, 0.88, 0.40, "shared\ndirections C", "blue",
         "blue_fill", 6.4, True)
    mini(slide, 6.13, 1.55, 0.82, 0.40, "coeff +\nowner", "orange",
         "orange_fill", 6.3, True)
    cells(slide, 5.26, 2.14, 2, 7, 0.15, 0.12, "blue", "blue_fill")
    for i in range(4):
        box(slide, 6.31 + i * 0.15, 2.14, 0.12, 0.12, "orange", "orange_fill", radius=False, lw=0.5)
    arrow(slide, 5.62, 1.96, 5.62, 2.07, "blue", 0.75)
    arrow(slide, 6.53, 1.96, 6.53, 2.07, "orange", 0.75)
    text(slide, 5.22, 2.52, 1.76, 0.20, "merge cost Δ(h,l)", size=6.0, color="muted")
    mini(slide, 5.22, 2.87, 1.66, 0.30, "C [P×m]; α, owner ∈ R^d", "teal", "green_fill", 5.9, True)

    # Cache strip.
    box(slide, 0.38, 4.08, 6.76, 1.84, "teal", "teal_fill", radius=True, lw=1.0)
    text(slide, 0.58, 4.22, 2.10, 0.24, "CoFeatKV KV cache", size=9.0, color="ink", bold=True, align=PP_ALIGN.LEFT)
    for x, label in [(0.68, "P1"), (1.78, "P2"), (2.88, "P3")]:
        box(slide, x, 4.72, 0.88, 0.58, "blue", "blue_fill", radius=True, lw=0.75)
        cells(slide, x + 0.10, 4.89, 2, 5, 0.10, 0.08, "blue", "blue_fill")
        text(slide, x, 5.36, 0.88, 0.16, f"{label} sealed", size=5.9, color="muted")
    text(slide, 3.98, 4.91, 0.22, 0.25, "+", size=14, color="muted")
    box(slide, 4.31, 4.72, 1.06, 0.58, "green", "green_fill", radius=True, lw=0.75)
    cells(slide, 4.47, 4.89, 2, 4, 0.11, 0.08, "green", "green_fill")
    text(slide, 4.31, 5.36, 1.06, 0.16, "active dense", size=5.9, color="muted")
    mini(slide, 5.58, 4.79, 1.32, 0.40, "sealed: C, α, owner\nV: INT8 + scale", "purple", "purple_fill", 5.9)
    arrow(slide, 1.46, 3.48, 1.46, 4.06, "teal", 1.0)
    arrow(slide, 6.05, 3.48, 6.05, 4.06, "teal", 1.0)

    # Stage transition.
    arrow(slide, 7.40, 3.26, 7.88, 3.26, "red", 1.0)
    arrow(slide, 7.08, 4.80, 8.12, 3.70, "teal", 1.0, dashed=True)
    text(slide, 7.02, 4.17, 1.15, 0.20, "sealed page factors", size=6.0, color="teal")

    # 4. Direct attention.
    module(slide, 7.88, 1.05, 3.05, 2.42, "Direct compressed-space attention", "4",
           "purple", "purple_fill")
    mini(slide, 8.12, 1.58, 0.78, 0.38, "current q", "orange", "orange_fill", 7.0, True)
    mini(slide, 9.05, 1.58, 1.64, 0.38, "scores = (qA)C^T", "blue", "blue_fill", 6.4, True)
    arrow(slide, 8.93, 1.77, 9.05, 1.77, "orange", 0.8)
    mini(slide, 8.12, 2.18, 1.08, 0.35, "sealed pages", "blue", "blue_fill", 6.4, True)
    mini(slide, 9.55, 2.18, 1.20, 0.35, "active dense page", "green", "green_fill", 6.2, True)
    arrow(slide, 9.00, 2.18, 9.00, 2.04, "blue", 0.75)
    arrow(slide, 10.10, 2.18, 10.10, 2.04, "green", 0.75)
    mini(slide, 8.48, 2.79, 1.84, 0.30, "FP32 online softmax; no K reconstruction", "red", "red_fill", 5.8, True)
    arrow(slide, 9.40, 2.55, 9.40, 2.75, "red", 0.75)

    # 5. Page update.
    module(slide, 11.13, 1.05, 1.82, 2.42, "Page update", "5", "red", "red_fill")
    mini(slide, 11.38, 1.58, 0.72, 0.35, "append\ntoken", "orange", "orange_fill", 6.5, True)
    diamond(slide, 12.11, 2.06, 0.65, 0.53, "full?")
    arrow(slide, 11.94, 1.94, 12.00, 1.96, "orange", 0.75)
    mini(slide, 11.37, 2.60, 0.78, 0.30, "No: dense", "green", "green_fill", 5.9)
    mini(slide, 12.36, 2.60, 0.42, 0.30, "Yes", "teal", "teal_fill", 6.0, True)
    mini(slide, 11.78, 3.03, 1.00, 0.30, "seal + compress", "teal", "teal_fill", 6.1, True)
    arrow(slide, 12.11, 2.34, 12.12, 3.00, "teal", 0.75)
    arrow(slide, 11.78, 3.18, 6.25, 3.48, "teal", 0.9, dashed=True)
    text(slide, 8.62, 3.29, 2.15, 0.18, "only the active page is updated", size=6.0, color="teal")

    # Final attention output.
    box(slide, 7.88, 4.08, 5.07, 1.84, "red", "red_fill", radius=True, lw=1.0)
    text(slide, 8.10, 4.22, 2.10, 0.24, "Final attention output", size=9.0, color="ink", bold=True, align=PP_ALIGN.LEFT)
    mini(slide, 8.19, 4.73, 0.96, 0.42, "compressed\nscores", "purple", "purple_fill", 6.3, True)
    arrow(slide, 9.25, 4.94, 9.48, 4.94, "red", 0.8)
    mini(slide, 9.53, 4.73, 0.97, 0.42, "online\nsoftmax", "red", "red_fill", 6.3, True)
    arrow(slide, 10.58, 4.94, 10.81, 4.94, "red", 0.8)
    mini(slide, 10.86, 4.73, 0.92, 0.42, "gathered V", "green", "green_fill", 6.3, True)
    arrow(slide, 11.88, 4.94, 12.11, 4.94, "red", 0.8)
    mini(slide, 12.16, 4.73, 0.58, 0.42, "output", "blue", "blue_fill", 6.2, True)
    arrow(slide, 9.40, 3.50, 9.40, 4.05, "red", 0.9)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    prs.save(OUT)


if __name__ == "__main__":
    build()
