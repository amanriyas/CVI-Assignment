"""
generate_report.py - CVI Assignment 3-Page PDF Report Generator

Produces a 3-page A4 PDF:
  - Font: Arial Narrow 11pt (falls back to Helvetica if not found)
  - Margins: 2cm all sides
  - Header: Student ID 2717450 | Module: Computer Vision and Imaging [06 32578]
  - Structure per task: Aim -> Method -> Results -> Conclusions

Run after all task scripts have completed successfully.
"""

import os
import sys
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import cm
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
    from reportlab.lib import colors
    from reportlab.platypus import (
        BaseDocTemplate, PageTemplate, Frame,
        Paragraph, Spacer, Image as RLImage,
        Table, TableStyle, PageBreak, KeepTogether,
        HRFlowable,
    )
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
except ImportError:
    print("reportlab not found. Install with:  pip install reportlab")
    sys.exit(1)

OUTPUT_DIR  = "output"
REPORT_PATH = os.path.join(OUTPUT_DIR, "CVI_Report_2717450.pdf")
STUDENT_ID  = "2717450"
MODULE      = "Computer Vision and Imaging [06 32578]"
os.makedirs(OUTPUT_DIR, exist_ok=True)

PAGE_W, PAGE_H = A4
MARGIN         = 2.0 * cm
CONTENT_W      = PAGE_W - 2 * MARGIN

FONT_SIZE   = 11
SMALL_SIZE  = 8.5
TINY_SIZE   = 7.5
HEAD_SIZE   = 11.5

# Register Arial Narrow from Windows system fonts; fall back to Helvetica
_FONT_DIR = r"C:\Windows\Fonts"
_FONT_MAP = {
    "ArialNarrow":        "arialn.ttf",
    "ArialNarrow-Bold":   "arialnb.ttf",
    "ArialNarrow-Italic": "arialni.ttf",
}
_fonts_ok = all(
    os.path.exists(os.path.join(_FONT_DIR, f)) for f in _FONT_MAP.values()
)
if _fonts_ok:
    for name, fname in _FONT_MAP.items():
        pdfmetrics.registerFont(TTFont(name, os.path.join(_FONT_DIR, fname)))
    BODY = "ArialNarrow"
    BOLD = "ArialNarrow-Bold"
    ITAL = "ArialNarrow-Italic"
else:
    print("Arial Narrow not found on this system - using Helvetica.")
    BODY = "Helvetica"
    BOLD = "Helvetica-Bold"
    ITAL = "Helvetica-Oblique"


# Paragraph styles
def _s(name, **kw):
    defaults = dict(fontName=BODY, fontSize=FONT_SIZE, leading=12.5,
                    alignment=TA_JUSTIFY, spaceAfter=2, spaceBefore=0)
    defaults.update(kw)
    return ParagraphStyle(name, **defaults)

S_BODY  = _s("body")
S_HEAD  = _s("head",  fontName=BOLD, fontSize=HEAD_SIZE, leading=14,
             spaceBefore=7, spaceAfter=3, alignment=TA_LEFT)
S_SUB   = _s("sub",   fontName=BOLD, fontSize=FONT_SIZE, leading=12.5,
             spaceBefore=4,  spaceAfter=2, alignment=TA_LEFT)
S_CAP   = _s("cap",   fontName=ITAL, fontSize=SMALL_SIZE, leading=10,
             spaceBefore=2, spaceAfter=4, alignment=TA_CENTER)
S_TH    = _s("th",    fontName=BOLD, fontSize=TINY_SIZE, leading=9,
             alignment=TA_CENTER)
S_TC    = _s("tc",    fontName=BODY, fontSize=TINY_SIZE, leading=9,
             alignment=TA_CENTER)

SP = lambda n: Spacer(1, n)


def _header_footer(canvas, doc):
    """Draw header line and page number on every page."""
    canvas.saveState()
    canvas.setFont(BOLD, 8.5)
    txt = f"Student ID: {STUDENT_ID}    |    {MODULE}"
    canvas.drawString(MARGIN, PAGE_H - MARGIN + 10, txt)
    canvas.setLineWidth(0.5)
    canvas.line(MARGIN, PAGE_H - MARGIN + 6,
                PAGE_W - MARGIN, PAGE_H - MARGIN + 6)
    canvas.setFont(BODY, 8)
    canvas.drawRightString(PAGE_W - MARGIN, MARGIN - 14,
                           f"Page {doc.page}")
    canvas.restoreState()


def _img_rl(path, max_w=None):
    """Return a reportlab Image flowable scaled to fit the content width."""
    if max_w is None:
        max_w = CONTENT_W
    im = cv2.imread(path)
    if im is None:
        raise FileNotFoundError(f"Cannot load figure: {path}")
    h_px, w_px = im.shape[:2]
    height = max_w * h_px / w_px
    return RLImage(path, width=max_w, height=height)


def _make_report_fig1():
    """Task 1 figure: 3 rows x 5 cols - all three images, each showing Original | Sobel | Prewitt | Laplacian | GT."""
    from task1_edge_detection import (load_image_pair, preprocess,
                                       apply_sobel, apply_prewitt,
                                       apply_laplacian, IMAGE_PAIRS)
    fig, axes = plt.subplots(3, 5, figsize=(13, 6.6))
    shorts = ["10905 JL", "43590 AM", "9343 AM"]
    for row, ((img_f, gt_f, _), short) in enumerate(zip(IMAGE_PAIRS, shorts)):
        img_gray, gt_bin = load_image_pair(img_f, gt_f)
        sm = preprocess(img_gray, sigma=1.0)
        _, sob, _ = apply_sobel(sm)
        _, pre, _ = apply_prewitt(sm)
        _, lap, _ = apply_laplacian(img_gray)
        for col, (im, t) in enumerate(zip(
                [img_gray, sob, pre, lap, gt_bin],
                [f"Orig ({short})", "Sobel", "Prewitt", "Laplacian", "GT"])):
            axes[row, col].imshow(im, cmap="gray", vmin=0, vmax=255)
            axes[row, col].set_title(t, fontsize=7)
            axes[row, col].axis("off")
    plt.tight_layout(pad=0.25)
    out = os.path.join(OUTPUT_DIR, "rep_fig1_task1.png")
    plt.savefig(out, dpi=95, bbox_inches="tight")
    plt.close()
    return out


def _make_report_fig2():
    """
    Task 2 figure: 2 rows.
    Row 1: Best Canny result for each of the three images.
    Row 2: Threshold sensitivity on 10905 JL (four threshold pairs).
    """
    from task1_edge_detection import load_image_pair, preprocess, IMAGE_PAIRS
    from task2_canny import apply_canny

    best_params = [
        (IMAGE_PAIRS[0], 30, 60,  "10905 JL"),
        (IMAGE_PAIRS[1], 10, 20,  "43590 AM"),
        (IMAGE_PAIRS[2], 20, 40,  "9343 AM"),
    ]

    fig, axes = plt.subplots(2, 4, figsize=(13, 5.2))

    for col, ((img_f, gt_f, _), tl, th, short) in enumerate(best_params):
        img_gray, gt_bin = load_image_pair(img_f, gt_f)
        best_e = apply_canny(img_gray, tl, th, sigma=1.0)
        axes[0, col].imshow(best_e, cmap="gray", vmin=0, vmax=255)
        axes[0, col].set_title(f"Canny ({short})\nt_low={tl}", fontsize=7)
        axes[0, col].axis("off")
    axes[0, 3].axis("off")

    img0, _ = load_image_pair(IMAGE_PAIRS[0][0], IMAGE_PAIRS[0][1])
    for col, (tl, th) in enumerate([(5, 10), (10, 20), (20, 40), (30, 60)]):
        e = apply_canny(img0, tl, th, sigma=1.0)
        axes[1, col].imshow(e, cmap="gray", vmin=0, vmax=255)
        axes[1, col].set_title(f"10905 JL\nt_low={tl}", fontsize=7)
        axes[1, col].axis("off")

    plt.tight_layout(pad=0.25)
    out = os.path.join(OUTPUT_DIR, "rep_fig2_task2.png")
    plt.savefig(out, dpi=100, bbox_inches="tight")
    plt.close()
    return out


def _make_report_fig3():
    """Task 3 figure: 1 row x 4 panels - Original | Canny | Hough lines | Accumulator heatmap."""
    from task2_canny import apply_canny
    img_bgr = cv2.imread("UoBDImage.jpg")
    gray    = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    edges   = apply_canny(gray, 50, 150, sigma=1.5)
    lines   = cv2.HoughLines(edges, 1, np.pi / 180, 200)
    overlay = img_bgr.copy()
    n_lines = 0
    if lines is not None:
        n_lines = len(lines)
        for ln in lines:
            rho, theta = ln[0]
            a, b = np.cos(theta), np.sin(theta)
            x0, y0 = a * rho, b * rho
            x1 = int(x0 + 2000 * (-b)); y1 = int(y0 + 2000 * a)
            x2 = int(x0 - 2000 * (-b)); y2 = int(y0 - 2000 * a)
            cv2.line(overlay, (x1, y1), (x2, y2), (0, 0, 220), 2)

    # Compact accumulator on downscaled edge map
    h, w    = gray.shape
    scale   = 0.35
    e_small = cv2.resize(edges, (int(w * scale), int(h * scale)))
    D       = int(np.ceil(np.sqrt(e_small.shape[0]**2 + e_small.shape[1]**2)))
    thetas  = np.deg2rad(np.arange(-90, 90, 1))
    cos_t, sin_t = np.cos(thetas), np.sin(thetas)
    acc     = np.zeros((2 * D + 1, len(thetas)), dtype=np.float32)
    for y_e, x_e in np.argwhere(e_small > 0):
        rhos = np.round(x_e * cos_t + y_e * sin_t + D).astype(int)
        valid = (rhos >= 0) & (rhos < acc.shape[0])
        acc[rhos[valid], np.where(valid)[0]] += 1
    acc_disp = cv2.normalize(acc, None, 0, 255,
                             cv2.NORM_MINMAX).astype(np.uint8)

    fig, axes = plt.subplots(1, 4, figsize=(16, 3.8))
    panels = [
        (cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB),
         "Original (UoBDImage.jpg)"),
        (edges,
         "Canny Edges\n(t_low=50, t_high=150)"),
        (cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB),
         f"Hough Lines\n(threshold=200, n={n_lines})"),
    ]
    for ax, (im, t) in zip(axes[:3], panels):
        ax.imshow(im, cmap="gray" if len(im.shape) == 2 else None,
                  vmin=0 if len(im.shape) == 2 else None,
                  vmax=255 if len(im.shape) == 2 else None)
        ax.set_title(t, fontsize=7.5)
        ax.axis("off")
    im_acc = axes[3].imshow(acc_disp, cmap="hot", aspect="auto",
                             extent=[-90, 90, -1, 1])
    axes[3].set_title("Hough Accumulator\n(bright = many votes)", fontsize=7.5)
    axes[3].set_xlabel("Theta (deg)", fontsize=6)
    axes[3].set_ylabel("Rho (norm.)", fontsize=6)
    axes[3].tick_params(labelsize=5)
    plt.tight_layout(pad=0.3)
    out = os.path.join(OUTPUT_DIR, "rep_fig3_task3.png")
    plt.savefig(out, dpi=115, bbox_inches="tight")
    plt.close()
    return out


def _tbl_style(header_color):
    return TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0),  header_color),
        ("TEXTCOLOR",     (0, 0), (-1, 0),  colors.white),
        ("FONTNAME",      (0, 0), (-1, 0),  BOLD),
        ("FONTSIZE",      (0, 0), (-1, -1), TINY_SIZE),
        ("BACKGROUND",    (0, 1), (-1, -1), colors.HexColor("#F9F9F9")),
        ("GRID",          (0, 0), (-1, -1), 0.4, colors.HexColor("#CCCCCC")),
        ("ALIGN",         (0, 0), (-1, -1), "CENTER"),
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING",    (0, 0), (-1, -1), 2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
        ("LEFTPADDING",   (0, 0), (-1, -1), 3),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 3),
    ])


def _task1_table():
    """Task 1 metrics table with merged image-name cells."""
    cw = [CONTENT_W * f for f in [0.185, 0.135, 0.17, 0.17, 0.17, 0.17]]
    header = [[Paragraph(h, S_TH) for h in
               ["Image", "Detector", "Precision", "Recall", "F1", "IoU"]]]
    rows = [
        ["10905 JL", "Sobel",     "0.369", "0.916", "0.526", "0.357"],
        ["",         "Prewitt",   "0.364", "0.918", "0.521", "0.353"],
        ["",         "Laplacian", "0.148", "0.400", "0.216", "0.121"],
        ["43590 AM", "Sobel",     "0.402", "0.764", "0.527", "0.358"],
        ["",         "Prewitt",   "0.406", "0.763", "0.530", "0.360"],
        ["",         "Laplacian", "0.134", "0.411", "0.202", "0.112"],
        ["9343 AM",  "Sobel",     "0.390", "0.850", "0.535", "0.365"],
        ["",         "Prewitt",   "0.386", "0.854", "0.532", "0.362"],
        ["",         "Laplacian", "0.193", "0.236", "0.212", "0.119"],
    ]
    data = header + [[Paragraph(c, S_TC) for c in r] for r in rows]
    tbl  = Table(data, colWidths=cw, repeatRows=1)
    st   = _tbl_style(colors.HexColor("#1565C0"))
    st.add("BACKGROUND", (0, 1), (-1, 3),  colors.HexColor("#EEF4FF"))
    st.add("BACKGROUND", (0, 4), (-1, 6),  colors.HexColor("#F9F9F9"))
    st.add("BACKGROUND", (0, 7), (-1, 9),  colors.HexColor("#EEF4FF"))
    st.add("SPAN",       (0, 1), (0, 3))
    st.add("SPAN",       (0, 4), (0, 6))
    st.add("SPAN",       (0, 7), (0, 9))
    tbl.setStyle(st)
    return tbl


def _task2_table():
    """Task 2 best-Canny metrics table."""
    cw = [CONTENT_W * f for f in [0.185, 0.09, 0.09, 0.16, 0.16, 0.155, 0.16]]
    header = [[Paragraph(h, S_TH) for h in
               ["Image", "t_low", "t_high", "Precision", "Recall", "F1", "IoU"]]]
    rows = [
        ["10905 JL", "30", "60",  "0.728", "0.348", "0.470", "0.308"],
        ["43590 AM", "10", "20",  "0.472", "0.272", "0.345", "0.208"],
        ["9343 AM",  "20", "40",  "0.472", "0.323", "0.383", "0.237"],
    ]
    data = header + [[Paragraph(c, S_TC) for c in r] for r in rows]
    tbl  = Table(data, colWidths=cw, repeatRows=1)
    st   = _tbl_style(colors.HexColor("#E65100"))
    st.add("BACKGROUND", (0, 1), (-1, 1), colors.HexColor("#FFF3E0"))
    st.add("BACKGROUND", (0, 2), (-1, 2), colors.HexColor("#F9F9F9"))
    st.add("BACKGROUND", (0, 3), (-1, 3), colors.HexColor("#FFF3E0"))
    tbl.setStyle(st)
    return tbl


def build_report(fig1, fig2, fig3):
    doc = BaseDocTemplate(
        REPORT_PATH,
        pagesize=A4,
        leftMargin=MARGIN,
        rightMargin=MARGIN,
        topMargin=MARGIN + 16,
        bottomMargin=MARGIN + 10,
    )
    frame = Frame(doc.leftMargin, doc.bottomMargin,
                  doc.width, doc.height, id="main")
    doc.addPageTemplates([
        PageTemplate(id="main", frames=[frame], onPage=_header_footer)
    ])

    story = []

    # TASK 1
    story.append(Paragraph("TASK 1: EDGE DETECTION", S_HEAD))

    story.append(Paragraph("<b>Aim</b>", S_SUB))
    story.append(Paragraph(
        "To apply three gradient-based edge detectors - Sobel, Prewitt, and "
        "Laplacian - to three HEp-2 immunofluorescence cell images (1280x1024 pixels, "
        "colour BMP) and evaluate their ability to capture manually annotated cell "
        "boundaries using Precision, Recall, F1-score, and Intersection over Union (IoU).",
        S_BODY))

    story.append(Paragraph("<b>Method</b>", S_SUB))
    story.append(Paragraph(
        "Images were converted from 3-channel colour to grayscale. Ground truth maps "
        "were inverted (white background, dark edges; mean pixel ~234) and "
        "automatically corrected so that edge pixels = 255. "
        "A Gaussian filter (sigma=1.0, 7x7 kernel) was applied before Sobel and "
        "Prewitt detection: the observation step revealed non-zero background variance "
        "(std 10-25), indicating sensor noise. Gaussian filtering "
        "G(x,y) = exp(-(x^2+y^2)/(2*sigma^2)) is preferred over box filtering as it "
        "preserves edge sharpness. The Laplacian used sigma=1.5 (9x9 kernel) because "
        "second-order derivatives amplify noise more aggressively than first-order "
        "methods.", S_BODY))
    story.append(Paragraph(
        "Edge detection followed three steps: (1) Compute Gx and Gy by convolution - "
        "Sobel weights the centre row/column x2 (Gx=[[-1,0,1],[-2,0,2],[-1,0,1]]) "
        "for better noise suppression; Prewitt uses uniform weights, making it simpler "
        "but slightly more noise-sensitive. "
        "(2) Gradient magnitude M(G) = sqrt(Gx^2+Gy^2) and direction "
        "alpha = arctan(Gy/Gx) are computed. "
        "(3) Otsu's thresholding is applied to produce a binary edge map - Otsu "
        "minimises intra-class variance between edge and non-edge pixels, providing a "
        "data-driven threshold. The Laplacian operator "
        "nabla^2 f = f(x+1,y)+f(x-1,y)+f(x,y+1)+f(x,y-1)-4f(x,y) detects edges in "
        "all directions; absolute values capture both positive and negative "
        "zero-crossings.", S_BODY))

    story.append(Paragraph("<b>Results</b>", S_SUB))
    story.append(_img_rl(fig1))
    story.append(Paragraph(
        "Figure 1: Sobel, Prewitt and Laplacian binary edge maps for all three "
        "HEp-2 images alongside the corresponding ground truth (GT).",
        S_CAP))
    story.append(SP(3))
    story.append(_task1_table())
    story.append(Paragraph(
        "Table 1: Quantitative evaluation (Precision, Recall, F1, IoU) for all three "
        "Task 1 detectors on all three images.",
        S_CAP))

    story.append(Paragraph("<b>Conclusions</b>", S_SUB))
    story.append(Paragraph(
        "Sobel and Prewitt achieved comparable F1 (~0.53) across all images, "
        "substantially outperforming the Laplacian (F1 ~0.21). Both first-order "
        "detectors show high recall (0.76-0.92) but moderate precision (0.37-0.41), "
        "as the absence of non-maximum suppression produces thick multi-pixel edges "
        "that overlap many ground truth pixels. Sobel's centre-weighted kernels gave "
        "marginally higher precision; Prewitt's uniform weights gave slightly higher "
        "recall, consistent with its greater sensitivity to subtle transitions. "
        "The Laplacian performed worst due to its second-order derivative amplifying "
        "residual noise despite stronger smoothing, producing noisy and doubled edges. "
        "Adding non-maximum suppression would improve precision without sacrificing "
        "recall.",
        S_BODY))

    story.append(SP(3))

    # TASK 2
    story.append(Paragraph("TASK 2: ADVANCED EDGE DETECTION - CANNY", S_HEAD))

    story.append(Paragraph("<b>Aim</b>", S_SUB))
    story.append(Paragraph(
        "To apply the Canny edge detector to the same three HEp-2 images, investigate "
        "the sensitivity of results to hysteresis threshold selection, and compare "
        "quantitative performance against the Task 1 methods.",
        S_BODY))

    story.append(Paragraph("<b>Method</b>", S_SUB))
    story.append(Paragraph(
        "Canny (1986) defines three criteria: (1) Good detection - low probability of "
        "missing real or falsely marking non-edges; (2) Good localisation - marked "
        "points close to the true edge centre; (3) Single response per true edge. "
        "The algorithm implements these via four steps: "
        "<b>Step 1</b> - Gaussian filtering (sigma=1.0, 7x7 kernel) suppresses noise. "
        "<b>Step 2</b> - Sobel operators yield M = sqrt(Gx^2+Gy^2) and "
        "alpha = arctan(Gy/Gx), identical to Task 1.",
        S_BODY))
    story.append(Paragraph(
        "<b>Step 3 - Non-maximum suppression (NMS):</b> Each pixel is compared against "
        "its two neighbours along the gradient direction; if not the local maximum it is "
        "suppressed to zero, producing single-pixel-wide edges (criterion 3). "
        "<b>Step 4 - Hysteresis thresholding:</b> t_high = 2*t_low. Pixels above "
        "t_high are strong edges; below t_low are rejected; between are accepted only "
        "if 8-connected to a strong edge, ensuring continuity (criterion 1) - unlike "
        "single-threshold approaches that cannot bridge edge gaps.",
        S_BODY))
    story.append(Paragraph(
        "HEp-2 images have low mean intensities (7-19/255), producing small gradient "
        "magnitudes. Four threshold pairs (t_low in {5, 10, 20, 30}, t_high=2*t_low) "
        "were tested to find the optimal setting per image.",
        S_BODY))

    story.append(Paragraph("<b>Results</b>", S_SUB))
    story.append(_img_rl(fig2))
    story.append(Paragraph(
        "Figure 2 - Top: Best Canny result per image (t_low=30/10/20 respectively). "
        "Bottom: Threshold sensitivity on 10905 JL - higher t_low raises precision "
        "but rapidly reduces recall.",
        S_CAP))
    story.append(SP(3))
    story.append(_task2_table())
    story.append(Paragraph(
        "Table 2: Best Canny threshold pair per image (selected by highest F1 score). "
        "Sobel F1 comparison: 10905 JL=0.526, 43590 AM=0.527, 9343 AM=0.535.",
        S_CAP))

    story.append(Paragraph("<b>Conclusions</b>", S_SUB))
    story.append(Paragraph(
        "Canny achieves higher precision than Task 1 (0.47-0.73 vs. 0.36-0.41) "
        "due to NMS producing single-pixel-wide edges. However, overall F1 (0.35-0.47) "
        "remains below Sobel/Prewitt (~0.53) because NMS and hysteresis conservatively "
        "suppress faint cell boundaries in these dark fluorescence images. "
        "As t_low increases, precision rises while recall falls sharply. "
        "The optimal t_low varies per image (10-30): dark 43590 AM (mean=7.1) needed "
        "t_low=10 while 10905 JL (mean=18.6) used t_low=30, confirming that threshold "
        "selection must account for image contrast.",
        S_BODY))

    story.append(SP(3))

    # TASK 3
    story.append(Paragraph("TASK 3: HOUGH TRANSFORM FOR LINE DETECTION", S_HEAD))

    story.append(Paragraph("<b>Aim</b>", S_SUB))
    story.append(Paragraph(
        "To detect straight lines in a university building image (UoBDImage.jpg) "
        "by applying the Hough Transform to Canny-detected edge points, and to "
        "evaluate the effect of accumulator threshold selection on detection quality.",
        S_BODY))

    story.append(Paragraph("<b>Method</b>", S_SUB))
    story.append(Paragraph(
        "Canny edge detection (sigma=1.5, t_low=50, t_high=150) was applied to "
        "UoBDImage.jpg (1200x675 pixels, colour JPEG). A stronger sigma than Tasks "
        "1-2 was used to suppress JPEG compression artefacts - high-frequency noise "
        "that would otherwise generate spurious accumulator votes for lines that do "
        "not exist in the scene.",
        S_BODY))
    story.append(Paragraph(
        "The Hough Transform uses the normal line parameterisation: "
        "rho = x*cos(theta) + y*sin(theta), where rho is the perpendicular distance "
        "from the image origin to the line and theta is the angle of the perpendicular. "
        "An accumulator H(rho, theta) was initialised to zero with rho quantised at "
        "1-pixel resolution and theta at 1-degree steps over [-90, 90]. "
        "For each Canny edge pixel (x, y), rho was computed for every theta value and "
        "the corresponding bin incremented. Peaks in H above a threshold correspond to "
        "lines supported by many collinear edge points, making the method robust to "
        "partial occlusion since remaining visible edge points still accumulate "
        "sufficient votes.",
        S_BODY))
    story.append(Paragraph(
        "Three accumulator thresholds (120, 150, 200) were tested to observe the "
        "trade-off between clutter and missed lines. The probabilistic Hough Transform "
        "(HoughLinesP, threshold=100, minLineLength=50, maxLineGap=15) was also "
        "applied, returning finite segments rather than infinite lines.",
        S_BODY))

    story.append(Paragraph("<b>Results</b>", S_SUB))
    story.append(_img_rl(fig3))
    story.append(Paragraph(
        "Figure 3: Original image; Canny edge map; Hough lines overlaid "
        "(threshold=200, 238 lines); Hough accumulator (bright = high vote count).",
        S_CAP))
    story.append(Paragraph(
        "Threshold=120 detected 4091 lines (severe over-detection); "
        "threshold=150 detected 1782 lines; threshold=200 detected 238 lines "
        "(primary result). HoughLinesP detected 869 line segments at threshold=100.",
        S_BODY))

    story.append(Paragraph("<b>Conclusions</b>", S_SUB))
    story.append(Paragraph(
        "At threshold=200 the Hough Transform successfully identified dominant "
        "architectural features - horizontal and vertical structural elements including "
        "walls, window frames, and roof edges. False positives arise from textured "
        "surfaces (brickwork, glass reflections) whose edge pixels accumulate spurious "
        "votes: consistent with the lecture observation that clutter increases the "
        "number of false accumulator peaks. Reducing the threshold to 120 increased "
        "detections 17-fold, confirming that the threshold directly controls the "
        "precision-recall balance. Missing lines (e.g., faint secondary edges) result "
        "from insufficient Canny support after NMS and hysteresis suppression. "
        "The probabilistic variant provided cleaner segment-level output - useful for "
        "identifying specific structural elements - but missed some lines due to "
        "random sub-sampling of edge points. Increasing the Gaussian sigma further "
        "or raising the accumulator threshold would reduce false positives, while "
        "lowering the Canny thresholds would recover weaker but genuine edges.",
        S_BODY))

    doc.build(story)
    print(f"\nReport saved to: {REPORT_PATH}")


if __name__ == "__main__":
    print("Generating compact report figures...")
    f1 = _make_report_fig1()
    f2 = _make_report_fig2()
    f3 = _make_report_fig3()
    print("Building PDF report...")
    build_report(f1, f2, f3)
