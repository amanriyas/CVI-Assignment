"""
task3_hough.py - Hough Transform for Line Detection (Task 3)

Applies the Hough Transform to detect straight lines in UoBDImage.jpg
using the normal parameterisation: rho = x*cos(theta) + y*sin(theta).
"""

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)
HOUGH_IMAGE = "UoBDImage.jpg"


def load_hough_image(path):
    """Load the building image and return (BGR, grayscale)."""
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Cannot load: {path}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img, gray


def get_canny_edges(gray, t_low=50, t_high=150, sigma=1.5):
    """
    Canny edge detection for the Hough input.
    sigma=1.5 is used to suppress JPEG compression artefacts that would
    otherwise generate spurious accumulator votes.
    """
    ksize    = int(6 * sigma) | 1
    smoothed = cv2.GaussianBlur(gray, (ksize, ksize), sigma)
    edges    = cv2.Canny(smoothed, t_low, t_high)
    return edges


def compute_hough_accumulator(edges):
    """
    Manually compute the Hough accumulator (rho-theta) for visualisation.
    rho range: [-D, +D], theta range: [-90, +90] degrees in 1-degree steps.
    Returns accumulator normalised to 0-255 for display.
    """
    h, w      = edges.shape
    D         = int(np.ceil(np.sqrt(h**2 + w**2)))
    thetas    = np.deg2rad(np.arange(-90, 90, 1))
    rho_range = np.arange(-D, D + 1, 1)

    cos_t = np.cos(thetas)
    sin_t = np.sin(thetas)

    accumulator = np.zeros((len(rho_range), len(thetas)), dtype=np.float32)

    edge_pts = np.argwhere(edges > 0)  # (y, x) format
    for y, x in edge_pts:
        rhos = x * cos_t + y * sin_t
        rho_indices = np.round(rhos + D).astype(int)
        valid = (rho_indices >= 0) & (rho_indices < len(rho_range))
        accumulator[rho_indices[valid], np.where(valid)[0]] += 1

    acc_norm = cv2.normalize(accumulator, None, 0, 255,
                             cv2.NORM_MINMAX).astype(np.uint8)
    return acc_norm, accumulator


def draw_hough_lines(img_bgr, lines, color=(0, 0, 255), thickness=2):
    """Draw infinite lines from cv2.HoughLines onto a copy of the image."""
    out = img_bgr.copy()
    if lines is None:
        return out
    for line in lines:
        rho, theta = line[0]
        a  = np.cos(theta)
        b  = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(round(x0 + 2000 * (-b)))
        y1 = int(round(y0 + 2000 *   a))
        x2 = int(round(x0 - 2000 * (-b)))
        y2 = int(round(y0 - 2000 *   a))
        cv2.line(out, (x1, y1), (x2, y2), color, thickness)
    return out


def draw_hough_lines_p(img_bgr, lines_p, color=(0, 255, 0), thickness=2):
    """Draw line segments from cv2.HoughLinesP onto a copy of the image."""
    out = img_bgr.copy()
    if lines_p is None:
        return out
    for seg in lines_p:
        x1, y1, x2, y2 = seg[0]
        cv2.line(out, (x1, y1), (x2, y2), color, thickness)
    return out


def main():
    print("=" * 65)
    print("TASK 3 - HOUGH TRANSFORM FOR LINE DETECTION")
    print("=" * 65)

    img_bgr, gray = load_hough_image(HOUGH_IMAGE)
    h, w = gray.shape
    print(f"\nImage: {HOUGH_IMAGE}  ({w} x {h} pixels)")

    # Step 1: Canny edge detection
    T_LOW, T_HIGH, SIGMA = 50, 150, 1.5
    print(f"\nCanny parameters: t_low={T_LOW}, t_high={T_HIGH}, sigma={SIGMA}")
    edges = get_canny_edges(gray, T_LOW, T_HIGH, SIGMA)
    edge_density = (edges > 0).mean() * 100
    print(f"  Edge pixel density: {edge_density:.2f}%")

    # Step 2: Standard Hough Transform at three thresholds
    # Higher threshold = fewer but more prominent lines; lower = more lines including false positives.
    HOUGH_THRESHOLDS = [120, 150, 200]
    hough_results    = {}
    for thresh in HOUGH_THRESHOLDS:
        lines = cv2.HoughLines(
            edges,
            rho=1,
            theta=np.pi/180,
            threshold=thresh
        )
        n_lines = 0 if lines is None else len(lines)
        hough_results[thresh] = lines
        print(f"  HoughLines threshold={thresh}: {n_lines} lines detected")

    # Step 3: Probabilistic Hough Transform for comparison
    lines_p = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi/180,
        threshold=100,
        minLineLength=50,
        maxLineGap=15
    )
    n_segs = 0 if lines_p is None else len(lines_p)
    print(f"\n  HoughLinesP (probabilistic): {n_segs} line segments detected")

    # Compute accumulator on downscaled edge map for visualisation speed
    print("\nComputing Hough accumulator...")
    scale        = 0.4
    edges_small  = cv2.resize(edges, (int(w * scale), int(h * scale)))
    acc_norm, _  = compute_hough_accumulator(edges_small)
    print(f"  Accumulator size: {acc_norm.shape[1]} theta x {acc_norm.shape[0]} rho bins")

    # ---------------------------------------------------------------
    # Figure 1: Original | Canny | Hough lines | Probabilistic Hough
    # ---------------------------------------------------------------
    print("\nGenerating figures...")
    best_thresh = 200
    lines_draw  = hough_results[best_thresh]
    img_lines   = draw_hough_lines(img_bgr, lines_draw)
    img_lines_p = draw_hough_lines_p(img_bgr, lines_p)

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    panels = [
        (cv2.cvtColor(img_bgr,    cv2.COLOR_BGR2RGB), "Original Image"),
        (edges,                                        "Canny Edge Map\n"
                                                       f"(t_low={T_LOW}, t_high={T_HIGH})"),
        (cv2.cvtColor(img_lines,  cv2.COLOR_BGR2RGB), f"Standard Hough Lines\n"
                                                       f"(threshold={best_thresh}, "
                                                       f"n={0 if lines_draw is None else len(lines_draw)})"),
        (cv2.cvtColor(img_lines_p, cv2.COLOR_BGR2RGB), f"Probabilistic Hough\n"
                                                        f"(threshold=80, "
                                                        f"n={n_segs} segments)"),
    ]
    for ax, (im, title) in zip(axes, panels):
        if len(im.shape) == 2:
            ax.imshow(im, cmap="gray", vmin=0, vmax=255)
        else:
            ax.imshow(im)
        ax.set_title(title, fontsize=8)
        ax.axis("off")
    plt.suptitle("Task 3 - Hough Transform Line Detection (UoBDImage)",
                 fontsize=11)
    plt.tight_layout()
    fig1_path = os.path.join(OUTPUT_DIR, "task3_hough_main.png")
    plt.savefig(fig1_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Figure saved: {fig1_path}")

    # ---------------------------------------------------------------
    # Figure 2: Effect of accumulator threshold on detected lines
    # ---------------------------------------------------------------
    fig, axes = plt.subplots(1, len(HOUGH_THRESHOLDS) + 1, figsize=(16, 5))
    axes[0].imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Original", fontsize=9)
    axes[0].axis("off")

    for ax, thresh in zip(axes[1:], HOUGH_THRESHOLDS):
        lns   = hough_results[thresh]
        drawn = draw_hough_lines(img_bgr, lns)
        n     = 0 if lns is None else len(lns)
        ax.imshow(cv2.cvtColor(drawn, cv2.COLOR_BGR2RGB))
        ax.set_title(f"Hough threshold={thresh}\n({n} lines)", fontsize=8)
        ax.axis("off")

    plt.suptitle(
        "Task 3 - Effect of Accumulator Threshold on Detected Lines",
        fontsize=10)
    plt.tight_layout()
    fig2_path = os.path.join(OUTPUT_DIR, "task3_threshold_effect.png")
    plt.savefig(fig2_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Figure saved: {fig2_path}")

    # ---------------------------------------------------------------
    # Figure 3: Hough accumulator space (rho-theta)
    # ---------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].imshow(edges, cmap="gray")
    axes[0].set_title("Canny Edge Map (input to Hough)", fontsize=9)
    axes[0].axis("off")

    im_acc = axes[1].imshow(acc_norm, cmap="hot", aspect="auto",
                            extent=[-90, 90, -1, 1])
    axes[1].set_title("Hough Accumulator Space\n"
                      "(bright = many votes = prominent line)", fontsize=9)
    axes[1].set_xlabel("Theta (degrees)")
    axes[1].set_ylabel("Rho (normalised)")
    plt.colorbar(im_acc, ax=axes[1], fraction=0.03, label="Votes (normalised)")
    plt.suptitle("Task 3 - Hough Accumulator (rho-theta parameter space)",
                 fontsize=10)
    plt.tight_layout()
    fig3_path = os.path.join(OUTPUT_DIR, "task3_accumulator.png")
    plt.savefig(fig3_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Figure saved: {fig3_path}")

    # ---------------------------------------------------------------
    # Figure 4: Effect of Canny parameters on Hough input
    # ---------------------------------------------------------------
    canny_params = [(30, 90), (50, 150), (80, 240)]
    fig, axes    = plt.subplots(1, len(canny_params) + 1, figsize=(16, 4))
    axes[0].imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Original", fontsize=9)
    axes[0].axis("off")

    for ax, (tl, th) in zip(axes[1:], canny_params):
        e     = get_canny_edges(gray, tl, th, sigma=1.5)
        lns   = cv2.HoughLines(e, 1, np.pi/180, 120)
        drawn = draw_hough_lines(img_bgr, lns)
        n     = 0 if lns is None else len(lns)
        ax.imshow(cv2.cvtColor(drawn, cv2.COLOR_BGR2RGB))
        ax.set_title(f"Canny t_low={tl}, t_high={th}\n"
                     f"Hough thresh=120 ({n} lines)", fontsize=8)
        ax.axis("off")

    plt.suptitle(
        "Task 3 - Effect of Canny Parameters on Hough Line Detection",
        fontsize=10)
    plt.tight_layout()
    fig4_path = os.path.join(OUTPUT_DIR, "task3_canny_sensitivity.png")
    plt.savefig(fig4_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Figure saved: {fig4_path}")

    # Summary
    print("\n" + "=" * 65)
    print("TASK 3 SUMMARY")
    print("=" * 65)
    print(f"  Image size           : {w} x {h}")
    print(f"  Canny edge density   : {edge_density:.2f}%")
    print(f"  Canny parameters     : t_low={T_LOW}, t_high={T_HIGH}, sigma={SIGMA}")
    print(f"  Hough rho resolution : 1 pixel")
    print(f"  Hough theta res.     : 1 degree")
    print()
    for thresh in HOUGH_THRESHOLDS:
        lns = hough_results[thresh]
        n   = 0 if lns is None else len(lns)
        print(f"  Standard Hough (threshold={thresh}): {n:3d} lines")
    print(f"  Probabilistic Hough  : {n_segs} segments "
          f"(threshold=80, minLen=40, maxGap=10)")
    print(f"\nTask 3 complete. Outputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
