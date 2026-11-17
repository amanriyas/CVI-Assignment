"""
task1_edge_detection.py - Basic Edge Detection (Task 1)

Applies Sobel, Prewitt, and Laplacian edge detectors to 3 HEp-2 cell images.
Evaluates against ground truth using Precision, Recall, F1, and IoU.
"""

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

from utils import compute_metrics, print_metrics_table

IMAGE_PAIRS = [
    ("10905 JL.bmp",   "10905 JL Edges.bmp",  "10905_JL"),
    ("43590 AM.bmp",   "43590 AM Edges.bmp",  "43590_AM"),
    ("9343 AM.bmp",    "9343 AM Edges.bmp",   "9343_AM"),
]
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_image_pair(img_path, gt_path):
    """Load image and ground truth as uint8 grayscale. Auto-inverts GT if background is white."""
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    gt  = cv2.imread(gt_path,  cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Cannot load: {img_path}")
    if gt is None:
        raise FileNotFoundError(f"Cannot load: {gt_path}")

    img_gray = (cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                if len(img.shape) == 3 else img.copy())
    gt_gray  = (cv2.cvtColor(gt,  cv2.COLOR_BGR2GRAY)
                if len(gt.shape)  == 3 else gt.copy())

    _, gt_bin = cv2.threshold(gt_gray, 127, 255, cv2.THRESH_BINARY)
    if gt_bin.mean() > 127:
        gt_bin = cv2.bitwise_not(gt_bin)

    return img_gray, gt_bin


def preprocess(img_gray, sigma=1.0):
    """Gaussian blur to suppress noise before gradient computation. Kernel size = 6*sigma rounded to odd."""
    ksize = int(6 * sigma) | 1
    if ksize < 3:
        ksize = 3
    return cv2.GaussianBlur(img_gray, (ksize, ksize), sigma)


def otsu_threshold(magnitude):
    """Apply Otsu's thresholding to a gradient magnitude map. Returns (threshold_value, binary_map)."""
    norm = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    thresh_val, binary = cv2.threshold(norm, 0, 255,
                                       cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh_val, binary


def apply_sobel(img_gray, ksize=3):
    """
    Sobel edge detector.
    Computes Gx, Gy using weighted 3x3 kernels, then M = sqrt(Gx^2 + Gy^2).
    Thresholded with Otsu. Returns (magnitude, binary, threshold).
    """
    Gx = cv2.Sobel(img_gray, cv2.CV_32F, 1, 0, ksize=ksize)
    Gy = cv2.Sobel(img_gray, cv2.CV_32F, 0, 1, ksize=ksize)
    magnitude = np.sqrt(Gx**2 + Gy**2)
    thresh_val, binary = otsu_threshold(magnitude)
    return magnitude, binary, thresh_val


def apply_prewitt(img_gray):
    """
    Prewitt edge detector.
    Uses uniform 3x3 kernels via cv2.filter2D (no built-in Prewitt in OpenCV).
    Returns (magnitude, binary, threshold).
    """
    kernel_Gx = np.array([[-1, 0, 1],
                           [-1, 0, 1],
                           [-1, 0, 1]], dtype=np.float32)
    kernel_Gy = np.array([[-1, -1, -1],
                           [ 0,  0,  0],
                           [ 1,  1,  1]], dtype=np.float32)

    img_f = img_gray.astype(np.float32)
    Gx = cv2.filter2D(img_f, -1, kernel_Gx)
    Gy = cv2.filter2D(img_f, -1, kernel_Gy)
    magnitude = np.sqrt(Gx**2 + Gy**2)
    thresh_val, binary = otsu_threshold(magnitude)
    return magnitude, binary, thresh_val


def apply_laplacian(img_gray, ksize=3):
    """
    Laplacian edge detector.
    Uses stronger pre-smoothing (sigma=1.5) because second-order derivatives
    amplify noise more than first-order methods.
    Absolute value taken to capture both positive and negative zero-crossings.
    Returns (abs_response, binary, threshold).
    """
    smoothed = cv2.GaussianBlur(img_gray, (9, 9), 1.5)
    lap = cv2.Laplacian(smoothed, cv2.CV_32F, ksize=ksize)
    abs_lap = np.abs(lap)
    thresh_val, binary = otsu_threshold(abs_lap)
    return abs_lap, binary, thresh_val


def main():
    print("=" * 65)
    print("TASK 1 - BASIC EDGE DETECTION")
    print("Detectors: Sobel, Prewitt, Laplacian")
    print("=" * 65)

    GAUSS_SIGMA = 1.0

    all_metrics = {
        "Sobel":     {},
        "Prewitt":   {},
        "Laplacian": {},
    }

    grid_images = []
    grid_titles = []

    for img_file, gt_file, label in IMAGE_PAIRS:
        print(f"\n--- Image: {label} ---")
        img_gray, gt_bin = load_image_pair(img_file, gt_file)

        # Gaussian pre-smoothing for Sobel and Prewitt
        smoothed = preprocess(img_gray, sigma=GAUSS_SIGMA)
        print(f"  Preprocessing: Gaussian blur sigma={GAUSS_SIGMA} applied")

        # Sobel
        sob_mag, sob_bin, sob_thresh = apply_sobel(smoothed)
        sob_metrics = compute_metrics(sob_bin, gt_bin)
        all_metrics["Sobel"][label] = sob_metrics
        print(f"  Sobel   | Otsu threshold={sob_thresh:.0f} | "
              f"P={sob_metrics['Precision']:.4f}  "
              f"R={sob_metrics['Recall']:.4f}  "
              f"F1={sob_metrics['F1']:.4f}  "
              f"IoU={sob_metrics['IoU']:.4f}")

        # Prewitt
        pre_mag, pre_bin, pre_thresh = apply_prewitt(smoothed)
        pre_metrics = compute_metrics(pre_bin, gt_bin)
        all_metrics["Prewitt"][label] = pre_metrics
        print(f"  Prewitt | Otsu threshold={pre_thresh:.0f} | "
              f"P={pre_metrics['Precision']:.4f}  "
              f"R={pre_metrics['Recall']:.4f}  "
              f"F1={pre_metrics['F1']:.4f}  "
              f"IoU={pre_metrics['IoU']:.4f}")

        # Laplacian (uses its own internal Gaussian)
        lap_mag, lap_bin, lap_thresh = apply_laplacian(img_gray)
        lap_metrics = compute_metrics(lap_bin, gt_bin)
        all_metrics["Laplacian"][label] = lap_metrics
        print(f"  Laplace | Otsu threshold={lap_thresh:.0f} | "
              f"P={lap_metrics['Precision']:.4f}  "
              f"R={lap_metrics['Recall']:.4f}  "
              f"F1={lap_metrics['F1']:.4f}  "
              f"IoU={lap_metrics['IoU']:.4f}")

        grid_images.extend([img_gray, sob_bin, pre_bin, lap_bin, gt_bin])
        grid_titles.extend([
            f"Original\n{label}",
            f"Sobel\nF1={sob_metrics['F1']:.3f}",
            f"Prewitt\nF1={pre_metrics['F1']:.3f}",
            f"Laplacian\nF1={lap_metrics['F1']:.3f}",
            "Ground Truth",
        ])

    # ---------------------------------------------------------------
    # Figure 1: Comparison grid (3 rows x 5 cols)
    # ---------------------------------------------------------------
    print("\nGenerating figures...")
    fig, axes = plt.subplots(3, 5, figsize=(17, 11))
    for idx, (im, title) in enumerate(zip(grid_images, grid_titles)):
        r, c = divmod(idx, 5)
        axes[r, c].imshow(im, cmap="gray", vmin=0, vmax=255)
        axes[r, c].set_title(title, fontsize=8)
        axes[r, c].axis("off")
    plt.suptitle(
        "Task 1 - Sobel, Prewitt and Laplacian Edge Detection on HEp-2 Cells",
        fontsize=11)
    plt.tight_layout()
    fig1_path = os.path.join(OUTPUT_DIR, "task1_comparison.png")
    plt.savefig(fig1_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Figure saved: {fig1_path}")

    # ---------------------------------------------------------------
    # Figure 2: Metrics bar chart
    # ---------------------------------------------------------------
    detector_names = list(all_metrics.keys())
    image_labels   = [lbl for _, _, lbl in IMAGE_PAIRS]
    metric_keys    = ["Precision", "Recall", "F1", "IoU"]

    fig, axes = plt.subplots(1, 4, figsize=(16, 5))
    x        = np.arange(len(image_labels))
    bar_w    = 0.25
    colors   = ["#2196F3", "#FF9800", "#4CAF50"]

    for met_idx, met_key in enumerate(metric_keys):
        ax = axes[met_idx]
        for det_idx, det_name in enumerate(detector_names):
            vals = [all_metrics[det_name][lbl][met_key]
                    for lbl in image_labels]
            ax.bar(x + det_idx * bar_w, vals, bar_w,
                   label=det_name, color=colors[det_idx], alpha=0.85)
        ax.set_title(met_key, fontsize=10)
        ax.set_xticks(x + bar_w)
        ax.set_xticklabels(image_labels, rotation=15, fontsize=7)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Score")
        ax.legend(fontsize=7)
        ax.grid(axis="y", alpha=0.3)

    plt.suptitle("Task 1 - Metrics Comparison: Sobel vs Prewitt vs Laplacian",
                 fontsize=11)
    plt.tight_layout()
    fig2_path = os.path.join(OUTPUT_DIR, "task1_metrics_chart.png")
    plt.savefig(fig2_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Figure saved: {fig2_path}")

    # ---------------------------------------------------------------
    # Figure 3: Gradient magnitude maps (first image)
    # ---------------------------------------------------------------
    img_gray_0, gt_bin_0 = load_image_pair(IMAGE_PAIRS[0][0], IMAGE_PAIRS[0][1])
    smoothed_0 = preprocess(img_gray_0, sigma=GAUSS_SIGMA)
    sob_mag_0, sob_bin_0, _ = apply_sobel(smoothed_0)
    pre_mag_0, pre_bin_0, _ = apply_prewitt(smoothed_0)
    lap_mag_0, lap_bin_0, _ = apply_laplacian(img_gray_0)

    def norm_display(m):
        return cv2.normalize(m, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    fig, axes = plt.subplots(2, 4, figsize=(14, 7))
    row0 = [img_gray_0,
            norm_display(sob_mag_0),
            norm_display(pre_mag_0),
            norm_display(lap_mag_0)]
    row1 = [gt_bin_0, sob_bin_0, pre_bin_0, lap_bin_0]
    titles0 = ["Original (10905 JL)",
               "Sobel - Gradient Magnitude",
               "Prewitt - Gradient Magnitude",
               "Laplacian - Abs Response"]
    titles1 = ["Ground Truth",
               "Sobel - Binary Edges",
               "Prewitt - Binary Edges",
               "Laplacian - Binary Edges"]
    for c, (im, t) in enumerate(zip(row0, titles0)):
        axes[0, c].imshow(im, cmap="gray", vmin=0, vmax=255)
        axes[0, c].set_title(t, fontsize=8)
        axes[0, c].axis("off")
    for c, (im, t) in enumerate(zip(row1, titles1)):
        axes[1, c].imshow(im, cmap="gray", vmin=0, vmax=255)
        axes[1, c].set_title(t, fontsize=8)
        axes[1, c].axis("off")
    plt.suptitle(
        "Task 1 - Gradient Magnitudes and Binary Edge Maps (10905 JL)",
        fontsize=10)
    plt.tight_layout()
    fig3_path = os.path.join(OUTPUT_DIR, "task1_magnitudes.png")
    plt.savefig(fig3_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Figure saved: {fig3_path}")

    # ---------------------------------------------------------------
    # Figure 4: Gradient direction (alpha = arctan2(Gy, Gx))
    # HSV colourmap used because direction is periodic (0 = 360 deg).
    # ---------------------------------------------------------------
    img_gray_d, _ = load_image_pair(IMAGE_PAIRS[0][0], IMAGE_PAIRS[0][1])
    smoothed_d    = preprocess(img_gray_d, sigma=GAUSS_SIGMA)

    Gx_sob = cv2.Sobel(smoothed_d, cv2.CV_32F, 1, 0, ksize=3)
    Gy_sob = cv2.Sobel(smoothed_d, cv2.CV_32F, 0, 1, ksize=3)

    k_Gx = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float32)
    k_Gy = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=np.float32)
    img_f = smoothed_d.astype(np.float32)
    Gx_pre = cv2.filter2D(img_f, -1, k_Gx)
    Gy_pre = cv2.filter2D(img_f, -1, k_Gy)

    def direction_to_display(Gx, Gy):
        """Map gradient direction [-pi, pi] to 0-255 for display."""
        alpha = np.arctan2(Gy, Gx)
        alpha_norm = (alpha + np.pi) / (2 * np.pi)
        return (alpha_norm * 255).astype(np.uint8)

    dir_sob = direction_to_display(Gx_sob, Gy_sob)
    dir_pre = direction_to_display(Gx_pre, Gy_pre)

    def signed_to_display(g):
        return cv2.normalize(g, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    fig, axes = plt.subplots(2, 4, figsize=(15, 7))

    row0 = [img_gray_d,
            signed_to_display(Gx_sob),
            signed_to_display(Gy_sob),
            dir_sob]
    titles0 = ["Original (10905 JL)",
               "Sobel Gx (horizontal gradient)",
               "Sobel Gy (vertical gradient)",
               "Sobel Direction alpha = arctan2(Gy,Gx)"]

    row1 = [norm_display(sob_mag_0),
            signed_to_display(Gx_pre),
            signed_to_display(Gy_pre),
            dir_pre]
    titles1 = ["Sobel Magnitude M = sqrt(Gx^2+Gy^2)",
               "Prewitt Gx (horizontal gradient)",
               "Prewitt Gy (vertical gradient)",
               "Prewitt Direction alpha = arctan2(Gy,Gx)"]

    for c, (im, t) in enumerate(zip(row0, titles0)):
        cmap = "hsv" if c == 3 else "gray"
        axes[0, c].imshow(im, cmap=cmap, vmin=0, vmax=255)
        axes[0, c].set_title(t, fontsize=7.5)
        axes[0, c].axis("off")

    for c, (im, t) in enumerate(zip(row1, titles1)):
        cmap = "hsv" if c == 3 else "gray"
        axes[1, c].imshow(im, cmap=cmap, vmin=0, vmax=255)
        axes[1, c].set_title(t, fontsize=7.5)
        axes[1, c].axis("off")

    plt.suptitle(
        "Task 1 - Gradient Components (Gx, Gy), Magnitude and Direction (10905 JL)",
        fontsize=10)
    plt.tight_layout()
    fig4_path = os.path.join(OUTPUT_DIR, "task1_gradient_direction.png")
    plt.savefig(fig4_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Figure saved: {fig4_path}")

    # Summary
    print_metrics_table(all_metrics)

    print("\nBest detector per image (highest F1):")
    for lbl in [lbl for _, _, lbl in IMAGE_PAIRS]:
        best_det = max(all_metrics,
                       key=lambda d: all_metrics[d][lbl]["F1"])
        best_f1  = all_metrics[best_det][lbl]["F1"]
        print(f"  {lbl}: {best_det}  (F1={best_f1:.4f})")

    print("\nTask 1 complete. Outputs saved to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
