"""
task2_canny.py - Advanced Edge Detection using Canny (Task 2)

Applies the Canny edge detector to the 3 HEp-2 cell images.
Evaluates against ground truth and compares with the best Task 1 method.
"""

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

from utils import compute_metrics, print_metrics_table

IMAGE_PAIRS = [
    ("10905 JL.bmp",  "10905 JL Edges.bmp",  "10905_JL"),
    ("43590 AM.bmp",  "43590 AM Edges.bmp",  "43590_AM"),
    ("9343 AM.bmp",   "9343 AM Edges.bmp",   "9343_AM"),
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


def apply_canny(img_gray, t_low, t_high, sigma=1.0):
    """
    Apply the Canny edge detector (4 steps):
      1. Gaussian filtering (sigma, ksize = 6*sigma rounded to odd).
      2. Sobel gradients - magnitude and direction.
      3. Non-maximum suppression - thins edges to single-pixel width.
      4. Hysteresis thresholding - t_high accepts strong edges, t_low sets the
         minimum for edges connected to a strong edge.
    cv2.Canny handles steps 2-4 internally.
    """
    ksize = int(6 * sigma) | 1
    if ksize < 3:
        ksize = 3
    smoothed = cv2.GaussianBlur(img_gray, (ksize, ksize), sigma)
    edges = cv2.Canny(smoothed, t_low, t_high)
    return edges


def main():
    print("=" * 65)
    print("TASK 2 - CANNY EDGE DETECTION")
    print("=" * 65)

    # Low thresholds needed: HEp-2 images are very dark (mean pixel 7-19),
    # so gradient magnitudes are small. t_high = 2 * t_low throughout.
    THRESHOLD_PAIRS = [
        (5,   10,  "t_low=5"),
        (10,  20,  "t_low=10"),
        (20,  40,  "t_low=20"),
        (30,  60,  "t_low=30"),
    ]

    GAUSS_SIGMA = 1.0

    all_metrics_by_thresh = {tp[2]: {} for tp in THRESHOLD_PAIRS}
    best_params_per_image = {}

    # Run Canny with all threshold pairs on all images
    for img_file, gt_file, label in IMAGE_PAIRS:
        print(f"\n--- Image: {label} ---")
        img_gray, gt_bin = load_image_pair(img_file, gt_file)

        best_f1   = -1
        best_data = None

        for t_low, t_high, thresh_label in THRESHOLD_PAIRS:
            edges   = apply_canny(img_gray, t_low, t_high, sigma=GAUSS_SIGMA)
            metrics = compute_metrics(edges, gt_bin)
            all_metrics_by_thresh[thresh_label][label] = metrics

            print(f"  [{thresh_label:12s}]  t_high={t_high:3d}  | "
                  f"P={metrics['Precision']:.4f}  "
                  f"R={metrics['Recall']:.4f}  "
                  f"F1={metrics['F1']:.4f}  "
                  f"IoU={metrics['IoU']:.4f}")

            if metrics["F1"] > best_f1:
                best_f1   = metrics["F1"]
                best_data = (t_low, t_high, thresh_label, edges.copy(), metrics)

        best_params_per_image[label] = best_data
        print(f"  Best: {best_data[2]}  (F1={best_f1:.4f})")

    # ---------------------------------------------------------------
    # Figure 1: Threshold tuning on representative image (43590 AM)
    # ---------------------------------------------------------------
    print("\nGenerating figures...")
    rep_label  = "43590_AM"
    rep_img_f, rep_gt_f, _ = IMAGE_PAIRS[1]
    rep_img, rep_gt = load_image_pair(rep_img_f, rep_gt_f)

    tune_images = [rep_img]
    tune_titles = [f"Original\n{rep_label}"]
    for t_low, t_high, thresh_label in THRESHOLD_PAIRS:
        edges   = apply_canny(rep_img, t_low, t_high, sigma=GAUSS_SIGMA)
        metrics = compute_metrics(edges, rep_gt)
        tune_images.append(edges)
        tune_titles.append(f"Canny {thresh_label}\n(F1={metrics['F1']:.3f})")
    tune_images.append(rep_gt)
    tune_titles.append("Ground Truth")

    fig, axes = plt.subplots(1, len(tune_images), figsize=(3.2 * len(tune_images), 4))
    for ax, im, title in zip(axes, tune_images, tune_titles):
        ax.imshow(im, cmap="gray", vmin=0, vmax=255)
        ax.set_title(title, fontsize=8)
        ax.axis("off")
    plt.suptitle(
        "Task 2 - Canny Threshold Sensitivity (43590 AM)", fontsize=10)
    plt.tight_layout()
    fig1_path = os.path.join(OUTPUT_DIR, "task2_canny_tuning.png")
    plt.savefig(fig1_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Figure saved: {fig1_path}")

    # ---------------------------------------------------------------
    # Figure 2: Best Canny vs best Task 1 (Sobel) vs ground truth
    # ---------------------------------------------------------------
    from task1_edge_detection import apply_sobel, preprocess

    grid_images = []
    grid_titles = []

    for img_file, gt_file, label in IMAGE_PAIRS:
        img_gray, gt_bin = load_image_pair(img_file, gt_file)
        smoothed = preprocess(img_gray, sigma=1.0)

        _, sob_bin, _ = apply_sobel(smoothed)
        sob_metrics   = compute_metrics(sob_bin, gt_bin)

        t_low, t_high, t_lbl, best_edges, best_metrics = best_params_per_image[label]

        grid_images.extend([img_gray, sob_bin, best_edges, gt_bin])
        grid_titles.extend([
            f"Original\n{label}",
            f"Best Task 1 (Sobel)\nF1={sob_metrics['F1']:.3f}",
            f"Canny {t_lbl}\nF1={best_metrics['F1']:.3f}",
            "Ground Truth",
        ])

    fig, axes = plt.subplots(3, 4, figsize=(14, 11))
    for idx, (im, title) in enumerate(zip(grid_images, grid_titles)):
        r, c = divmod(idx, 4)
        axes[r, c].imshow(im, cmap="gray", vmin=0, vmax=255)
        axes[r, c].set_title(title, fontsize=8)
        axes[r, c].axis("off")
    plt.suptitle(
        "Task 2 - Canny vs Best Task 1 (Sobel) vs Ground Truth",
        fontsize=11)
    plt.tight_layout()
    fig2_path = os.path.join(OUTPUT_DIR, "task2_canny_comparison.png")
    plt.savefig(fig2_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Figure saved: {fig2_path}")

    # ---------------------------------------------------------------
    # Figure 3: Metrics bar chart across threshold pairs
    # ---------------------------------------------------------------
    rep_label2 = "10905_JL"
    metric_keys = ["Precision", "Recall", "F1", "IoU"]
    fig, axes = plt.subplots(1, 4, figsize=(14, 4))
    x     = np.arange(len(THRESHOLD_PAIRS))
    cmap  = ["#2196F3", "#FF9800", "#4CAF50", "#E91E63"]

    for met_idx, met_key in enumerate(metric_keys):
        ax = axes[met_idx]
        vals = [all_metrics_by_thresh[tp[2]][rep_label2][met_key]
                for tp in THRESHOLD_PAIRS]
        bars = ax.bar(x, vals, color=cmap, alpha=0.85, width=0.6)
        ax.set_title(f"{met_key}\n({rep_label2})", fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels([tp[2] for tp in THRESHOLD_PAIRS],
                           rotation=20, fontsize=7)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Score")
        ax.grid(axis="y", alpha=0.3)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, val + 0.02,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=7)

    plt.suptitle(
        "Task 2 - Effect of Canny Threshold on Metrics (10905 JL)",
        fontsize=10)
    plt.tight_layout()
    fig3_path = os.path.join(OUTPUT_DIR, "task2_threshold_metrics.png")
    plt.savefig(fig3_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Figure saved: {fig3_path}")

    # Summary tables
    for thresh_label in [tp[2] for tp in THRESHOLD_PAIRS]:
        print(f"\nMetrics for {thresh_label}:")
        print_metrics_table({"Canny-" + thresh_label: all_metrics_by_thresh[thresh_label]})

    print("\nBest Canny parameters per image (highest F1):")
    for label in [lbl for _, _, lbl in IMAGE_PAIRS]:
        t_low, t_high, t_lbl, _, m = best_params_per_image[label]
        print(f"  {label}: t_low={t_low}, t_high={t_high}  "
              f"(F1={m['F1']:.4f}  P={m['Precision']:.4f}  "
              f"R={m['Recall']:.4f}  IoU={m['IoU']:.4f})")

    print("\nTask 2 complete. Outputs saved to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
