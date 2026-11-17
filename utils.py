"""
utils.py - Shared utility functions for CVI Assignment.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


def compute_metrics(detected, ground_truth):
    """Compute Precision, Recall, F1, and IoU between a detected edge map and ground truth."""
    det = (detected > 0).astype(np.uint8)
    gt  = (ground_truth > 0).astype(np.uint8)

    TP = int(np.sum((det == 1) & (gt == 1)))
    FP = int(np.sum((det == 1) & (gt == 0)))
    FN = int(np.sum((det == 0) & (gt == 1)))

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall    = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)
    iou       = TP / (TP + FP + FN) if (TP + FP + FN) > 0 else 0.0

    return {
        "Precision": round(precision, 4),
        "Recall":    round(recall,    4),
        "F1":        round(f1,        4),
        "IoU":       round(iou,       4),
    }


def display_comparison(images, titles, rows, cols, save_path, dpi=150, figsize=None):
    """Save a grid of images with titles as a PNG."""
    if figsize is None:
        figsize = (cols * 3.5, rows * 3.2)

    fig, axes = plt.subplots(rows, cols, figsize=figsize)

    # Flatten axes for uniform indexing
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes[np.newaxis, :]
    elif cols == 1:
        axes = axes[:, np.newaxis]

    for idx, (img, title) in enumerate(zip(images, titles)):
        r, c = divmod(idx, cols)
        ax = axes[r, c]

        if img is None:
            ax.axis("off")
            ax.set_title(title, fontsize=9)
            continue

        # BGR to RGB for colour images
        if len(img.shape) == 3 and img.shape[2] == 3:
            ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        else:
            ax.imshow(img, cmap="gray", vmin=0, vmax=255)

        ax.set_title(title, fontsize=9)
        ax.axis("off")

    # Hide unused subplots
    for idx in range(len(images), rows * cols):
        r, c = divmod(idx, cols)
        axes[r, c].axis("off")

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"  Figure saved: {save_path}")


def print_metrics_table(results_dict):
    """Print a formatted metrics table for all detectors and images."""
    image_names = []
    for det_results in results_dict.values():
        for img_name in det_results:
            if img_name not in image_names:
                image_names.append(img_name)

    metrics = ["Precision", "Recall", "F1", "IoU"]
    col_w   = 10

    header_img  = "Image".ljust(20)
    header_det  = "Detector".ljust(14)
    header_met  = "".join(m.ljust(col_w) for m in metrics)
    separator   = "-" * (20 + 14 + col_w * len(metrics))

    print("\n" + separator)
    print(f"{header_img}{header_det}{header_met}")
    print(separator)

    for img_name in image_names:
        for detector, det_results in results_dict.items():
            if img_name in det_results:
                m = det_results[img_name]
                row = (
                    img_name[:19].ljust(20)
                    + detector[:13].ljust(14)
                    + "".join(f"{m[k]:.4f}".ljust(col_w) for k in metrics)
                )
                print(row)
        print(separator)
