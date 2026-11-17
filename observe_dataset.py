"""
observe_dataset.py - Initial dataset observation (run this first)

Loads all 3 HEp-2 cell images and their ground truth edge maps,
prints key properties, and saves a side-by-side display figure.
Observations here directly inform preprocessing choices in Tasks 1-3.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# ---------------------------------------------------------------
# File paths (relative to this script's working directory)
# ---------------------------------------------------------------
IMAGE_PAIRS = [
    ("10905 JL.bmp",   "10905 JL Edges.bmp"),
    ("43590 AM.bmp",   "43590 AM Edges.bmp"),
    ("9343 AM.bmp",    "9343 AM Edges.bmp"),
]
HOUGH_IMAGE = "UoBDImage.jpg"
OUTPUT_DIR  = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def analyse_image(path, label):
    """Load image and print key properties."""
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        print(f"  ERROR: Could not load {path}")
        return None

    h, w = img.shape[:2]
    channels = 1 if len(img.shape) == 2 else img.shape[2]
    dtype    = img.dtype
    pmin     = int(img.min())
    pmax     = int(img.max())

    print(f"\n  [{label}]  {path}")
    print(f"    Dimensions  : {w} x {h} pixels")
    print(f"    Channels    : {channels}")
    print(f"    Data type   : {dtype}")
    print(f"    Pixel range : {pmin} - {pmax}")

    # Check if a 3-channel image is effectively grayscale
    if channels == 3:
        max_ch_diff = max(
            int(np.max(np.abs(img[:, :, 0].astype(np.int32)
                              - img[:, :, 1].astype(np.int32)))),
            int(np.max(np.abs(img[:, :, 1].astype(np.int32)
                              - img[:, :, 2].astype(np.int32)))),
        )
        is_gray = max_ch_diff == 0
        print(f"    Effectively grayscale: {is_gray}  "
              f"(max inter-channel diff = {max_ch_diff})")
    else:
        print(f"    Effectively grayscale: True  (single-channel)")

    # Noise estimate: standard deviation of a flat region can indicate noise,
    # but a simple global std gives a rough sense of contrast/noise level.
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if channels == 3 else img
    print(f"    Mean intensity  : {gray.mean():.1f}")
    print(f"    Std deviation   : {gray.std():.1f}")

    # Edge density in ground truth gives context for how much of the image
    # consists of actual edges.
    return img


def main():
    print("=" * 60)
    print("DATASET OBSERVATION")
    print("=" * 60)

    fig_images  = []
    fig_titles  = []

    for i, (img_file, gt_file) in enumerate(IMAGE_PAIRS):
        print(f"\nImage set {i + 1}:")
        img = analyse_image(img_file, "Original")
        gt  = analyse_image(gt_file,  "Ground Truth Edges")

        if img is not None and gt is not None:
            # Convert to grayscale for display consistency
            img_gray = (cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                        if len(img.shape) == 3 else img)
            gt_gray  = (cv2.cvtColor(gt, cv2.COLOR_BGR2GRAY)
                        if len(gt.shape) == 3 else gt)

            # Edge density: fraction of pixels that are edges in ground truth
            gt_bin = (gt_gray > 127).astype(np.uint8)
            edge_density = gt_bin.mean() * 100
            print(f"    GT edge density : {edge_density:.2f}% of pixels")

            fig_images.extend([img_gray, gt_gray])
            label = img_file.replace(".bmp", "")
            fig_titles.extend([f"Original: {label}",
                                f"GT Edges: {label}"])

    # Also inspect the Hough image
    print(f"\nHough Transform image:")
    hough_img = analyse_image(HOUGH_IMAGE, "UoBDImage")

    # ---------------------------------------------------------------
    # Save side-by-side observation figure (3 rows x 2 cols)
    # ---------------------------------------------------------------
    if len(fig_images) == 6:
        fig, axes = plt.subplots(3, 2, figsize=(8, 11))
        for idx, (im, title) in enumerate(zip(fig_images, fig_titles)):
            r, c = divmod(idx, 2)
            axes[r, c].imshow(im, cmap="gray", vmin=0, vmax=255)
            axes[r, c].set_title(title, fontsize=9)
            axes[r, c].axis("off")
        plt.suptitle("Dataset Observation - HEp-2 Cell Images and Ground Truth",
                     fontsize=11, y=1.01)
        plt.tight_layout()
        out_path = os.path.join(OUTPUT_DIR, "obs_hep2_pairs.png")
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"\n  Figure saved: {out_path}")

    # Hough image display
    if hough_img is not None:
        hough_gray = (cv2.cvtColor(hough_img, cv2.COLOR_BGR2GRAY)
                      if len(hough_img.shape) == 3 else hough_img)
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        ax.imshow(cv2.cvtColor(hough_img, cv2.COLOR_BGR2RGB)
                  if len(hough_img.shape) == 3 else hough_img,
                  cmap="gray" if len(hough_img.shape) == 2 else None)
        ax.set_title("UoBDImage - Hough Transform Input", fontsize=9)
        ax.axis("off")
        plt.tight_layout()
        out_path = os.path.join(OUTPUT_DIR, "obs_hough_image.png")
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Figure saved: {out_path}")

    # ---------------------------------------------------------------
    # Summary of observations informing preprocessing decisions
    # ---------------------------------------------------------------
    print("\n" + "=" * 60)
    print("OBSERVATIONS SUMMARY (informs preprocessing in Tasks 1-3)")
    print("=" * 60)
    print("""
  1. Image type: HEp-2 cell images stored as BMP.
     - Check above output: if channels = 3 but effectively grayscale,
       cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) is safe to use.
     - If already single-channel, load directly as grayscale.

  2. Noise level: examine std deviation and visual appearance.
     - If std is moderate and cells are relatively smooth, light
       Gaussian smoothing (sigma=1.0) may be applied to suppress
       sensor noise before gradient computation.
     - Instructor note: only preprocess if genuinely needed.

  3. Ground truth format: check min/max values.
     - If GT pixels are 0 and 255, standard binary thresholding applies.
     - If inverted (edges = 0, background = 255), invert before metrics.

  4. Edge density from ground truth guides expected recall range.

  5. UoBDImage: colour JPEG of a building - rich in straight lines,
     suitable for Hough Transform line detection.
""")


if __name__ == "__main__":
    main()
