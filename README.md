# Edge Detection & Hough Transform (CVI Assignment)

Classical computer-vision pipeline built with OpenCV and NumPy. It covers three tasks:

1. **Basic edge detection** – Sobel, Prewitt and Laplacian operators on HEp-2 cell images.
2. **Canny edge detection** – hysteresis threshold tuning, compared against the Task 1 detectors.
3. **Hough transform** – straight-line detection on a building photograph (`UoBDImage.jpg`).

Detected edges in Tasks 1–2 are scored against hand-annotated ground-truth edge maps using
Precision, Recall, F1 and IoU.

## Project structure

| File | Purpose |
|------|---------|
| `observe_dataset.py` | Inspects the images (size, dtype, intensity range) and saves overview figures. Run first. |
| `task1_edge_detection.py` | Sobel / Prewitt / Laplacian with Otsu thresholding, metrics, magnitude and gradient-direction plots. |
| `task2_canny.py` | Canny with a sweep over low/high thresholds and a comparison with Task 1. |
| `task3_hough.py` | Standard and probabilistic Hough transform, accumulator visualisation, threshold and Canny sensitivity. |
| `utils.py` | Shared helpers: `compute_metrics`, `print_metrics_table`, figure grid display. |
| `generate_report.py` | Assembles the figures in `output/` into report figures. |
| `output/` | Generated figures (PNG). |

The `*.bmp` files are the HEp-2 cell images (`<id>.bmp`) with their ground-truth edge maps (`<id> Edges.bmp`).

## Setup

```bash
python -m venv venv
venv\Scripts\activate        # Windows  (source venv/bin/activate on Linux/macOS)
pip install -r requirements.txt
```

## Usage

Run from the repository root (file paths are relative):

```bash
python observe_dataset.py
python task1_edge_detection.py
python task2_canny.py
python task3_hough.py
```

Figures are written to `output/`.

## Results

### Task 1 – Sobel / Prewitt / Laplacian (F1 score)

| Image | Sobel | Prewitt | Laplacian |
|-------|-------|---------|-----------|
| 10905_JL | **0.526** | 0.521 | 0.216 |
| 43590_AM | 0.527 | **0.530** | 0.202 |
| 9343_AM | **0.535** | 0.532 | 0.212 |

Sobel and Prewitt behave almost identically (high recall, low precision since they produce thick edges);
the Laplacian is much weaker because it is highly noise-sensitive.

### Task 2 – Canny

Best per-image thresholds (F1): `10905_JL` 0.470 (30/60), `43590_AM` 0.345 (10/20), `9343_AM` 0.383 (20/40).
Canny gives much higher precision (thin, well-localised edges) but lower recall than Sobel/Prewitt, so its F1 is
lower on this dataset, where the ground truth contains faint cell boundaries. Results are very sensitive to the thresholds.

### Task 3 – Hough transform

Canny (50/150, σ = 1.5) feeds a 1° / 1 px Hough accumulator on the 1200×675 image:

| Method | Lines |
|--------|-------|
| Standard, threshold 120 | 4091 |
| Standard, threshold 150 | 1782 |
| Standard, threshold 200 | 238 |
| Probabilistic (threshold 80, minLen 40, maxGap 10) | 869 segments |

## Tech stack

Python 3, OpenCV, NumPy, Matplotlib.
