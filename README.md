# Building Outline Extraction from Satellite Imagery

Classical computer-vision pipeline that detects building outlines from satellite
imagery using only NumPy

**Pipeline:** NDBI Suppression → Canny Edge Detection → Morphological Cleanup → Hough Transform

The repository ships with a Flask web UI on top of the pipeline so you can
upload images, tweak hyperparameters, and watch the run progress live in the
browser.

---

## Repository layout

```
.
├── app.py                     # Flask web UI (entry point)
├── building_extraction.py     # Core pipeline — Canny / morph / Hough from scratch
├── sentinel_pipeline.py       # Sentinel-2 wrapper (true NDBI from B11/B08)
├── sat_image/                 # Sample Sentinel-2 scene (5 TIFFs); used by 'sample data' mode
│   ├── *_B02_(Raw).tiff       # Blue
│   ├── *_B03_(Raw).tiff       # Green
│   ├── *_B04_(Raw).tiff       # Red
│   ├── *_B08_(Raw).tiff       # NIR
│   └── *_B11_(Raw).tiff       # SWIR
├── samples/                   # Test images for the upload mode (synthetic + scene PNGs)
├── outputs/                   # Pipeline writes results here (auto-created)
│   └── stages/                # Per-stage diagnostic PNGs
├── uploads/                   # Per-job upload temp dirs (auto-created, auto-cleaned)
├── requirements.txt
├── setup.sh                   # Linux / macOS environment bootstrap
├── setup.bat                  # Windows environment bootstrap
└── README.md                  # (this file)
```

---

## Quick start

### Linux / macOS

```bash
bash setup.sh
source venv/bin/activate
python app.py
```

### Windows

```bat
setup.bat
venv\Scripts\activate
python app.py
```

Then open **http://localhost:5000** in your browser.

The setup script creates a virtual environment in `./venv`, upgrades pip, and
installs everything in `requirements.txt`. You only need to run it once.

---

## Dependencies

Installed automatically by `setup.sh` / `setup.bat`. If you'd rather install
them manually:

```bash
pip install -r requirements.txt
```

| Package        | Purpose                                                  |
|----------------|----------------------------------------------------------|
| `numpy`        | All CV math (Canny, Sobel, morphology, Hough)            |
| `opencv-python`| Image file I/O only (`cv2.imread`, `cv2.imwrite`)        |
| `matplotlib`   | 8-panel diagnostic figure generation                     |
| `Pillow`       | Fallback image I/O when OpenCV isn't available           |
| `tifffile`     | Reading multi-band Sentinel-2 GeoTIFFs                   |
| `Flask`        | Web UI (`app.py`)                                        |
| `Werkzeug`     | Secure file upload handling for the web UI               |

Python 3.8+ required.

---

## Usage

### 1. Web UI (recommended)

```bash
python app.py
```

Open `http://localhost:5000`. Two modes:

- **Upload Image from Device** — drag/drop any PNG / JPG / BMP / WebP / TIFF (up to 256 MB).
  Falls back to the RGB-proxy NDBI (`grayness × brightness`).
- **Sentinel-2 SAT Data** — runs the multi-band pipeline with true NDBI from
  `B11 (SWIR) / B08 (NIR)`. Two sub-modes:
    - **Use sample data** (default) — runs on whatever's already in `./sat_image/`.
      The repository ships with a sample 5-band scene there.
    - **Upload my own bands** — drop the 5 TIFFs (`B02`, `B03`, `B04`, `B08`, `B11`)
      directly, or a single `.zip` containing them. Required bands are validated
      both client-side (visual warning) and server-side (HTTP 400 with a list of
      what's missing) before the pipeline starts.

Toggle **Custom Hyperparameters** to override the 5 tunable knobs:

| Parameter (UI label)          | CLI flag             | Default | Pipeline stage                     |
|-------------------------------|----------------------|---------|------------------------------------|
| NDBI Threshold                | `--ndbi-threshold`   | `0.1`   | Stage 1 — NDBI suppression         |
| Gaussian Kernel Size          | `--gauss-ksize`      | `5`     | Stage 2 — Canny / Gaussian blur    |
| Gaussian Sigma (σ)            | `--gauss-sigma`      | `1.4`   | Stage 2 — Canny / Gaussian blur    |
| Structuring Element Type      | `--morph-shape`      | `square`| Stage 3 — Morphological cleanup    |
| Structuring Element Size      | `--morph-size`       | `3`     | Stage 3 — Morphological cleanup    |

Both upload mode and Sentinel-2 mode honour these overrides; with the toggle
off, each pipeline uses its own tuned defaults.

### 2. Command line

```bash
# Single RGB image (synthetic / Google Earth / aerial photo)
python building_extraction.py path/to/image.png

# Synthetic demo scene
python building_extraction.py --syn

# Sentinel-2 multi-band stack — true NDBI from B11/B08
python sentinel_pipeline.py sat_image
```

All hyperparameter flags from the table above work on the command line too:

```bash
python sentinel_pipeline.py sat_image \
    --ndbi-threshold 0.05 \
    --gauss-sigma 1.0 \
    --morph-shape cross \
    --morph-size 5
```

---

## Outputs

After a successful run, `outputs/` contains:

- `final_outlines.png` — input image with detected Hough segments overlaid
- `building_extraction_pipeline.png` — 8-panel diagnostic (one per stage)
- `sentinel2_rgb.png` — RGB composite (Sentinel-2 mode only)
- `stages/` — individual PNGs for each pipeline stage:
  - `1_original.png`
  - `2_ndbi.png`
  - `3_builtup_mask.png`
  - `4_suppressed_gray.png`
  - `5_canny_edges.png`
  - `6_after_morphology.png`
  - `7_hough_accumulator.png`
  - `8_detected_outlines.png`

