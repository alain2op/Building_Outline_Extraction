"""
sentinel_pipeline.py
====================

Usage:
  python sentinel_pipeline.py                         # defaults to ./sat_image
  python sentinel_pipeline.py path/to/folder
  python sentinel_pipeline.py path/to/folder \
         --gauss-ksize 5 --gauss-sigma 1.4 \
         --morph-shape square --morph-size 3 \
         --ndbi-threshold 0.20

These hyperparameter flags mirror building_extraction.py so that the
Flask UI ('Custom Hyperparameters' toggle) can drive both pipelines
through the same `build_script_args` helper in app.py.
"""

import os
import sys
import glob
import argparse
import numpy as np

try:
    import tifffile
except ImportError:
    print("ERROR: 'tifffile' not installed.  Run:  pip install tifffile")
    sys.exit(1)

import cv2

# Reuse the existing pipeline
from building_extraction import extract_buildings


# ----------------------------------------------------------------------
# File discovery + band loading
# ----------------------------------------------------------------------

def find_band(src_dir, band_name):
    """Return the first TIFF in src_dir whose filename contains `band_name`."""
    # Copernicus naming uses parentheses like 'B02_(Raw).tiff', glob loosely.
    for pattern in (f'*_{band_name}_*.tif*',
                    f'*{band_name}_*.tif*',
                    f'*{band_name}*.tif*'):
        matches = sorted(glob.glob(os.path.join(src_dir, pattern)))
        if matches:
            return matches[0]
    return None


def load_band(path):
    """Read a TIFF as a float32 numpy array."""
    return tifffile.imread(path).astype(np.float32)


def percentile_stretch(arr, lo_pct=2, hi_pct=98):
    """Stretch to [0, 255] using robust percentiles. Returns uint8."""
    if (arr == 0).all():
        return np.zeros_like(arr, dtype=np.uint8)
    pos = arr[arr > 0]
    if pos.size == 0:
        return np.zeros_like(arr, dtype=np.uint8)
    lo = np.percentile(pos, lo_pct)
    hi = np.percentile(pos, hi_pct)
    out = np.clip((arr - lo) / (hi - lo + 1e-9), 0, 1)
    return (out * 255).astype(np.uint8)


def check_data(arr, name):
    """Print stats and return True if the band has non-trivial content."""
    nonzero = int((arr != 0).sum())
    pct = 100.0 * nonzero / arr.size
    print(f"   {name:14s} shape={arr.shape}  "
          f"range=[{arr.min():.4f}, {arr.max():.4f}]  nonzero={pct:5.1f}%")
    return pct > 1.0


def build_rgb(src_dir):
    """Prefer an existing True_color TIFF; otherwise stack B04/B03/B02."""
    tc_matches = sorted(glob.glob(os.path.join(src_dir, '*True_color*.tif*')))
    if tc_matches:
        print(f"   using true-colour composite: "
              f"{os.path.basename(tc_matches[0])}")
        tc = load_band(tc_matches[0])
        if tc.ndim == 2:
            tc = np.stack([tc, tc, tc], axis=-1)
        tc = tc[..., :3]
        rgb = np.stack([percentile_stretch(tc[..., i]) for i in range(3)],
                       axis=-1)
        return rgb

    print("   no true-colour TIFF found - building RGB from B04/B03/B02")
    paths = {b: find_band(src_dir, b) for b in ('B04', 'B03', 'B02')}
    for name, p in paths.items():
        if p is None:
            raise FileNotFoundError(f"RGB band {name} not found in {src_dir}")
    r, g, b = (load_band(paths['B04']),
               load_band(paths['B03']),
               load_band(paths['B02']))
    rgb = np.stack([percentile_stretch(r),
                    percentile_stretch(g),
                    percentile_stretch(b)], axis=-1)
    return rgb


def match_resolution(arr, target_shape):
    """Bilinear resample `arr` to target_shape (H, W)."""
    if arr.shape[:2] == target_shape:
        return arr
    return cv2.resize(arr, (target_shape[1], target_shape[0]),
                      interpolation=cv2.INTER_LINEAR)


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main(src_dir,
         out_dir='outputs',
         gauss_ksize=5,
         gauss_sigma=1.0,
         morph_shape='square',
         morph_size=3,
         ndbi_threshold=0.20):
    """
    Sentinel-2 entry point.

    Note: defaults here intentionally differ from building_extraction.py
    because Sentinel-2 is 10 m/px real imagery (jagged edges, dense urban
    structure), so the original tuned values are kept as DEFAULTS.
    Anything the caller passes in (e.g. via the Flask UI) overrides them.
    """
    os.makedirs(out_dir, exist_ok=True)
    print(f"Source directory: {src_dir}")
    print(f"Hyperparams: ndbi={ndbi_threshold}  "
          f"gauss_ksize={gauss_ksize}  gauss_sigma={gauss_sigma}  "
          f"morph={morph_shape}/{morph_size}")

    # ---- 1. Build RGB ----
    print("\n[1/4] Building RGB composite ...")
    rgb = build_rgb(src_dir)
    rgb_ok = check_data(rgb, 'RGB')
    H, W = rgb.shape[:2]

    # ---- 2. NIR + SWIR for NDBI ----
    print("\n[2/4] Loading NIR (B08) and SWIR (B11) ...")
    b08 = find_band(src_dir, 'B08')
    b11 = find_band(src_dir, 'B11')
    if b08 is None or b11 is None:
        print(f"   ERROR: required bands missing in {src_dir}")
        print(f"     B08 (NIR) : {b08}")
        print(f"     B11 (SWIR): {b11}")
        sys.exit(1)
    nir  = load_band(b08)
    swir = load_band(b11)
    nir_ok  = check_data(nir,  'NIR (B08)')
    swir_ok = check_data(swir, 'SWIR (B11)')

    print("\n[3/4] Aligning resolutions to RGB grid ...")
    nir  = match_resolution(nir,  (H, W))
    swir = match_resolution(swir, (H, W))
    print(f"   final grid: {(H, W)}  (H x W)")

    # ---- 4. Sanity check ----
    if not (rgb_ok and nir_ok and swir_ok):
        print("\n" + "=" * 64)
        print("!! DATA IS EMPTY - download has zero pixels.")
        print("   Re-download from Copernicus Browser making sure to:")
        print("     - DRAW an Area of Interest on the map first")
        print("     - Set Resolution to HIGH (or CUSTOM 10 m/px)")
        print("     - Use Analytical -> TIFF (32-bit float)")
        print("=" * 64)
        sys.exit(1)

    rgb_path = os.path.join(out_dir, 'sentinel2_rgb.png')
    cv2.imwrite(rgb_path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    print(f"   saved RGB preview: {rgb_path}")

    print("\n[4/4] Running building-extraction pipeline (true NDBI) ...\n")
    extract_buildings(
        rgb_path,
        out_dir=out_dir,
        swir=swir,
        nir=nir,

        # ---- Hyperparameters: forwarded from CLI / Flask UI ----
        ndbi_threshold=ndbi_threshold,
        gauss_ksize=gauss_ksize,
        canny_sigma=gauss_sigma,
        morph_shape=morph_shape,
        morph_size=morph_size,

        # ---- Sentinel-2-specific tuning kept fixed (real-image edges) ----
        canny_low=0.06,
        canny_high=0.15,
        min_segment_length=2,
        merge_angle_tol=8,
        merge_perp_tol=4,
        merge_gap_tol=12,
        num_hough_peaks=100,

        resize_max=2000,
    )
    print("\nDone. Check 'outputs/' for pipeline visualisation and outlines.")


def _parse_args():
    parser = argparse.ArgumentParser(
        description='Sentinel-2 building-outline extraction pipeline')
    parser.add_argument('src_dir', nargs='?', default='sat_image',
                        help='Folder containing Sentinel-2 TIFFs (default: sat_image)')
    parser.add_argument('--gauss-ksize',    type=int,   default=5,
                        help='Gaussian kernel size (odd, default: 5)')
    parser.add_argument('--gauss-sigma',    type=float, default=1.0,
                        help='Gaussian sigma / canny_sigma (default: 1.0)')
    parser.add_argument('--morph-shape',    type=str,   default='square',
                        choices=['square', 'cross'],
                        help='Morphological structuring element shape (default: square)')
    parser.add_argument('--morph-size',     type=int,   default=3,
                        help='Morphological structuring element size (odd, default: 3)')
    parser.add_argument('--ndbi-threshold', type=float, default=0.20,
                        help='NDBI threshold for built-up mask (default: 0.20)')
    return parser.parse_args()


if __name__ == '__main__':
    args = _parse_args()

    if not os.path.isdir(args.src_dir):
        print(f"Error: directory not found: {args.src_dir}")
        print(f"Usage: python {sys.argv[0]} [path/to/sat_image]")
        sys.exit(1)

    # Force kernel + structuring-element sizes to be odd, mirroring
    # building_extraction.py's __main__ guard.
    gauss_ksize = args.gauss_ksize if args.gauss_ksize % 2 == 1 else args.gauss_ksize + 1
    morph_size  = args.morph_size  if args.morph_size  % 2 == 1 else args.morph_size  + 1

    main(args.src_dir,
         gauss_ksize=gauss_ksize,
         gauss_sigma=args.gauss_sigma,
         morph_shape=args.morph_shape,
         morph_size=morph_size,
         ndbi_threshold=args.ndbi_threshold)
