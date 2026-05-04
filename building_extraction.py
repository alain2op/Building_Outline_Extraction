"""
Building Outline Extraction from High-Resolution Satellite Imagery
===================================================================
Pipeline:
  1. NDBI-based suppression of non-built-up areas
  2. Canny Edge Detection
  3. Binary Morphological Operations (clean-up)
  4. Hough Transform (line/outline detection)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import os
import sys

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
    from PIL import Image


# ============================================================
# STEP 0: IMAGE I/O 
# ============================================================

def read_image(path):
    """Read an image as an RGB numpy array."""
    if HAS_CV2:
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Cannot read: {path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        img = np.array(Image.open(path).convert('RGB'))
    return img


def rgb_to_grayscale(img_rgb):
    """Luminance conversion (ITU-R BT.601)."""
    return (0.299 * img_rgb[..., 0] +
            0.587 * img_rgb[..., 1] +
            0.114 * img_rgb[..., 2]).astype(np.float32)


# ============================================================
# STEP 1: NDBI  (Normalized Difference Built-up Index)
# ============================================================
# Real NDBI needs SWIR (Short-Wave IR) and NIR (Near IR) bands:
#       NDBI = (SWIR - NIR) / (SWIR + NIR)
#
# For RGB images, we use a proxy that mimics NDBI behaviour based on the
# observation that built-up areas tend to be bright and gray (low saturation):
def compute_ndbi(swir, nir):
    swir = swir.astype(np.float32)
    nir = nir.astype(np.float32)
    denom = swir + nir
    denom[denom == 0] = 1e-6
    return (swir - nir) / denom


def compute_ndbi_rgb_proxy(img_rgb):
    """
    RGB proxy for NDBI when multispectral bands aren't available.
    grayness  = 1 - (max(r,g,b) - min(r,g,b)) / max(r,g,b)
    brightness= (r + g + b) / 3
    proxy     = grayness * brightness
    Result is then rescaled to [-1, 1] so the threshold behaves like
    a true NDBI.
    """
    r = img_rgb[..., 0].astype(np.float32) / 255.0
    g = img_rgb[..., 1].astype(np.float32) / 255.0
    b = img_rgb[..., 2].astype(np.float32) / 255.0

    max_c = np.maximum(np.maximum(r, g), b)
    min_c = np.minimum(np.minimum(r, g), b)
    grayness = 1.0 - (max_c - min_c) / (max_c + 1e-6)
    brightness = (r + g + b) / 3.0

    proxy = grayness * brightness
    # Normalise to [-1, 1]
    lo, hi = proxy.min(), proxy.max()
    proxy = 2 * (proxy - lo) / (hi - lo + 1e-9) - 1
    return proxy


def suppress_non_builtup(gray, ndbi, threshold=0.0):
    """Zero-out pixels whose NDBI is below the threshold."""
    mask = (ndbi >= threshold).astype(np.float32)
    return gray * mask, mask


# ============================================================
# STEP 2: CANNY EDGE DETECTION
# ============================================================

def gaussian_kernel(size=5, sigma=1.4):
    """Create a 2D Gaussian kernel."""
    k = size // 2
    y, x = np.mgrid[-k:k+1, -k:k+1]
    g = np.exp(-(x**2 + y**2) / (2.0 * sigma**2))
    return g / g.sum()




def convolve2d_fast(image, kernel):
    """
    Vectorised convolution using stride tricks
    """
    from numpy.lib.stride_tricks import sliding_window_view
    iH, iW = image.shape
    kH, kW = kernel.shape
    pH, pW = kH // 2, kW // 2
    padded = np.pad(image, ((pH, pH), (pW, pW)), mode='edge')
    windows = sliding_window_view(padded, (kH, kW))
    kflip = kernel[::-1, ::-1]
    return np.einsum('ijkl,kl->ij', windows, kflip).astype(np.float32)


def sobel_gradients(image):
    """Compute gradient magnitude and direction using Sobel."""
    Kx = np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]], dtype=np.float32)
    Ky = np.array([[-1, -2, -1],
                   [ 0,  0,  0],
                   [ 1,  2,  1]], dtype=np.float32)
    Gx = convolve2d_fast(image, Kx)
    Gy = convolve2d_fast(image, Ky)
    magnitude = np.hypot(Gx, Gy)
    magnitude = magnitude / (magnitude.max() + 1e-9) * 255.0
    direction = np.arctan2(Gy, Gx)  # radians, in (-pi, pi]
    return magnitude, direction


def non_max_suppression(mag, direction):
    """
    Thin the edges: keep pixel only if it's a local max along
    the gradient direction.
    """
    H, W = mag.shape
    out = np.zeros((H, W), dtype=np.float32)
    angle = np.rad2deg(direction) % 180  # map to [0, 180)

    for i in range(1, H - 1):
        for j in range(1, W - 1):
            a = angle[i, j]
            # Choose two neighbours along the gradient direction
            if (0 <= a < 22.5) or (157.5 <= a < 180):
                n1, n2 = mag[i, j-1], mag[i, j+1]
            elif 22.5 <= a < 67.5:
                n1, n2 = mag[i-1, j+1], mag[i+1, j-1]
            elif 67.5 <= a < 112.5:
                n1, n2 = mag[i-1, j], mag[i+1, j]
            else:  # 112.5..157.5
                n1, n2 = mag[i-1, j-1], mag[i+1, j+1]

            if mag[i, j] >= n1 and mag[i, j] >= n2:
                out[i, j] = mag[i, j]
    return out


def double_threshold(img, low_ratio=0.05, high_ratio=0.15):
    """Classify pixels as strong (255), weak (75), or zero."""
    high = img.max() * high_ratio
    low = high * (low_ratio / high_ratio) if high_ratio > 0 else img.max() * low_ratio
    strong, weak = 255, 75
    res = np.zeros_like(img, dtype=np.uint8)
    res[img >= high] = strong
    res[(img >= low) & (img < high)] = weak
    return res, weak, strong


def hysteresis(img, weak=75, strong=255):
    """
    Promote weak pixels to strong if they are 8-connected
    to any strong pixel; otherwise drop them.
    """
    H, W = img.shape
    out = img.copy()
    # Iterate until stable
    changed = True
    while changed:
        changed = False
        for i in range(1, H - 1):
            for j in range(1, W - 1):
                if out[i, j] == weak:
                    neigh = out[i-1:i+2, j-1:j+2]
                    if (neigh == strong).any():
                        out[i, j] = strong
                        changed = True
    out[out != strong] = 0
    return out


def canny_edge_detector(gray, sigma=1.4, ksize=5,
                        low_ratio=0.05, high_ratio=0.15, verbose=True):
    """Full Canny pipeline."""
    if verbose: print("   [Canny] Gaussian blur ...")
    blurred = convolve2d_fast(gray, gaussian_kernel(ksize, sigma))

    if verbose: print("   [Canny] Sobel gradients ...")
    mag, direction = sobel_gradients(blurred)

    if verbose: print("   [Canny] Non-max suppression ...")
    nms = non_max_suppression(mag, direction)

    if verbose: print("   [Canny] Double thresholding ...")
    thresh, weak, strong = double_threshold(nms, low_ratio, high_ratio)

    if verbose: print("   [Canny] Hysteresis tracking ...")
    edges = hysteresis(thresh, weak, strong)
    return edges


# ============================================================
# STEP 3: BINARY MORPHOLOGICAL OPERATIONS
# ============================================================

def make_struct_elem(shape='square', size=3):
    if shape == 'square':
        return np.ones((size, size), dtype=np.uint8)
    if shape == 'cross':
        se = np.zeros((size, size), dtype=np.uint8)
        se[size // 2, :] = 1
        se[:, size // 2] = 1
        return se
    raise ValueError(shape)


def binary_dilate(img, se):
    """Binary dilation: a pixel is 1 if any SE hit is 1."""
    from numpy.lib.stride_tricks import sliding_window_view
    bin_img = (img > 0).astype(np.uint8)
    kH, kW = se.shape
    pad = kH // 2
    padded = np.pad(bin_img, pad, mode='constant', constant_values=0)
    windows = sliding_window_view(padded, (kH, kW))
    # Dilation = max over neighbourhood where SE == 1
    mask = se.astype(bool)
    result = (windows[..., mask].max(axis=-1) > 0).astype(np.uint8) * 255
    return result


def binary_erode(img, se):
    """Binary erosion: a pixel is 1 only if all SE hits are 1."""
    from numpy.lib.stride_tricks import sliding_window_view
    bin_img = (img > 0).astype(np.uint8)
    kH, kW = se.shape
    pad = kH // 2
    padded = np.pad(bin_img, pad, mode='constant', constant_values=0)
    windows = sliding_window_view(padded, (kH, kW))
    mask = se.astype(bool)
    result = (windows[..., mask].min(axis=-1) > 0).astype(np.uint8) * 255
    return result


def binary_open(img, se):   # erosion and then dilation  (removes small noise)
    return binary_dilate(binary_erode(img, se), se)


def binary_close(img, se):  # dilation and then erosion (fills small gaps)
    return binary_erode(binary_dilate(img, se), se)


def zhang_suen_thinning(img, max_iter=50):
    """
    At each pass we do two sub-iterations; a pixel P1 is removed if:
      (1) it has 2..6 non-zero 8-neighbours
      (2) exactly one 0->1 transition in the ordered ring P2..P9,P2
      (3) at least one of {P2,P4,P6} is 0   (sub-iter 1)
          at least one of {P2,P4,P8} is 0   (sub-iter 2)
      (4) at least one of {P4,P6,P8} is 0   (sub-iter 1)
          at least one of {P2,P6,P8} is 0   (sub-iter 2)
    Iterate until no pixels are removed.
    """
    img = (img > 0).astype(np.uint8)

    def neighbours(I):
        p = np.pad(I, 1, constant_values=0)
        P2 = p[0:-2, 1:-1]; P3 = p[0:-2, 2:  ]; P4 = p[1:-1, 2:  ]
        P5 = p[2:  , 2:  ]; P6 = p[2:  , 1:-1]; P7 = p[2:  , 0:-2]
        P8 = p[1:-1, 0:-2]; P9 = p[0:-2, 0:-2]
        return P2, P3, P4, P5, P6, P7, P8, P9

    def transitions(P2, P3, P4, P5, P6, P7, P8, P9):
        seq = [P2, P3, P4, P5, P6, P7, P8, P9, P2]
        A = np.zeros_like(P2, dtype=np.int32)
        for k in range(8):
            A += ((seq[k] == 0) & (seq[k + 1] == 1)).astype(np.int32)
        return A

    for _ in range(max_iter):
        removed = False

        # ---- sub-iteration 1 ----
        P2, P3, P4, P5, P6, P7, P8, P9 = neighbours(img)
        B = P2 + P3 + P4 + P5 + P6 + P7 + P8 + P9
        A = transitions(P2, P3, P4, P5, P6, P7, P8, P9)
        cond = ((img == 1) & (B >= 2) & (B <= 6) & (A == 1) &
                ((P2 * P4 * P6) == 0) & ((P4 * P6 * P8) == 0))
        if cond.any():
            img = img.copy()
            img[cond] = 0
            removed = True

        # ---- sub-iteration 2 ----
        P2, P3, P4, P5, P6, P7, P8, P9 = neighbours(img)
        B = P2 + P3 + P4 + P5 + P6 + P7 + P8 + P9
        A = transitions(P2, P3, P4, P5, P6, P7, P8, P9)
        cond = ((img == 1) & (B >= 2) & (B <= 6) & (A == 1) &
                ((P2 * P4 * P8) == 0) & ((P2 * P6 * P8) == 0))
        if cond.any():
            img = img.copy()
            img[cond] = 0
            removed = True

        if not removed:
            break

    return (img * 255).astype(np.uint8)


def morphological_cleanup(edges, morph_shape='square', morph_size=3, verbose=True):
    """
      1. Closing  -> join small breaks (double-lines from Canny, gaps)
      2. Thinning 
      3. Isolated-pixel removal -> drop remaining noise specks
    """
    se3 = make_struct_elem(morph_shape, morph_size)

    if verbose: print("   [Morph] Closing (dilate->erode) to bridge gaps ...")
    closed = binary_close(edges, se3)

    if verbose: print("   [Morph] Zhang-Suen thinning -> 1-pixel skeleton ...")
    skeleton = zhang_suen_thinning(closed)

    if verbose: print("   [Morph] Removing isolated pixels ...")
    from numpy.lib.stride_tricks import sliding_window_view
    bin_img = (skeleton > 0).astype(np.uint8)
    padded = np.pad(bin_img, 1, mode='constant')
    windows = sliding_window_view(padded, (3, 3))
    neigh_count = windows.sum(axis=(-1, -2)) - bin_img
    cleaned = np.where(neigh_count >= 1, skeleton, 0).astype(np.uint8)

    return cleaned


# ============================================================
# STEP 4: HOUGH TRANSFORM 
# ============================================================

def hough_line_transform(edges, theta_step=1.0, rho_step=1.0):
    """
    Classical Hough transform for lines: rho = x*cos(theta) + y*sin(theta)
    Returns accumulator, thetas (rad), rhos.
    """
    H, W = edges.shape
    diag = int(np.ceil(np.sqrt(H**2 + W**2)))
    rhos = np.arange(-diag, diag + 1, rho_step)
    thetas = np.deg2rad(np.arange(-90, 90, theta_step))

    cos_t = np.cos(thetas)
    sin_t = np.sin(thetas)
    n_thetas = len(thetas)

    accumulator = np.zeros((len(rhos), n_thetas), dtype=np.int32)

    ys, xs = np.nonzero(edges)
    for x, y in zip(xs, ys):
        rho_vals = x * cos_t + y * sin_t
        rho_idx = np.round(rho_vals + diag).astype(int)   
        valid = (rho_idx >= 0) & (rho_idx < len(rhos))
        accumulator[rho_idx[valid], np.arange(n_thetas)[valid]] += 1
    return accumulator, thetas, rhos


def hough_peaks(accumulator, num_peaks=80, threshold=None, nhood=(5, 3)):
    """
    Find local maxima in the Hough accumulator.
    With a 1-pixel-wide skeleton input, each real edge produces ONE
    sharp peak, so we can use a small suppression neighbourhood.
    """
    if threshold is None:
        threshold = 0.12 * accumulator.max()
    peaks, acc = [], accumulator.copy()
    nh_r, nh_t = nhood
    for _ in range(num_peaks):
        idx = np.argmax(acc)
        r, t = np.unravel_index(idx, acc.shape)
        if acc[r, t] < threshold:
            break
        peaks.append((r, t))
        r0, r1 = max(0, r - nh_r), min(acc.shape[0], r + nh_r + 1)
        t0, t1 = max(0, t - nh_t), min(acc.shape[1], t + nh_t + 1)
        acc[r0:r1, t0:t1] = 0
    return peaks


def hough_peaks_to_segments(edges, peaks, thetas, rhos,
                            rho_tolerance=1.5, max_gap=6, min_length=15):
    ys, xs = np.nonzero(edges)
    xs = xs.astype(np.float32)
    ys = ys.astype(np.float32)
    segments = []

    for r_idx, t_idx in peaks:
        rho = rhos[r_idx]
        theta = thetas[t_idx]
        cos_t, sin_t = np.cos(theta), np.sin(theta)

        # Edge pixels on this line (within rho_tolerance)
        mask = np.abs(xs * cos_t + ys * sin_t - rho) < rho_tolerance
        if mask.sum() < 2:
            continue
        lx, ly = xs[mask], ys[mask]

        # Parameter along the line direction
        t_vals = lx * (-sin_t) + ly * cos_t
        order = np.argsort(t_vals)
        tv, sx, sy = t_vals[order], lx[order], ly[order]

        # Walk through sorted pixels, emitting every run that's long enough
        start = 0
        for i in range(1, len(tv)):
            if tv[i] - tv[i-1] > max_gap:
                if tv[i-1] - tv[start] >= min_length:
                    segments.append(((int(sx[start]), int(sy[start])),
                                     (int(sx[i-1]),   int(sy[i-1]))))
                start = i
        if tv[-1] - tv[start] >= min_length:
            segments.append(((int(sx[start]), int(sy[start])),
                             (int(sx[-1]),    int(sy[-1]))))
    return segments


def suppress_duplicate_segments(segments, angle_tol_deg=12, perp_dist_tol=5,
                                min_overlap_ratio=0.3):
    """
    Segment-level non-maximum suppression.
    Sort segments by length (longest first). For each, drop all later
    segments that are:
      - nearly parallel   (angle within angle_tol_deg)
      - close in rho      (perpendicular distance < perp_dist_tol)
      - overlapping along the line direction (>= min_overlap_ratio)
    """
    if len(segments) <= 1:
        return list(segments)

    # Sort by length descending
    enriched = []
    for (x1, y1), (x2, y2) in segments:
        L = float(np.hypot(x2 - x1, y2 - y1))
        ang = np.arctan2(y2 - y1, x2 - x1)   # direction along the line
        enriched.append((L, ang, (x1, y1), (x2, y2)))
    enriched.sort(key=lambda t: t[0], reverse=True)

    kept = []
    for L, ang, p1, p2 in enriched:
        x1, y1 = p1; x2, y2 = p2
        is_duplicate = False
        for kL, kang, kp1, kp2 in kept:
            # 1) angle difference (mod pi)
            da = abs(ang - kang)
            da = min(da, np.pi - da)
            if np.rad2deg(da) > angle_tol_deg:
                continue

            # 2) perpendicular distance from this segment's midpoint
            #    to the kept segment's line
            kx1, ky1 = kp1; kx2, ky2 = kp2
            klen = np.hypot(kx2 - kx1, ky2 - ky1)
            if klen < 1e-6:
                continue
            mx, my = (x1 + x2) / 2, (y1 + y2) / 2
            perp = abs((mx - kx1) * (ky2 - ky1) -
                       (my - ky1) * (kx2 - kx1)) / klen
            if perp > perp_dist_tol:
                continue

            # 3) overlap along the kept segment's direction
            kdx, kdy = (kx2 - kx1) / klen, (ky2 - ky1) / klen
            # Project all four endpoints onto the kept line's axis
            t_kept = sorted([0.0, klen])
            t_this = sorted([(x1 - kx1) * kdx + (y1 - ky1) * kdy,
                             (x2 - kx1) * kdx + (y2 - ky1) * kdy])
            overlap = max(0, min(t_kept[1], t_this[1]) -
                             max(t_kept[0], t_this[0]))
            ratio = overlap / max(L, 1e-6)
            if ratio >= min_overlap_ratio:
                is_duplicate = True
                break

        if not is_duplicate:
            kept.append((L, ang, p1, p2))

    return [(p1, p2) for _, _, p1, p2 in kept]


def merge_collinear_segments(segments, angle_tol_deg=6, perp_dist_tol=6,
                             endpoint_gap_tol=18):
    """
    Merge segments that are nearly collinear and touch/overlap.
    angle_tol_deg  : max angle difference between segments (degrees)
    perp_dist_tol  : max perpendicular distance between the two lines
    endpoint_gap_tol : max gap between nearest endpoints
    """
    segs = [tuple(map(tuple, s)) for s in segments]
    changed = True
    while changed and len(segs) > 1:
        changed = False
        out = []
        used = [False] * len(segs)
        for i in range(len(segs)):
            if used[i]:
                continue
            (x1, y1), (x2, y2) = segs[i]
            ang_i = np.arctan2(y2 - y1, x2 - x1)
            merged_with = -1
            for j in range(i + 1, len(segs)):
                if used[j]:
                    continue
                (x3, y3), (x4, y4) = segs[j]
                ang_j = np.arctan2(y4 - y3, x4 - x3)
                # angle difference (mod pi because direction is irrelevant)
                da = abs(ang_i - ang_j)
                da = min(da, np.pi - da)
                if np.rad2deg(da) > angle_tol_deg:
                    continue
                # perpendicular distance from (x3,y3) to the first line
                L = np.hypot(x2 - x1, y2 - y1)
                if L < 1e-6:
                    continue
                perp = abs((x3 - x1) * (y2 - y1) - (y3 - y1) * (x2 - x1)) / L
                if perp > perp_dist_tol:
                    continue
                # endpoint gap (min over 4 pairs)
                d = min(np.hypot(x3 - x1, y3 - y1), np.hypot(x4 - x1, y4 - y1),
                        np.hypot(x3 - x2, y3 - y2), np.hypot(x4 - x2, y4 - y2))
                if d > endpoint_gap_tol:
                    continue
                # Merge: take the two most extreme points along ang_i
                pts = [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
                dx, dy = np.cos(ang_i), np.sin(ang_i)
                projs = [p[0] * dx + p[1] * dy for p in pts]
                lo, hi = int(np.argmin(projs)), int(np.argmax(projs))
                out.append((pts[lo], pts[hi]))
                used[i] = used[j] = True
                merged_with = j
                changed = True
                break
            if merged_with == -1 and not used[i]:
                out.append(segs[i])
                used[i] = True
        segs = out
    return segs


def draw_segments(img_rgb, segments, color=(255, 0, 0), thickness=2):
    """Draw list of ((x1,y1),(x2,y2)) segments on a copy of img_rgb."""
    out = img_rgb.copy()
    if out.ndim == 2:
        out = np.stack([out] * 3, axis=-1)
    for (x1, y1), (x2, y2) in segments:
        _draw_line(out, (x1, y1), (x2, y2), color, thickness)
    return out


def _draw_line(img, p1, p2, color, thickness=1):
    x1, y1 = p1; x2, y2 = p2
    H, W = img.shape[:2]
    dx = abs(x2 - x1); dy = abs(y2 - y1)
    sx = 1 if x1 < x2 else -1
    sy = 1 if y1 < y2 else -1
    err = dx - dy
    x, y = x1, y1
    while True:
        for tx in range(-thickness // 2, thickness // 2 + 1):
            for ty in range(-thickness // 2, thickness // 2 + 1):
                xx, yy = x + tx, y + ty
                if 0 <= xx < W and 0 <= yy < H:
                    img[yy, xx] = color
        if x == x2 and y == y2: break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy; x += sx
        if e2 < dx:
            err += dx; y += sy


# ============================================================
# END-TO-END PIPELINE
# ============================================================

def extract_buildings(image_path, out_dir='outputs',
                      use_ndbi_proxy=True,
                      swir=None, nir=None,
                      ndbi_threshold= 0.1,
                      gauss_ksize=5,
                      canny_sigma=1.4,
                      canny_low=0.05, canny_high=0.15,
                      morph_shape='square',
                      morph_size=3,
                      num_hough_peaks=400,
                      min_segment_length=15,
                      merge_angle_tol=5,
                      merge_perp_tol=4,
                      merge_gap_tol=15,
                      resize_max=400):
    """Full pipeline. Saves intermediate + final results."""
    os.makedirs(out_dir, exist_ok=True)

    print("[1/6] Reading image ...")
    print(image_path)
    img = read_image(image_path)

    # Optional downscale for tractable runtime
    # No resizing

    gray = rgb_to_grayscale(img)

    print("[2/6] Computing NDBI ...")
    if swir is not None and nir is not None:
        ndbi = compute_ndbi(swir, nir)
    else:
        ndbi = compute_ndbi_rgb_proxy(img)
        print("      (using RGB proxy - no SWIR/NIR provided)")

    print("[3/6] Suppressing non-built-up areas ...")
    gray_suppressed, builtup_mask = suppress_non_builtup(gray, ndbi, ndbi_threshold)

    print("[4/6] Canny edge detection (from-scratch) ...")
    edges = canny_edge_detector(gray_suppressed,
                                sigma=canny_sigma,
                                ksize=gauss_ksize,
                                low_ratio=canny_low,
                                high_ratio=canny_high)

    mask_u8 = (builtup_mask * 255).astype(np.uint8)

    mask_neighborhood = binary_dilate(mask_u8, make_struct_elem('square', 3))
    edges = (edges * (mask_neighborhood > 0)).astype(np.uint8)

    print("[5/6] Morphological clean-up ...")
    edges_clean = morphological_cleanup(edges, morph_shape=morph_shape, morph_size=morph_size)

    print("[6/6] Hough transform ...")
    acc, thetas, rhos = hough_line_transform(edges_clean)
    peaks = hough_peaks(acc, num_peaks=num_hough_peaks)
    print(f"      found {len(peaks)} raw Hough peaks")

    segments = hough_peaks_to_segments(edges_clean, peaks, thetas, rhos,
                                       rho_tolerance=1.0, max_gap=6,
                                       min_length=min_segment_length)
    print(f"      {len(segments)} segments after line-walking")

    segments = merge_collinear_segments(segments,
                                        angle_tol_deg=merge_angle_tol,
                                        perp_dist_tol=merge_perp_tol,
                                        endpoint_gap_tol=merge_gap_tol)
    print(f"      {len(segments)} segments after merging duplicates")


    segments = suppress_duplicate_segments(segments,
                                           angle_tol_deg=12,
                                           perp_dist_tol=5,
                                           min_overlap_ratio=0.3)
    print(f"      {len(segments)} segments after duplicate suppression")

    result = draw_segments(img, segments, color=(255, 0, 0), thickness=1)

    stages_dir = os.path.join(out_dir, 'stages')
    os.makedirs(stages_dir, exist_ok=True)

    def _save_stage(idx, name, arr, cmap=None):
        """Save a single stage at native resolution (no axes/whitespace)."""
        path = os.path.join(stages_dir, f"{idx}_{name}.png")
        if cmap is None:
            plt.imsave(path, arr)
        else:
            plt.imsave(path, arr, cmap=cmap)
        return path

    stage_paths = []
    stage_paths.append(_save_stage(1, 'original',          img))
    stage_paths.append(_save_stage(2, 'ndbi',              ndbi, cmap='RdYlGn_r'))
    stage_paths.append(_save_stage(3, 'builtup_mask',      builtup_mask, cmap='gray'))
    stage_paths.append(_save_stage(4, 'suppressed_gray',   gray_suppressed, cmap='gray'))
    stage_paths.append(_save_stage(5, 'canny_edges',       edges, cmap='gray'))
    stage_paths.append(_save_stage(6, 'after_morphology',  edges_clean, cmap='gray'))
    stage_paths.append(_save_stage(7, 'hough_accumulator', np.log1p(acc), cmap='hot'))
    stage_paths.append(_save_stage(8, 'detected_outlines', result))
    print(f"\n✓ {len(stage_paths)} per-stage PNGs saved to: {stages_dir}/")


    H_img, W_img = img.shape[:2]
    aspect = W_img / max(H_img, 1)
    fig_w, fig_h = 4 * 5 * aspect, 2 * 5 
    fig, axes = plt.subplots(2, 4, figsize=(fig_w, fig_h))
    axes[0, 0].imshow(img);                       axes[0, 0].set_title("1. Original")
    axes[0, 1].imshow(ndbi, cmap='RdYlGn_r');     axes[0, 1].set_title("2. NDBI (red=built-up)")
    axes[0, 2].imshow(builtup_mask, cmap='gray'); axes[0, 2].set_title("3. Built-up mask")
    axes[0, 3].imshow(gray_suppressed, cmap='gray'); axes[0, 3].set_title("4. Suppressed grayscale")
    axes[1, 0].imshow(edges, cmap='gray');        axes[1, 0].set_title("5. Canny edges")
    axes[1, 1].imshow(edges_clean, cmap='gray');  axes[1, 1].set_title("6. After morphology")
    axes[1, 2].imshow(np.log1p(acc), cmap='hot', aspect='auto',
                     extent=[np.rad2deg(thetas[0]), np.rad2deg(thetas[-1]),
                             rhos[-1], rhos[0]])
    axes[1, 2].set_title("7. Hough accumulator")
    axes[1, 2].set_xlabel(r"$\theta$ (deg)"); axes[1, 2].set_ylabel(r"$\rho$")
    axes[1, 3].imshow(result);                    axes[1, 3].set_title("8. Detected outlines")
    for ax in axes.flat:
        if ax is not axes[1, 2]: ax.axis('off')
    plt.tight_layout()
    out_path = os.path.join(out_dir, 'building_extraction_pipeline.png')
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"✓ Combined pipeline figure saved: {out_path}")

    # Save final image separately too (full resolution, no axes)
    final_path = os.path.join(out_dir, 'final_outlines.png')
    plt.imsave(final_path, result)
    print(f"✓ Final result saved: {final_path}")

    return {
        'original': img, 'ndbi': ndbi, 'mask': builtup_mask,
        'edges': edges, 'edges_clean': edges_clean,
        'accumulator': acc, 'peaks': peaks, 'result': result,
    }


# ============================================================
# DEMO: generate a synthetic "satellite" scene if user has no image
# ============================================================
def make_synthetic_scene(H=200, W=300, seed=0):
    """Simple scene with buildings (gray rectangles), vegetation (green),
       water (blue) - to demo the pipeline without a real image."""
    rng = np.random.default_rng(seed)
    img = np.zeros((H, W, 3), dtype=np.uint8)

    # Vegetation background (greenish with noise)
    img[..., 0] = 60  + rng.integers(0, 20, (H, W))
    img[..., 1] = 120 + rng.integers(0, 30, (H, W))
    img[..., 2] = 50  + rng.integers(0, 20, (H, W))

    # A water body (blue patch)
    img[140:180, 20:90, 0] = 30
    img[140:180, 20:90, 1] = 60
    img[140:180, 20:90, 2] = 150

    # Buildings: gray rectangles of various sizes
    buildings = [(30, 40, 60, 90), (30, 130, 55, 175),
                 (90, 40, 120, 80), (90, 160, 115, 230),
                 (30, 220, 80, 280), (150, 110, 185, 150),
                 (150, 200, 180, 270)]
    for y1, x1, y2, x2 in buildings:
        shade = rng.integers(140, 200)
        img[y1:y2, x1:x2] = [shade, shade, shade - 10]
        # Dark border to simulate roof edges
        img[y1:y1+1, x1:x2] = [40, 40, 40]
        img[y2-1:y2, x1:x2] = [40, 40, 40]
        img[y1:y2, x1:x1+1] = [40, 40, 40]
        img[y1:y2, x2-1:x2] = [40, 40, 40]
    return img


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Building Outline Extraction Pipeline')
    parser.add_argument('image_path', nargs='?', default='--syn',
                        help='Path to input image, or --syn for synthetic demo')
    parser.add_argument('--gauss-ksize',      type=int,   default=5,       help='Gaussian kernel size (odd, default: 5)')
    parser.add_argument('--gauss-sigma',      type=float, default=1.4,     help='Gaussian sigma (default: 1.4)')
    parser.add_argument('--morph-shape',      type=str,   default='square',
                        choices=['square', 'cross'],                        help='Morphological structuring element shape (default: square)')
    parser.add_argument('--morph-size',       type=int,   default=3,       help='Morphological structuring element size (odd, default: 3)')
    parser.add_argument('--ndbi-threshold',   type=float, default=0.1,     help='NDBI threshold for built-up mask (default: 0.1)')
    args = parser.parse_args()

    if args.image_path == '--syn':
        print("Generating synthetic satellite scene ...\n")
        synth = make_synthetic_scene()
        img_path = 'synthetic_demo.png'
        if HAS_CV2:
            cv2.imwrite(img_path, cv2.cvtColor(synth, cv2.COLOR_RGB2BGR))
        else:
            Image.fromarray(synth).save(img_path)
    else:
        img_path = args.image_path
        if not os.path.isfile(img_path):
            print(f"Error: file not found: {img_path}")
            sys.exit(1)

    # Ensure gauss_ksize and morph_size are odd
    gauss_ksize = args.gauss_ksize if args.gauss_ksize % 2 == 1 else args.gauss_ksize + 1
    morph_size  = args.morph_size  if args.morph_size  % 2 == 1 else args.morph_size  + 1

    extract_buildings(img_path,
                      out_dir='outputs',
                      gauss_ksize=gauss_ksize,
                      canny_sigma=args.gauss_sigma,
                      ndbi_threshold=args.ndbi_threshold,
                      morph_shape=args.morph_shape,
                      morph_size=morph_size,
                      resize_max=600)