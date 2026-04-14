"""
difference_map.py
=================
Per-frame difference maps, blur maps, and confidence maps.

Paper: Section 6.1, Equations (7), (8), (9)

Equation (7) – Difference map:
    D_i(x, y, r) = Σ_{x',y'} w(x'-x, y'-y) * |Î_i(x',y') - Î^r_0(x',y')|

    where w is a 2-D Gaussian kernel with variance (μ², μ²)
    (paper uses μ = 15 for this step, Section 8)

Equation (8) – Blur map:
    B_i(x, y) = δ_i * argmin_r  D_i(x, y, r)

Equation (9) – Confidence map:
    C_i(x, y) ∝ ( mean_{r'} D_i(x,y,r')  -  min_{r'} D_i(x,y,r') )^α

    (paper uses α = 2, Section 8)
"""

import numpy as np
from scipy.ndimage import gaussian_filter


# ---------------------------------------------------------------------------
# Equation (7) – Difference map
# ---------------------------------------------------------------------------

def compute_difference_map(aligned_frame: np.ndarray,
                            blur_stack:    np.ndarray,
                            mu:            float = 15.0) -> np.ndarray:
    """
    Compute D_i(x, y, r) for one aligned focal stack frame.

    D_i(x, y, r) = Gaussian_σ=μ * |Î_i - Î^r_0|

    Parameters
    ----------
    aligned_frame : (H x W) or (H x W x C) float32  –  aligned frame Î_i
    blur_stack    : (N_r x H x W) or (N_r x H x W x C)  –  pre-rendered blurs
    mu            : Gaussian kernel std  (paper: μ = 15)

    Returns
    -------
    D : (N_r x H x W) float32 difference map
    """
    N_r = blur_stack.shape[0]

    # Convert to grayscale for the difference computation
    if aligned_frame.ndim == 3:
        frame_g = aligned_frame.mean(axis=2).astype(np.float32)
    else:
        frame_g = aligned_frame.astype(np.float32)

    H, W = frame_g.shape
    D = np.zeros((N_r, H, W), dtype=np.float32)

    for k in range(N_r):
        if blur_stack.ndim == 4:
            blurred_g = blur_stack[k].mean(axis=2).astype(np.float32)
        else:
            blurred_g = blur_stack[k].astype(np.float32)

        abs_diff = np.abs(frame_g - blurred_g)
        # Gaussian-weighted patch sum  ≡  Gaussian filter
        D[k] = gaussian_filter(abs_diff, sigma=mu, mode='reflect')

    return D   # (N_r, H, W)


# ---------------------------------------------------------------------------
# Equation (8) – Blur map
# Equation (9) – Confidence map
# ---------------------------------------------------------------------------

def compute_blur_map(D:       np.ndarray,
                     radii:   np.ndarray,
                     delta_i: float = 1.0) -> np.ndarray:
    """
    Compute B_i(x, y) = δ_i * argmin_r D_i(x, y, r)   [Equation 8]

    Parameters
    ----------
    D       : (N_r x H x W) difference map
    radii   : (N_r,) blur radius values
    delta_i : magnification scaling constant for frame i (default 1.0)

    Returns
    -------
    B : (H x W) float32 estimated blur radius per pixel
    """
    best_idx = np.argmin(D, axis=0)          # (H, W)
    B = delta_i * radii[best_idx]            # Equation (8)
    return B.astype(np.float32)


def compute_confidence_map(D:     np.ndarray,
                            alpha: float = 2.0) -> np.ndarray:
    """
    Compute C_i(x, y) ∝ (mean_r D_i - min_r D_i)^α   [Equation 9]

    A high confidence means the minimum of D is well-defined (sharp dip),
    indicating a reliable blur estimate at that pixel.

    Parameters
    ----------
    D     : (N_r x H x W) difference map
    alpha : exponent  (paper: α = 2)

    Returns
    -------
    C : (H x W) float32 confidence map, normalised to [0, 1]
    """
    mean_D = D.mean(axis=0)   # (H, W)
    min_D  = D.min(axis=0)    # (H, W)
    C = (mean_D - min_D) ** alpha   # Equation (9)

    # Normalise to [0, 1]
    c_max = C.max()
    if c_max > 1e-9:
        C = C / c_max

    return C.astype(np.float32)


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------

def process_frame(aligned_frame: np.ndarray,
                  blur_stack:    np.ndarray,
                  radii:         np.ndarray,
                  mu:            float = 15.0,
                  delta_i:       float = 1.0,
                  alpha:         float = 2.0) -> tuple:
    """
    Compute D_i, B_i, C_i for one aligned frame.

    Returns
    -------
    D : (N_r x H x W) difference map
    B : (H x W) blur map
    C : (H x W) confidence map
    """
    D = compute_difference_map(aligned_frame, blur_stack, mu=mu)
    B = compute_blur_map(D, radii, delta_i=delta_i)
    C = compute_confidence_map(D, alpha=alpha)
    return D, B, C
