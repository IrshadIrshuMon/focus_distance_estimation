"""
psf_model.py
============
Point Spread Function (PSF) model.

Paper: "Depth from Focus with Your Mobile Phone"
       Suwajanakorn, Hernandez, Seitz - CVPR 2015

Equation (4) - Blur radius:
    b_i(s) = A * |f_i - s| / s * F / (f_i - F)

Equation (5) - Frame re-rendering (locally shift-invariant approx):
    ~I_i(x,y) = integral integral I0(u,v) * D(x-u, y-v, b_i(x,y)) du dv
"""

import numpy as np
from scipy.ndimage import uniform_filter


def compute_blur_radius(A: float,
                        F: float,
                        f_i: float,
                        depth_map: np.ndarray) -> np.ndarray:
    """
    Compute per-pixel blur radius using Equation (4).

        b_i(s) = A * |f_i - s| / s * F / (f_i - F)

    Parameters
    ----------
    A         : aperture size (scalar)
    F         : focal length  (scalar)
    f_i       : focal depth of frame i (scalar)
    depth_map : scene depth map (H x W), all values > 0

    Returns
    -------
    b : blur radius map (H x W), non-negative float32
    """
    s = np.maximum(depth_map, 1e-6).astype(np.float64)
    denom = f_i - F

    if abs(denom) < 1e-9:
        return np.zeros_like(s, dtype=np.float32)

    b = A * (np.abs(f_i - s) / s) * (F / denom)
    return np.abs(b).astype(np.float32)


def apply_disc_psf(image: np.ndarray, radius: float) -> np.ndarray:
    """
    Blur a 2-D single-channel image with a uniform disc PSF.

    The paper uses a uniform disc-shaped PSF.  We approximate it
    with a box filter of half-width = radius (fast, no extra deps).

    Parameters
    ----------
    image  : 2-D float32 array (H x W), values in [0, 1]
    radius : PSF radius in pixels (>= 0)

    Returns
    -------
    blurred : same shape as image, float32
    """
    image = image.astype(np.float32)
    r = int(round(max(radius, 0.0)))
    if r == 0:
        return image.copy()
    size = 2 * r + 1
    return uniform_filter(image, size=size, mode='reflect').astype(np.float32)


def render_frame(all_in_focus: np.ndarray,
                 depth_map: np.ndarray,
                 A: float,
                 F: float,
                 f_i: float) -> np.ndarray:
    """
    Re-render a focal stack frame using the locally shift-invariant
    approximation of Equation (5).

    For each pixel we compute its blur radius from the depth map and
    apply the corresponding disc PSF.

    Parameters
    ----------
    all_in_focus : (H x W) or (H x W x C) float32 in [0, 1]
    depth_map    : (H x W) float32 scene depth
    A, F, f_i    : camera parameters

    Returns
    -------
    rendered : same shape as all_in_focus, float32
    """
    b_map = compute_blur_radius(A, F, f_i, depth_map)  # (H x W)

    # Quantise radius to 0.25-pixel steps (same as blur stack)
    step = 0.25
    r_max = float(b_map.max())
    radii = np.arange(0.0, r_max + step, step)

    if all_in_focus.ndim == 2:
        channels = [all_in_focus]
    else:
        channels = [all_in_focus[:, :, c] for c in range(all_in_focus.shape[2])]

    rendered_channels = []
    for ch in channels:
        out = np.zeros_like(ch, dtype=np.float32)
        for r in radii:
            mask = (b_map >= r - step / 2) & (b_map < r + step / 2)
            if not mask.any():
                continue
            blurred = apply_disc_psf(ch, r)
            out[mask] = blurred[mask]
        rendered_channels.append(out)

    if all_in_focus.ndim == 2:
        return rendered_channels[0]
    return np.stack(rendered_channels, axis=2).astype(np.float32)
