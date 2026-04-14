"""
blur_stack.py
=============
Build the pre-rendered blur stack from the all-in-focus image.

Paper: Section 6.1

"we generate a stack with blur radius increasing by 0.25 pixels
 between consecutive frames"  (Section 8)

Blur range: r = 0, 0.25, 0.50, …, 6.5  (paper Section 8)
"""

import numpy as np
from psf_model import apply_disc_psf


def build_blur_stack(all_in_focus: np.ndarray,
                     r_min:  float = 0.0,
                     r_max:  float = 6.5,
                     r_step: float = 0.25,
                     verbose: bool = False) -> tuple:
    """
    Pre-render the all-in-focus image at every blur radius level.

    Parameters
    ----------
    all_in_focus : (H x W) or (H x W x C) float32 in [0, 1]
    r_min        : minimum blur radius  (paper: 0)
    r_max        : maximum blur radius  (paper: 6.5)
    r_step       : radius increment     (paper: 0.25)
    verbose      : print progress

    Returns
    -------
    blur_stack : (N_r x H x W) or (N_r x H x W x C) float32
    radii      : (N_r,) float32 array of blur radii
    """
    radii = np.arange(r_min, r_max + r_step / 2.0, r_step, dtype=np.float32)
    N_r = len(radii)

    if all_in_focus.ndim == 2:
        H, W = all_in_focus.shape
        stack = np.zeros((N_r, H, W), dtype=np.float32)
        for k, r in enumerate(radii):
            stack[k] = apply_disc_psf(all_in_focus.astype(np.float32), float(r))
            if verbose:
                print(f"    blur r={r:.2f}  ({k+1}/{N_r})")
    else:
        H, W, C = all_in_focus.shape
        stack = np.zeros((N_r, H, W, C), dtype=np.float32)
        for k, r in enumerate(radii):
            for c in range(C):
                stack[k, :, :, c] = apply_disc_psf(
                    all_in_focus[:, :, c].astype(np.float32), float(r))
            if verbose:
                print(f"    blur r={r:.2f}  ({k+1}/{N_r})")

    return stack, radii
