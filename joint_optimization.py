"""
joint_optimization.py
=====================
Joint optimisation for aperture A, focal length F, focal depths f_1…f_n,
and scene depth map s.

Paper: Section 6.1, Equation (10)

    min_{A, s, F, f_1,…,f_n}  Σ_i Σ_{x,y}  [ b_i(s_{x,y}) - B_i(x,y) ]²  ·  C_i(x,y)²

where  b_i(s) = A * |f_i - s| / s * F / (f_i - F)   [Equation 4]

Solved with scipy least_squares (Trust-Region Reflective ≈ bounded L-M).

Initialisation (Section 6.1 / Section 8):
    focal depths : linear span over [depth_near, depth_far]
    depth map    : constant mid-depth
    A            : 3   (paper default, Section 8)
    F            : 2   (paper default, Section 8)
    depth range  : [10, 32]   (paper default, Section 8)
"""

import numpy as np
from scipy.optimize import least_squares
from psf_model import compute_blur_radius


# ---------------------------------------------------------------------------
# Parameter packing / unpacking
# ---------------------------------------------------------------------------

def _pack(A: float, F: float,
          focal_depths: np.ndarray,
          depth_map: np.ndarray) -> np.ndarray:
    """
    Flatten [A, F, f_1,…,f_n, s_flat] into a 1-D vector.
    """
    return np.concatenate([[A, F],
                            focal_depths.ravel(),
                            depth_map.ravel()]).astype(np.float64)


def _unpack(params: np.ndarray, n: int, H: int, W: int) -> tuple:
    """
    Unpack 1-D vector back to (A, F, focal_depths, depth_map).
    """
    A            = float(params[0])
    F            = float(params[1])
    focal_depths = params[2: 2 + n]
    depth_map    = params[2 + n:].reshape(H, W)
    return A, F, focal_depths, depth_map


# ---------------------------------------------------------------------------
# Residual function  –  Equation (10)
# ---------------------------------------------------------------------------

def _residuals(params:     np.ndarray,
               blur_maps:  list,
               conf_maps:  list,
               n:          int,
               H:          int,
               W:          int) -> np.ndarray:
    """
    Weighted residuals for least_squares:

        r_{i,x,y} = ( b_i(s_{x,y}) - B_i(x,y) )  *  C_i(x,y)

    scipy minimises  Σ r²  which equals the objective in Equation (10).
    """
    A, F, focal_depths, depth_map = _unpack(params, n, H, W)

    # Hard clamp to keep physics valid
    A         = max(A, 1e-4)
    F         = max(F, 1e-4)
    depth_map = np.maximum(depth_map, 1e-4)

    res_parts = []
    for i in range(n):
        f_i   = float(focal_depths[i])
        b_pred = compute_blur_radius(A, F, f_i, depth_map)   # (H, W)
        r_i    = (b_pred - blur_maps[i]) * conf_maps[i]       # Eq. (10)
        res_parts.append(r_i.ravel())

    return np.concatenate(res_parts).astype(np.float64)


# ---------------------------------------------------------------------------
# Initialisation  –  Section 6.1 / Section 8
# ---------------------------------------------------------------------------

def _initialise(n: int, H: int, W: int,
                depth_near: float, depth_far: float,
                A_init: float, F_init: float) -> np.ndarray:
    """
    Initialise parameter vector.

    Focal depths: linear from depth_near to depth_far  (Section 6.1)
    Depth map   : constant mid-depth                    (Section 6.1)
    A, F        : paper defaults                        (Section 8)
    """
    fd_init = np.linspace(depth_near, depth_far, n)
    dm_init = np.full((H, W), 0.5 * (depth_near + depth_far))
    return _pack(A_init, F_init, fd_init, dm_init)


# ---------------------------------------------------------------------------
# Main optimisation
# ---------------------------------------------------------------------------

def joint_optimize(blur_maps:   list,
                   conf_maps:   list,
                   depth_near:  float = 10.0,
                   depth_far:   float = 32.0,
                   A_init:      float = 3.0,
                   F_init:      float = 2.0,
                   max_nfev:    int   = 500,
                   verbose:     bool  = True) -> dict:
    """
    Jointly optimise A, F, f_1…f_n, and depth map s  (Equation 10).

    Parameters
    ----------
    blur_maps   : list of n arrays (H x W) – B_i per frame
    conf_maps   : list of n arrays (H x W) – C_i per frame
    depth_near  : minimum scene depth          (paper: 10)
    depth_far   : maximum scene depth          (paper: 32)
    A_init      : initial aperture size        (paper:  3)
    F_init      : initial focal length         (paper:  2)
    max_nfev    : max function evaluations
    verbose     : print progress

    Returns
    -------
    dict with keys:
        'A'            : float  – optimised aperture
        'F'            : float  – optimised focal length
        'focal_depths' : (n,) ndarray  – optimised focal depths
        'depth_map'    : (H x W) ndarray  – optimised depth map
        'result'       : raw scipy OptimizeResult
    """
    n    = len(blur_maps)
    H, W = blur_maps[0].shape

    x0 = _initialise(n, H, W, depth_near, depth_far, A_init, F_init)

    # Bounds: all parameters > 0; depth and focal depth within scene range
    lo = np.full_like(x0, 1e-4)
    hi = np.full_like(x0, np.inf)
    lo[2: 2 + n] = depth_near * 0.5
    hi[2: 2 + n] = depth_far  * 2.0
    lo[2 + n:]   = depth_near * 0.5
    hi[2 + n:]   = depth_far  * 2.0

    if verbose:
        print(f"  [JointOpt] {n} frames, {H}×{W} image, "
              f"{len(x0)} parameters …")

    result = least_squares(
        _residuals,
        x0,
        bounds        = (lo, hi),
        method        = 'trf',          # Trust-Region Reflective (bounded L-M)
        args          = (blur_maps, conf_maps, n, H, W),
        max_nfev      = max_nfev,
        ftol          = 1e-6,
        xtol          = 1e-6,
        gtol          = 1e-6,
        verbose       = 2 if verbose else 0,
    )

    A_opt, F_opt, fd_opt, dm_opt = _unpack(result.x, n, H, W)

    if verbose:
        print(f"  [JointOpt] Converged: cost={result.cost:.6f}")
        print(f"             A={A_opt:.5f}  F={F_opt:.5f}")
        print(f"             depth range [{dm_opt.min():.3f}, "
              f"{dm_opt.max():.3f}]")

    return {
        'A':            float(A_opt),
        'F':            float(F_opt),
        'focal_depths': fd_opt.astype(np.float64),
        'depth_map':    dm_opt.astype(np.float32),
        'result':       result,
    }
