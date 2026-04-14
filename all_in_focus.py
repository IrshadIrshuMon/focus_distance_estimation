"""
all_in_focus.py
===============
All-in-focus image stitching from an aligned focal stack.

Paper: Section 5, Equation (2)

    E(x) = Σ_{i∈V} E_i(x_i)  +  λ Σ_{(i,j)∈E} E_{ij}(x_i, x_j)

Unary term  E_i(x_i):
    Measures defocus — defined as the Gaussian-weighted sum of
    exp|∇I(u,v)| in a patch around the pixel.
    LOWER value = MORE in focus (used as cost).

Pairwise term  E_{ij}(x_i, x_j):
    Total variation in frame labels: |x_i - x_j|  (submodular)

Two backends:
    'greedy'  –  winner-takes-all per pixel (fast, always available)
    'mrf'     –  α-expansion via PyMaxflow  (better quality, optional)

Implementation constants (Section 8):
    λ = 0.04,   μ = 13
"""

import numpy as np
from scipy.ndimage import gaussian_filter


# ---------------------------------------------------------------------------
# Sharpness / focus measure  –  unary term E_i
# ---------------------------------------------------------------------------

def sharpness_map(frame: np.ndarray, mu: float = 13.0) -> np.ndarray:
    """
    Compute per-pixel sharpness for one focal stack frame.

    Unary cost  E_i(x_i) = - Σ_{patch} w * exp|∇I|
    (negated so that LOWER cost = sharper = better label)

    Parameters
    ----------
    frame : (H x W) or (H x W x C) float32 in [0, 1]
    mu    : Gaussian patch std   (paper: μ = 13)

    Returns
    -------
    cost : (H x W) float32, LOWER = more in-focus
    """
    if frame.ndim == 3:
        gray = frame.mean(axis=2).astype(np.float32)
    else:
        gray = frame.astype(np.float32)

    gy = np.gradient(gray, axis=0)
    gx = np.gradient(gray, axis=1)
    grad_mag = np.sqrt(gx ** 2 + gy ** 2)

    # Gaussian-weighted sum of exp|∇I| — higher = sharper
    sharpness = gaussian_filter(np.exp(grad_mag), sigma=mu, mode='reflect')

    # Convert to cost (lower = better)
    return (-sharpness).astype(np.float32)


def build_cost_volume(focal_stack: list, mu: float = 13.0) -> np.ndarray:
    """
    Build the unary cost volume for all frames.

    Returns
    -------
    cost_vol : (n_frames x H x W) float32
    """
    costs = [sharpness_map(f, mu) for f in focal_stack]
    return np.stack(costs, axis=0)   # (n, H, W)


# ---------------------------------------------------------------------------
# Greedy stitching  (fast fallback)
# ---------------------------------------------------------------------------

def stitch_greedy(focal_stack: list, cost_vol: np.ndarray) -> np.ndarray:
    """
    Winner-takes-all: pick the lowest-cost (sharpest) frame per pixel.

    Parameters
    ----------
    focal_stack : list of n (H x W x C) frames
    cost_vol    : (n x H x W) unary costs

    Returns
    -------
    aif : (H x W x C) or (H x W) float32 all-in-focus image
    """
    best_idx = np.argmin(cost_vol, axis=0)   # (H, W)
    H, W = best_idx.shape
    sample = focal_stack[0]

    if sample.ndim == 3:
        C = sample.shape[2]
        ch_stack = np.stack([f.astype(np.float32) for f in focal_stack], axis=0)
        # ch_stack : (n, H, W, C)
        aif = ch_stack[best_idx, np.arange(H)[:, None], np.arange(W)[None, :], :]
    else:
        ch_stack = np.stack([f.astype(np.float32) for f in focal_stack], axis=0)
        aif = ch_stack[best_idx, np.arange(H)[:, None], np.arange(W)[None, :]]

    return aif.astype(np.float32)


# ---------------------------------------------------------------------------
# MRF stitching with graph-cut α-expansion  (preferred)
# ---------------------------------------------------------------------------

def stitch_mrf(focal_stack: list,
               cost_vol: np.ndarray,
               lam: float = 0.04) -> np.ndarray:
    """
    MRF-based stitching using Equation (2).

    Requires PyMaxflow:  pip install PyMaxflow
    Falls back to greedy if not installed.

    Parameters
    ----------
    focal_stack : list of n frames
    cost_vol    : (n x H x W) unary costs  (lower = sharper)
    lam         : smoothness weight  (paper: λ = 0.04)

    Returns
    -------
    aif : all-in-focus image, same shape as one frame
    """
    try:
        import maxflow
    except ImportError:
        print("  [AiF] PyMaxflow not installed → using greedy stitching.")
        print("        To enable MRF:  pip install PyMaxflow")
        return stitch_greedy(focal_stack, cost_vol)

    n_frames, H, W = cost_vol.shape
    n_pixels = H * W
    SCALE = 10000   # integer scaling for maxflow

    # Normalise costs to [0, 1]
    c_min = cost_vol.min()
    c_max = cost_vol.max()
    cost_norm = (cost_vol - c_min) / (c_max - c_min + 1e-9)   # (n, H, W)

    # Initialise labels with greedy solution
    best_labels = np.argmin(cost_vol, axis=0).astype(np.int32)  # (H, W)

    # α-expansion iterations
    for iteration in range(5):
        changed = False
        for alpha in range(n_frames):
            g = maxflow.Graph[int](n_pixels, 4 * n_pixels)
            nodes = g.add_nodes(n_pixels)

            # Unary / terminal edges
            for idx in range(n_pixels):
                r, c = divmod(idx, W)
                cur = int(best_labels[r, c])
                t_source = int(cost_norm[cur,   r, c] * SCALE)  # keep current
                t_sink   = int(cost_norm[alpha,  r, c] * SCALE)  # switch to α
                g.add_tedge(nodes[idx], t_source, t_sink)

            # Pairwise edges (4-connected grid)
            for r in range(H):
                for c in range(W):
                    idx = r * W + c
                    cur = int(best_labels[r, c])
                    # right neighbour
                    if c + 1 < W:
                        nb  = int(best_labels[r, c + 1])
                        w   = int(lam * abs(cur - nb) * SCALE)
                        g.add_edge(nodes[idx], nodes[idx + 1], w, w)
                    # bottom neighbour
                    if r + 1 < H:
                        nb  = int(best_labels[r + 1, c])
                        w   = int(lam * abs(cur - nb) * SCALE)
                        g.add_edge(nodes[idx], nodes[idx + W], w, w)

            g.maxflow()

            for idx in range(n_pixels):
                r, c = divmod(idx, W)
                if g.get_segment(nodes[idx]) == 1:  # switch to alpha
                    if best_labels[r, c] != alpha:
                        best_labels[r, c] = alpha
                        changed = True

        if not changed:
            break

    # Stitch with optimised label map
    return _gather_pixels(focal_stack, best_labels)


def _gather_pixels(focal_stack: list, label_map: np.ndarray) -> np.ndarray:
    """Pick pixels from frames according to label_map."""
    H, W = label_map.shape
    sample = focal_stack[0]
    if sample.ndim == 3:
        ch_stack = np.stack([f.astype(np.float32) for f in focal_stack], axis=0)
        aif = ch_stack[label_map,
                       np.arange(H)[:, None],
                       np.arange(W)[None, :], :]
    else:
        ch_stack = np.stack([f.astype(np.float32) for f in focal_stack], axis=0)
        aif = ch_stack[label_map,
                       np.arange(H)[:, None],
                       np.arange(W)[None, :]]
    return aif.astype(np.float32)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def compute_all_in_focus(aligned_stack: list,
                          mu:      float = 13.0,
                          lam:     float = 0.04,
                          method:  str   = 'greedy',
                          verbose: bool  = True) -> np.ndarray:
    """
    Compute all-in-focus image from an aligned focal stack.

    Parameters
    ----------
    aligned_stack : list of aligned frames (H x W x C) or (H x W),
                    float32 in [0, 1]
    mu            : Gaussian patch std for sharpness   (paper: 13)
    lam           : MRF smoothness weight              (paper: 0.04)
    method        : 'greedy' (fast) or 'mrf' (better, needs PyMaxflow)
    verbose       : print progress

    Returns
    -------
    aif : (H x W x C) or (H x W) float32 all-in-focus image
    """
    if verbose:
        print(f"  [AiF] Building cost volume (μ={mu}) …")

    cost_vol = build_cost_volume(aligned_stack, mu=mu)  # (n, H, W)

    if verbose:
        print(f"  [AiF] Stitching ({method}) …")

    if method == 'mrf':
        aif = stitch_mrf(aligned_stack, cost_vol, lam=lam)
    else:
        aif = stitch_greedy(aligned_stack, cost_vol)

    if verbose:
        print(f"  [AiF] Done. Shape={aif.shape}  "
              f"range=[{aif.min():.3f}, {aif.max():.3f}]")

    return aif
