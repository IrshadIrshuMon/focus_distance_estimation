"""
align_stack.py
==============
Focal stack alignment via optical flow concatenation.

Paper: Section 4 - Focal Stack Alignment

Key idea (Section 4):
    Instead of aligning each frame directly to the reference (which fails
    when defocus is large), we compute flow between CONSECUTIVE frames and
    concatenate them:

        F^1_i = F^{i-1}_i  ∘  F^1_{i-1}

    where  ∘  is the flow concatenation operator:
        S_x = F'_x + W_{F'}(F_x)
        S_y = F'_y + W_{F'}(F_y)

This ensures consecutive frames always look similar (small defocus diff),
making optical flow reliable.

Dependencies
------------
    opencv-python  (cv2)   for optical flow (Farneback)
    numpy
"""

import numpy as np

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


# ---------------------------------------------------------------------------
# Warp helpers
# ---------------------------------------------------------------------------

def warp_image(image: np.ndarray, flow: np.ndarray) -> np.ndarray:
    """
    Warp image I according to flow field F.

    Implements Equation (1):
        W_F(I(u,v)) = I(u + F(u,v)_x,  v + F(u,v)_y)

    Parameters
    ----------
    image : (H x W) or (H x W x C) float32
    flow  : (H x W x 2) float32  [flow_x, flow_y]

    Returns
    -------
    warped : same shape as image, float32
    """
    H, W = image.shape[:2]
    # Build absolute map:  map_x[r,c] = c + flow_x[r,c]
    grid_x, grid_y = np.meshgrid(np.arange(W, dtype=np.float32),
                                  np.arange(H, dtype=np.float32))
    map_x = grid_x + flow[:, :, 0]
    map_y = grid_y + flow[:, :, 1]

    if HAS_CV2:
        if image.ndim == 2:
            warped = cv2.remap(image.astype(np.float32), map_x, map_y,
                               interpolation=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_REFLECT)
        else:
            C = image.shape[2]
            warped = np.zeros_like(image, dtype=np.float32)
            for c in range(C):
                warped[:, :, c] = cv2.remap(
                    image[:, :, c].astype(np.float32),
                    map_x, map_y,
                    interpolation=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_REFLECT)
    else:
        # Fallback: nearest-neighbour remap without cv2
        map_xi = np.clip(np.round(map_x).astype(int), 0, W - 1)
        map_yi = np.clip(np.round(map_y).astype(int), 0, H - 1)
        if image.ndim == 2:
            warped = image[map_yi, map_xi]
        else:
            warped = image[map_yi, map_xi, :]

    return warped.astype(np.float32)


def concatenate_flows(F_ab: np.ndarray, F_bc: np.ndarray) -> np.ndarray:
    """
    Concatenate two flow fields so that the result warps directly from
    frame a to frame c.

    Paper formula:
        S_x = F'_x + W_{F'}(F_x)
        S_y = F'_y + W_{F'}(F_y)

    where F' = F_ab (a→b) and F = F_bc (b→c).

    Parameters
    ----------
    F_ab : (H x W x 2) flow from a to b
    F_bc : (H x W x 2) flow from b to c

    Returns
    -------
    F_ac : (H x W x 2) flow from a to c
    """
    # Warp F_bc through F_ab to align it to frame-a coordinates
    Fx_warped = warp_image(F_bc[:, :, 0], F_ab)
    Fy_warped = warp_image(F_bc[:, :, 1], F_ab)

    F_ac = np.stack([
        F_ab[:, :, 0] + Fx_warped,
        F_ab[:, :, 1] + Fy_warped
    ], axis=2)
    return F_ac.astype(np.float32)


# ---------------------------------------------------------------------------
# Optical flow between two frames
# ---------------------------------------------------------------------------

def compute_optical_flow(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """
    Compute dense optical flow from src to dst.

    Uses cv2.calcOpticalFlowFarneback when available,
    otherwise returns a zero flow field (identity warp).

    Parameters
    ----------
    src, dst : (H x W) or (H x W x C) float32 images in [0, 1]

    Returns
    -------
    flow : (H x W x 2) float32  [flow_x, flow_y]
    """
    def to_gray_u8(img):
        if img.ndim == 3:
            gray = img.mean(axis=2)
        else:
            gray = img
        return (np.clip(gray, 0, 1) * 255).astype(np.uint8)

    H, W = src.shape[:2]

    if not HAS_CV2:
        print("  [align] cv2 not found – returning zero flow (no alignment).")
        return np.zeros((H, W, 2), dtype=np.float32)

    src_u8 = to_gray_u8(src)
    dst_u8 = to_gray_u8(dst)

    # Farneback parameters tuned close to paper's optical flow settings
    flow = cv2.calcOpticalFlowFarneback(
        src_u8, dst_u8,
        None,
        pyr_scale=0.85,   # close to paper's ratio=0.85
        levels=4,
        winsize=15,
        iterations=4,
        poly_n=5,
        poly_sigma=1.2,
        flags=0
    )
    # flow shape from cv2: (H, W, 2) where [:,:,0]=x, [:,:,1]=y
    return flow.astype(np.float32)


# ---------------------------------------------------------------------------
# Global affine pre-alignment (magnification / rolling-shutter correction)
# ---------------------------------------------------------------------------

def compute_affine_transform(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """
    Estimate a 2x3 affine matrix that aligns src to dst.

    Paper: "we compute a global affine warp … to compensate for
    magnification changes or rolling-shutter effects before computing
    the optical flow"  (Section 4)

    Returns
    -------
    M : (2 x 3) affine matrix, or identity if estimation fails
    """
    if not HAS_CV2:
        return np.eye(2, 3, dtype=np.float32)

    def to_gray_u8(img):
        if img.ndim == 3:
            gray = img.mean(axis=2)
        else:
            gray = img
        return (np.clip(gray, 0, 1) * 255).astype(np.uint8)

    H, W = src.shape[:2]
    src_u8 = to_gray_u8(src)
    dst_u8 = to_gray_u8(dst)

    try:
        warp_mode = cv2.MOTION_AFFINE
        criteria  = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                     50, 1e-4)
        M = np.eye(2, 3, dtype=np.float32)
        _, M = cv2.findTransformECC(dst_u8, src_u8, M, warp_mode, criteria)
        return M
    except cv2.error:
        return np.eye(2, 3, dtype=np.float32)


def apply_affine(image: np.ndarray, M: np.ndarray) -> np.ndarray:
    """Apply a 2x3 affine warp to an image."""
    if not HAS_CV2:
        return image.copy()
    H, W = image.shape[:2]
    if image.ndim == 2:
        return cv2.warpAffine(image.astype(np.float32), M, (W, H),
                               flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_REFLECT)
    C = image.shape[2]
    out = np.zeros_like(image, dtype=np.float32)
    for c in range(C):
        out[:, :, c] = cv2.warpAffine(image[:, :, c].astype(np.float32),
                                        M, (W, H),
                                        flags=cv2.INTER_LINEAR,
                                        borderMode=cv2.BORDER_REFLECT)
    return out


# ---------------------------------------------------------------------------
# Main alignment pipeline  –  Section 4
# ---------------------------------------------------------------------------

def align_focal_stack(focal_stack: list,
                      verbose: bool = True) -> list:
    """
    Align all frames in the focal stack to the reference frame (frame 0).

    Algorithm (Section 4):
        1. For each consecutive pair (i, i+1):
           a. Compute global affine warp for magnification correction
           b. Compute optical flow F^i_{i+1} on affine-corrected frames
        2. Concatenate flows to get F^1_i for every frame i
        3. Warp each frame using its concatenated flow

    Parameters
    ----------
    focal_stack : list of n frames (H x W x C) or (H x W), float32 [0,1]
    verbose     : print progress

    Returns
    -------
    aligned_stack : list of n aligned frames, same shapes as input
    """
    n = len(focal_stack)
    if n == 1:
        return focal_stack[:]

    H, W = focal_stack[0].shape[:2]

    if verbose:
        print(f"  [align] Aligning {n} frames to reference (frame 1) …")

    # Step 1 – compute consecutive flows F^i_{i+1}
    consecutive_flows = []          # consecutive_flows[i] = flow from i to i+1
    for i in range(n - 1):
        src = focal_stack[i]
        dst = focal_stack[i + 1]

        # 1a. Global affine pre-alignment
        M = compute_affine_transform(dst, src)
        dst_affine = apply_affine(dst, M)

        # 1b. Optical flow on affine-corrected pair
        flow_i = compute_optical_flow(src, dst_affine)   # (H, W, 2)
        consecutive_flows.append(flow_i)

        if verbose:
            mag = np.sqrt(flow_i[:,:,0]**2 + flow_i[:,:,1]**2).mean()
            print(f"    Flow {i+1}→{i+2}  mean magnitude={mag:.3f} px")

    # Step 2 – concatenate flows: F^1_i = F^{i-1}_i ∘ F^1_{i-1}
    cumulative_flows = [None]       # cumulative_flows[0] = None (reference)
    cum = np.zeros((H, W, 2), dtype=np.float32)   # identity flow at reference
    for i in range(n - 1):
        cum = concatenate_flows(cum, consecutive_flows[i])
        cumulative_flows.append(cum.copy())

    # Step 3 – warp each frame
    aligned = [focal_stack[0].copy()]               # reference is unchanged
    for i in range(1, n):
        warped = warp_image(focal_stack[i], cumulative_flows[i])
        aligned.append(warped)
        if verbose:
            print(f"    Frame {i+1} aligned.")

    if verbose:
        print(f"  [align] Done.\n")

    return aligned
