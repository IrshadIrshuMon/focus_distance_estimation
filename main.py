"""
main.py
=======
Depth-from-Focus Calibration Pipeline.

Paper: "Depth from Focus with Your Mobile Phone"
       Suwajanakorn, Hernandez, Seitz – CVPR 2015

ONLY INPUT NEEDED:
    --focal_stack_dir   folder of aligned focal stack images (PNG/JPG)

Pipeline:
    Step 1  align_stack.py        Align frames (Section 4)
    Step 2  all_in_focus.py       Compute all-in-focus image (Section 5, Eq.2)
    Step 3  blur_stack.py         Build pre-rendered blur stack (Section 6.1)
    Step 4  difference_map.py     D_i, B_i, C_i per frame (Eq. 7,8,9)
    Step 5  joint_optimization.py Joint optimise A,F,f_i,depth (Eq. 10)
    Step 6  depth_refinement.py   Anisotropic refinement (Eq. 11-15)
    Step 7  Save outputs + print calibration parameters

Usage:
    python main.py --focal_stack_dir focal_stack/

    Optional:
        --output_dir   results/
        --depth_near   10        (paper default)
        --depth_far    32        (paper default)
        --A_init       3         (paper default)
        --F_init       2         (paper default)
        --lam          0.001     (paper default)
        --aif_method   greedy    ('greedy' or 'mrf')
        --no_align               skip alignment (if frames already aligned)

Folder format:
    focal_stack/
        frame_01.png
        frame_02.png
        ...
    (sorted alphabetically, all same resolution, PNG or JPG)

Outputs saved to --output_dir:
    all_in_focus.png            computed all-in-focus image
    depth_coarse.npy / .png     depth from joint optimisation
    depth_refined.npy / .png    depth after anisotropic refinement
    calibration_parameters.txt  A, F, f_1…f_n and depth stats
    summary.png                 side-by-side figure
"""

import argparse
import os
import glob
import sys
import numpy as np

# ── optional display libraries ───────────────────────────────────────────────
try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_PLT = True
except ImportError:
    HAS_PLT = False

# ── pipeline modules ─────────────────────────────────────────────────────────
from align_stack       import align_focal_stack
from all_in_focus      import compute_all_in_focus
from blur_stack        import build_blur_stack
from difference_map    import process_frame
from joint_optimization import joint_optimize
from depth_refinement  import refine_depth_map


# =============================================================================
# I/O helpers
# =============================================================================

def load_image(path: str) -> np.ndarray:
    """Load image as float32 in [0, 1], RGB."""
    if HAS_CV2:
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Cannot read: {path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    else:
        try:
            from PIL import Image as PILImage
            img = np.array(PILImage.open(path).convert('RGB'),
                           dtype=np.float32) / 255.0
        except ImportError:
            sys.exit("ERROR: Install opencv-python or Pillow to load images.")
    return img


def load_focal_stack(directory: str) -> list:
    """
    Load all images in a folder as the aligned focal stack.
    Files are sorted alphabetically – name them frame_01.png, frame_02.png …
    """
    exts = ('*.png', '*.jpg', '*.jpeg', '*.tif', '*.tiff')
    paths = []
    for ext in exts:
        paths.extend(glob.glob(os.path.join(directory, ext)))
    paths = sorted(paths)

    if len(paths) == 0:
        sys.exit(f"ERROR: No images found in '{directory}'.\n"
                 f"       Supported: PNG, JPG, JPEG, TIF, TIFF\n"
                 f"       Name files so they sort correctly:\n"
                 f"         frame_01.png, frame_02.png, …")

    print(f"[Load] {len(paths)} frames found in '{directory}':")
    for p in paths:
        print(f"         {os.path.basename(p)}")

    stack = []
    for p in paths:
        img = load_image(p)
        stack.append(img)

    H, W = stack[0].shape[:2]
    print(f"[Load] Image resolution: {H} × {W}\n")

    # Sanity check: all frames same size
    for i, f in enumerate(stack):
        if f.shape[:2] != (H, W):
            sys.exit(f"ERROR: Frame {i+1} has different size "
                     f"{f.shape[:2]} vs expected ({H},{W}).")
    return stack


def save_image(img: np.ndarray, path: str):
    """Save float32 [0,1] image as PNG."""
    img_u8 = np.clip(img * 255, 0, 255).astype(np.uint8)
    if HAS_CV2:
        cv2.imwrite(path, cv2.cvtColor(img_u8, cv2.COLOR_RGB2BGR))
    else:
        from PIL import Image as PILImage
        PILImage.fromarray(img_u8).save(path)


def save_depth(depth: np.ndarray, name: str, output_dir: str):
    """
    Save depth map as:
        <name>.npy  – raw float32 values (load with np.load)
        <name>.png  – normalised 8-bit for visual inspection
    """
    npy_path = os.path.join(output_dir, f'{name}.npy')
    png_path = os.path.join(output_dir, f'{name}.png')

    np.save(npy_path, depth.astype(np.float32))

    d_norm = (depth - depth.min()) / (depth.max() - depth.min() + 1e-9)
    d_u8   = (d_norm * 255).astype(np.uint8)

    if HAS_CV2:
        cv2.imwrite(png_path, d_u8)
    else:
        from PIL import Image as PILImage
        PILImage.fromarray(d_u8).save(png_path)

    print(f"    {npy_path}  (raw float32)")
    print(f"    {png_path}  (8-bit visual)")


def save_calibration_txt(A:              float,
                          F:              float,
                          focal_depths:   np.ndarray,
                          depth_coarse:   np.ndarray,
                          depth_refined:  np.ndarray,
                          output_dir:     str):
    """Write all recovered calibration parameters to a text file."""
    path = os.path.join(output_dir, 'calibration_parameters.txt')
    lines = [
        "DfF Calibration Parameters",
        "Suwajanakorn, Hernandez, Seitz - CVPR 2015",
        "=" * 52,
        "",
        f"Aperture size   A  =  {A:.8f}",
        f"Focal length    F  =  {F:.8f}",
        "",
        "Focal depth of each frame:",
    ]
    for i, fd in enumerate(focal_depths):
        lines.append(f"  Frame {i+1:02d}:  f_{i+1} = {fd:.8f}")

    lines += [
        "",
        "Coarse depth map (from Joint Optimisation, Eq. 10):",
        f"  min  = {depth_coarse.min():.6f}",
        f"  max  = {depth_coarse.max():.6f}",
        f"  mean = {depth_coarse.mean():.6f}",
        "",
        "Refined depth map (after Anisotropic Regularisation, Eq. 11-15):",
        f"  min  = {depth_refined.min():.6f}",
        f"  max  = {depth_refined.max():.6f}",
        f"  mean = {depth_refined.mean():.6f}",
        "",
        "NOTE: A and F are recovered up to an affine ambiguity in inverse",
        "depth (Section 6). Relative depth ordering is geometrically correct.",
    ]

    with open(path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f"    {path}")


def print_calibration(A, F, focal_depths, depth_coarse, depth_refined):
    """Pretty-print calibration results to the console."""
    sep = "=" * 58
    print(f"\n{sep}")
    print("  CALIBRATION RESULTS")
    print(sep)
    print(f"  Aperture size   A  =  {A:.6f}")
    print(f"  Focal length    F  =  {F:.6f}")
    print()
    print("  Focal depth per frame:")
    for i, fd in enumerate(focal_depths):
        print(f"    Frame {i+1:02d}:  f_{i+1} = {fd:.6f}")
    print()
    print(f"  Coarse depth :  min={depth_coarse.min():.4f}  "
          f"max={depth_coarse.max():.4f}  mean={depth_coarse.mean():.4f}")
    print(f"  Refined depth:  min={depth_refined.min():.4f}  "
          f"max={depth_refined.max():.4f}  mean={depth_refined.mean():.4f}")
    print()
    print("  NOTE: A and F are up to an affine ambiguity in inverse")
    print("  depth.  Relative scene structure is correct.")
    print(f"{sep}\n")


def save_summary_figure(aif, depth_coarse, depth_refined, output_dir):
    """Save a side-by-side PNG summary."""
    if not HAS_PLT:
        return
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].imshow(np.clip(aif, 0, 1))
    axes[0].set_title('All-in-Focus\n(Section 5, Eq. 2)', fontsize=11)
    axes[0].axis('off')

    im1 = axes[1].imshow(depth_coarse, cmap='plasma')
    axes[1].set_title('Coarse Depth Map\n(Joint Opt., Eq. 10)', fontsize=11)
    axes[1].axis('off')
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    im2 = axes[2].imshow(depth_refined, cmap='plasma')
    axes[2].set_title('Refined Depth Map\n(Anisotropic Reg., Eq. 11-15)',
                       fontsize=11)
    axes[2].axis('off')
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    plt.tight_layout()
    out = os.path.join(output_dir, 'summary.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    {out}")


# =============================================================================
# Synthetic demo  (runs when no --focal_stack_dir is given)
# =============================================================================

def make_synthetic_stack(n_frames: int = 6,
                          H: int = 48,
                          W: int = 64) -> list:
    """
    Create a small synthetic focal stack for testing.
    Returns list of (H x W x 3) float32 frames in [0, 1].
    """
    from psf_model import apply_disc_psf, compute_blur_radius

    print("[Demo] Building synthetic focal stack …")
    rng = np.random.default_rng(0)

    gt_depth = np.full((H, W), 25.0, dtype=np.float32)
    gt_depth[:H // 2, :] = 12.0      # top half closer

    aif_texture = rng.uniform(0.2, 0.8, (H, W, 3)).astype(np.float32)

    A_gt, F_gt = 3.0, 2.0
    focal_depths_gt = np.linspace(10.0, 32.0, n_frames)

    stack = []
    for idx, f_i in enumerate(focal_depths_gt):
        frame = np.zeros_like(aif_texture)
        b_map = compute_blur_radius(A_gt, F_gt, f_i, gt_depth)
        for c in range(3):
            for r_val in np.unique(np.round(b_map * 4) / 4.0):
                mask    = np.abs(b_map - r_val) < 0.13
                blurred = apply_disc_psf(aif_texture[:, :, c], float(r_val))
                frame[:, :, c][mask] = blurred[mask]
        stack.append(frame)
        print(f"  frame {idx+1}/{n_frames}  f_i={f_i:.1f}")

    print()
    return stack


# =============================================================================
# Main pipeline
# =============================================================================

def run_pipeline(focal_stack: list,
                 output_dir:  str,
                 depth_near:  float = 10.0,
                 depth_far:   float = 32.0,
                 A_init:      float = 3.0,
                 F_init:      float = 2.0,
                 lam:         float = 0.001,
                 aif_method:  str   = 'greedy',
                 do_align:    bool  = True,
                 verbose:     bool  = True) -> dict:
    """
    Run the full DfF calibration pipeline.

    Parameters
    ----------
    focal_stack : list of (H x W x C) float32 frames
    output_dir  : folder to write all results
    depth_near  : closest scene depth    (paper: 10)
    depth_far   : farthest scene depth   (paper: 32)
    A_init      : initial aperture       (paper:  3)
    F_init      : initial focal length   (paper:  2)
    lam         : refinement data weight (paper:  0.001)
    aif_method  : 'greedy' or 'mrf'
    do_align    : run focal stack alignment (Step 1)
    verbose     : detailed console output

    Returns
    -------
    dict: A, F, focal_depths, depth_coarse, depth_refined, aif
    """
    os.makedirs(output_dir, exist_ok=True)

    n = len(focal_stack)
    H, W = focal_stack[0].shape[:2]

    print(f"\n{'='*58}")
    print(f"  DfF Calibration Pipeline")
    print(f"  Frames: {n}   Resolution: {H}×{W}")
    print(f"{'='*58}\n")

    # ------------------------------------------------------------------
    # Step 1 – Focal stack alignment  (Section 4)
    # ------------------------------------------------------------------
    print("[Step 1] Focal stack alignment (Section 4) …")
    if do_align:
        aligned = align_focal_stack(focal_stack, verbose=verbose)
    else:
        print("  (skipped – using frames as provided)\n")
        aligned = focal_stack
    print()

    # ------------------------------------------------------------------
    # Step 2 – All-in-focus image  (Section 5, Equation 2)
    # ------------------------------------------------------------------
    print("[Step 2] All-in-focus stitching (Section 5, Eq. 2) …")
    aif = compute_all_in_focus(aligned, mu=13.0, lam=0.04,
                                method=aif_method, verbose=verbose)
    aif_path = os.path.join(output_dir, 'all_in_focus.png')
    save_image(aif, aif_path)
    print(f"  Saved: {aif_path}\n")

    # ------------------------------------------------------------------
    # Step 3 – Blur stack  (Section 6.1)
    # ------------------------------------------------------------------
    print("[Step 3] Building blur stack (Section 6.1) …")
    blur_stk, radii = build_blur_stack(aif, r_min=0.0, r_max=6.5,
                                        r_step=0.25, verbose=False)
    print(f"  Shape: {blur_stk.shape}   "
          f"Radii: {radii[0]:.2f} … {radii[-1]:.2f}\n")

    # ------------------------------------------------------------------
    # Step 4 – Difference maps, blur maps, confidence maps (Eq. 7, 8, 9)
    # ------------------------------------------------------------------
    print("[Step 4] Computing D_i (Eq.7), B_i (Eq.8), C_i (Eq.9) …")
    diff_maps, blur_maps, conf_maps = [], [], []
    for i, frame in enumerate(aligned):
        D_i, B_i, C_i = process_frame(frame, blur_stk, radii,
                                        mu=15.0, delta_i=1.0, alpha=2.0)
        diff_maps.append(D_i)
        blur_maps.append(B_i)
        conf_maps.append(C_i)
        if verbose:
            print(f"  Frame {i+1:02d}/{n}  "
                  f"B=[{B_i.min():.3f},{B_i.max():.3f}]  "
                  f"C=[{C_i.min():.3f},{C_i.max():.3f}]")
    print()

    # ------------------------------------------------------------------
    # Step 5 – Joint optimisation  (Equation 10)
    #           → recovers A, F, f_1…f_n, coarse depth map
    # ------------------------------------------------------------------
    print("[Step 5] Joint optimisation – A, F, focal depths, "
          "depth map (Eq. 10) …")
    opt = joint_optimize(blur_maps, conf_maps,
                          depth_near=depth_near,
                          depth_far=depth_far,
                          A_init=A_init,
                          F_init=F_init,
                          verbose=verbose)

    A_opt        = opt['A']
    F_opt        = opt['F']
    focal_depths = opt['focal_depths']
    depth_coarse = opt['depth_map']
    print()

    print("  Saving coarse depth map …")
    save_depth(depth_coarse, 'depth_coarse', output_dir)
    print()

    # ------------------------------------------------------------------
    # Step 6 – Depth map refinement  (Equations 11–15)
    # ------------------------------------------------------------------
    print("[Step 6] Depth map refinement (Eq. 11-15) …")
    depth_refined = refine_depth_map(
        coarse_depth    = depth_coarse,
        reference_image = aligned[0],
        diff_maps       = diff_maps,
        radii           = radii,
        A               = A_opt,
        F               = F_opt,
        focal_depths    = focal_depths,
        lam             = lam,
        alpha_t         = 20.0,
        beta_t          = 1.0,
        eps             = 0.005,
        theta_0         = 2.0,
        theta_min       = 0.005,
        n_bins          = 32,
        verbose         = verbose,
    )
    print()

    print("  Saving refined depth map …")
    save_depth(depth_refined, 'depth_refined', output_dir)
    print()

    # ------------------------------------------------------------------
    # Step 7 – Save calibration parameters + figures
    # ------------------------------------------------------------------
    print("[Step 7] Saving calibration parameters …")
    save_calibration_txt(A_opt, F_opt, focal_depths,
                          depth_coarse, depth_refined, output_dir)

    print("\n[Step 8] Saving summary figure …")
    save_summary_figure(aif, depth_coarse, depth_refined, output_dir)

    print_calibration(A_opt, F_opt, focal_depths, depth_coarse, depth_refined)

    print(f"{'='*58}")
    print(f"  All outputs written to: {output_dir}/")
    print(f"{'='*58}\n")

    return dict(A=A_opt, F=F_opt, focal_depths=focal_depths,
                depth_coarse=depth_coarse, depth_refined=depth_refined,
                aif=aif)


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description='DfF Calibration – Suwajanakorn et al. CVPR 2015\n'
                    'Input: a folder of focal stack frames. That is all.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            'Example:\n'
            '  python main.py --focal_stack_dir focal_stack/ --output_dir results/\n\n'
            'Focal stack folder format:\n'
            '  focal_stack/\n'
            '      frame_01.png\n'
            '      frame_02.png\n'
            '      ...\n'
            '  (sorted alphabetically, all same resolution)'
        )
    )
    p.add_argument('--focal_stack_dir', type=str, default=None,
                   help='Folder containing focal stack frames (PNG/JPG)')
    p.add_argument('--output_dir',  type=str,   default='results',
                   help='Output folder (default: results/)')
    p.add_argument('--depth_near',  type=float, default=10.0,
                   help='Closest scene depth (default: 10)')
    p.add_argument('--depth_far',   type=float, default=32.0,
                   help='Farthest scene depth (default: 32)')
    p.add_argument('--A_init',      type=float, default=3.0,
                   help='Initial aperture (default: 3)')
    p.add_argument('--F_init',      type=float, default=2.0,
                   help='Initial focal length (default: 2)')
    p.add_argument('--lam',         type=float, default=0.001,
                   help='Refinement data weight (default: 0.001)')
    p.add_argument('--aif_method',  type=str,   default='greedy',
                   choices=['greedy', 'mrf'],
                   help="All-in-focus method: 'greedy' (default) or 'mrf'")
    p.add_argument('--no_align',    action='store_true', default=False,
                   help='Skip alignment step (if frames are already aligned)')
    p.add_argument('--verbose',     action='store_true', default=True)
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()

    if args.focal_stack_dir is None:
        print("[Info] No --focal_stack_dir provided → running synthetic demo.\n")
        focal_stack = make_synthetic_stack(n_frames=6, H=32, W=48)
    else:
        focal_stack = load_focal_stack(args.focal_stack_dir)

    run_pipeline(
        focal_stack = focal_stack,
        output_dir  = args.output_dir,
        depth_near  = args.depth_near,
        depth_far   = args.depth_far,
        A_init      = args.A_init,
        F_init      = args.F_init,
        lam         = args.lam,
        aif_method  = args.aif_method,
        do_align    = not args.no_align,
        verbose     = args.verbose,
    )
