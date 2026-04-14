"""
depth_refinement.py
===================
Depth map refinement using anisotropic Huber-norm regularisation.

Paper: Section 6.2, Equations (11)–(15)

Data term (11):
    U(x, Q(x)) = (1/n) Σ_i  D_i(x, b_i(Q(x)))

Energy functional (12):
    E_Q = ∫_Ω  λ·U(x,Q(x))  +  ‖T(x)∇Q(x)‖_ε  dx

Diffusion tensor T(x)  (Section 6.2):
    T(x) = exp(-α|∇I|^β) n⃗n⃗ᵀ + n⃗⊥n⃗⊥ᵀ

Huber norm (13):
    ‖z‖_ε = { z²/(2ε)      if |z| ≤ ε
             { |z| - ε/2   otherwise

Decoupled alternating minimisation (14)–(15):
    E_A = ∫ λ·U(x,A(x)) + (1/2θ)(A(x)−Q(x))² dx   → data step
    E_Q = ∫ (1/2θ)(A(x)−Q(x))² + ‖T(x)∇Q(x)‖_ε dx → TV step

θ schedule (Section 8):
    θ_0 = 2,   θ_{k+1} = θ_k*(1 − 0.01*k)   until θ ≤ 0.005

Implementation constants (Section 8):
    32 inverse-depth bins,  λ=0.001,  α=20,  β=1,  ε=0.005
"""

import numpy as np
from psf_model import compute_blur_radius


# ---------------------------------------------------------------------------
# Huber norm  –  Equation (13)
# ---------------------------------------------------------------------------

def huber(z: np.ndarray, eps: float = 0.005) -> np.ndarray:
    """
    Element-wise Huber norm ‖z‖_ε.

    ‖z‖_ε = z²/(2ε)    if |z| ≤ ε
           = |z| - ε/2  otherwise
    """
    az = np.abs(z)
    return np.where(az <= eps,
                    az ** 2 / (2.0 * eps),
                    az - eps / 2.0)


# ---------------------------------------------------------------------------
# Anisotropic diffusion tensor  –  Section 6.2
# ---------------------------------------------------------------------------

def diffusion_tensor(reference_image: np.ndarray,
                     alpha: float = 20.0,
                     beta:  float = 1.0) -> tuple:
    """
    Compute per-pixel 2×2 diffusion tensor T(x).

        T(x) = exp(-α|∇I|^β) n⃗n⃗ᵀ + n⃗⊥n⃗⊥ᵀ

    where  n⃗ = ∇I/|∇I|   and   n⃗⊥ ⊥ n⃗.

    Parameters
    ----------
    reference_image : (H x W) or (H x W x C) float32
    alpha, beta     : tensor constants  (paper: α=20, β=1)

    Returns
    -------
    T_xx, T_xy, T_yx, T_yy : four (H x W) float32 tensor components
    """
    if reference_image.ndim == 3:
        I = reference_image.mean(axis=2).astype(np.float32)
    else:
        I = reference_image.astype(np.float32)

    Iy, Ix = np.gradient(I)
    grad_mag = np.sqrt(Ix ** 2 + Iy ** 2) + 1e-9

    w  = np.exp(-alpha * (grad_mag ** beta))   # scalar weight

    # Unit vectors
    nx = Ix / grad_mag   # gradient direction
    ny = Iy / grad_mag
    # perpendicular: n⃗⊥ = (-ny, nx)

    # T = w * n⃗n⃗ᵀ + n⃗⊥n⃗⊥ᵀ
    # n⃗⊥n⃗⊥ᵀ = [[-ny], [nx]] * [[-ny, nx]] = [[ny², -nx*ny], [-nx*ny, nx²]]
    T_xx = w * nx * nx + ny * ny
    T_xy = w * nx * ny - nx * ny
    T_yx = T_xy.copy()
    T_yy = w * ny * ny + nx * nx

    return (T_xx.astype(np.float32),
            T_xy.astype(np.float32),
            T_yx.astype(np.float32),
            T_yy.astype(np.float32))


# ---------------------------------------------------------------------------
# Data term  –  Equation (11)
# ---------------------------------------------------------------------------

def data_term(Q:            np.ndarray,
              diff_maps:    list,
              radii:        np.ndarray,
              A:            float,
              F:            float,
              focal_depths: np.ndarray) -> np.ndarray:
    """
    Compute U(x, Q(x)) = (1/n) Σ_i D_i(x, b_i(Q(x)))

    Q is the INVERSE depth map (Q = 1/s).

    Parameters
    ----------
    Q            : (H x W) inverse depth
    diff_maps    : list of D_i arrays (N_r x H x W)
    radii        : (N_r,) blur radius values
    A, F         : camera parameters
    focal_depths : (n,) focal depth per frame

    Returns
    -------
    U : (H x W) float32 data term
    """
    n    = len(diff_maps)
    H, W = Q.shape
    U    = np.zeros((H, W), dtype=np.float32)
    r_step = float(radii[1] - radii[0]) if len(radii) > 1 else 0.25

    for i, D_i in enumerate(diff_maps):
        s     = 1.0 / np.maximum(Q, 1e-6)              # scene depth
        b     = compute_blur_radius(A, F, float(focal_depths[i]), s)
        b     = np.clip(b, radii[0], radii[-1])
        idx   = np.round((b - radii[0]) / r_step).astype(np.int32)
        idx   = np.clip(idx, 0, len(radii) - 1)
        # Lookup D_i at predicted blur radius per pixel
        rows  = np.arange(H, dtype=np.int32)
        cols  = np.arange(W, dtype=np.int32)
        U_i   = D_i[idx, rows[:, None], cols[None, :]]
        U    += U_i

    return (U / n).astype(np.float32)


# ---------------------------------------------------------------------------
# Data step  –  Equation (14)  (point-wise search + Newton step)
# ---------------------------------------------------------------------------

def data_step(Q:            np.ndarray,
              diff_maps:    list,
              radii:        np.ndarray,
              A:            float,
              F:            float,
              focal_depths: np.ndarray,
              theta:        float,
              lam:          float = 0.001,
              n_bins:       int   = 32) -> np.ndarray:
    """
    Minimise E_A  (Equation 14) w.r.t. A(x) via point-wise search
    over quantised inverse-depth labels, then one Newton step.

    Parameters
    ----------
    Q            : current inverse depth map (H x W)
    diff_maps    : list of D_i (N_r x H x W)
    radii        : (N_r,) blur radii
    A, F         : camera parameters (fixed in refinement)
    focal_depths : (n,) fixed in refinement
    theta        : current decoupling constant
    lam          : data term weight  (paper: 0.001)
    n_bins       : number of inverse-depth labels  (paper: 32)

    Returns
    -------
    A_aux : (H x W) auxiliary variable
    """
    H, W   = Q.shape
    q_min  = float(Q.min())
    q_max  = float(Q.max())

    # Guard against degenerate range
    if q_max - q_min < 1e-9:
        return Q.copy()

    labels = np.linspace(q_min, q_max, n_bins, dtype=np.float64)
    delta  = float(labels[1] - labels[0]) if n_bins > 1 else 1e-4

    best_E     = np.full((H, W), np.inf, dtype=np.float64)
    best_label = Q.copy().astype(np.float64)

    for label in labels:
        Q_label = np.full((H, W), label, dtype=np.float32)
        U       = data_term(Q_label, diff_maps, radii, A, F, focal_depths)
        E       = lam * U.astype(np.float64) + \
                  (1.0 / (2.0 * theta)) * (label - Q.astype(np.float64)) ** 2
        mask = E < best_E
        best_E[mask]     = E[mask]
        best_label[mask] = label

    # Single Newton step for sub-label accuracy  (Section 6.2)
    def energy_at(q_val: float) -> np.ndarray:
        Qv = np.full((H, W), q_val, dtype=np.float32)
        U  = data_term(Qv, diff_maps, radii, A, F, focal_depths)
        return lam * U.astype(np.float64) + \
               (1.0 / (2.0 * theta)) * (q_val - Q.astype(np.float64)) ** 2

    E_plus  = energy_at(np.clip(best_label + delta, q_min, q_max).mean())
    E_minus = energy_at(np.clip(best_label - delta, q_min, q_max).mean())
    E_0     = best_E

    f1 = (E_plus - E_minus) / (2.0 * delta)
    f2 = (E_plus - 2.0 * E_0 + E_minus) / (delta ** 2)
    f2 = np.where(np.abs(f2) < 1e-12, 1e-12, f2)

    A_aux = best_label - f1 / f2
    A_aux = np.clip(A_aux, q_min, q_max)
    return A_aux.astype(np.float32)


# ---------------------------------------------------------------------------
# TV / regularisation step  –  Equation (15)
# ---------------------------------------------------------------------------

def tv_step(Q:     np.ndarray,
            A_aux: np.ndarray,
            T_xx:  np.ndarray,
            T_xy:  np.ndarray,
            T_yx:  np.ndarray,
            T_yy:  np.ndarray,
            theta: float,
            eps:   float = 0.005,
            iters: int   = 30) -> np.ndarray:
    """
    Minimise E_Q  (Equation 15) w.r.t. Q using gradient descent.

    E_Q = ∫ (1/2θ)(A-Q)² + ‖T(x)∇Q‖_ε dx

    Uses a proximal gradient step.  The paper uses the primal-dual
    algorithm of Chambolle & Pock (2011).

    Parameters
    ----------
    Q, A_aux      : (H x W) float32
    T_xx..T_yy    : (H x W) tensor components
    theta         : decoupling constant
    eps           : Huber constant  (paper: 0.005)
    iters         : number of gradient descent iterations

    Returns
    -------
    Q_new : (H x W) float32 updated inverse depth
    """
    Q_new = Q.copy().astype(np.float64)
    A_d   = A_aux.astype(np.float64)
    dt    = 0.05 / (1.0 / (2.0 * theta) + 4.0 + 1e-9)

    for _ in range(iters):
        Qy, Qx = np.gradient(Q_new)

        # Anisotropic: T * ∇Q
        TQx = T_xx.astype(np.float64) * Qx + T_xy.astype(np.float64) * Qy
        TQy = T_yx.astype(np.float64) * Qx + T_yy.astype(np.float64) * Qy

        mag = np.sqrt(TQx ** 2 + TQy ** 2) + 1e-9

        # Huber shrinkage
        shrink = np.where(mag <= eps, 1.0 / eps, 1.0 / mag)

        _, div_x = np.gradient(shrink * TQx)
        div_y, _ = np.gradient(shrink * TQy)
        div      = div_x + div_y

        data_grad = (Q_new - A_d) / theta
        Q_new     = Q_new - dt * (data_grad - div)

    return Q_new.astype(np.float32)


# ---------------------------------------------------------------------------
# Full refinement pipeline  –  Section 6.2
# ---------------------------------------------------------------------------

def refine_depth_map(coarse_depth:    np.ndarray,
                     reference_image: np.ndarray,
                     diff_maps:       list,
                     radii:           np.ndarray,
                     A:               float,
                     F:               float,
                     focal_depths:    np.ndarray,
                     lam:       float = 0.001,
                     alpha_t:   float = 20.0,
                     beta_t:    float = 1.0,
                     eps:       float = 0.005,
                     theta_0:   float = 2.0,
                     theta_min: float = 0.005,
                     n_bins:    int   = 32,
                     verbose:   bool  = True) -> np.ndarray:
    """
    Refine the coarse depth map using anisotropic Huber-norm regularisation.

    Implements alternating minimisation of Equations (14) and (15)
    with the θ schedule from Section 8.

    Parameters
    ----------
    coarse_depth    : (H x W) initial depth map from joint optimisation
    reference_image : reference frame for tensor computation
    diff_maps       : list of D_i (N_r x H x W each)
    radii           : (N_r,) blur radius array
    A, F            : camera parameters  (fixed during refinement)
    focal_depths    : (n,) focal depths  (fixed during refinement)
    lam             : data weight   (paper: 0.001)
    alpha_t, beta_t : tensor constants  (paper: α=20, β=1)
    eps             : Huber constant    (paper: 0.005)
    theta_0         : initial θ         (paper: 2.0)
    theta_min       : stop when θ ≤ θ_min  (paper: 0.005)
    n_bins          : inverse-depth quantisation bins  (paper: 32)
    verbose         : print progress

    Returns
    -------
    depth_refined : (H x W) float32 refined SCENE depth (not inverse depth)
    """
    # Work in inverse depth space  (paper optimises inverse depth Q = 1/s)
    Q = (1.0 / np.maximum(coarse_depth, 1e-6)).astype(np.float32)

    # Precompute diffusion tensor (fixed – depends only on reference image)
    T_xx, T_xy, T_yx, T_yy = diffusion_tensor(
        reference_image, alpha=alpha_t, beta=beta_t)

    theta = float(theta_0)
    k     = 0

    while theta > theta_min:
        # --- data step: Equation (14) ---
        A_aux = data_step(Q, diff_maps, radii, A, F,
                          focal_depths, theta, lam=lam, n_bins=n_bins)

        # --- TV step: Equation (15) ---
        Q = tv_step(Q, A_aux, T_xx, T_xy, T_yx, T_yy, theta, eps=eps)

        # --- θ schedule: Section 8 ---
        theta = theta * (1.0 - 0.01 * k)
        k    += 1

        if verbose and k % 10 == 0:
            print(f"    iter={k:3d}  θ={theta:.5f}  "
                  f"Q=[{Q.min():.4f}, {Q.max():.4f}]")

    if verbose:
        print(f"  [Refine] Converged after {k} iterations.")

    # Convert inverse depth back to scene depth
    depth_refined = 1.0 / np.maximum(Q, 1e-6)
    return depth_refined.astype(np.float32)
