"""Numba CPU finite differences for the modified 3D publisher-subscriber blocks.
The default output contains v_3(t,x0,s,s) for the paper's partial-diagonal
comparisons. --full-reference saves the central three-dimensional crop.
"""

#%%
import argparse
import os

os.environ.setdefault("NUMBA_NUM_THREADS", os.environ.get("HJ_NUMBA_THREADS", "8"))

from pathlib import Path

import numpy as np
from numba import njit, prange

DATA_DIR = Path(__file__).resolve().parent / "data"


@njit(fastmath=True)
def _apply_bc3d_neumann_inplace(u):
    Nx, Ny, Nz = u.shape
    u[0, :, :] = u[1, :, :]
    u[Nx - 1, :, :] = u[Nx - 2, :, :]
    u[:, 0, :] = u[:, 1, :]
    u[:, Ny - 1, :] = u[:, Ny - 2, :]
    u[:, :, 0] = u[:, :, 1]
    u[:, :, Nz - 1] = u[:, :, Nz - 2]


@njit(parallel=True, fastmath=True)
def _step_one_hji3d(v_prev, X1, X2, X3, dx, dt, a, b, c, alpha, beta, s11, s22, s33, s12, s13, s23, theta=np.pi / 4, bc_neumann=True):
    Nx, Ny, Nz = v_prev.shape
    v_new = v_prev.copy()

    inv2dx = 1.0 / (2.0 * dx)
    invdx2 = 1.0 / (dx * dx)
    inv4dx2 = 1.0 / (4.0 * dx * dx)
    R11, R12 = np.cos(theta), -np.sin(theta)
    R21, R22 = np.sin(theta), np.cos(theta)
    for i in prange(1, Nx - 1):
        for j in range(1, Ny - 1):
            for k in range(1, Nz - 1):
                dv_dx = (v_prev[i + 1, j, k] - v_prev[i - 1, j, k]) * inv2dx
                dv_dy = (v_prev[i, j + 1, k] - v_prev[i, j - 1, k]) * inv2dx
                dv_dz = (v_prev[i, j, k + 1] - v_prev[i, j, k - 1]) * inv2dx

                x1, x2, x3 = X1[i], X2[j], X3[k]
                Ax1, Ax2, Ax3 = a * x1, -x1 + a * x2, -x1 + a * x3
                phi1 = alpha * np.sin(x1) * (x1 * x1)
                phi2 = -beta * x1 * (x2 * x2)
                phi3 = -beta * x1 * (x3 * x3)
                f1, f2, f3 = Ax1 + phi1, Ax2 + phi2, Ax3 + phi3

                drift = dv_dx * f1 + dv_dy * f2 + dv_dz * f3
                term1 = -b * (np.abs(dv_dy) + np.abs(dv_dz))
                q1 = R11 * dv_dy + R21 * dv_dz
                q2 = R12 * dv_dy + R22 * dv_dz
                term2 = c * (np.abs(q1) + np.abs(q2))
                H = drift + term1 + term2

                vc = v_prev[i, j, k]
                vxx = (v_prev[i + 1, j, k] - 2.0 * vc + v_prev[i - 1, j, k]) * invdx2
                vyy = (v_prev[i, j + 1, k] - 2.0 * vc + v_prev[i, j - 1, k]) * invdx2
                vzz = (v_prev[i, j, k + 1] - 2.0 * vc + v_prev[i, j, k - 1]) * invdx2
                vxy = (v_prev[i + 1, j + 1, k] - v_prev[i + 1, j - 1, k] - v_prev[i - 1, j + 1, k] + v_prev[i - 1, j - 1, k]) * inv4dx2
                vxz = (v_prev[i + 1, j, k + 1] - v_prev[i + 1, j, k - 1] - v_prev[i - 1, j, k + 1] + v_prev[i - 1, j, k - 1]) * inv4dx2
                vyz = (v_prev[i, j + 1, k + 1] - v_prev[i, j + 1, k - 1] - v_prev[i, j - 1, k + 1] + v_prev[i, j - 1, k - 1]) * inv4dx2
                diffusion = s11 * vxx + s22 * vyy + s33 * vzz + 2 * (s12 * vxy + s13 * vxz + s23 * vyz)

                v_new[i, j, k] = v_prev[i, j, k] + dt * (H + 0.5 * diffusion)

    if bc_neumann:
        _apply_bc3d_neumann_inplace(v_new)
    return v_new


def solve_hji_stochastic_fdm_3d_numba(alpha=-2.0, beta=2.0, a=1.0, b=1.0, c=0.5,
                                       sigma=np.eye(3) * 0.1, N_dim=3, r=1.0,
                                       domain_t=(0.0, 1.0), num_t=101,
                                       domain_x=(-1.0, 1.0), num_x=65, dtype=np.float64,
                                       nu_h=0.0, output="full"):
    if output not in ("full", "crop", "slice"):
        raise ValueError("output must be full, crop or slice")
    t_grid = np.linspace(*domain_t, num_t, dtype=dtype)
    x_grid = np.linspace(*domain_x, num_x, dtype=dtype)
    dx = (x_grid[1] - x_grid[0]).item()
    dt = (t_grid[1] - t_grid[0]).item()

    px = np.linspace(domain_x[0] - dx, domain_x[1] + dx, num_x + 2, dtype=dtype)
    X1, X2, X3 = px, px, px

    Sigma = np.asarray(sigma, dtype=dtype) @ np.asarray(sigma, dtype=dtype).T + nu_h * np.eye(3)
    s11, s22, s33 = Sigma[0, 0], Sigma[1, 1], Sigma[2, 2]
    s12, s13, s23 = Sigma[0, 1], Sigma[0, 2], Sigma[1, 2]

    # g(x) = 0.5 * ((N-1)*x1^2 + x2^2 + x3^2 - (N-1)*r^2); (x1,x2,x3) = (publisher, subscriber_i, subscriber_j)
    V_T = np.zeros((num_x + 2, num_x + 2, num_x + 2), dtype=dtype)
    term = 0.5 * ((N_dim - 1) * X1[:, None, None]**2 + X2[None, :, None]**2 + X3[None, None, :]**2 - (N_dim - 1) * (r * r))
    V_T[1:-1, 1:-1, 1:-1] = term[1:-1, 1:-1, 1:-1]
    _apply_bc3d_neumann_inplace(V_T)

    time_list = np.array([0.00, 0.10, 0.20, 0.30, 0.40], dtype=dtype)
    snap_idx = np.rint((time_list - domain_t[0]) / dt).astype(np.int64)
    snap_idx = np.clip(snap_idx, 0, num_t - 2)
    rev_idx = (num_t - 2) - snap_idx

    def snapshot(v):
        if output == "full":
            return v[1:-1, 1:-1, 1:-1].copy()
        if (num_x - 1) % 6:
            raise ValueError("The central crop requires a grid size divisible by 6")
        q = (num_x - 1) // 6
        idx = np.arange(2*q+1, 4*q+2)
        if output == "slice":
            return v[idx[:, None], idx[None, :], idx[None, :]].copy()
        return v[2*q+1:4*q+2, 2*q+1:4*q+2, 2*q+1:4*q+2].copy()

    snaps = np.empty((6, *snapshot(V_T).shape), dtype=dtype)
    snaps[5] = snapshot(V_T)
    v_prev = V_T.copy()
    for i in range(num_t - 1):
        v_next = _step_one_hji3d(v_prev, X1, X2, X3, dx, dt, a, b, c, alpha, beta, s11, s22, s33, s12, s13, s23, theta=np.pi / 4, bc_neumann=True)
        for j in range(5):
            if i == rev_idx[j]:
                snaps[j] = snapshot(v_next)
        v_prev = v_next

    return snaps


def sigma_from_text(sigma_txt):
    """'01_00_01' -> diag(0.1, 0.0, 0.1); '03_00_03' -> diag(0.3, 0.0, 0.3); ..."""
    vals = [int(tok) / 10.0 for tok in sigma_txt.split("_")]
    return np.diag(vals)


def generate_reference(sigma_txt, h=600, output_dir=None, full_reference=False):
    """Solve on [-1.5,1.5]^3 and store the central evaluation data."""
    if h < 6 or h % 6:
        raise ValueError("Grid size must be a positive multiple of 6")
    if sigma_txt not in ("01_00_01", "03_00_03", "05_00_05"):
        raise ValueError("Use 01_00_01, 03_00_03 or 05_00_05")
    spacing = 3.0 / h
    nu_h = 6.0 * spacing
    # T / (h_x / (1152 * 3)) = 576 * Nx.
    steps = 576 * h
    values = solve_hji_stochastic_fdm_3d_numba(
        alpha=-1/(1.5**2*np.sin(1.5)), beta=1/1.5**3,
        a=1., b=1., c=.5, sigma=sigma_from_text(sigma_txt), N_dim=3, r=1.,
        domain_t=(0., .5), num_t=steps+1, domain_x=(-1.5, 1.5), num_x=h+1,
        dtype=np.float64, nu_h=nu_h, output="crop" if full_reference else "slice")
    if not np.isfinite(values).all():
        raise RuntimeError("Nonfinite FDM solution")
    out_dir = Path(output_dir) if output_dir is not None else DATA_DIR.parent / "generated_reference"
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "" if h == 600 else f"_h{h}"
    kind = "3d" if full_reference else "slice"
    path = out_dir / f"fdm_reference_{kind}_sigma{sigma_txt.split('_')[0]}{suffix}.npy"
    np.save(path, values)
    print(f"saved: {path}")
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sigma", choices=["01_00_01", "03_00_03", "05_00_05"], default="01_00_01")
    parser.add_argument("--h", type=int, default=600, help="Cells per axis; paper resolution: 600.")
    parser.add_argument("--full-reference", action="store_true", help="Store the central 3D crop instead of its partial-diagonal slice.")
    parser.add_argument("--output-dir", type=Path, default=DATA_DIR.parent / "generated_reference")
    args = parser.parse_args()
    generate_reference(args.sigma, args.h, args.output_dir, args.full_reference)
