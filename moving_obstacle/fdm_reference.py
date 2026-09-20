"""Explicit FDM reference for the moving-obstacle problem.

Arrays use (time, x2, x1) ordering on an xy grid.
"""
import argparse
import os
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", os.environ.get("HJ_GPU_ID", "0"))
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_default_matmul_precision", "highest")

def _apply_bc(u, bc_flag):
    """Apply BC on a padded (H,W) grid with 1 ghost layer.
       bc_flag: 0 -> Neumann (edge copy), 1 -> linear extrapolation.
    """
    def neumann(uu):
        uu = uu.at[0, :].set(uu[1, :])
        uu = uu.at[-1, :].set(uu[-2, :])
        uu = uu.at[:, 0].set(uu[:, 1])
        uu = uu.at[:, -1].set(uu[:, -2])
        return uu

    def extrap(uu):
        uu = uu.at[0, :].set(2.0 * uu[1, :]    - uu[2, :])
        uu = uu.at[-1, :].set(2.0 * uu[-2, :]  - uu[-3, :])
        uu = uu.at[:, 0].set(2.0 * uu[:, 1]    - uu[:, 2])
        uu = uu.at[:, -1].set(2.0 * uu[:, -2]  - uu[:, -3])
        return uu

    return lax.cond(bc_flag == 0, neumann, extrap, u)

def _step_one(v_prev, t, params):
    """
    v_prev : (padded_n, padded_n) with 1-cell ghost layer
    t      : scalar time
    params : dict with grids/coeffs
    """
    X1, X2        = params["X1"], params["X2"]
    dx, dt        = params["dx"], params["dt"]
    l1, l2        = params["lambda1"], params["lambda2"]
    delta         = params["delta"]
    eps           = params["epsilon"]
    s11, s12, s22 = params["s11"], params["s12"], params["s22"]
    bc_flag       = params["bc_flag"]

    # xy indexing: x1 varies across columns, x2 across rows.
    dv_dx = (v_prev[1:-1, 2:] - v_prev[1:-1, :-2]) * (0.5 / dx)
    dv_dy = (v_prev[2:, 1:-1] - v_prev[:-2, 1:-1]) * (0.5 / dx)
    grad_norm = jnp.sqrt(dv_dx*dv_dx + dv_dy*dv_dy + 1e-20)

    # moving obstacle potential phi(t, x)
    x_obs0 = 0.5 * jnp.cos(jnp.pi * t)
    x_obs1 = 0.5 * jnp.sin(jnp.pi * t)
    dist2  = (X1[1:-1, 1:-1] - x_obs0)**2 + (X2[1:-1, 1:-1] - x_obs1)**2
    phi    = jnp.exp(-dist2 / (2.0 * (eps**2)))

    # piecewise Hamiltonian
    H_quad = -(grad_norm**2) / (4.0 * l1) + l2*phi + delta*grad_norm
    H_lin  = -grad_norm + l1 + l2*phi + delta*grad_norm
    H      = jnp.where(grad_norm <= 2.0*l1, H_quad, H_lin)

    # second derivatives (interior)
    d2v_xx = (v_prev[1:-1, 2:] - 2.0*v_prev[1:-1, 1:-1] + v_prev[1:-1, :-2]) / (dx*dx)
    d2v_yy = (v_prev[2:, 1:-1] - 2.0*v_prev[1:-1, 1:-1] + v_prev[:-2, 1:-1]) / (dx*dx)
    d2v_xy = (v_prev[2:, 2:] - v_prev[2:, :-2] - v_prev[:-2, 2:] + v_prev[:-2, :-2]) / (4.0*dx*dx)

    diffusion = s11*d2v_xx + 2.0*s12*d2v_xy + s22*d2v_yy
    rhs = -(H + 0.5*diffusion)

    v_new = v_prev.at[1:-1, 1:-1].set(v_prev[1:-1, 1:-1] - dt*rhs)
    v_new = _apply_bc(v_new, bc_flag)
    return v_new

def solve_hji_fdm_explicit_jax_memlite(
    lambda_1=0.1, lambda_2=1.0, lambda_3=0.1,
    epsilon=0.3, sigma=np.zeros((2, 2)), delta=0.1,
    xgoal=(0.9, 0.9),
    domain_t=(0.0, 1.0), num_t=11,
    domain_x=(-1.0, 1.0), num_x=11,
    nu_h=0.0,
    bc='neumann',           # {'neumann','extrap'}
    terminal_case='one',    # {'one','two'}
    dtype=jnp.float64
):
    # grids
    t_grid = jnp.linspace(domain_t[0], domain_t[1], num_t, dtype=dtype)
    x_grid = jnp.linspace(domain_x[0], domain_x[1], num_x, dtype=dtype)
    dx = x_grid[1] - x_grid[0]
    dt = t_grid[1] - t_grid[0]

    padded_n = num_x + 2
    x_pad = jnp.linspace(domain_x[0]-dx, domain_x[1]+dx, padded_n, dtype=dtype)
    X1, X2 = jnp.meshgrid(x_pad, x_pad, indexing='xy')

    sigma = jnp.asarray(sigma, dtype=dtype)
    sigma2 = sigma @ sigma.T + nu_h * jnp.eye(sigma.shape[0], dtype=dtype)
    s11, s12, s22 = sigma2[0, 0], sigma2[0, 1], sigma2[1, 1]

    # terminal condition (padded)
    v_T = jnp.zeros((padded_n, padded_n), dtype=dtype)
    if terminal_case == 'one':
        xg = jnp.asarray(xgoal, dtype=dtype)
        term = lambda_3 * ((X1[1:-1,1:-1]-xg[0])**2 + (X2[1:-1,1:-1]-xg[1])**2)
        v_T = v_T.at[1:-1,1:-1].set(term)
    elif terminal_case == 'two':
        xg1 = jnp.asarray(xgoal, dtype=dtype)
        xg2 = jnp.array([-xg1[0], xg1[1]], dtype=dtype)
        target1 = lambda_3 * ((X1[1:-1,1:-1]-xg1[0])**2 + (X2[1:-1,1:-1]-xg1[1])**2)
        target2 = lambda_3 * ((X1[1:-1,1:-1]-xg2[0])**2 + (X2[1:-1,1:-1]-xg2[1])**2)
        mask = (X1[1:-1,1:-1] >= 0.0)
        v_T = v_T.at[1:-1,1:-1].set(jnp.where(mask, target1, target2))
    else:
        raise ValueError("terminal_case must be 'one' or 'two'.")
    bc_flag = 0 if bc == 'neumann' else 1
    v_T = _apply_bc(v_T, bc_flag)

    # snapshot times (ascending): 0.00, 0.25, 0.50, 0.75
    time_list = jnp.array([0.00, 0.25, 0.50, 0.75], dtype=dtype)
    snap_idx = (time_list * (num_t - 1)).astype(jnp.int32)
    snap_idx = jnp.clip(snap_idx, 0, num_t-2)   # exclude T
    # reversed stepping index i corresponds to t = t_grid[num_t-2 - i]
    rev_idx = (num_t - 2) - snap_idx            # length 4

    t_seq = t_grid[:-1][::-1]  # times for backward stepping

    params = dict(
        X1=X1, X2=X2,
        dx=dx, dt=dt,
        lambda1=lambda_1, lambda2=lambda_2, delta=delta,
        epsilon=epsilon,
        s11=s11, s12=s12, s22=s22,
        bc_flag=bc_flag
    )

    def run_loop(v0, t_seq, rev_idx, params):
        snaps = jnp.zeros((4, padded_n, padded_n), dtype=dtype)

        def body(i, state):
            v_prev, snaps = state
            v_next = _step_one(v_prev, t_seq[i], params)

            # Store the selected time snapshots.
            def write_if(snaps, j):
                return lax.cond(
                    i == rev_idx[j],
                    lambda s: s.at[j].set(v_next),
                    lambda s: s,
                    snaps
                )

            snaps = write_if(snaps, 0)
            snaps = write_if(snaps, 1)
            snaps = write_if(snaps, 2)
            snaps = write_if(snaps, 3)
            return (v_next, snaps)

        v_last, snaps = lax.fori_loop(0, t_seq.shape[0], body, (v0, snaps))
        return snaps, v_last

    run_loop_jit = jax.jit(run_loop)
    snaps, v0 = run_loop_jit(v_T, t_seq, rev_idx, params)

    # Return snapshots at t=0, 0.25, 0.5, 0.75, 1.
    v_out = jnp.concatenate([snaps[:,1:-1,1:-1], v_T[jnp.newaxis, 1:-1, 1:-1]], axis=0)
    return v_out, t_grid, x_grid, num_t

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nx", type=int, default=800, help="Spatial cells per axis on [-2,2].")
    parser.add_argument("--sigma", type=float, nargs=2, choices=(0.0, 0.1), default=(0.0, 0.0),
                        metavar=("SIGMA1", "SIGMA2"), help="Diagonal diffusion amplitudes.")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "generated_reference")
    args = parser.parse_args()
    if args.nx < 4 or args.nx % 4:
        parser.error("nx must be a positive multiple of 4")
    nx = args.nx
    values, _, _, _ = solve_hji_fdm_explicit_jax_memlite(
        lambda_1=0.1, lambda_2=1.0, lambda_3=0.1,
        epsilon=0.3, sigma=np.diag(args.sigma), delta=0.1,
        xgoal=(0.9, 0.9), domain_t=(0.0, 1.0), num_t=100*nx+1,
        domain_x=(-2.0, 2.0), num_x=nx+1, nu_h=5/nx,
        bc="neumann", terminal_case="one", dtype=jnp.float64)
    q = nx // 4
    crop = np.asarray(values[:, q:3*q+1, q:3*q+1])
    if not np.isfinite(crop).all():
        raise SystemExit("FDM produced non-finite values")
    tag = "_".join("01" if value == 0.1 else "00" for value in args.sigma)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / f"fdm_reference_sigma_{tag}_nx{nx}.npy"
    np.save(path, crop)
    print(f"Saved {path}: shape={crop.shape}, nx={nx}, steps={100*nx}, nu={5/nx}")


if __name__ == "__main__":
    main()
