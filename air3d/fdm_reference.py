"""Explicit JAX finite-difference solver for the AIR3D (Air3D) pursuit-evasion
HJI equation. Produces the FDM reference solutions consumed by `train_pinn.py`
and `make_figures.py`.

GPU selection: set the HJ_GPU_ID environment variable (defaults to "0"), e.g.
    HJ_GPU_ID=1 python fdm_reference.py --sigma 0 0 0
    HJ_GPU_ID=1 python fdm_reference.py --sigma 0.03 0.05 0.0
"""

#%%
import argparse
import os

GPU_ID = os.environ.get("HJ_GPU_ID", "0")
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ.setdefault("CUDA_VISIBLE_DEVICES", GPU_ID)
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", ".9")

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_default_matmul_precision", "highest")

DATA_DIR = Path(__file__).resolve().parent / "data"


# --------------------------
# 3D Boundary handling
#   - x,y: bc_flag 0->Neumann, 1->Extrap
#   - psi: periodic (always)
# --------------------------
def _apply_bc3d(u, bc_flag):
    def neumann(uu):
        uu = uu.at[0, :, :].set(uu[1, :, :])
        uu = uu.at[-1, :, :].set(uu[-2, :, :])
        uu = uu.at[:, 0, :].set(uu[:, 1, :])
        uu = uu.at[:, -1, :].set(uu[:, -2, :])
        return uu

    def extrap(uu):
        uu = uu.at[0, :, :].set(2.0 * uu[1, :, :] - uu[2, :, :])
        uu = uu.at[-1, :, :].set(2.0 * uu[-2, :, :] - uu[-3, :, :])
        uu = uu.at[:, 0, :].set(2.0 * uu[:, 1, :] - uu[:, 2, :])
        uu = uu.at[:, -1, :].set(2.0 * uu[:, -2, :] - uu[:, -3, :])
        return uu

    u = lax.cond(bc_flag == 0, neumann, extrap, u)
    # The physical grid includes both -pi and pi; synchronize the duplicate.
    u = u.at[:, :, -2].set(u[:, :, 1])
    u = u.at[:, :, 0].set(u[:, :, -3])
    u = u.at[:, :, -1].set(u[:, :, 2])
    return u


def _step_one_air3d(v_prev, t, P):
    x1, x2, x3 = P["X"], P["Y"], P["PSI"]
    dx, dy, dz = P["dx"], P["dy"], P["dz"]
    dt = P["dt"]
    ve, vp = P["ve"], P["vp"]
    obar = P["obar"]
    beta = P["beta"]
    s11, s12, s13 = P["s11"], P["s12"], P["s13"]
    s22, s23, s33 = P["s22"], P["s23"], P["s33"]
    bc_flag = P["bc_flag"]

    dv_dx1 = (v_prev[2:, 1:-1, 1:-1] - v_prev[:-2, 1:-1, 1:-1]) / (2 * dx)
    dv_dx2 = (v_prev[1:-1, 2:, 1:-1] - v_prev[1:-1, :-2, 1:-1]) / (2 * dy)
    dv_dx3 = (v_prev[1:-1, 1:-1, 2:] - v_prev[1:-1, 1:-1, :-2]) / (2 * dz)

    H = (
        dv_dx1 * (-ve + vp * jnp.cos(x3[None, None, 1:-1]))
        + dv_dx2 * (vp * jnp.sin(x3[None, None, 1:-1]))
        + obar * jnp.abs(dv_dx1 * x2[None, 1:-1, None] - dv_dx2 * x1[1:-1, None, None] - dv_dx3)
        - obar * jnp.abs(dv_dx3)
    )

    v_c = v_prev[1:-1, 1:-1, 1:-1]
    vxx = (v_prev[2:, 1:-1, 1:-1] - 2.0 * v_c + v_prev[:-2, 1:-1, 1:-1]) / (dx * dx)
    vyy = (v_prev[1:-1, 2:, 1:-1] - 2.0 * v_c + v_prev[1:-1, :-2, 1:-1]) / (dy * dy)
    vzz = (v_prev[1:-1, 1:-1, 2:] - 2.0 * v_c + v_prev[1:-1, 1:-1, :-2]) / (dz * dz)

    vxy = (v_prev[2:, 2:, 1:-1] - v_prev[2:, :-2, 1:-1] - v_prev[:-2, 2:, 1:-1] + v_prev[:-2, :-2, 1:-1]) / (4.0 * dx * dy)
    vxz = (v_prev[2:, 1:-1, 2:] - v_prev[2:, 1:-1, :-2] - v_prev[:-2, 1:-1, 2:] + v_prev[:-2, 1:-1, :-2]) / (4.0 * dx * dz)
    vyz = (v_prev[1:-1, 2:, 2:] - v_prev[1:-1, 2:, :-2] - v_prev[1:-1, :-2, 2:] + v_prev[1:-1, :-2, :-2]) / (4.0 * dy * dz)

    diffusion = s11 * vxx + s22 * vyy + s33 * vzz + 2.0 * (s12 * vxy + s13 * vxz + s23 * vyz)

    rhs = -(H + 0.5 * diffusion)
    v_new = v_prev.at[1:-1, 1:-1, 1:-1].set(v_c - dt * rhs)
    v_new = _apply_bc3d(v_new, bc_flag)
    return v_new


def solve_air3d_fdm_explicit_jax_memlite(
    ve=0.5, vp=0.5, omega_bar=1.0, beta=0.5,
    sigma=None,
    nu_h=0.0,
    domain_t=(0.0, 1.0), num_t=101,
    domain_x=(-1.5, 1.5), num_x=65,
    domain_y=(-1.5, 1.5), num_y=65,
    domain_psi=(-jnp.pi, jnp.pi), num_psi=65,
    bc="neumann",
    dtype=jnp.float64,
    slice_only=False,
):
    # Building the default sigma at call time (not as a default-argument
    # value) avoids touching the JAX backend merely by importing this module.
    if sigma is None:
        sigma = jnp.diag(jnp.array([0.03, 0.05, 0.00]))
    t_grid = jnp.linspace(*domain_t, num_t, dtype=dtype)
    x1_grid = jnp.linspace(*domain_x, num_x, dtype=dtype)
    x2_grid = jnp.linspace(*domain_y, num_y, dtype=dtype)
    x3_grid = jnp.linspace(*domain_psi, num_psi, dtype=dtype)

    dx = x1_grid[1] - x1_grid[0]
    dy = x2_grid[1] - x2_grid[0]
    dz = x3_grid[1] - x3_grid[0]
    dt = t_grid[1] - t_grid[0]

    px = jnp.linspace(domain_x[0] - dx, domain_x[1] + dx, num_x + 2, dtype=dtype)
    py = jnp.linspace(domain_y[0] - dy, domain_y[1] + dy, num_y + 2, dtype=dtype)
    pz = jnp.linspace(domain_psi[0] - dz, domain_psi[1] + dz, num_psi + 2, dtype=dtype)

    X, Y, PSI = px, py, pz

    sigma = jnp.asarray(sigma, dtype=dtype)
    Sigma = sigma @ sigma.T + nu_h * jnp.eye(3, dtype=dtype)
    s11, s12, s13 = Sigma[0, 0], Sigma[0, 1], Sigma[0, 2]
    s22, s23 = Sigma[1, 1], Sigma[1, 2]
    s33 = Sigma[2, 2]

    V_T = jnp.zeros((num_x + 2, num_y + 2, num_psi + 2), dtype=dtype)
    term_xy = jnp.sqrt(X[1:-1, None, None] ** 2 + Y[None, 1:-1, None] ** 2) - beta
    V_T = V_T.at[1:-1, 1:-1, 1:-1].set(term_xy)
    bc_flag = 0 if bc == "neumann" else 1
    V_T = _apply_bc3d(V_T, bc_flag)

    time_list = jnp.array([0.00, 0.25, 0.50, 0.75], dtype=dtype)
    snap_idx = (time_list * (num_t - 1)).astype(jnp.int64)
    snap_idx = jnp.clip(snap_idx, 0, num_t - 2)
    rev_idx = (num_t - 2) - snap_idx
    t_seq = t_grid[:-1][::-1]

    P = dict(
        X=X, Y=Y, PSI=PSI, dx=dx, dy=dy, dz=dz, dt=dt,
        ve=ve, vp=vp, obar=omega_bar, beta=beta,
        s11=s11, s12=s12, s13=s13, s22=s22, s23=s23, s33=s33, bc_flag=bc_flag,
    )

    def snapshot(v):
        if slice_only:
            q = (num_x - 1) // 4
            psi_index = round(0.75 * (num_psi - 1)) + 1
            return v[q+1:3*q+2, q+1:3*q+2, psi_index]
        return v[1:-1, 1:-1, 1:-1]

    def run_loop(v0, t_seq, rev_idx, P):
        snaps = jnp.zeros((4, *snapshot(v0).shape), dtype=dtype)

        def body(i, state):
            v_prev, snaps = state
            v_next = _step_one_air3d(v_prev, t_seq[i], P)

            def write_if(snaps, j):
                return lax.cond(i == rev_idx[j], lambda s: s.at[j].set(snapshot(v_next)), lambda s: s, snaps)

            for j in range(4):
                snaps = write_if(snaps, j)
            return (v_next, snaps)

        v_last, snaps = lax.fori_loop(0, t_seq.shape[0], body, (v0, snaps))
        return snaps, v_last

    snaps, V0 = run_loop(V_T, t_seq, rev_idx, P)
    V_out = jnp.concatenate([snaps, snapshot(V_T)[None]], axis=0)
    return V_out


def generate_reference(nx, sigma_vec, output_dir):
    from settings import problem, data_suffix
    P = problem(nx, sigma_vec)
    nt = int(np.ceil(1 / P["tau_h"]))
    values = solve_air3d_fdm_explicit_jax_memlite(
        ve=.5, vp=.5, omega_bar=1.5, beta=.5,
        sigma=P["sigma"], nu_h=P["nu_h"],
        domain_t=(0., 1.), num_t=nt+1,
        domain_x=(-3., 3.), num_x=nx+1,
        domain_y=(-3., 3.), num_y=nx+1,
        domain_psi=(-np.pi, np.pi), num_psi=nx+1,
        dtype=jnp.float64, slice_only=True)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"fdm_reference_h{nx}{data_suffix(sigma_vec)}.npy"
    values = np.asarray(values)
    if not np.isfinite(values).all():
        raise RuntimeError("Nonfinite FDM reference")
    np.save(path, values)
    print(f"saved: {path}")
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nx", type=int, choices=[200, 400, 800], default=800)
    parser.add_argument("--sigma", type=float, nargs=3, default=[0., 0., 0.])
    parser.add_argument("--output-dir", type=Path, default=DATA_DIR.parent / "generated_reference")
    args = parser.parse_args()
    generate_reference(args.nx, args.sigma, args.output_dir)
