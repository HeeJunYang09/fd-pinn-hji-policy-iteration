"""Train the PI-PINN value function for the publisher-subscriber HJI equation
at a given dimension, against the (dimension-reduced) 3D FDM reference
produced by `fdm_reference.py`.

GPU selection: set the HJ_GPU_ID environment variable (defaults to "0"), e.g.
    HJ_GPU_ID=1 python train_pinn.py --dim 11 --sigma-val 0.1
    HJ_GPU_ID=1 python train_pinn.py --dim 3  --sigma-val 0.3
    HJ_GPU_ID=1 python train_pinn.py --dim 51 --sigma-val 0.3
"""

#%%
import argparse
import os

GPU_ID = os.environ.get("HJ_GPU_ID", "0")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", GPU_ID)
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", ".9")

import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(__file__).resolve().parent / "data"
if str(REPO_ROOT / "common") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "common"))

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
import pickle
from jax import random, jit, vmap, config, value_and_grad, tree_util
from jax.nn import initializers
from tqdm import tqdm

config.update("jax_default_matmul_precision", "highest")


#%%
@jit
def forward(params, data):
    x = data
    for (w, b) in params[:-1]:
        x = jnp.sin(jnp.matmul(x, w) + b)
    (w, b) = params[-1]
    return jnp.matmul(x, w) + b


def init_params(layers_dims, key):
    initializer = initializers.glorot_uniform()
    keys = random.split(key, len(layers_dims) - 1)

    def initial_(in_dim, out_dim, key):
        w = initializer(key, (in_dim, out_dim))
        b = jnp.zeros(shape=(out_dim,))
        return (w, b)

    return [initial_(in_dim, out_dim, key) for in_dim, out_dim, key in zip(layers_dims[:-1], layers_dims[1:], keys)]


def count_params(params):
    return sum(w.size + b.size for w, b in params)


def grid_sample(key, step_size, domain):
    return random.uniform(key, shape=(), minval=domain[0] + step_size, maxval=domain[1] - step_size)


grid_sample = vmap(grid_sample, in_axes=(0, 0, None))


def sample_grid_collocation(num_interior, key, N=3, minnum_h=200, maxnum_h=6400, domain_t=(0.0, 0.5), domain_x=(-1.5, 1.5)):
    keys = random.split(key, N + 3)
    Lx = domain_x[1] - domain_x[0]
    min_h, max_h = Lx / maxnum_h, Lx / minnum_h
    h_collection = random.uniform(keys[1], (num_interior,), minval=min_h, maxval=max_h)
    tau_collection = jnp.sqrt(h_collection / (1152 * N))
    nu_h_collection = h_collection * 6.0

    t_key = random.split(keys[2], num_interior)
    ts = grid_sample(t_key, tau_collection, domain_t)
    x_list = []
    for k in range(N):
        x_key = random.split(keys[k + 3], num_interior)
        x_list.append(grid_sample(x_key, h_collection, domain_x))
    return jnp.stack([ts, *x_list], axis=-1), tau_collection[:, None], h_collection[:, None], nu_h_collection[:, None]


#%%
def gen_v_single(N, r, tf):
    def v_single(data, params):
        t, x0, xi = data[0], data[1], data[2:]
        nn_output = forward(params, data)[0]
        g = 0.5 * ((N - 1) * x0**2 + jnp.sum(xi**2) - (N - 1) * r**2)
        return (tf - t) * nn_output + g  # terminal-condition ansatz

    return v_single


def gen_PINN_loss(v_single, N=3, theta=jnp.pi / 4):
    assert N % 2 == 1 and N > 1
    NS = (N - 1) // 2
    R = jnp.array([[jnp.cos(theta), -jnp.sin(theta)], [jnp.sin(theta), jnp.cos(theta)]])

    def central_grad_laplacian(tx, params, h, u_xc):
        t = tx[:1]
        x_base = tx[1:]
        e = jnp.eye(N)

        def one_dim_deri(idx):
            ei = e[idx]
            u_xp = v_single(jnp.concatenate([t, x_base + h * ei]), params)
            u_xm = v_single(jnp.concatenate([t, x_base - h * ei]), params)
            return (u_xp - u_xm) / (2 * h), (u_xp - 2.0 * u_xc + u_xm) / (h * h)

        return vmap(one_dim_deri)(jnp.arange(N))

    def PINN_loss(params, grid_tx, P, u_n, d_n, tau, h, nu_h):
        a, b, c = P["a"], P["b"], P["c"]
        alpha, beta = P["alpha"], P["beta"]
        sigma = P["sigma"].reshape(-1)

        def residual(tx, u, d, tau, h, nu_h):
            t, x = tx[:1], tx[1:]
            tau, h, nu_h = tau[0], h[0], nu_h[0]
            u_value = v_single(tx, params)
            u_back = v_single(jnp.concatenate([t - tau, x]), params)
            dv_dt = (u_value - u_back) / tau

            dv_dx, d2v_xx = central_grad_laplacian(tx, params, h, u_value)
            dv_dxi = dv_dx[1:]

            x0 = x[0]
            f0 = a * x0 + alpha * jnp.sin(x0) * x0**2
            fi = -x0 + a * x[1:] - beta * x0 * (x[1:] ** 2)
            f = jnp.concatenate([jnp.array([f0]), fi], axis=0)

            Bu_term = b * jnp.sum(u * dv_dxi)
            p_blocks = dv_dxi.reshape(NS, 2)
            d_blocks = d.reshape(NS, 2)
            q_blocks = p_blocks @ R
            Cd_term = c * jnp.sum(d_blocks * q_blocks)
            diffusion = 0.5 * jnp.sum((sigma**2 + nu_h) * d2v_xx)

            res = dv_dt + jnp.dot(dv_dx, f) + Bu_term + Cd_term + diffusion
            return res**2

        return jnp.mean(vmap(residual)(grid_tx, u_n, d_n, tau, h, nu_h))

    return PINN_loss


def gen_policy_fn(v_single, N, theta=jnp.pi / 4):
    assert N % 2 == 1 and N > 1
    NS = (N - 1) // 2
    R = jnp.array([[jnp.cos(theta), -jnp.sin(theta)], [jnp.sin(theta), jnp.cos(theta)]])

    def policy_fn(params, data_tx, h):
        def sign_pm1(x):
            return jnp.where(x >= 0, 1.0, -1.0)

        def find_policy(tx, h):
            t = tx[:1]
            x_base = tx[1:]
            e = jnp.eye(N)
            h = h[0]

            def one_dim_deri(idx):
                ei = e[idx]
                u_xp = v_single(jnp.concatenate([t, x_base + h * ei]), params)
                u_xm = v_single(jnp.concatenate([t, x_base - h * ei]), params)
                return (u_xp - u_xm) / (2 * h)

            dv_dx = vmap(one_dim_deri)(jnp.arange(N))
            p_sub = dv_dx[1:]
            u = -sign_pm1(p_sub)
            p_blocks = p_sub.reshape(NS, 2)
            q_blocks = p_blocks @ R
            d = sign_pm1(q_blocks).reshape(-1)
            return u, d

        return vmap(find_policy, in_axes=(0, 0))(data_tx, h)

    return policy_fn


#%%
def eval_nn_partial_diag_slice(params, t_val, xg, N, tf=0.5, r=1.0):
    """Evaluate v_N(t, x0, s, s, ..., s) on a 2D grid (s vs x0). N must be odd, >= 3."""
    assert (N % 2 == 1) and (N >= 3)
    v_singleN = gen_v_single(N=N, r=r, tf=tf)
    xg = np.asarray(xg)
    nx = len(xg)
    I, J = np.meshgrid(np.arange(nx), np.arange(nx), indexing="ij")
    x0, s = xg[J], xg[I]
    XN = np.full((nx, nx, N), s[..., None], dtype=np.float64)
    XN[..., 0] = x0
    XN_flat = XN.reshape(-1, N)
    tcol = jnp.full((XN_flat.shape[0], 1), t_val)
    tx = jnp.concatenate([tcol, jnp.asarray(XN_flat)], axis=1)
    v_flat = vmap(lambda z: v_singleN(z, params))(tx)
    return np.asarray(v_flat).reshape(nx, nx)


def ref_from_3d_fdm_partial_diag(V3, N):
    """v_N(t, x0, s, ..., s) ~= NS * v_3D(t, x0, s, s), NS=(N-1)/2 -- the paper's
    decomposition-based reference reused for all higher-dimensional slices."""
    assert (N % 2 == 1) and (N >= 3)
    NS = (N - 1) // 2
    nx = V3.shape[0]
    idx = np.arange(nx)
    A = V3[:, idx, idx]
    return (NS * A).T


def error_metrics(v_nn, v_fdm, eps=1e-12):
    diff = v_nn - v_fdm
    l2 = np.linalg.norm(diff.ravel())
    l2_ref = np.linalg.norm(v_fdm.ravel()) + eps
    return dict(rel_l2=l2 / l2_ref, linf=np.max(np.abs(diff)), mse=np.mean(diff**2), mae=np.mean(np.abs(diff)))


def sigma_alt(dim, val=1.0, dtype=jnp.float32):
    """Alternating diagonal diffusion diag(val, 0, val, 0, ...) used throughout the paper."""
    idx = jnp.arange(dim)
    return val * (idx % 2 == 0).astype(dtype)


#%%
def train_value_and_update_policy_with_sample(Ni, P, minnum_h=200, maxnum_h=2000, num_iters=100, num_epochs=500, lr=1e-3,
                                               verbose=True, seed=42, layer=(3, 64, 64, 1), dim=2, v_fdm=None):
    def initial_sample_ball(key, radius=1.0, dim=2):
        key_dir, key_rad = random.split(key)
        normal = random.normal(key_dir, shape=(dim,))
        unit_vec = normal / jnp.linalg.norm(normal)
        r = random.uniform(key_rad, shape=()) ** (1.0 / dim)
        return radius * r * unit_vec

    init_keys = random.split(random.key(seed), 4)
    u_n = vmap(initial_sample_ball, in_axes=(0, None, None))(random.split(init_keys[0], num=Ni), 1.0, dim - 1).reshape(Ni, dim - 1)
    d_n = vmap(initial_sample_ball, in_axes=(0, None, None))(random.split(init_keys[1], num=Ni), 1.0, dim - 1).reshape(Ni, dim - 1)

    key = init_keys[2]
    params = init_params(list(layer), key)

    sigma = jnp.asarray(P["sigma"])
    P["sigma"] = jnp.diag(sigma) if sigma.ndim == 2 else sigma

    optimizer = optax.adam(lr)
    opt_state = optimizer.init(params)

    v_single = gen_v_single(N=dim, r=1.0, tf=P["domain_t"][1])
    PINN_loss = jit(gen_PINN_loss(v_single, N=dim))
    policy_update = jit(gen_policy_fn(v_single, N=dim))

    @jit
    def step(params, opt_state, data_tx, P, u_n, d_n, tau, h, nu_h):
        loss, grads = value_and_grad(PINN_loss)(params, data_tx, P, u_n, d_n, tau, h, nu_h)
        updates, opt_state = optimizer.update(grads, opt_state)
        return optax.apply_updates(params, updates), opt_state, loss

    mse_history, l2_history = [], []
    iter_key = random.split(init_keys[-1], 3)
    data_tx, tau, h, nu_h = sample_grid_collocation(Ni, iter_key[-1], minnum_h=minnum_h, maxnum_h=maxnum_h, domain_t=P["domain_t"], domain_x=P["domain_x"], N=dim)

    def track_error():
        nx = len(v_fdm[0])
        xg = np.linspace(-0.5, 0.5, nx)
        vN = eval_nn_partial_diag_slice(params, 0.0, xg, dim, tf=0.5, r=1.0)
        vref = ref_from_3d_fdm_partial_diag(v_fdm[0], dim)
        m = error_metrics(vN, vref)
        mse_history.append(m["mse"])
        l2_history.append(m["rel_l2"])

    if v_fdm is not None:
        track_error()

    for n in tqdm(range(num_iters), desc="Policy Iteration", position=0):
        params_current = params.copy()
        epoch_bar = tqdm(range(num_epochs), desc=f"Train (Iter {n})", leave=False)
        for epoch in epoch_bar:
            if (epoch + 1) % (num_epochs // 10) == 0 or epoch < 1:
                iter_key = random.split(iter_key[1], 2)
                keys = random.split(iter_key[0], 3)
                data_tx, tau, h, nu_h = sample_grid_collocation(Ni, keys[0], minnum_h=minnum_h, maxnum_h=maxnum_h, domain_t=P["domain_t"], domain_x=P["domain_x"], N=dim)
                if n == 0:
                    u_n = vmap(initial_sample_ball, in_axes=(0, None, None))(random.split(keys[1], num=Ni), 1.0, dim - 1).reshape(Ni, dim - 1)
                    d_n = vmap(initial_sample_ball, in_axes=(0, None, None))(random.split(keys[2], num=Ni), 1.0, dim - 1).reshape(Ni, dim - 1)
                else:
                    u_n, d_n = policy_update(params_current, data_tx, h)

            params, opt_state, loss = step(params, opt_state, data_tx, P, u_n, d_n, tau, h, nu_h)
            if verbose and epoch % 10 == 0:
                epoch_bar.set_postfix(loss=f"{loss:.4e}")

        if v_fdm is not None:
            track_error()

        u_current, d_current = u_n, d_n
        u_n, d_n = policy_update(params, data_tx, h)

    return params, u_n, d_n, params_current, u_current, d_current, mse_history, l2_history


#%%
def run(dim, sigma_val):
    Nx = 600  # matches the paper's FDM reference resolution (h=600)
    domain_x, domain_t = (-1.5, 1.5), (0.0, 0.5)
    Lx = domain_x[1] - domain_x[0]

    P = {
        "a": 1.0, "b": 1.0, "c": 0.5,
        "alpha": -1 / (np.sin(domain_x[1]) * domain_x[1] ** 2), "beta": 1 / (domain_x[1] ** 3),
        "domain_t": domain_t, "domain_x": domain_x,
    }
    P["sigma"] = sigma_alt(dim, val=sigma_val)
    sigma_tag = f"{int(round(sigma_val * 10)):02d}"  # 0.1 -> "01", 0.3 -> "03", 0.5 -> "05"

    v_fdm = np.load(DATA_DIR / f"fdm_reference_3d_sigma{sigma_tag}.npy")
    minnum_h = maxnum_h = Nx
    nu_h = 6.0 * (Lx / Nx)

    num_iters, num_epoch = 500, 5000
    Ni = 2000 * dim
    unit = 64

    start_time = time.time()
    params, u_n, d_n, params_current, u_current, d_current, mse_history, l2_history = train_value_and_update_policy_with_sample(
        Ni, P, dim=dim, num_iters=num_iters, num_epochs=num_epoch, layer=[dim + 1, unit, unit, unit, 1],
        v_fdm=v_fdm, minnum_h=minnum_h, maxnum_h=maxnum_h,
    )
    first_leaf = tree_util.tree_leaves(params)[0]
    first_leaf.block_until_ready()
    train_time = time.time() - start_time

    out_dir = DATA_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    params_num = count_params(params)
    print(f"dim={dim} N={Ni} iters={num_iters} epochs={num_epoch} params={params_num} nu_h={nu_h:.4f} train_time={train_time:.0f}s")
    stem = f"trained_pinn_{dim}d_sigma{sigma_tag}"
    with open(out_dir / f"{stem}.pkl", "wb") as f:
        pickle.dump(
            {"params": params, "u_n": u_n, "d_n": d_n, "params_current": params_current,
             "u_current": u_current, "d_current": d_current, "sigma": P["sigma"],
             "train_time": train_time, "mse_history": mse_history, "l2_history": l2_history},
            f,
        )

    ss = Nx // 6
    xg = np.linspace(-0.5, 0.5, 2 * ss + 1)
    nx = len(xg)
    t_list = [0.00, 0.10, 0.20, 0.30, 0.40, 0.50]
    V = v_fdm
    assert V.shape[1] == nx, f"FDM nx={V.shape[1]} vs requested nx={nx} mismatch."

    fig, axs = plt.subplots(3, 6, figsize=(28, 12))
    for ti, tval in enumerate(t_list):
        levels = 20
        vN = eval_nn_partial_diag_slice(params, tval, xg, dim, tf=0.5, r=1.0)
        vref = ref_from_3d_fdm_partial_diag(V[ti], dim)
        m = error_metrics(vN, vref)

        cf_PINN = axs[0][ti].contourf(xg, xg, vN, levels=levels)
        axs[0][ti].set_title(r"$v_{NN}(t,x;\theta_N)$" + fr"at $t$={tval:.2f}")
        fig.colorbar(cf_PINN, ax=axs[0][ti])
        cf_FDM = axs[1][ti].contourf(xg, xg, vref, levels=levels)
        axs[1][ti].set_title(fr"Ref $V(t,x)$ at $t$={tval:.2f}")
        fig.colorbar(cf_FDM, ax=axs[1][ti])
        if ti == 5:
            levels = 0
        cf_diff = axs[2][ti].contourf(xg, xg, abs(vN - vref), levels=levels, cmap="magma")
        c_bar = fig.colorbar(cf_diff, ax=axs[2][ti])
        if ti == 5:
            cf_diff.set_clim(1e-20, 1e-4)
            c_bar.ax.yaxis.get_offset_text().set_x(3.4)
        axs[2][ti].text(0.5, -0.24, f"MSE: {m['mse']:.2e}\nRelative $L^2$-error: {m['rel_l2']:.2e}",
                         transform=axs[2][ti].transAxes, ha="center", va="top", fontsize=16)
        for k in range(3):
            axs[k][ti].set_xticks([-0.5, -0.25, 0.0, 0.25, 0.5])
            axs[k][ti].set_yticks([-0.5, -0.25, 0.0, 0.25, 0.5])
            axs[k][ti].set_rasterized(True)

    plt.tight_layout(h_pad=0.1)
    plt.savefig(out_dir / f"{stem}.png", bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {out_dir / stem}.pkl")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dim", type=int, required=True, help="Problem dimension: 3, 11, or 51 in the paper.")
    parser.add_argument("--sigma-val", type=float, required=True, help="Diffusion magnitude: 0.1, 0.3, or 0.5 in the paper.")
    args = parser.parse_args()
    run(args.dim, args.sigma_val)
