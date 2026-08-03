"""Train the PI-PINN value function for the Air3D pursuit-evasion HJI equation
against the FDM reference produced by `fdm_reference.py`.

GPU selection: set the HJ_GPU_ID environment variable (defaults to "0"), e.g.
    HJ_GPU_ID=0 python train_pinn.py --seed 0
"""

#%%
import argparse
import os

GPU_ID = os.environ.get("HJ_GPU_ID", "0")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", GPU_ID)
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", ".9")

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(__file__).resolve().parent / "data"
if str(REPO_ROOT / "common") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "common"))

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
import pickle
from jax import random, jit, vmap, value_and_grad
from jax.nn import initializers
from tqdm import tqdm

from utils import compute_error_metrics

jax.config.update("jax_default_matmul_precision", "highest")


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


def grid_sample(key, N, step, domain):
    index = random.randint(key, (), 1, N - 1)
    return domain[0] + index * step


grid_sample = vmap(grid_sample, in_axes=(0, 0, 0, None))


def sample_grid_collocation(N, key, Nx, nu_h, tau_h, domain_t=(0.0, 1.0), domain_x=(-1.5, 1.5), domain_psi=(-jnp.pi, jnp.pi)):
    keys = random.split(key, 6)
    tau_count = int(jnp.ceil((domain_t[1] - domain_t[0]) / tau_h))

    h_vector = jnp.full((N,), Nx, dtype=jnp.int32)
    tau_vector = jnp.full((N,), tau_count, dtype=jnp.int32)
    h_step = jnp.full((N,), (domain_x[1] - domain_x[0]) / Nx)
    h_psi_step = jnp.full((N,), (domain_psi[1] - domain_psi[0]) / Nx)
    tau_step = jnp.full((N,), tau_h)
    nu_h_step = jnp.full((N,), nu_h)

    t_key = random.split(keys[2], N)
    x1_key = random.split(keys[3], N)
    x2_key = random.split(keys[4], N)
    x3_key = random.split(keys[5], N)

    ts = grid_sample(t_key, tau_vector, tau_step, domain_t)
    x1 = grid_sample(x1_key, h_vector, h_step, domain_x)
    x2 = grid_sample(x2_key, h_vector, h_step, domain_x)
    x3 = grid_sample(x3_key, h_vector, h_psi_step, domain_psi)
    return jnp.stack([ts, x1, x2, x3], axis=-1), tau_step[:, None], h_step[:, None], h_psi_step[:, None], nu_h_step[:, None]


def sample_periodic_points(NP, key, domain_t=(0.0, 1.0), domain_x=(-1.5, 1.5)):
    keys = random.split(key, 3)
    t_periodic = random.uniform(keys[0], shape=(NP,), minval=domain_t[0], maxval=domain_t[1])
    x1_periodic = random.uniform(keys[1], shape=(NP,), minval=domain_x[0], maxval=domain_x[1])
    x2_periodic = random.uniform(keys[2], shape=(NP,), minval=domain_x[0], maxval=domain_x[1])
    return jnp.stack([t_periodic, x1_periodic, x2_periodic], axis=-1)


def gen_v_single(beta):
    def v_single(data, params):
        t, x1, x2, x3 = data
        nn_output = forward(params, data)[0]
        g = jnp.sqrt(x1**2 + x2**2) - beta
        return (1.0 - t) * nn_output + g

    return v_single


def value_function_from_params_at_t(t_val, params, x1, x2, v_single, x3_value):
    num_x = len(x1)
    x_flat = jnp.stack([x1.flatten(), x2.flatten()], axis=-1)
    t_fixed = jnp.full((x_flat.shape[0], 1), t_val)
    x3_fixed = jnp.full((x_flat.shape[0], 1), x3_value)
    tx_flat = jnp.concatenate([t_fixed, x_flat, x3_fixed], axis=1)
    v_vals = vmap(lambda tx: v_single(tx, params))(tx_flat)
    return v_vals.reshape((num_x, num_x))


def make_pinn_loss(v_single):
    def residual(tx, omega_e, omega_p, params, tau, h, h_psi, nu_h, sigma):
        t, x1, x2, x3 = tx
        tau, h, h_psi, nu_h = tau[0], h[0], h_psi[0], nu_h[0]
        u_value = v_single(tx, params)

        u_back = v_single(jnp.array([t - tau, x1, x2, x3]), params)
        dv_dt = (u_value - u_back) / tau

        u_x1p = v_single(jnp.array([t, x1 + h, x2, x3]), params)
        u_x1m = v_single(jnp.array([t, x1 - h, x2, x3]), params)
        dv_dx1 = (u_x1p - u_x1m) / (2 * h)
        u_x2p = v_single(jnp.array([t, x1, x2 + h, x3]), params)
        u_x2m = v_single(jnp.array([t, x1, x2 - h, x3]), params)
        dv_dx2 = (u_x2p - u_x2m) / (2 * h)
        u_x3p = v_single(jnp.array([t, x1, x2, x3 + h_psi]), params)
        u_x3m = v_single(jnp.array([t, x1, x2, x3 - h_psi]), params)
        dv_dx3 = (u_x3p - u_x3m) / (2 * h_psi)

        u_x1p_x2p = v_single(jnp.array([t, x1 + h, x2 + h, x3]), params)
        u_x1p_x2m = v_single(jnp.array([t, x1 + h, x2 - h, x3]), params)
        u_x1m_x2p = v_single(jnp.array([t, x1 - h, x2 + h, x3]), params)
        u_x1m_x2m = v_single(jnp.array([t, x1 - h, x2 - h, x3]), params)
        d2v_dx1x2 = (u_x1p_x2p - u_x1p_x2m - u_x1m_x2p + u_x1m_x2m) / (4 * h * h)

        u_x1p_x3p = v_single(jnp.array([t, x1 + h, x2, x3 + h_psi]), params)
        u_x1p_x3m = v_single(jnp.array([t, x1 + h, x2, x3 - h_psi]), params)
        u_x1m_x3p = v_single(jnp.array([t, x1 - h, x2, x3 + h_psi]), params)
        u_x1m_x3m = v_single(jnp.array([t, x1 - h, x2, x3 - h_psi]), params)
        d2v_dx1x3 = (u_x1p_x3p - u_x1p_x3m - u_x1m_x3p + u_x1m_x3m) / (4 * h * h_psi)

        u_x2p_x3p = v_single(jnp.array([t, x1, x2 + h, x3 + h_psi]), params)
        u_x2p_x3m = v_single(jnp.array([t, x1, x2 + h, x3 - h_psi]), params)
        u_x2m_x3p = v_single(jnp.array([t, x1, x2 - h, x3 + h_psi]), params)
        u_x2m_x3m = v_single(jnp.array([t, x1, x2 - h, x3 - h_psi]), params)
        d2v_dx2x3 = (u_x2p_x3p - u_x2p_x3m - u_x2m_x3p + u_x2m_x3m) / (4 * h * h_psi)

        d2v_dx1x1 = (u_x1p - 2 * u_value + u_x1m) / (h**2)
        d2v_dx2x2 = (u_x2p - 2 * u_value + u_x2m) / (h**2)
        d2v_dx3x3 = (u_x3p - 2 * u_value + u_x3m) / (h_psi**2)
        hess_v = jnp.array(
            [
                [d2v_dx1x1, d2v_dx1x2, d2v_dx1x3],
                [d2v_dx1x2, d2v_dx2x2, d2v_dx2x3],
                [d2v_dx1x3, d2v_dx2x3, d2v_dx3x3],
            ]
        )

        drift1 = -sigma["ve"] + sigma["vp"] * jnp.cos(x3) + omega_e * x2
        drift2 = sigma["vp"] * jnp.sin(x3) - omega_e * x1
        drift3 = omega_p - omega_e
        drift_dot_grad = drift1 * dv_dx1 + drift2 * dv_dx2 + drift3 * dv_dx3

        sigma_tot = sigma["sigma"] @ sigma["sigma"].T + nu_h * jnp.eye(sigma["sigma"].shape[0])
        diffusion = jnp.trace(sigma_tot @ hess_v)

        res = dv_dt + drift_dot_grad + 0.5 * diffusion
        return res**2

    @jit
    def pinn_loss(params, grid_tx, grid_Px, P, omega_e_n, omega_p_n, tau, h, h_psi, nu_h):
        interior_loss = jnp.mean(
            vmap(
                lambda tx, a, b, tau, h, h_psi, nu_h: residual(tx, a, b, params, tau, h, h_psi, nu_h, P),
                in_axes=(0, 0, 0, 0, 0, 0, 0),
            )(grid_tx, omega_e_n, omega_p_n, tau, h, h_psi, nu_h)
        )

        def periodic_residual(tx):
            v_mpsi = v_single(jnp.array([tx[0], tx[1], tx[2], -jnp.pi]), params)
            v_ppsi = v_single(jnp.array([tx[0], tx[1], tx[2], jnp.pi]), params)
            return (v_ppsi - v_mpsi) ** 2

        periodic_loss = jnp.mean(vmap(periodic_residual, in_axes=0)(grid_Px))
        return interior_loss + P["periodic_weight"] * periodic_loss

    return pinn_loss


def make_policy_fn(v_single):
    def find_policy(tx, params, h, h_psi, omega_bar):
        t, x1, x2, x3 = tx
        h, h_psi = h[0], h_psi[0]
        v_x1p = v_single(jnp.array([t, x1 + h, x2, x3]), params)
        v_x1m = v_single(jnp.array([t, x1 - h, x2, x3]), params)
        dv_dx1 = (v_x1p - v_x1m) / (2 * h)
        v_x2p = v_single(jnp.array([t, x1, x2 + h, x3]), params)
        v_x2m = v_single(jnp.array([t, x1, x2 - h, x3]), params)
        dv_dx2 = (v_x2p - v_x2m) / (2 * h)
        v_x3p = v_single(jnp.array([t, x1, x2, x3 + h_psi]), params)
        v_x3m = v_single(jnp.array([t, x1, x2, x3 - h_psi]), params)
        dv_dx3 = (v_x3p - v_x3m) / (2 * h_psi)

        a = x2 * dv_dx1 - x1 * dv_dx2 - dv_dx3
        b = dv_dx3
        omega_e = omega_bar * jnp.sign(a)
        omega_p = -omega_bar * jnp.sign(b)
        return omega_e, omega_p

    def gen_policy_fn(params, omega_bar):
        @jit
        def policy_fn(data_tx, h, h_psi):
            return vmap(find_policy, in_axes=(0, None, 0, 0, None))(data_tx, params, h, h_psi, omega_bar)

        return policy_fn

    return gen_policy_fn


#%%
def train_value_and_update_policy_with_sample(
    N, P, v_single, pinn_loss, gen_policy_fn,
    NP=1000, num_iters=100, num_epochs=500, lr=1e-3, verbose=True, seed=42, layer=(4, 64, 64, 1), v_fdm=None,
):
    def initial_sample_ball(key, radius=1.0, dim=1):
        normal = random.normal(key, shape=(dim,))
        unit_vec = normal / jnp.linalg.norm(normal)
        r = random.uniform(key, shape=()) ** (1.0 / dim)
        return radius * r * unit_vec

    init_keys = random.split(random.key(seed), 4)
    omega_e_n = vmap(initial_sample_ball, in_axes=(0, None, None))(random.split(init_keys[0], num=N), P["omega_bar"], 1)
    omega_p_n = vmap(initial_sample_ball, in_axes=(0, None, None))(random.split(init_keys[1], num=N), P["omega_bar"], 1)

    key = init_keys[2]
    params = init_params(list(layer), key)

    optimizer = optax.adam(lr)
    opt_state = optimizer.init(params)

    @jit
    def step(params, opt_state, data_tx, data_Px, P, omega_e_n, omega_p_n, tau, h, h_psi, nu_h):
        loss, grads = value_and_grad(pinn_loss)(params, data_tx, data_Px, P, omega_e_n, omega_p_n, tau, h, h_psi, nu_h)
        updates, opt_state = optimizer.update(grads, opt_state)
        new_params = optax.apply_updates(params, updates)
        return new_params, opt_state, loss

    def compute_error_at_t0(v_fdm, params):
        x3_value = jnp.pi / 2
        test_num_x = 101
        x = jnp.linspace(-1.5, 1.5, test_num_x)
        x1, x2 = jnp.meshgrid(x, x, indexing="xy")
        v_pinn = value_function_from_params_at_t(0.0, params, x1, x2, v_single, x3_value)
        errors = compute_error_metrics(v_pinn, v_fdm[0].T)
        return errors["MSE"], errors["Relative L2"]

    mse_history, l2_history = [], []
    outer_key = random.split(init_keys[-1], 4)
    if v_fdm is not None:
        mse, l2_error = compute_error_at_t0(v_fdm, params)
        mse_history.append(mse)
        l2_history.append(l2_error)

    for n in tqdm(range(num_iters), desc="Policy Iteration", position=0):
        params_current = params.copy()
        epoch_bar = tqdm(range(num_epochs), desc=f"Train (Iter {n})", leave=False)
        policy_fn = gen_policy_fn(params_current, P["omega_bar"])
        iter_key = random.split(outer_key[1], 2)
        loss_log = []
        for epoch in epoch_bar:
            if (epoch + 1) % (num_epochs // 10) == 0 or epoch < 1:
                iter_key = random.split(iter_key[1], 2)
                keys = random.split(iter_key[0], 3)
                data_tx, tau, h, h_psi, nu_h = sample_grid_collocation(
                    N, keys[0], Nx=P["Nx"], nu_h=P["nu_h"], tau_h=P["tau_h"], domain_x=P["domain_x"]
                )
                data_Px = sample_periodic_points(NP, keys[1], domain_t=P["domain_t"], domain_x=P["domain_x"])
                omega_e_n_batch, omega_p_n_batch = policy_fn(data_tx, h, h_psi)

            params, opt_state, loss = step(
                params, opt_state, data_tx, data_Px, P, omega_e_n_batch, omega_p_n_batch, tau, h, h_psi, nu_h
            )
            loss_log.append(loss)

            if epoch >= 100:
                past_loss = loss_log[epoch - 100]
                rel_change = abs(loss - past_loss) / max(abs(past_loss), 1e-8)
                if rel_change < 1e-6:
                    break
            if verbose and epoch % 10 == 0:
                epoch_bar.set_postfix(loss=f"{loss:.4e}")

        outer_key = random.split(outer_key[0], 3)
        if v_fdm is not None:
            mse, l2_error = compute_error_at_t0(v_fdm, params)
            mse_history.append(mse)
            l2_history.append(l2_error)

    policy_fn = gen_policy_fn(params, P["omega_bar"])
    alpha_n, beta_n = policy_fn(data_tx, h, h_psi)
    return params, alpha_n, beta_n, mse_history, l2_history


#%%
def plot_value_function_at_t(t_val, v_vals, ax, x1, x2, levels=20):
    cf = ax.contourf(x1, x2, v_vals, levels=levels)
    ax.set_title(fr"Value function $v(t={t_val:.1f}, x)$")
    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")
    ax.set_aspect("equal")
    return cf


def run(seed):
    Nx = 400  # matches FDM h_400 reference; used for both figures and PARAMS_PKL naming
    step = 2
    domain_t = [0.0, 1.0]
    domain_x = [-3.0, 3.0]
    lx = domain_x[1] - domain_x[0]

    h_x = lx / Nx
    nu_h = 5.5 * h_x
    tau_h = min(h_x**2 / (3 * nu_h), nu_h / 17424)
    tau_count = int(np.ceil(1.0 / tau_h))

    sigma = jnp.zeros((3, 3))
    ve, vp, omega_bar, beta = 0.5, 0.5, 1.5, 0.5

    P = {
        "sigma": sigma, "ve": ve, "vp": vp, "omega_bar": omega_bar, "beta": beta,
        "h_x": h_x, "h_psi": 2 * jnp.pi / Nx, "nu_h": nu_h, "tau_h": tau_h, "Nx": Nx,
        "domain_t": domain_t, "domain_x": domain_x, "periodic_weight": 100.0,
    }

    v_single = gen_v_single(beta)
    pinn_loss = make_pinn_loss(v_single)
    gen_policy_fn = make_policy_fn(v_single)

    v_fdm = np.load(DATA_DIR / f"fdm_reference_h{Nx}.npy")
    v_fdm = v_fdm[:, ::step, ::step]

    num_iters, num_epoch, N, NP = 500, 1000, 4000, 1000
    print(f"h={Nx} iters={num_iters} epochs={num_epoch} nu_h={nu_h:.5f} tau_count={tau_count}")
    params, alpha_n, beta_n, mse_history, l2_history = train_value_and_update_policy_with_sample(
        N, P, v_single, pinn_loss, gen_policy_fn, NP=NP, seed=seed,
        num_iters=num_iters, num_epochs=num_epoch, layer=[4, 64, 64, 64, 64, 1], v_fdm=v_fdm,
    )

    out_dir = DATA_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"trained_pinn_h{Nx}_seed{seed}"
    with open(out_dir / f"{stem}.pkl", "wb") as f:
        pickle.dump({"params": params, "alpha_n": alpha_n, "beta_n": beta_n, "mse_history": mse_history, "l2_history": l2_history}, f)

    test_num_x = 101
    x = jnp.linspace(-1.5, 1.5, test_num_x)
    x1, x2 = jnp.meshgrid(x, x, indexing="xy")
    x3_value = jnp.pi / 2
    fig, axs = plt.subplots(3, 5, figsize=(24, 12))
    for i, t in enumerate([0.0, 0.25, 0.5, 0.75, 1.0]):
        levels = 20
        v_pinn = value_function_from_params_at_t(t, params, x1, x2, v_single, x3_value)
        v_ref_t = v_fdm[i].T
        diff = v_ref_t - v_pinn
        errors = compute_error_metrics(v_pinn, v_ref_t)

        cf_PINN = plot_value_function_at_t(t, v_pinn, axs[0][i], x1, x2)
        fig.colorbar(cf_PINN, ax=axs[0][i])
        cf_FDM = plot_value_function_at_t(t, v_ref_t, axs[1][i], x1, x2)
        fig.colorbar(cf_FDM, ax=axs[1][i])
        if i == 4:
            levels = 0
        cf_diff = plot_value_function_at_t(t, abs(diff), axs[2][i], x1, x2, levels=levels)
        c_bar = fig.colorbar(cf_diff, ax=axs[2][i])
        if i == 4:
            cf_diff.set_clim(1e-20, 1e-4)
            c_bar.ax.yaxis.get_offset_text().set_x(3.4)
        axs[2][i].text(
            0.5, -0.24, f"MSE: {errors['MSE']:.2e}\nRelative $L^2$-error: {errors['Relative L2']:.2e}",
            transform=axs[2][i].transAxes, ha="center", va="top", fontsize=16,
        )
        for k in range(3):
            axs[k][i].set_xticks([-1, -0.5, 0, 0.5, 1])
            axs[k][i].set_yticks([-1, -0.5, 0, 0.5, 1])
            axs[k][i].set_rasterized(True)

    plt.tight_layout(h_pad=0.1)
    plt.savefig(out_dir / f"{stem}.png", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.semilogy(np.arange(num_iters + 1), l2_history)
    ax.set_xlabel("Outer iteration")
    ax.set_ylabel(r"Relative $L^2$-error")
    ax.set_xlim([-1, num_iters])
    ax.set_ylim([10**-3, 10**3])
    plt.tight_layout(h_pad=0.1)
    plt.savefig(out_dir / f"{stem}_l2_history.png", bbox_inches="tight")
    plt.close(fig)

    print(f"saved: {out_dir / stem}.pkl")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0, help="Training seed. The paper figure uses seed=0.")
    args = parser.parse_args()
    run(args.seed)
