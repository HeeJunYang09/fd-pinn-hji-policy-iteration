"""Train the PI-PINN value function for the moving-obstacle problem and reproduce
the paper figures: `moving_obstacle_main.pdf`, `moving_obstacle_sigma_01_00.pdf`,
`moving_obstacle_sigma_00_01.pdf`.

GPU selection: set the HJ_GPU_ID environment variable (defaults to "0"), e.g.
    HJ_GPU_ID=2 python train_and_plot.py
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

from utils import value_function_from_params_at_t, compute_error_metrics

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
    total = 0
    for w, b in params:
        total += w.size + b.size
    return total


def grid_sample(key, step_size, domain):
    return random.uniform(key, shape=(), minval=domain[0] + step_size, maxval=domain[1] - step_size)


grid_sample = vmap(grid_sample, in_axes=(0, 0, None))


def sample_grid_collocation(N, key, minnum_h=200, maxnum_h=2000, domain_t=(0.0, 1.0), domain_x=(-2.0, 2.0)):
    keys = random.split(key, 5)

    Lx = domain_x[1] - domain_x[0]
    min_h = Lx / maxnum_h
    max_h = Lx / minnum_h
    h_collection = random.uniform(keys[1], (N,), minval=min_h, maxval=max_h)
    tau_collection = h_collection / 400
    nu_h_collection = h_collection * 1.25

    t_key = random.split(keys[2], N)
    x1_key = random.split(keys[3], N)
    x2_key = random.split(keys[4], N)

    ts = grid_sample(t_key, tau_collection, domain_t)
    x1 = grid_sample(x1_key, h_collection, domain_x)
    x2 = grid_sample(x2_key, h_collection, domain_x)
    return jnp.stack([ts, x1, x2], axis=-1), tau_collection[:, None], h_collection[:, None], nu_h_collection[:, None]


#%%
def make_v_single(lambda_3, x_goal1):
    def v_single(data, params):
        t, x1, x2 = data[0], data[1], data[2]
        x = jnp.array([x1, x2])
        nn_output = forward(params, data)[0]
        g = lambda_3 * jnp.sum((x - x_goal1) ** 2)  # terminal-condition ansatz
        return (1.0 - t) * nn_output + g

    return v_single


def make_pinn_loss(v_single, lambda_1, lambda_2, epsilon):
    def phi_fn(t, x):
        x_obs = jnp.array([0.5 * jnp.cos(jnp.pi * t), 0.5 * jnp.sin(jnp.pi * t)])
        return jnp.exp(-jnp.linalg.norm(x - x_obs) ** 2 / (2 * epsilon**2))

    def residual(tx, alpha, beta, params, tau, h, nu_h, sigma):
        t, x1, x2 = tx
        tau, h, nu_h = tau[0], h[0], nu_h[0]
        u_value = v_single(tx, params)

        tx_back = jnp.array([t - tau, x1, x2])
        dv_dt = (u_value - v_single(tx_back, params)) / tau

        u_x1p = v_single(jnp.array([t, x1 + h, x2]), params)
        u_x1m = v_single(jnp.array([t, x1 - h, x2]), params)
        dv_dx1 = (u_x1p - u_x1m) / (2 * h)
        u_x2p = v_single(jnp.array([t, x1, x2 + h]), params)
        u_x2m = v_single(jnp.array([t, x1, x2 - h]), params)
        dv_dx2 = (u_x2p - u_x2m) / (2 * h)

        u_x1p_x2p = v_single(jnp.array([t, x1 + h, x2 + h]), params)
        u_x1p_x2m = v_single(jnp.array([t, x1 + h, x2 - h]), params)
        u_x1m_x2p = v_single(jnp.array([t, x1 - h, x2 + h]), params)
        u_x1m_x2m = v_single(jnp.array([t, x1 - h, x2 - h]), params)
        d2v_dx1x2 = (u_x1p_x2p - u_x1p_x2m - u_x1m_x2p + u_x1m_x2m) / (4 * h * h)
        d2v_dx1x1 = (u_x1p - 2 * u_value + u_x1m) / (h**2)
        d2v_dx2x2 = (u_x2p - 2 * u_value + u_x2m) / (h**2)
        hess_v = jnp.array([[d2v_dx1x1, d2v_dx1x2], [d2v_dx1x2, d2v_dx2x2]])

        x = jnp.array([x1, x2])
        phi_val = phi_fn(t, x)
        policy_dot_grad = dv_dx1 * alpha[0] + dv_dx2 * alpha[1] + dv_dx1 * beta[0] + dv_dx2 * beta[1]
        sigma_tot = sigma @ sigma.T + nu_h * jnp.eye(sigma.shape[0])
        diffusion = jnp.trace(sigma_tot @ hess_v)

        res = dv_dt + lambda_1 * jnp.sum(alpha**2) + lambda_2 * phi_val + policy_dot_grad + 0.5 * diffusion
        return res**2

    @jit
    def pinn_loss(params, grid_tx, sigma, alpha_n, beta_n, tau, h, nu_h):
        return jnp.mean(
            vmap(lambda tx, a, b, tau, h, nu_h: residual(tx, a, b, params, tau, h, nu_h, sigma), in_axes=(0, 0, 0, 0, 0, 0))(
                grid_tx, alpha_n, beta_n, tau, h, nu_h
            )
        )

    return pinn_loss


def make_policy_fn(v_single, lambda_1, delta):
    def find_policy(tx, params, h):
        t, x1, x2 = tx
        h = h[0]
        v_x1p = v_single(jnp.array([t, x1 + h, x2]), params)
        v_x1m = v_single(jnp.array([t, x1 - h, x2]), params)
        dv_dx1 = (v_x1p - v_x1m) / (2 * h)
        v_x2p = v_single(jnp.array([t, x1, x2 + h]), params)
        v_x2m = v_single(jnp.array([t, x1, x2 - h]), params)
        dv_dx2 = (v_x2p - v_x2m) / (2 * h)

        dv_dx = jnp.stack([dv_dx1, dv_dx2], axis=0)
        grad_norm = jnp.linalg.norm(dv_dx) + 1e-10
        alpha_unclipped = -dv_dx / (2 * lambda_1)
        alpha = jnp.where(grad_norm <= 2 * lambda_1, alpha_unclipped, -dv_dx / grad_norm)
        beta = delta * dv_dx / grad_norm
        return alpha, beta

    def gen_policy_fn(params):
        @jit
        def policy_fn(data_tx, h):
            return vmap(find_policy, in_axes=(0, None, 0))(data_tx, params, h)

        return policy_fn

    return gen_policy_fn


#%%
def train_value_and_update_policy_with_sample(
    N, sigma, v_single, pinn_loss, gen_policy_fn,
    minnum_h=200, maxnum_h=2000, num_iters=100, num_epochs=1000, lr=1e-3,
    verbose=True, seed=42, delta=0.1, layer=(3, 64, 64, 1), v_fdm=None,
):
    """Alternating PI-PINN training: fit v(t,x) under a fixed policy, then update
    the policy from the current value function's gradient."""

    def initial_sample_ball(key, radius=1.0, dim=2):
        normal = random.normal(key, shape=(dim,))
        unit_vec = normal / jnp.linalg.norm(normal)
        r = random.uniform(key, shape=()) ** (1.0 / dim)
        return radius * r * unit_vec

    init_keys = random.split(random.key(seed), 4)
    init_alpha_keys = random.split(init_keys[0], num=N)
    init_beta_keys = random.split(init_keys[1], num=N)
    alpha_n = vmap(initial_sample_ball, in_axes=(0, None, None))(init_alpha_keys, 1.0, 2)
    beta_n = vmap(initial_sample_ball, in_axes=(0, None, None))(init_beta_keys, delta, 2)

    key = init_keys[2]
    params = init_params(list(layer), key)

    optimizer = optax.adam(lr)
    opt_state = optimizer.init(params)

    @jit
    def step(params, opt_state, data_tx, sigma, alpha_n, beta_n, tau, h, nu_h):
        loss, grads = value_and_grad(pinn_loss)(params, data_tx, sigma, alpha_n, beta_n, tau, h, nu_h)
        updates, opt_state = optimizer.update(grads, opt_state)
        new_params = optax.apply_updates(params, updates)
        return new_params, opt_state, loss

    def compute_error_at_t0(v_fdm, params):
        test_num_x = 101
        x = jnp.linspace(-1, 1, test_num_x)
        x1, x2 = jnp.meshgrid(x, x, indexing="xy")
        v_pinn = value_function_from_params_at_t(0.0, params, x1, x2, v_single)
        errors = compute_error_metrics(v_pinn, v_fdm[0])
        return errors["MSE"], errors["Relative L2"]

    data_test, _, _, _ = sample_grid_collocation(10000, random.key(9999), minnum_h=minnum_h, maxnum_h=maxnum_h)
    mse_history, l2_history, v_inf_history = [], [], []
    outer_key = random.split(init_keys[-1], 4)
    v_prev = forward(params, data_test)
    if v_fdm is not None:
        mse, l2_error = compute_error_at_t0(v_fdm, params)
        mse_history.append(mse)
        l2_history.append(l2_error)

    refresh_interval = max(1, num_epochs // 10)
    for n in tqdm(range(num_iters), desc="Policy Iteration", position=0):
        params_current = params.copy()
        epoch_bar = tqdm(range(num_epochs), desc=f"Train (Iter {n})", leave=False)
        policy_fn = gen_policy_fn(params_current)
        iter_key = random.split(outer_key[1], 2)
        for epoch in epoch_bar:
            if (epoch + 1) % refresh_interval == 0 or epoch < 1:
                iter_key = random.split(iter_key[1], 2)
                keys = random.split(iter_key[0], 3)
                data_tx, tau, h, nu_h = sample_grid_collocation(N, keys[0], minnum_h=minnum_h, maxnum_h=maxnum_h)
                alpha_batch, beta_batch = policy_fn(data_tx, h)

            params, opt_state, loss = step(params, opt_state, data_tx, sigma, alpha_batch, beta_batch, tau, h, nu_h)
            if verbose and epoch % 10 == 0:
                epoch_bar.set_postfix(loss=f"{loss:.4e}")

        v_next = forward(params, data_test)
        v_inf_history.append(jnp.max(abs(v_next - v_prev)))
        v_prev = v_next
        outer_key = random.split(outer_key[0], 3)
        if v_fdm is not None:
            mse, l2_error = compute_error_at_t0(v_fdm, params)
            mse_history.append(mse)
            l2_history.append(l2_error)

    policy_fn = gen_policy_fn(params)
    alpha_n, beta_n = policy_fn(data_tx, h)
    return params, alpha_n, beta_n, mse_history, l2_history, v_inf_history


#%%
def plot_value_function_at_t(t_val, v_vals, ax, x1, x2, levels=20):
    cf = ax.contourf(x1, x2, v_vals, levels=levels)
    ax.set_title(fr"Value function $v(t={t_val:.1f}, x)$")
    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")
    ax.set_aspect("equal")
    return cf


def load_paper_reference(sigma_text):
    """Load the Nx=800 reference on the 101x101 evaluation grid.

    The file contains five snapshots cropped to [-1,1]^2 (401x401 nodes).
    The full FDM solve uses [-2,2]^2, h=4/800, dt=1/80000 and nu=0.00625.
    Arrays use xy indexing: axis 1 is x1 and axis 0 is x2.
    """
    path = DATA_DIR / f"fdm_reference_{sigma_text}.npy"
    values = np.load(path)
    if values.shape != (5, 401, 401):
        raise ValueError(
            f"{path.name}: expected an Nx=800 crop (5, 401, 401), "
            f"got {values.shape}. Use the Nx=800 reference shipped with this example."
        )
    return values[:, ::4, ::4]


def run_one_sigma(sigma, sigma_text, output_dir=None):
    minnum_h = maxnum_h = 800  # matches the paper run: min_nu = max_nu = 1.25*4/800 = 0.00625

    lambda_1, lambda_2, lambda_3 = 0.1, 1, 0.1
    delta, epsilon = 0.1, 0.3
    x_goal1 = jnp.array([0.9, 0.9])

    v_fdm = load_paper_reference(sigma_text)

    num_iters, num_epoch, N = 1000, 1000, 2000
    min_nu = 1.25 * (4 / minnum_h)
    max_nu = 1.25 * (4 / maxnum_h)

    v_single = make_v_single(lambda_3, x_goal1)
    pinn_loss = make_pinn_loss(v_single, lambda_1, lambda_2, epsilon)
    gen_policy_fn = make_policy_fn(v_single, lambda_1, delta)

    params, alpha_n, beta_n, mse_history, l2_history, v_inf_history = train_value_and_update_policy_with_sample(
        N, sigma, v_single, pinn_loss, gen_policy_fn,
        num_iters=num_iters, num_epochs=num_epoch, layer=[3, 64, 64, 64, 64, 1],
        v_fdm=v_fdm, minnum_h=minnum_h, maxnum_h=maxnum_h,
    )

    out_dir = Path(output_dir) if output_dir is not None else DATA_DIR.parent / "trained"
    out_dir.mkdir(parents=True, exist_ok=True)
    params_num = count_params(params)
    print(f"[{sigma_text}] N={N} iters={num_iters} epochs={num_epoch} params={params_num} min_nu={min_nu} max_nu={max_nu}")
    stem = f"trained_pinn_{sigma_text}"

    with open(out_dir / f"{stem}.pkl", "wb") as f:
        pickle.dump(
            {"params": params, "alpha_n": alpha_n, "beta_n": beta_n,
             "mse_history": mse_history, "l2_history": l2_history, "v_inf_history": v_inf_history},
            f,
        )

    plot_checkpoint(params, v_fdm, v_single, out_dir / f"{stem}.png")

    x_range = np.arange(num_iters + 1)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.semilogy(x_range, l2_history)
    ax.set_xlabel("Outer iteration")
    ax.set_ylabel(r"Relative $L^2$-error")
    ax.set_xlim([0, num_iters])
    ax.set_ylim([10**-3, 10**3])
    plt.tight_layout(h_pad=0.1)
    plt.savefig(out_dir / f"{stem}_l2_history.png", bbox_inches="tight")
    plt.close(fig)

    print(f"[{sigma_text}] saved: {out_dir / stem}.pkl / .png")


def plot_checkpoint(params, v_fdm, v_single, output_path):
    """Plot the same five snapshots for a trained or released checkpoint."""
    test_num_x = 101
    x = jnp.linspace(-1, 1, test_num_x)
    x1, x2 = jnp.meshgrid(x, x, indexing="xy")
    fig, axs = plt.subplots(3, 5, figsize=(24, 12))
    for i, t in enumerate([0.0, 0.25, 0.5, 0.75, 1.0]):
        levels = 20
        v_pinn = value_function_from_params_at_t(t, params, x1, x2, v_single)
        v_ref_t = v_fdm[i]
        diff = v_ref_t - v_pinn
        errors = compute_error_metrics(v_pinn, v_ref_t)

        cf_PINN = plot_value_function_at_t(t, v_pinn, axs[0][i], x1, x2)
        fig.colorbar(cf_PINN, ax=axs[0][i])
        axs[0][i].set_title(r"$v^{\mathcal{S}}_N(t,x;\theta_N)$" + fr"at $t$={t:.2f}")

        cf_FDM = plot_value_function_at_t(t, v_ref_t, axs[1][i], x1, x2)
        fig.colorbar(cf_FDM, ax=axs[1][i])
        axs[1][i].set_title(fr"Ref $V(t,x)$ at $t$={t:.2f}")

        if i == 4:
            levels = 0
        cf_diff = plot_value_function_at_t(t, abs(diff), axs[2][i], x1, x2, levels=levels)
        axs[2][i].set_title(fr"$\left|\text{{Ref }} v(t,x)" + r"- v^{\mathcal{S}}_N(t,x;\theta_N)\right|$")
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
    plt.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


#%%
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DATA_DIR.parent / "trained")
    args = parser.parse_args()
    # sigma_00_00 -> moving_obstacle_main.pdf
    # sigma_01_00 -> moving_obstacle_sigma_01_00.pdf
    # sigma_00_01 -> moving_obstacle_sigma_00_01.pdf
    sigma_list = [jnp.array([[0.0, 0.0], [0.0, 0.0]]), jnp.array([[0.1, 0.0], [0.0, 0.0]]), jnp.array([[0.0, 0.0], [0.0, 0.1]])]
    sigma_text_list = ["sigma_00_00", "sigma_01_00", "sigma_00_01"]
    for sigma, sigma_text in zip(sigma_list, sigma_text_list):
        run_one_sigma(sigma, sigma_text, output_dir=args.output_dir)
