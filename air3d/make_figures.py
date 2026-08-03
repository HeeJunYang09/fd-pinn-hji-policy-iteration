"""Build the four Air3D paper figures from the FDM reference (fdm_reference.py)
and trained PINN params (train_pinn.py):
    air3d_brs_3d_tube.pdf, air3d_slice_comparison_t0.pdf,
    air3d_zero_levelset_fdm.pdf, air3d_zero_levelset_fdpinn.pdf
"""

#%%
import os

GPU_ID = os.environ.get("HJ_GPU_ID", "0")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", GPU_ID)
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", ".9")

#%%
from pathlib import Path
import pickle

import jax.numpy as jnp
from jax import jit, vmap
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np

try:
    from skimage import measure
except ImportError as exc:
    raise ImportError("Please install scikit-image to render the 3D Air3D tube figure.") from exc


#%%
HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
FIGURE_DIR = HERE.parents[0] / "figure"

NX = 400
SEED = 0

PARAMS_PKL = DATA_DIR / f"trained_pinn_h{NX}_seed{SEED}.pkl"
FDM_NPY = DATA_DIR / f"fdm_reference_h{NX}.npy"

BETA = 0.5
TIMES = (1.0, 0.5, 0.0)
FDM_TIME_INDICES = (4, 2, 0)
SLICE_X3 = np.pi / 2.0
DOMAIN_X = (-1.5, 1.5)
DOMAIN_PSI = (-np.pi, np.pi)
PINN_GRID = (101, 101, 101)
PINN_BATCH = 200_000
PINN_COLORS = ((0.6, 0.35, 0.75), (0.8, 0.25, 0.45), (1.0, 0.15, 0.15))
PINN_ALPHAS = (0.65, 0.75, 0.85)
PINN_LABELS = (r"$t=1.0$", r"$t=0.5$", r"$t=0.0$")

SAVE_3D = FIGURE_DIR / "air3d_brs_3d_tube"
SAVE_FDPINN_2D = FIGURE_DIR / "air3d_zero_levelset_fdpinn"
SAVE_FDM_2D = FIGURE_DIR / "air3d_zero_levelset_fdm"
SAVE_COMPARE_T0 = FIGURE_DIR / "air3d_slice_comparison_t0"


#%%
def load_params_pkl(file_path):
    with open(file_path, "rb") as f:
        load_data = pickle.load(f)
    if isinstance(load_data, dict):
        if "params" not in load_data:
            raise KeyError(f"'params' not found in {file_path}")
        return load_data["params"], load_data
    return load_data, {}


def downsample_fdm_to_match(v_fdm, target_size):
    if v_fdm.ndim != 3:
        raise ValueError(f"Expected FDM array of shape (num_times, nx, ny), got {v_fdm.shape}")
    nx = v_fdm.shape[1]
    if nx != v_fdm.shape[2]:
        raise ValueError(f"Expected square FDM slices, got {v_fdm.shape}")
    if nx == target_size:
        return v_fdm
    if (nx - 1) % (target_size - 1) != 0:
        raise ValueError(f"Cannot downsample FDM grid {nx} to {target_size} with uniform stride.")
    stride = (nx - 1) // (target_size - 1)
    return v_fdm[:, ::stride, ::stride]


#%%
@jit
def forward(params, data):
    x = data
    for (w, b) in params[:-1]:
        x = jnp.sin(jnp.matmul(x, w) + b)
    (w, b) = params[-1]
    return jnp.matmul(x, w) + b


def gen_v_single(beta):
    def v_single(data, params):
        t, x1, x2, x3 = data
        del x3
        nn_output = forward(params, data)[0]
        g = jnp.sqrt(x1**2 + x2**2) - beta
        return (1.0 - t) * nn_output + g

    return v_single


def make_v_func(v_single, params):
    f_batch = jit(vmap(lambda row: v_single(row, params)))

    def v_func(batch_np):
        return np.array(f_batch(jnp.array(batch_np)))

    return v_func


#%%
def plot_brs3d_from_jax(v_single, params, *, t0=0.0, x1=DOMAIN_X, x2=DOMAIN_X, x3=DOMAIN_PSI,
                         n=PINN_GRID, batch=PINN_BATCH, color=(1.0, 0.15, 0.15), alpha=0.85, ax=None):
    v_func = make_v_func(v_single, params)

    x1g = np.linspace(*x1, n[0], dtype=np.float32)
    x2g = np.linspace(*x2, n[1], dtype=np.float32)
    x3g = np.linspace(x3[0], x3[1], n[2], endpoint=False, dtype=np.float32)

    X1, X2, X3 = np.meshgrid(x1g, x2g, x3g, indexing="xy")
    num_total = X1.size
    flat = np.column_stack([np.full(num_total, t0, np.float32), X1.ravel(), X2.ravel(), X3.ravel()])

    V = np.empty(num_total, dtype=np.float32)
    for start in range(0, num_total, batch):
        stop = min(start + batch, num_total)
        V[start:stop] = v_func(flat[start:stop])
    V = V.reshape(X1.shape)

    if ax is None:
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection="3d")

    dy, dx, dz = x2g[1] - x2g[0], x1g[1] - x1g[0], x3g[1] - x3g[0]
    verts, faces, _, _ = measure.marching_cubes(V, level=0.0, spacing=(dy, dx, dz))
    verts[:, 0] += x2[0]
    verts[:, 1] += x1[0]
    verts[:, 2] += x3[0]

    tri = verts[faces]
    normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    normals /= np.linalg.norm(normals, axis=1, keepdims=True) + 1e-12
    light = np.array([-0.3, -0.2, 0.9], dtype=float)
    light /= np.linalg.norm(light) + 1e-12
    intensity = 0.25 + 0.75 * np.clip(normals @ light, 0.0, 1.0)
    base_rgb = np.array(color, float)
    face_colors = np.empty((tri.shape[0], 4))
    face_colors[:, :3] = base_rgb * intensity[:, None]
    face_colors[:, 3] = alpha

    mesh = Poly3DCollection(tri, facecolors=face_colors, edgecolor="none")
    ax.add_collection3d(mesh)
    ax.set_xlim(x1[1], x1[0])
    ax.set_ylim(*x2)
    ax.set_zlim(*x3)
    ax.set_box_aspect((x1[1] - x1[0], x2[1] - x2[0], x3[1] - x3[0]))
    ax.set_proj_type("ortho")
    ax.set_xticks([x1[0], 0.0, x1[1]])
    ax.set_yticks([0.0, x2[1]])
    ax.tick_params(axis="x", pad=-1)
    ax.tick_params(axis="y", pad=-1)
    ax.set_xlabel(r"$x_2$", labelpad=-1)
    ax.set_ylabel(r"$x_1$", labelpad=-2)
    ax.set_zlabel(r"$x_3$")
    ax.xaxis.set_rotate_label(False)
    ax.yaxis.set_rotate_label(False)
    ax.zaxis.set_rotate_label(False)
    ax.view_init(elev=15, azim=-25)
    return V, (x1g, x2g, x3g)


def nearest_x3_idx(x3g, target):
    two_pi = 2.0 * np.pi
    delta = (x3g - target + np.pi) % two_pi - np.pi
    return int(np.argmin(np.abs(delta)))


def plot_slice_x3_multi(V_list, x1g, x2g, x3g, *, x3_target=SLICE_X3, colors=PINN_COLORS,
                         alphas=PINN_ALPHAS, labels=PINN_LABELS, ax=None, method="FDM"):
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))

    k = nearest_x3_idx(x3g, x3_target)
    handles = []
    order = np.argsort(alphas)
    for idx in order:
        V = V_list[idx]
        color, alpha, label = colors[idx], alphas[idx], labels[idx]
        Vsl = V if V.ndim == 2 else V[:, :, k]
        fill_rgba = mcolors.to_rgba(color, alpha)
        ax.contourf(x1g, x2g, Vsl, levels=[float(Vsl.min()), 0.0], colors=[fill_rgba], zorder=1)
        handles.append(mpatches.Patch(color=color, alpha=alpha, label=label))

    ax.legend(handles=handles, loc="upper right", framealpha=1.0, edgecolor="0.8")
    ax.set_title(rf"BRS at $x_3=\pi/2$ over $t$ ({method})")
    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$", rotation=0)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(x1g.min(), x1g.max())
    ax.set_ylim(x2g.min(), x2g.max())
    return ax


def save_current_figure(base_path):
    plt.savefig(base_path.with_suffix(".pdf"), bbox_inches="tight")


#%%
def main():
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    params, _ = load_params_pkl(PARAMS_PKL)
    v_fdm = np.load(FDM_NPY)
    v_single = gen_v_single(BETA)

    fig, ax = plt.subplots(figsize=(8, 6), subplot_kw={"projection": "3d"})
    V1, (x1g, x2g, x3g) = plot_brs3d_from_jax(v_single, params, alpha=PINN_ALPHAS[0], color=PINN_COLORS[0], t0=TIMES[0], n=PINN_GRID, ax=ax)
    V05, _ = plot_brs3d_from_jax(v_single, params, alpha=PINN_ALPHAS[1], color=PINN_COLORS[1], t0=TIMES[1], n=PINN_GRID, ax=ax)
    V0, _ = plot_brs3d_from_jax(v_single, params, alpha=PINN_ALPHAS[2], color=PINN_COLORS[2], t0=TIMES[2], n=PINN_GRID, ax=ax)
    h_t1 = mpatches.Patch(color=PINN_COLORS[0], alpha=PINN_ALPHAS[0], label=r"$t = 1.0$")
    h_t05 = mpatches.Patch(color=PINN_COLORS[1], alpha=PINN_ALPHAS[1], label=r"$t = 0.5$")
    h_t0 = mpatches.Patch(color=PINN_COLORS[2], alpha=PINN_ALPHAS[2], label=r"$t = 0.0$")
    ax.legend(handles=[h_t0, h_t05, h_t1], title=r"BRS at time $t$", loc="upper left", framealpha=1.0, edgecolor="0.8")
    plt.tight_layout()
    save_current_figure(SAVE_3D)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    plot_slice_x3_multi([V1, V05, V0], x1g, x2g, x3g, ax=ax, method="FD PINN")
    plt.tight_layout()
    save_current_figure(SAVE_FDPINN_2D)
    plt.close(fig)

    v_fdm_ds = downsample_fdm_to_match(v_fdm, len(x1g))
    fdm_slices = [v_fdm_ds[idx].T for idx in FDM_TIME_INDICES]
    fig, ax = plt.subplots(figsize=(6, 5))
    plot_slice_x3_multi(fdm_slices, x1g, x2g, x3g, ax=ax, method="FDM")
    plt.tight_layout()
    save_current_figure(SAVE_FDM_2D)
    plt.close(fig)

    titles = [r"BRS (FDM) $t=0$", r"BRS (FD PINN) $t=0$"]
    k_idx = nearest_x3_idx(x3g, SLICE_X3)
    datas = [v_fdm_ds[FDM_TIME_INDICES[-1]].T, V0[:, :, k_idx]]
    data_ref = datas[0]
    step = 0.01
    vmin_disp = np.floor(data_ref.min() / step) * step
    vmax_disp = np.ceil(data_ref.max() / step) * step
    norm = mcolors.TwoSlopeNorm(vmin=vmin_disp, vcenter=0.0, vmax=vmax_disp)

    fig, ax = plt.subplots(1, 3, figsize=(12, 3), dpi=200)
    for idx in range(2):
        im = ax[idx].imshow(datas[idx], origin="lower", extent=[DOMAIN_X[0], DOMAIN_X[1], DOMAIN_X[0], DOMAIN_X[1]],
                             cmap="bwr", norm=norm, interpolation="bicubic")
        ax[idx].set_aspect("equal", adjustable="box")
        ax[idx].set_xticks([DOMAIN_X[0], 0.0, DOMAIN_X[1]])
        ax[idx].set_yticks([DOMAIN_X[0], 0.0, DOMAIN_X[1]])
        ax[idx].set_title(titles[idx])

    im3 = ax[-1].imshow(np.abs(datas[0] - datas[1]), origin="lower", extent=[DOMAIN_X[0], DOMAIN_X[1], DOMAIN_X[0], DOMAIN_X[1]],
                         cmap="coolwarm", interpolation="bicubic")
    ax[-1].set_aspect("equal", adjustable="box")
    ax[-1].set_xticks([DOMAIN_X[0], 0.0, DOMAIN_X[1]])
    ax[-1].set_yticks([DOMAIN_X[0], 0.0, DOMAIN_X[1]])
    ax[-1].set_title("Absolute error")
    fig.colorbar(im3, ax=ax[-1])

    fig.subplots_adjust(left=0.055, right=0.97, bottom=0.14, top=0.88, wspace=0.38)
    fig.canvas.draw()
    pos = ax[0].get_position()
    cax = fig.add_axes([pos.x0 + 0.23 * pos.width, pos.y0 + 0.90 * pos.height, 0.45 * pos.width, 0.05 * pos.height])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.set_ticks([vmin_disp, 0.0, vmax_disp])
    cb.ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    cb.ax.tick_params(labelsize=7, pad=1)
    save_current_figure(SAVE_COMPARE_T0)
    plt.close(fig)

    print(f"Loaded params from: {PARAMS_PKL}")
    print(f"Loaded FDM from:    {FDM_NPY}")
    for p in (SAVE_3D, SAVE_FDPINN_2D, SAVE_FDM_2D, SAVE_COMPARE_T0):
        print(f"Saved: {p.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
