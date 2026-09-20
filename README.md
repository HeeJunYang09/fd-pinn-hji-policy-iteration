# fd-pinn-hji-policy-iteration

Code, checkpoints, and reference data for **"Finite-Difference-Based PINN Policy
Iteration for Degenerate Viscous Hamilton–Jacobi–Isaacs Equations"**.

```
fd-pinn-hji-policy-iteration/
├── common/utils.py              # shared helpers (error metrics, value-fn evaluation)
├── moving_obstacle/
│   ├── train_and_plot.py        # policy-iteration training and evaluation
│   ├── fdm_reference.py         # moving-obstacle FDM solver
│   ├── reproduce.py             # checkpoint evaluation and figures
│   └── data/                    # fdm_reference_sigma_*.npy + trained_pinn_sigma_*.{pkl,png} (all included)
├── air3d/
│   ├── fdm_reference.py         # explicit FD solver (JAX) for the Air3D pursuit-evasion HJI eq.
│   ├── train_pinn.py            # PI-PINN training with continuous uniform collocation
│   ├── make_figures.py          # builds the 4 Air3D figures (BRS tube, slices, level sets)
│   ├── reproduce.py             # checkpoint evaluation at each grid size
│   └── data/                    # FDM references and trained models for Nx=200,400,800
├── publisher_subscriber/
│   ├── fdm_reference.py         # Numba CPU solver for the modified 3D block equations
│   ├── train_pinn.py            # PI-PINN training by dimension and diffusion
│   ├── reproduce.py             # evaluation of all 12 models and paper figures
│   ├── plotting.py              # value, reference and error panels
│   └── data/                    # final models, reference slices and settings
├── build_paper_figures.py       # wraps the moving_obstacle/PS PNGs into PDFs alongside Air3D's direct PDFs
└── figure/                      # figure PDFs
```

## Figure → script mapping

| Paper figure | Produced by |
|---|---|
| `moving_obstacle_main.pdf`, `moving_obstacle_sigma_01_00.pdf`, `moving_obstacle_sigma_00_01.pdf` | `moving_obstacle/reproduce.py --plots` |
| `air3d_brs_3d_tube.pdf`, `air3d_slice_comparison_t0.pdf`, `air3d_zero_levelset_fdm.pdf`, `air3d_zero_levelset_fdpinn.pdf` | `air3d/make_figures.py` using the Nx=800, seed=0 checkpoint |
| `ps_3d_sigma01.pdf`, `ps_3d_sigma03.pdf`, `ps_11d_sigma01.pdf`, `ps_11d_sigma05.pdf`, `ps_51d_sigma01.pdf`, `ps_51d_sigma03.pdf` | `publisher_subscriber/reproduce.py --plots` |

The figure PDFs and the checkpoints used to generate them are included.

## Running the experiments

Use the Python 3.9 requirements listed for each example. Training GPU selection
uses `HJ_GPU_ID`, for example `HJ_GPU_ID=2 python publisher_subscriber/train_pinn.py --dim 3 --sigma-val 0.1`.
Checkpoint evaluation runs on CPU by default. The supplied data suffice to
reproduce the reported errors and figures without training or FDM generation.

## Moving obstacle

Python 3.9, JAX 0.4.23, NumPy 1.26.4, and Optax 0.2.2.
CPU dependencies are listed in `moving_obstacle/requirements-py39.txt`.
The CUDA 11 setup uses `jaxlib==0.4.23+cuda11.cudnn86`.

```bash
python -m pip install -r moving_obstacle/requirements-py39.txt
python moving_obstacle/reproduce.py --check --plots
```

`--check` verifies the supplied files and recomputed errors. `--plots` generates
PNG/PDF figures. Results are written to `moving_obstacle/reproduced/`;
use `--output-dir PATH` to select another directory. Evaluation runs on CPU by
default.

| Diffusion amplitude sigma | Relative L2 error at t=0 |
|---|---:|
| 0 | 5.09e-3 |
| diag(0.1, 0) | 5.25e-3 |
| diag(0, 0.1) | 3.55e-3 |

Checkpoints contain the final network parameters and sampled controls.
Settings and file hashes are in `moving_obstacle/data/provenance.json`.
The reproduction command writes per-snapshot errors to its output directory.

The FDM grid has 800 cells per axis on `[-2,2]^2`, with `h=4/800`,
`dt=1/80000`, `nu=0.00625`, and homogeneous Neumann boundary conditions.
The stored arrays have shape `(5,401,401)` and cover `[-1,1]^2` at
`t=0,0.25,0.5,0.75,1`. Arrays are indexed as `[time,x2,x1]`; sigma components
refer to `(x1,x2)`. Evaluation uses every fourth spatial point, giving a
`101 x 101` grid, and computes
`norm(v_PINN-v_FDM)/(norm(v_FDM)+1e-8)`.

Training uses sine layers `[3,64,64,64,64,1]`, a hard terminal condition,
Adam with learning rate `1e-3`, 2,000 interior samples, and 1,000 policy updates.
Each update takes 1,000 optimizer steps. Samples are drawn uniformly from the
domain reduced by one stencil width.

```bash
# Train all three diffusion cases
python moving_obstacle/train_and_plot.py

# Generate FDM references
python moving_obstacle/fdm_reference.py --sigma 0 0
python moving_obstacle/fdm_reference.py --sigma 0.1 0
python moving_obstacle/fdm_reference.py --sigma 0 0.1
```

## Air3D

Python 3.9, JAX 0.4.23, NumPy 1.26.4, Optax 0.2.2, and scikit-image 0.24.0.
The CUDA 11 setup uses `jaxlib==0.4.23+cuda11.cudnn86`.

```bash
python -m pip install -r air3d/requirements-py39.txt
python air3d/reproduce.py --check
python air3d/make_figures.py
```

The release includes six final checkpoints with seed 0 and their FDM references:
three resolutions for each diffusion diagonal. `reproduce.py --check` verifies
file hashes and recomputed errors at five times. Results are written to
`air3d/reproduced/errors.json`; use `--output-dir PATH` to change the location.

| Nx | Relative L2, sigma = 0 | Relative L2, sigma = diag(0.03, 0.05, 0) |
|---|---:|---:|
| 200 | 0.00523221 | 0.00634437 |
| 400 | 0.00570971 | 0.00943061 |
| 800 | 0.00595062 | 0.00758169 |

Each entry uses a separately trained model at that resolution and diffusion.
Errors are evaluated at `t=0`, exactly `x3=pi/2`, on a `101 x 101` grid in
`[-1.5,1.5]^2`, using `norm(v_PINN-v_FDM)/(norm(v_FDM)+1e-8)`.
Settings and file hashes are in `air3d/data/provenance.json`.
Checkpoints contain the final network parameters, sampled controls, and training settings.
The four Air3D PDFs in `figure/` use the Nx=800, sigma=0, seed=0 checkpoint.
The 2D panels evaluate the network directly at `x3=pi/2`; the 3D rendering uses
101 angular points without a duplicate endpoint.

Training uses `ve=vp=beta=0.5`, `omega_bar=1.5`, sine layers
`[4,64,64,64,64,1]`, a hard terminal condition, Adam with learning rate `1e-3`,
500 policy updates, and 1,000 optimizer steps per update. Each batch contains
4,000 interior space-time points and 1,000 periodic-boundary pairs, with
periodic penalty weight 100. Interior coordinates are sampled independently
from continuous uniform distributions, inset from the domain boundaries by
`tau_h`, `h_x`, and `h_psi` in the respective coordinates. Batches are refreshed
at epoch 0 and every 100th optimizer step within each policy update.

The computational domain is `[0,1] x [-3,3]^2 x [-pi,pi]`, with
`h_x=6/Nx`, `h_psi=2*pi/Nx`, and `nu_h=5.5*h_x`.
The backward time step is the minimum of `h_x**2/(3*nu_h+sum(sigma_i**2))`
and `nu_h/17424`, with the additional bound `4*h_x**2/(3*max(sigma_i**2))`
when diffusion is nonzero. FDM uses float64 and `dt=1/ceil(1/tau_h)`, with
the same physical diffusion and artificial viscosity as the corresponding
network. The angular boundary is periodic; the duplicate endpoints are synchronized.
Stored references contain five planar snapshots corresponding to
`t=0,0.25,0.5,0.75,1`, at FDM time indices `floor(t*n_t)`, where `n_t=ceil(1/tau_h)`, indexed as
`[time,x1,x2]` and transposed for comparison with the PINN's `xy` grid.

```bash
# Repeat with --nx 400 and --nx 800 for the other resolutions.
python air3d/train_pinn.py --nx 200 --sigma 0 0 0 --seed 0
python air3d/train_pinn.py --nx 200 --sigma 0.03 0.05 0 --seed 0

# Optional: generate FDM references.
python air3d/fdm_reference.py --nx 200 --sigma 0 0 0
python air3d/fdm_reference.py --nx 200 --sigma 0.03 0.05 0
```

Training writes to `air3d/trained/`; FDM generation writes to
`air3d/generated_reference/`. Both commands accept `--output-dir PATH`.

## Publisher–subscriber

Python 3.9, JAX 0.4.23, NumPy 1.26.4, Optax 0.2.2, and Numba 0.60.0.
The CUDA 11 setup uses `jaxlib==0.4.23+cuda11.cudnn86`.

```bash
python -m pip install -r publisher_subscriber/requirements-py39.txt
python publisher_subscriber/reproduce.py --check --plots
```

The release contains the ten main-table models (dimensions 3, 5, 11, 25, 51;
diffusion amplitudes 0.1 and 0.5), and the supplementary 3D and 51D models
with amplitude 0.3. All use seed 42 and 2,000 times the dimension interior
samples. Checkpoints contain final parameters, diffusion, training settings,
and elapsed training time.

| Dimension | Relative L2, sigma=0.1 | Relative L2, sigma=0.5 | Training time, sigma=0.5 [s] |
|---|---:|---:|---:|
| 3 | 0.00320075 | 0.03149334 | 3450 |
| 5 | 0.00786610 | 0.04392175 | 6070 |
| 11 | 0.00756106 | 0.04650151 | 20931 |
| 25 | 0.10968663 | 0.14021456 | 95929 |
| 51 | 0.14880880 | 0.19392969 | 418018 |

For sigma=0.3, the supplementary 3D and 51D errors are 0.00577288 and
0.12419060. All errors above are evaluated at `t=0` on the partial diagonal
`x=(x0,s,...,s)`, using a `201 x 201` grid over `[-0.5,0.5]^2` and
`norm(v_PINN-v_FDM)/(norm(v_FDM)+1e-12)`.

The three `fdm_reference_slice_sigma*.npy` files store `v_3(t,x0,s,s)` at
`t=0,0.1,0.2,0.3,0.4,0.5`, indexed as `[time,x0,s]`. Evaluation scales each
slice by `(N-1)/2` and transposes it to the network's `[s,x0]` ordering.
The slices are included directly; the full three-dimensional arrays are not
needed for evaluation or training. Settings, file hashes, and expected errors
are in `publisher_subscriber/data/provenance.json`.
`--check` verifies all 12 models at six times. `--plots` generates the six PS
paper figures. Outputs go to `publisher_subscriber/reproduced/`; use
`--output-dir PATH` to change the location.

Training uses sine layers `[N+1,64,64,64,1]`, a hard terminal condition,
Adam with learning rate `1e-3`, 500 policy updates, and 5,000 optimizer steps
per update. Continuous uniform samples stay within stencil margins of
`[0,0.5] x [-1.5,1.5]^N`. Batches refresh at epoch 0 and every 500th
optimizer step. The residual uses `h=3/600`, `nu_h=0.03`, and
`tau_h=sqrt(h/(1152*N))`. Reference values are used only for evaluation.

```bash
# Repeat for the desired dimension and diffusion amplitude.
python publisher_subscriber/train_pinn.py --dim 51 --sigma-val 0.3 --seed 42

# Optional CPU generation of reference slices; the paper uses --h 600.
python publisher_subscriber/fdm_reference.py --sigma 03_00_03 --h 600
```

Training writes to `publisher_subscriber/trained/`; reference generation writes
to `publisher_subscriber/generated_reference/`. Both accept `--output-dir PATH`.
The FDM solves on `[-1.5,1.5]^3` with homogeneous Neumann boundaries, exact
coefficients `alpha=-1/(1.5^2*sin(1.5))`, `beta=1/1.5^3`, diffusion
`sigma @ sigma.T + nu_h*I`, and `nu_h=6*h_x`. Its time step is
`dt=h_x/(1152*3)`, giving 345,600 steps at `h_x=3/600` over `[0,0.5]`.
The FDM time step differs from the neural residual's practical time quotient.
Use a smaller grid, for example `--h 12`, for a quick solver check.
`--full-reference` saves the central `[-0.5,0.5]^3` crop instead of its
partial-diagonal slice.

The full 3D reference arrays are also available separately (about 390 MB each):

| File | Download |
|---|---|
| `fdm_reference_3d_sigma01.npy` | [Google Drive](https://drive.google.com/file/d/1Gginq0FDwbae6vDQpFa4kxzkpBnJeA6J/view?usp=sharing) |
| `fdm_reference_3d_sigma03.npy` | [Google Drive](https://drive.google.com/file/d/16utVosA6ANiDgQ8-fYISqxNcKYdyA49G/view?usp=sharing) |
| `fdm_reference_3d_sigma05.npy` | [Google Drive](https://drive.google.com/file/d/1K6ag_qpAwlQo8gfhR_QrhzszBCRBp1bv/view?usp=sharing) |

## License

MIT — see [LICENSE](LICENSE).
