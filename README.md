# fd-pinn-hji-policy-iteration

Code (and the intermediate/final data needed to run it) that reproduces the
13 figures from **"Finite-Difference-Based PINN Policy Iteration for
Degenerate Viscous Hamilton–Jacobi–Isaacs Equations"** (`degen_PI_paper.pdf`,
`paper/figure/*.pdf`). Each subproblem follows the same pattern: an FDM
reference solver, a PI-PINN training script, and a figure-assembly step.

```
fd-pinn-hji-policy-iteration/
├── common/utils.py              # shared helpers (error metrics, value-fn evaluation)
├── moving_obstacle/
│   ├── train_and_plot.py        # FDM ref is precomputed data; trains 3 sigma settings, plots directly
│   └── data/                    # fdm_reference_sigma_*.npy + trained_pinn_sigma_*.{pkl,png} (all included)
├── air3d/
│   ├── fdm_reference.py         # explicit FD solver (JAX) for the Air3D pursuit-evasion HJI eq.
│   ├── train_pinn.py            # PI-PINN training against the FDM reference
│   ├── make_figures.py          # builds the 4 Air3D figures (BRS tube, slices, level sets)
│   └── data/                    # fdm_reference_h{50,100,200,400,800}.npy + trained_pinn_h400_seed0.{pkl,png} (all included)
├── publisher_subscriber/
│   ├── fdm_reference.py         # Numba CPU FD solver (3D; reused for 11D/51D by projection)
│   ├── train_pinn.py            # PI-PINN training + plot, parameterized by --dim / --sigma-val
│   └── data/                    # trained_pinn_{3d,11d,51d}_sigma{01,03,05}.{pkl,png} (see "Excluded files" below)
├── build_paper_figures.py       # wraps the moving_obstacle/PS PNGs into PDFs alongside Air3D's direct PDFs
└── figure/                      # the 13 output PDFs (already built; identical to paper/figure/)
```

## Figure → script mapping

| Paper figure | Produced by |
|---|---|
| `moving_obstacle_main.pdf`, `moving_obstacle_sigma_01_00.pdf`, `moving_obstacle_sigma_00_01.pdf` | `moving_obstacle/train_and_plot.py` → `build_paper_figures.py` |
| `air3d_brs_3d_tube.pdf`, `air3d_slice_comparison_t0.pdf`, `air3d_zero_levelset_fdm.pdf`, `air3d_zero_levelset_fdpinn.pdf` | `air3d/fdm_reference.py` → `air3d/train_pinn.py` → `air3d/make_figures.py` (writes PDFs directly) |
| `ps_3d_sigma01.pdf`, `ps_3d_sigma03.pdf`, `ps_11d_sigma01.pdf`, `ps_11d_sigma05.pdf`, `ps_51d_sigma01.pdf`, `ps_51d_sigma03.pdf` | `publisher_subscriber/fdm_reference.py` → `publisher_subscriber/train_pinn.py` → `build_paper_figures.py` |

`figure/` already contains all 13 PDFs, generated from the checked-in `data/`
folders — you don't need to retrain anything just to see the figures.

## Reproducing from scratch

GPU selection everywhere is via the `HJ_GPU_ID` environment variable
(defaults to `"0"`), e.g. `HJ_GPU_ID=2 python train_pinn.py ...`.

```bash
pip install -r requirements.txt

# Moving obstacle (loops over all 3 sigma settings itself)
python moving_obstacle/train_and_plot.py

# Air3D
python air3d/fdm_reference.py --sigma 0 0 0        # ~minutes on GPU
python air3d/train_pinn.py --seed 0                # the paper figure uses seed 0
python air3d/make_figures.py

# Publisher-subscriber (repeat for the (dim, sigma) pairs used in the paper)
python publisher_subscriber/fdm_reference.py --sigma 01_00_01 --h 600   # slow CPU solve, see note below
python publisher_subscriber/fdm_reference.py --sigma 03_00_03 --h 600
python publisher_subscriber/train_pinn.py --dim 3  --sigma-val 0.1
python publisher_subscriber/train_pinn.py --dim 3  --sigma-val 0.3
python publisher_subscriber/train_pinn.py --dim 11 --sigma-val 0.1
python publisher_subscriber/train_pinn.py --dim 11 --sigma-val 0.5   # needs sigma 05_00_05 FDM ref too
python publisher_subscriber/train_pinn.py --dim 51 --sigma-val 0.1
python publisher_subscriber/train_pinn.py --dim 51 --sigma-val 0.3

# Assemble the moving-obstacle / PS PNGs into PDFs (Air3D's are already written by make_figures.py)
python build_paper_figures.py
```

Per the paper, the publisher-subscriber references for 11D/51D are **not**
solved directly — `train_pinn.py` reuses the 3D FDM solution and projects it
(`ref_from_3d_fdm_partial_diag`), so only three 3D FDM solves are needed
(`sigma_01_00_01`, `sigma_03_00_03`, `sigma_05_00_05`).

## Excluded large files

A few files exceed GitHub's 100MB per-file limit and are **not** included in
this folder. Regenerate them with the commands above if you need to retrain
from scratch — the already-trained results (small `.pkl`/`.png`) needed to
reproduce the figures without retraining ARE included.

| File | Size | Regenerate with | Mirror |
|---|---|---|---|
| `publisher_subscriber/data/fdm_reference_3d_sigma01.npy` | 372MB | `publisher_subscriber/fdm_reference.py --sigma 01_00_01 --h 600` | [Google Drive](https://drive.google.com/file/d/1Gginq0FDwbae6vDQpFa4kxzkpBnJeA6J/view?usp=sharing) |
| `publisher_subscriber/data/fdm_reference_3d_sigma03.npy` | 372MB | `publisher_subscriber/fdm_reference.py --sigma 03_00_03 --h 600` | [Google Drive](https://drive.google.com/file/d/16utVosA6ANiDgQ8-fYISqxNcKYdyA49G/view?usp=sharing) |
| `publisher_subscriber/data/fdm_reference_3d_sigma05.npy` | 372MB | `publisher_subscriber/fdm_reference.py --sigma 05_00_05 --h 600` | [Google Drive](https://drive.google.com/file/d/1K6ag_qpAwlQo8gfhR_QrhzszBCRBp1bv/view?usp=sharing) |
| `publisher_subscriber/data/trained_pinn_51d_sigma03.pkl` | 156MB | `publisher_subscriber/train_pinn.py --dim 51 --sigma-val 0.3` | [Google Drive](https://drive.google.com/file/d/1ViEpbDAynng47tLpn9tmTcSV9iqkRYv7/view?usp=sharing) |

All four are also available as a folder: https://drive.google.com/drive/folders/1IJl1f3VmWVtcet5dmjgnwmKSIZ1hkvHy?usp=drive_link
(verified publicly downloadable — no sign-in required — with byte-exact sizes
matching the files staged at `../large_files_for_drive/`, a sibling folder
outside this repo).

Note: the FDM solve at `h=600` is expensive (the paper's own runs took a long
time on a multi-core CPU with Numba parallelism); use a smaller `--h` for a
quick smoke test of the solver itself.

## Where this code came from

Ported and cleaned up from the exploratory research scripts in the parent
`policy_iterative_discrete/` tree (which has many near-duplicate,
GPU-index-suffixed variants accumulated over the project's iteration). GPU
selection was hardcoded per-file there; here it's a single env var, and each
subproblem's training loop is one script instead of several near-identical
copies.

## License

MIT — see [LICENSE](LICENSE).
