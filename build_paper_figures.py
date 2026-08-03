"""Assemble the 13 paper figures (PDF) from each subproblem's raw PNG output.

Air3D's `air3d/make_figures.py` already writes PDFs directly into figure/, so
this script only needs to rasterize->PDF the moving-obstacle and
publisher-subscriber PNGs (same trick the original repo used: wrap the PNG in
a borderless PDF page).

Run after generating the pngs (see README.md for the full pipeline):
    python build_paper_figures.py
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-paper")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


HERE = Path(__file__).resolve().parent
FIGURE_DIR = HERE / "figure"

FIGURE_MAP = {
    "moving_obstacle/data/trained_pinn_sigma_00_00.png": "moving_obstacle_main.pdf",
    "moving_obstacle/data/trained_pinn_sigma_01_00.png": "moving_obstacle_sigma_01_00.pdf",
    "moving_obstacle/data/trained_pinn_sigma_00_01.png": "moving_obstacle_sigma_00_01.pdf",
    "publisher_subscriber/data/trained_pinn_11d_sigma01.png": "ps_11d_sigma01.pdf",
    "publisher_subscriber/data/trained_pinn_11d_sigma05.png": "ps_11d_sigma05.pdf",
    "publisher_subscriber/data/trained_pinn_3d_sigma01.png": "ps_3d_sigma01.pdf",
    "publisher_subscriber/data/trained_pinn_3d_sigma03.png": "ps_3d_sigma03.pdf",
    "publisher_subscriber/data/trained_pinn_51d_sigma01.png": "ps_51d_sigma01.pdf",
    "publisher_subscriber/data/trained_pinn_51d_sigma03.png": "ps_51d_sigma03.pdf",
}

# Air3D writes these PDFs directly (see air3d/make_figures.py); listed here only
# so this script can warn if `make_figures.py` hasn't been run yet.
AIR3D_DIRECT_PDFS = [
    "air3d_brs_3d_tube.pdf",
    "air3d_slice_comparison_t0.pdf",
    "air3d_zero_levelset_fdm.pdf",
    "air3d_zero_levelset_fdpinn.pdf",
]


def export_png_as_pdf(src: Path, dst: Path) -> None:
    image = plt.imread(src)
    height, width = image.shape[:2]
    dpi = 100
    fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi, frameon=False)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(image)
    ax.set_axis_off()
    fig.savefig(dst, format="pdf", dpi=dpi, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def main() -> None:
    FIGURE_DIR.mkdir(exist_ok=True)
    for rel_src, out_name in FIGURE_MAP.items():
        src = (HERE / rel_src).resolve()
        dst = FIGURE_DIR / out_name
        if not src.exists():
            print(f"SKIP (missing input, run the corresponding train_*.py first): {src}")
            continue
        export_png_as_pdf(src, dst)
        print(f"generated pdf from png: {src} -> {dst}")

    for name in AIR3D_DIRECT_PDFS:
        if not (FIGURE_DIR / name).exists():
            print(f"SKIP (missing, run `python air3d/make_figures.py` first): {name}")


if __name__ == "__main__":
    main()
