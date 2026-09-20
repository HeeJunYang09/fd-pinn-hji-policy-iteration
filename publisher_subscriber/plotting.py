"""Publisher-subscriber value, reference and error panels."""
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from train_pinn import eval_nn_partial_diag_slice, ref_from_3d_fdm_partial_diag, error_metrics


def plot_slices(params, reference, dimension, output_png):
    V = reference
    N, T = dimension, .5
    xg = np.linspace(-.5, .5, V.shape[1])
    t_list = [0., .1, .2, .3, .4, .5]
    fig, axs = plt.subplots(3, 6, figsize=(28, 12))
    for ti, tval in enumerate(t_list):
        levels = 20
        vN = eval_nn_partial_diag_slice(params, tval, xg, N, tf=T, r=1.0)

        V3 = V[ti]
        vref = ref_from_3d_fdm_partial_diag(V3, N)

        m = error_metrics(vN, vref)
        mse = m['mse']
        rel_l2 = m['rel_l2']

        cf_PINN = axs[0][ti].contourf(xg, xg, vN, levels=levels)
        axs[0][ti].set_title(r"$v_{NN}(t,x;\theta_N)$" + fr"at $t$={tval:.2f}")
        fig.colorbar(cf_PINN, ax=axs[0][ti])

        cf_FDM = axs[1][ti].contourf(xg, xg, vref, levels=levels)
        axs[1][ti].set_title(fr"Ref $V(t,x)$ at $t$={tval:.2f}")
        fig.colorbar(cf_FDM, ax=axs[1][ti])
        if ti == 5:
            levels = 0
        cf_diff = axs[2][ti].contourf(xg, xg, abs(vN - vref), levels=levels, cmap='magma')
        axs[2][ti].set_title(fr"$\left|\text{{Ref }} v(t,x)" + r"- v_{NN}(t,x;\theta_N)\right|$")
        c_bar = fig.colorbar(cf_diff, ax=axs[2][ti])
        if ti == 5:
            cf_diff.set_clim(1e-20, 1e-4)
            offset_text = c_bar.ax.yaxis.get_offset_text()
            offset_text.set_x(3.4)

        axs[2][ti].text(
            0.5, -0.24,
            f"MSE: {mse:.2e}\nRelative $L^2$-error: {rel_l2:.2e}",
            transform=axs[2][ti].transAxes,
            ha='center',
            va='top',
            fontsize=16
        )

        for k in range(3):
            axs[k][ti].set_xticks([-0.5, -0.25, 0., 0.25, 0.5])
            axs[k][ti].set_yticks([-0.5, -0.25, 0., 0.25, 0.5])
            axs[k][ti].set_rasterized(True)

    plt.tight_layout(h_pad=0.1)
    output_png = Path(output_png)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, bbox_inches="tight")
    plt.close(fig)


def export_pdf(source_png, output_pdf):
    image = plt.imread(source_png)
    height, width = image.shape[:2]
    fig = plt.figure(figsize=(width / 100, height / 100), dpi=100, frameon=False)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(image)
    ax.set_axis_off()
    fig.savefig(output_pdf, format="pdf", dpi=100, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
