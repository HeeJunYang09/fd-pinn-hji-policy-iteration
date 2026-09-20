"""Air3D experiment settings and data filenames."""
import numpy as np


def data_suffix(sigma):
    values = tuple(float(s) for s in sigma)
    if values == (0., 0., 0.):
        return ""
    if values == (0.03, 0.05, 0.):
        return "_sigma_003_005_000"
    raise ValueError("Supported diffusion diagonals: (0,0,0), (0.03,0.05,0)")


def problem(nx, sigma=(0., 0., 0.)):
    import jax.numpy as jnp
    data_suffix(sigma)
    h = 6 / nx
    nu = 5.5 * h
    tau = min(h*h / (3*nu + sum(s*s for s in sigma)), nu / 17424)
    if max(sigma) > 0:
        tau = min(tau, 4*h*h / (3*max(sigma)**2))
    return dict(sigma=jnp.diag(jnp.asarray(sigma)), ve=.5, vp=.5,
                omega_bar=1.5, beta=.5, h_x=h, h_psi=2*np.pi/nx,
                nu_h=nu, tau_h=tau, Nx=nx, domain_t=[0., 1.],
                domain_x=[-3., 3.], periodic_weight=100.)
