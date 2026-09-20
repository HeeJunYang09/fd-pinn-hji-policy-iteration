import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches
from IPython.display import HTML
import os
from jax import grad, random, jit, vmap, hessian, config
from jax.nn import initializers
import optax
import pickle
from tqdm.notebook import tqdm
from scipy.interpolate import RegularGridInterpolator





## ---------------------------------------------------------------------------------------------------##
# === simulate trajectory using interpolated policy ===

def simulate_trajectory_from_params(
    params, v_single, x0,
    sigma, lambda_1=0.1, delta=0.1,
    T=1.0, dt=0.01, seed=0
):
    """
    Simulates the trajectory given the neural network parameters for value function.
    
    Parameters:
    -----------
    params : list
        Neural network parameters.
    v_single : callable
        Function v(t, x) → scalar, computed from `params`.
    x0 : array_like
        Initial position.
    sigma : ndarray
        Diffusion matrix.
    lambda_1 : float
        Weight for control penalty in policy.
    delta : float
        Disturbance bound.
    T : float
        Total simulation time.
    dt : float
        Maximum time step.
    seed : int
        Random seed.

    Returns:
    --------
    ts : jnp.ndarray
        Time array.
    traj : jnp.ndarray
        Simulated trajectory.
    """
    if not np.isfinite(T) or T < 0:
        raise ValueError("T must be finite and nonnegative")
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("dt must be finite and positive")
    num_steps = int(np.ceil(T / dt))
    step_dt = T / num_steps if num_steps else 0.0
    ts = jnp.linspace(0.0, T, num_steps + 1)
    traj = [np.array(x0).reshape(-1)]
    key = random.PRNGKey(seed)

    def get_policy(t, x):
        tx = jnp.concatenate([jnp.array([t]), x])
        grad_v = grad(v_single)(tx, params)[1:]
        norm = jnp.linalg.norm(grad_v) + 1e-8
        alpha = jnp.where(norm <= 2 * lambda_1, -grad_v / (2 * lambda_1), -grad_v / norm)
        beta = delta * grad_v / norm
        return alpha, beta

    x = np.array(x0)
    for i in range(num_steps):
        t = float(ts[i])
        key, subkey = random.split(key)
        dBt = random.normal(subkey, shape=(2,)) * np.sqrt(step_dt)

        alpha, beta = get_policy(t, x)
        dx = (alpha + beta) * step_dt + sigma @ dBt
        x = x + dx
        traj.append(x)

    return ts, jnp.stack(traj)

## ---------------------------------------------------------------------------------------------------##

def x_obs(t):
    return jnp.array([
        0.5 * jnp.cos(jnp.pi * t),
        0.5 * jnp.sin(jnp.pi * t)
    ])

## ---------------------------------------------------------------------------------------------------##

def plot_trajectory(ts, traj, x_obs_func, ax=None, x_goal=(0.9, 0.9)):
    """
    Plot the trajectory of the agent and the obstacle path over time.

    Parameters:
    -----------
    ts : array-like
        A 1D array of time steps corresponding to the trajectory.

    traj : array-like, shape (N, 2)
        Agent trajectory positions over time (x_1, x_2) at each time step.

    x_obs_func : callable
        Function that takes a scalar time t and returns the obstacle position (2D array).

    """
    traj = np.array(traj)
    obs_path = np.array([x_obs_func(t) for t in ts])

    # Create figure and axis 
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))

    # Trajectory and obstacle path
    ax.plot(traj[:, 0], traj[:, 1], 'b--', linewidth=1.5, alpha=0.7, label='trajectory')
    ax.plot(obs_path[:, 0], obs_path[:, 1], 'r--', linewidth=1.5, alpha=0.7, label='obstacle path')
    ax.plot(traj[-1, 0], traj[-1, 1], 'bo', label='agent')
    ax.plot(obs_path[-1,0], obs_path[-1,1], 'ro', label='obstacle')
    ax.plot(*x_goal, 'ko', label='target')
    ax.plot(traj[0, 0], traj[0, 1], 'go', label='start')
    
    # Plot appearance
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_xlabel(r'$x_1$')
    ax.set_ylabel(r'$x_2$')
    ax.set_title('Trajectory under Learned Policy')
    ax.set_aspect('equal')
    ax.grid(True)
    ax.legend(loc='lower right')
    
    return ax    
    
    

## ---------------------------------------------------------------------------------------------------##

def animate_trajectory(ts, traj, x_obs_func):
    """
    Animate the agent's trajectory along with the moving obstacle.

    Parameters
    ----------
    ts : array-like
        A 1D array of time steps corresponding to the trajectory.

    traj : array-like, shape (N, 2)
        Agent trajectory positions over time (x_1, x_2) at each time step.

    x_obs_func : callable
        Function that takes a scalar time t and returns the obstacle position (2D array).

    Returns
    -------
    HTML
        An HTML5 video animation of the trajectory and obstacle movement.
        Intended for use in Jupyter Notebook environments.
    """
    traj = np.array(traj)
    obs_traj = []  # Store obstacle positions for tracing

    fig, ax = plt.subplots(figsize=(6, 6))

    # Initialize plot elements
    agent_dot, = ax.plot([], [], 'bo', label='agent')
    obs_dot, = ax.plot([], [], 'ro', label='obstacle')
    goal_dot = ax.plot(0.9, 0.9, 'ko', label='target')[0]
    agent_trace_line, = ax.plot([], [], 'b--', linewidth=1, alpha=0.7, label='trajectory')
    obs_trace_line, = ax.plot([], [], 'r--', linewidth=1, alpha=0.7, label='obstacle path')
    ax.scatter(traj[0, 0], traj[0, 1], color='green', label='start')

    # Set plot appearance
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.grid(True)
    ax.set_aspect('equal')
    ax.set_title('Trajectory Animation')
    ax.legend(loc='lower right')

    def init():
        agent_dot.set_data([], [])
        obs_dot.set_data([], [])
        agent_trace_line.set_data([], [])
        obs_trace_line.set_data([], [])
        return agent_dot, obs_dot, goal_dot, agent_trace_line, obs_trace_line

    def update(i):
        x = np.array(traj[i]).reshape(-1)[:2]
        obs = x_obs_func(ts[i])
        obs_traj.append(obs)
        obs_array = np.array(obs_traj)

        agent_trace_line.set_data(traj[:i+1, 0], traj[:i+1, 1])
        obs_trace_line.set_data(obs_array[:, 0], obs_array[:, 1])
        agent_dot.set_data([x[0]], [x[1]])
        obs_dot.set_data([obs[0]], [obs[1]])
        
        

        return agent_dot, obs_dot, goal_dot, agent_trace_line, obs_trace_line

    ani = animation.FuncAnimation(
        fig, update, frames=len(ts), init_func=init,
        blit=True, interval=50, repeat=False
    )

    plt.close(fig)
    return HTML(ani.to_html5_video())

## ---------------------------------------------------------------------------------------------------##

def plot_policy_from_params(
    params,
    v_single_fn,
    t_index,
    num_t,
    num_x,
    lambda_1=0.1,
    delta=0.1,
    domain_x=(-1, 1),
    label_alpha=r"\alpha_n",
    label_beta=r"\beta_n",
    ax=None
):
    """
    Visualize the control policies αₙ and βₙ computed from the gradient of a neural network value function
    at a specific time index. Plots both vector fields over the 2D spatial domain.

    Parameters
    ----------
    params : list
        Parameters of the trained neural network representing the value function v(t, x).

    v_single_fn : callable
        Function that computes the scalar value v(t, x) for a given input tx and parameters.

    t_index : int
        Index of the time step to visualize (0 ≤ t_index < num_t).

    num_t : int
        Total number of time steps used in the training or discretization.

    num_x : int
        Number of grid points in each spatial dimension.

    lambda_1 : float, optional
        Regularization coefficient for the alpha policy update (default is 0.1).

    delta : float, optional
        Disturbance magnitude bound used in beta policy update (default is 0.1).

    domain_x : tuple of float, optional
        Range of the spatial domain for both x₁ and x₂ axes (default is (-1, 1)).

    label_alpha : str, optional
        LaTeX string to label the α policy (default is r"\alpha_n").

    label_beta : str, optional
        LaTeX string to label the β policy (default is r"\beta_n").

    ax : list of matplotlib.axes.Axes, optional
        Optional axes to plot on; if None, new figure and axes will be created.

    Returns
    -------
    ax : list of matplotlib.axes.Axes
        The axes containing the quiver plots for αₙ and βₙ.
    """

    # Convert time index to actual time value
    t_val = t_index / (num_t - 1)

    # Create spatial meshgrid
    x = jnp.linspace(*domain_x, num_x)
    X1, X2 = jnp.meshgrid(x, x, indexing='ij')
    grid_points = jnp.stack([X1.flatten(), X2.flatten()], axis=-1)

    # Create time-augmented input (t, x) for network evaluation
    t_fixed = jnp.full((grid_points.shape[0], 1), t_val)
    tx_grid = jnp.concatenate([t_fixed, grid_points], axis=1)

    # Compute policies from gradient of value function
    def compute_policy(tx):
        grad_v = grad(v_single_fn)(tx, params)[1:]
        norm = jnp.linalg.norm(grad_v) + 1e-10
        alpha = jnp.where(norm <= 2 * lambda_1, -grad_v / (2 * lambda_1), -grad_v / norm)
        beta = delta * grad_v / norm
        return alpha, beta

    # Vectorize policy computation over the grid
    alpha_vals, beta_vals = vmap(compute_policy)(tx_grid)

    # Reshape into 2D fields
    alpha_u = alpha_vals[:, 0].reshape((num_x, num_x))
    alpha_v = alpha_vals[:, 1].reshape((num_x, num_x))
    beta_u = beta_vals[:, 0].reshape((num_x, num_x))
    beta_v = beta_vals[:, 1].reshape((num_x, num_x))

    # Prepare plotting axes
    if ax is None:
        fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    else:
        axs = ax

    # Plot αₙ and βₙ vector fields
    axs[0].quiver(X1, X2, alpha_u, alpha_v)
    axs[0].set_title(fr"Policy Vector Field: ${label_alpha}(t={t_val:.2f}, x)$")

    axs[1].quiver(X1, X2, beta_u, beta_v)
    axs[1].set_title(fr"Policy Vector Field: ${label_beta}(t={t_val:.2f}, x)$")

    # Common formatting
    for a in axs:
        a.set_xlabel(r"$x_1$")
        a.set_ylabel(r"$x_2$")
        a.set_aspect("equal")
        a.grid(True)

    return axs

## ---------------------------------------------------------------------------------------------------##

def compute_and_plot_brs_over_time_by_time(v_single, params, t_values, num_x, lambda_3, r, domain_x=(-1, 1), x_goal=(0.9, 0.9)):
    """
    Compute and visualize the Backward Reachable Set (BRS) over time using the value function
    obtained from a trained neural network.

    The BRS at time t is defined as the sublevel set:
        G(t) = { x ∈ R² | v(t,x) ≤ λ₃ r² }

    Parameters
    ----------
    v_single : callable
        Function that evaluates the scalar value v(t, x) from neural network parameters.
        Should have the signature v_single(tx, params), where tx ∈ R³ = [t, x1, x2].

    params : list
        Trained neural network parameters (e.g., list of weight/bias tuples).

    t_values : list of float
        List of actual time values in [0, 1] at which to compute and visualize the BRS.

    num_x : int
        Number of discretization points per axis for the 2D spatial grid.

    lambda_3 : float
        Coefficient used in the terminal cost function (typically large, e.g., 100).

    r : float
        Radius of the target set centered at the goal (used to define BRS threshold).

    domain_x : tuple, optional
        Tuple defining the spatial domain in both x₁ and x₂ directions (default: (-1, 1)).

    x_goal : tuple of float, optional
        Coordinates of the target center in 2D space (default: (0.9, 0.9)).

    Returns
    -------
    None
        Displays a matplotlib plot with the BRS contours at each specified time.
    """

    # Create 2D spatial grid
    x = jnp.linspace(*domain_x, num_x)
    X1, X2 = jnp.meshgrid(x, x, indexing='ij')
    grid_points = jnp.stack([X1.flatten(), X2.flatten()], axis=-1)

    # Threshold derived from the target set definition: v(t,x) ≤ λ₃ r²
    threshold = lambda_3 * r**2

    # Vectorized evaluation of v(t,x) for all grid points at fixed t
    @jit
    def evaluate_v_batch(t_val):
        t_vec = jnp.full((grid_points.shape[0], 1), t_val)
        tx = jnp.concatenate([t_vec, grid_points], axis=1)
        return vmap(lambda tx_: v_single(tx_, params))(tx)

    # Set up subplot layout
    fig, axs = plt.subplots(1, len(t_values), figsize=(5 * len(t_values), 5))
    if len(t_values) == 1:
        axs = [axs]

    # Loop over each time value to compute and plot the BRS
    for i, t_val in enumerate(t_values):
        v_vals = evaluate_v_batch(t_val).reshape(num_x, num_x)
        mask = v_vals <= threshold  # Indicator mask for BRS region

        ax = axs[i]
        # Fill the region corresponding to the BRS
        ax.contourf(X1, X2, mask.astype(float), levels=[0.5, 1.1], colors='lightblue', alpha=0.6)
        # Draw the level curve corresponding to the BRS threshold
        ax.contour(X1, X2, v_vals, levels=[threshold], colors='blue', linewidths=2)

        # Plot target location as a red dot
        ax.plot(*x_goal, 'ro', label='target point')

        # Plot target set G₀ as a transparent circle
        circle = patches.Circle(x_goal, radius=r, color='orange', fill=True, alpha=0.3, label=r'target set $\mathcal{G}_0$')
        ax.add_patch(circle)
        
        # Set titles and axes
        ax.set_title(f"BRS at $t = {t_val:.2f}$, $r = {r:.2f}$")
        ax.set_xlabel(r"$x_1$")
        ax.set_ylabel(r"$x_2$")
        ax.set_aspect('equal')
        ax.set_xlim(*domain_x)
        ax.set_ylim(*domain_x)
        ax.grid(True)
        ax.legend(loc='lower right')

    # Final layout adjustment and display
    plt.tight_layout()
    plt.show()

## ---------------------------------------------------------------------------------------------------##

def value_function_from_params_at_t(t_val, params, x1, x2, v_single):
    num_x = len(x1)
    # Generate spatial grid
    x_flat = jnp.stack([x1.flatten(), x2.flatten()], axis=-1)

    # Fix time input to t_val
    t_fixed = jnp.full((x_flat.shape[0], 1), t_val)
    tx_flat = jnp.concatenate([t_fixed, x_flat], axis=1)

    # Evaluate value function
    v_vals = vmap(lambda tx: v_single(tx, params))(tx_flat)
    v_vals = v_vals.reshape((num_x, num_x))
    return v_vals

def plot_value_function_at_t(t_val, v_vals, ax, x1, x2, levels = 20, cmap='viridis'):
    cf = ax.contourf(x1, x2, v_vals, levels=levels, cmap=cmap) 

    ax.set_title(fr"Value function $v(t={t_val:.1f}, x)$")
    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")
    ax.set_aspect("equal")

    return cf

def policy_params(params, t_grid, x_grid, policy_update):
  grid_t, grid_x1, grid_x2 = jnp.meshgrid(t_grid, x_grid, x_grid, indexing='ij')
  grid_tx = jnp.stack([grid_t, grid_x1, grid_x2], axis=-1).reshape(-1, 3)
  return policy_update(params, grid_tx)

## ---------------------------------------------------------------------------------------------------##

def compute_error_metrics(v_pinn, v_fdm):
    """
    Compute quantitative error metrics between PINN and FDM value functions.

    Parameters
    ----------
    v_pinn : array-like, shape (num_x, num_x)
        Value function predicted by PINN at a given time t.

    v_fdm : array-like, shape (num_x, num_x)
        Value function computed using FDM at the same time t.

    Returns
    -------
    metrics : dict
        Dictionary containing MSE, relative L2 error, and max absolute error.
    """
    error = v_pinn - v_fdm
    mse = jnp.mean((error)**2)
    rel_l2 = jnp.linalg.norm(error) / (jnp.linalg.norm(v_fdm) + 1e-8)  # prevent divide-by-zero
    max_abs = jnp.max(jnp.abs(error))

    return {
        'MSE': float(mse),
        'Relative L2': float(rel_l2),
        'Max Error': float(max_abs)
    }
