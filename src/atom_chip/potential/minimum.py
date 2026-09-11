from collections.abc import Callable

import jax
import jax.numpy as jnp
import optimistix as optx

BFGS_SOLVER = optx.BFGS(rtol=1e-10, atol=1e-10)


def solve_bounded_bfgs(
    objective: Callable[[jnp.ndarray], float],
    x0: jnp.ndarray,
    lower: jnp.ndarray,
    upper: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, optx.Solution]:
    """Bounded minimization: BFGS in sigmoid space with a Newton polish step.

    BFGS is unconstrained, so it optimizes an unbounded u; the sigmoid maps u into [lower, upper],
    so the bounds can never be violated. Its stopping test watches the change in f, which can be
    small near the minimum while the gradient is non-zero, so one saddle-free Newton polish step
    (`_newton_step`) is added and kept if it lowers the gradient norm. A stall exactly on a saddle
    (g=0) is left for the caller's guard to reject (a returned point with a negative Hessian
    eigenvalue is not a minimum).

    Returns (x, objective(x), sol): sol is the BFGS exit; x and value are post-polish.
    """
    # Scale f to ~1 at the seed so the solver's absolute atol is meaningful. This does not
    # affect the early stop above: that comes from the relative (rtol) test, which is
    # scale-invariant, so this constant cancels in |Δf|/|f|.
    value_scale = 1.0 / jnp.maximum(jnp.abs(objective(x0)), 1e-30)

    def wrapped(u: jnp.ndarray, _: None) -> jnp.ndarray:
        x = _to_bounded(u, lower, upper)
        return objective(x) * value_scale

    u0 = _to_unbounded(x0, lower, upper)
    sol = optx.minimise(wrapped, BFGS_SOLVER, u0, throw=False)
    x_bfgs = _to_bounded(sol.value, lower, upper)

    x_new = _newton_step(objective, x_bfgs, lower, upper)
    g_bfgs = jnp.linalg.norm(jax.grad(objective)(x_bfgs))
    g_new = jnp.linalg.norm(jax.grad(objective)(x_new))
    x = jnp.where(g_new < g_bfgs, x_new, x_bfgs)
    return x, objective(x), sol


def _newton_step(
    objective: Callable[[jnp.ndarray], float],
    x: jnp.ndarray,
    lower: jnp.ndarray,
    upper: jnp.ndarray,
) -> jnp.ndarray:
    """One saddle-free Newton step toward the minimum, clipped to the bounds.

    Where H is indefinite (as near the soft endpoint of the transport) a raw Newton step can jump
    to a saddle/maximum. The saddle-free variant (Dauphin et al. 2014) replaces H with |H|
    (eigenvalues reflected positive), so the step is always a descent; where H is already
    positive-definite it is an ordinary Newton step.
    """
    g = jax.grad(objective)(x)
    H = jax.hessian(objective)(x)
    evals, evecs = jnp.linalg.eigh(H)
    H_pd = evecs @ jnp.diag(jnp.abs(evals)) @ evecs.T          # |H|: reflect negative curvature
    return jnp.clip(x - jnp.linalg.solve(H_pd, g), lower, upper)


def _to_bounded(u: jnp.ndarray, lower: jnp.ndarray, upper: jnp.ndarray) -> jnp.ndarray:
    return lower + (upper - lower) * jax.nn.sigmoid(u)


def _to_unbounded(x: jnp.ndarray, lower: jnp.ndarray, upper: jnp.ndarray) -> jnp.ndarray:
    t = jnp.clip((x - lower) / (upper - lower), 1e-6, 1 - 1e-6)
    return jnp.log(t / (1 - t))
