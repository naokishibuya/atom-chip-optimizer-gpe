"""Thomas–Fermi scaling dynamics under time-dependent harmonic confinement."""
from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp

from ..schedule import CurrentSchedule


# fmt: off
@dataclass
class BreathingResult:
    T_total_s : float          # transport duration [s]
    B_traj    : np.ndarray     # (n_fine, 3, 3) B(t) scaling matrix in laboratory-axis coordinates
    det_B_traj: np.ndarray     # (n_fine,) det B(t) volume factor
    min_det_B : float          # minimum sampled det B
# fmt: on


def run_ermakov(sched: CurrentSchedule, T_total_s: float) -> BreathingResult:
    """Integrate B(t) using node-centered half-first/full-interior/half-last holds."""
    omega = 2.0 * np.pi * np.asarray(sched.omegas_hz, dtype=np.float64)   # (n_nodes, 3) [rad/s]
    V = np.asarray(sched.eigvecs, dtype=np.float64)                       # (n_nodes, 3, 3), columns = axes
    n_nodes = omega.shape[0]
    if n_nodes < 2:
        raise ValueError("Ermakov integration requires at least two schedule nodes")
    if T_total_s <= 0.0:
        raise ValueError("Ermakov integration requires a positive transport duration")

    # H has eigenvalues mω², so K = H/m = V diag(ω²) Vᵀ.
    K_nodes = np.einsum("tai,ti,tbi->tab", V, omega**2, V)                # (n_nodes, 3, 3) [1/s²]
    K0 = K_nodes[0]
    dt_ctrl = T_total_s / (n_nodes - 1)

    # Trap curvatures set the response timescale in the Ermakov equation.
    # Use 2ω_max as a heuristic breathing-frequency scale, targeting ~20 samples per cycle.
    # Keep at least n_nodes samples. Cap output sampling at 400,000 points.
    # The cap can reduce sampling density; not reached by the paper's schedules.
    # These are output sample times; solve_ivp chooses its integration steps separately.
    omega_max = float(np.max(omega))
    n_fine = int(np.clip(
        np.ceil(20.0 * (2.0 * omega_max / (2.0 * np.pi)) * T_total_s),
        n_nodes,
        400_000,
    ))
    t_fine = np.linspace(0.0, T_total_s, n_fine)

    sampled_states = np.empty((n_fine, 18), dtype=np.float64)  # flattened B and Bdot
    sampled = np.zeros(n_fine, dtype=bool)
    state = np.concatenate([np.eye(3).ravel(), np.zeros(9)])   # flattened B(0)=I and Bdot(0)=0

    for j, K_hold in enumerate(K_nodes):
        start = 0.0 if j == 0 else (j - 0.5) * dt_ctrl
        stop = T_total_s if j == n_nodes - 1 else (j + 0.5) * dt_ctrl

        def scaling_ode(
            _time_s: float,
            current_state: np.ndarray,
            held_curvature: np.ndarray = K_hold,
        ) -> np.ndarray:
            B = current_state[:9].reshape(3, 3)
            Bdot = current_state[9:].reshape(3, 3)
            Bddot = -held_curvature @ B + (np.linalg.inv(B).T @ K0) / np.linalg.det(B)
            return np.concatenate([Bdot.ravel(), Bddot.ravel()])

        # solve initial-value problem given the current state (B, Bdot) and an equation for its time derivative
        sol = solve_ivp(
            scaling_ode,
            [start, stop],
            state,
            dense_output=True,  # makes the solver provide an interpolation function, sol.sol(t)
            rtol=1e-8,
            atol=1e-10,
        )
        if not sol.success:
            raise RuntimeError(f"Ermakov integration failed at schedule node {j}: {sol.message}")

        mask = (t_fine >= start) & (t_fine <= stop) if j == n_nodes - 1 else \
               (t_fine >= start) & (t_fine < stop)
        sampled_states[mask] = sol.sol(t_fine[mask]).T
        sampled[mask] = True
        state = sol.y[:, -1]

    if not np.all(sampled):
        missing = np.flatnonzero(~sampled)
        raise RuntimeError(f"Ermakov integration left {missing.size} sample times unassigned")

    B_traj = sampled_states[:, :9].reshape(-1, 3, 3)   # (n_fine, 3, 3)
    det_B = np.linalg.det(B_traj)                      # (n_fine,)
    if not np.all(np.isfinite(B_traj)) or not np.all(np.isfinite(det_B)) or np.any(det_B <= 0.0):
        raise RuntimeError("Ermakov integration produced a non-finite state or non-positive det(B)")

    return BreathingResult(
        T_total_s  = T_total_s,
        B_traj     = B_traj,
        det_B_traj = det_B,
        min_det_B  = float(det_B.min()),
    )
