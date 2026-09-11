"""Solving the GPE on a box grid in SI: the discretization and the per-step evolvers."""
import logging
import math
from collections.abc import Callable
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from ..potential import constants
from .box import BoxGrid
from .fourier import evolve_psi, relax_psi, shift_box

logger = logging.getLogger("tdgpe.solver")

# Hard ceiling on the phase advance per substep.
STABILITY_THRESHOLD = 0.5

# Accuracy target, leaving margin below the stability ceiling.
_DT_ACCURACY_PHASE  = 0.4
_IMAG_SAFETY_FACTOR = 0.1
_RELAX_TOL          = 1e-6


@dataclass
class Stepping:
    """Temporal discretization for one GPE solve."""
    real_step_s: float  # real-time duration of one forward step [s]
    real_sub_s: float   # requested Strang substep [s]
    imag_step_s: float  # imaginary-time step [s]
    imag_iters: int     # imaginary-time iterations

    @property
    def n_sub_half(self) -> int:
        """Number of complete Strang steps in half a control-node interval."""
        return max(1, math.ceil(0.5 * self.real_step_s / self.real_sub_s - 1e-9))

    @property
    def n_sub(self) -> int:
        """Number of complete Strang steps between adjacent control nodes."""
        return 2 * self.n_sub_half

    @property
    def real_sub_realized_s(self) -> float:
        """Actual Strang step duration [s], adjusted to divide each control half-interval exactly."""
        return self.real_step_s / self.n_sub


def derive_stepping(
    grid: BoxGrid,
    mass_kg: float,
    omegas_hz: np.ndarray,
) -> tuple[float, float, int]:
    """Derive real- and imaginary-time stepping from the grid and trap."""
    omegas_rad = 2.0 * np.pi * np.asarray(omegas_hz, dtype=np.float64)
    omega_max, omega_min = float(np.max(omegas_rad)), float(np.min(omegas_rad))

    # Real-time step: use the grid kinetic and trap frequency scales; round down to 0.1 µs.
    # Stepping later adjusts this requested step to divide each control half-interval exactly.
    k2_max = sum((np.pi / dx) ** 2 for dx in grid.dx_m)
    omega_kin = float(constants.hbar) * k2_max / (2.0 * mass_kg)

    real_sub_s  = float(np.floor((_DT_ACCURACY_PHASE / (omega_kin + omega_max)) / 1e-7) * 1e-7)

    # Imaginary-time step: use the fastest trap frequency as a heuristic relaxation rate.
    # Round to the nearest 0.5 µs.
    imag_step_s = float(np.round((_IMAG_SAFETY_FACTOR / omega_max) / 5e-7) * 5e-7)

    # Iteration count: estimate squared-amplitude suppression using the slowest trap frequency
    # and the rounded imaginary-time step; round up to a multiple of 100.
    imag_iters  = int(np.ceil(np.ceil(-np.log(_RELAX_TOL) / (2.0 * omega_min * imag_step_s)) / 100) * 100)
    return real_sub_s, imag_step_s, imag_iters


def make_solvers(
    grid: BoxGrid,
    stepping: Stepping,
    *,
    U_func: Callable,
    mass_kg: float,
    g: float,
    omegas_hz: np.ndarray,
    linear: bool,
) -> tuple[Callable, Callable, Callable]:
    """Return ground-state, full-hold, and half-hold solvers."""
    kin_coeff = float(constants.hbar) / (2.0 * mass_kg)             # ℏ/2m [m²/s]: kinetic dispersion
    nonlinear_coeff = 0.0 if linear else g / float(constants.hbar)  # g/ℏ; 0.0 for a linear run

    real_sub_s = stepping.real_sub_realized_s
    imag_step_s = stepping.imag_step_s
    logger.info("constants: dV=%.2e m³  g=%.3g J·m³  ℏ/2m=%.3g m²/s", grid.dV_m3, g, kin_coeff)

    # Check the worst phase advance over the schedule.
    k2_max = float(jnp.max(jnp.sum(grid.k ** 2, axis=-1)))
    omega_kin = kin_coeff * k2_max
    omega_max = float(np.max(np.abs(2.0 * np.pi * np.asarray(omegas_hz))))
    phase = (omega_kin + omega_max) * real_sub_s
    if phase > STABILITY_THRESHOLD:
        raise RuntimeError(
            f"Stability check failed: phase advance per substep = {phase:.3f} > {STABILITY_THRESHOLD}. "
            "Increase n_sub to reduce real_sub."
        )

    def sample_U_on_grid(
        box_center_mm: jnp.ndarray,
        currents_A: jnp.ndarray,
    ) -> jnp.ndarray:
        """Sample U/ℏ on the box grid."""
        points_lab_mm = grid.r_local_m.reshape(-1, 3) * 1e3 + box_center_mm  # local metres → lab mm
        U_flat = U_func(points_lab_mm, currents_A)
        return (U_flat / float(constants.hbar)).reshape(grid.shape)

    def ground_state(
        box_center_mm: jnp.ndarray,
        currents_A: jnp.ndarray,
        sigma_init_m: jnp.ndarray,
    ) -> tuple[jnp.ndarray, float]:
        """Relax a ground state and return its stationary-state residual."""
        potential_coeff = sample_U_on_grid(box_center_mm, currents_A)
        psi = relax_psi(
            grid,
            potential_coeff=potential_coeff,
            dt_s=imag_step_s,
            n_iters=stepping.imag_iters,
            nonlinear_coeff=nonlinear_coeff,
            kin_coeff=kin_coeff,
            sigma_init_m=sigma_init_m,
        )
        # Compute ‖(H − μ)ψ‖/|μ|.
        k2 = jnp.sum(grid.k ** 2, axis=-1)
        H_psi = jnp.fft.ifftn(kin_coeff * k2 * jnp.fft.fftn(psi)) \
            + (potential_coeff + nonlinear_coeff * jnp.abs(psi) ** 2) * psi
        mu = jnp.real(jnp.sum(jnp.conj(psi) * H_psi) * grid.dV_m3)
        residual = jnp.sqrt(jnp.sum(jnp.abs(H_psi - mu * psi) ** 2) * grid.dV_m3) / jnp.abs(mu)
        return psi, float(residual)

    def _evolve_half(psi: jnp.ndarray, potential_coeff: jnp.ndarray) -> jnp.ndarray:
        return evolve_psi(
            grid,
            psi=psi,
            potential_coeff=potential_coeff,
            dt_s=real_sub_s,
            n_sub=stepping.n_sub_half,
            nonlinear_coeff=nonlinear_coeff,
            kin_coeff=kin_coeff,
        )

    @jax.jit
    def forward_half(
        psi: jnp.ndarray,
        box_center_mm: jnp.ndarray,
        prev_box_center_mm: jnp.ndarray,
        currents_A: jnp.ndarray,
    ) -> jnp.ndarray:
        """Advance one half hold."""
        potential_coeff = sample_U_on_grid(box_center_mm, currents_A)
        psi = shift_box(grid, psi=psi, delta_r_m=(box_center_mm - prev_box_center_mm) * 1e-3)
        return _evolve_half(psi, potential_coeff)

    @jax.jit
    def forward_step(
        psi: jnp.ndarray,
        box_center_mm: jnp.ndarray,
        prev_box_center_mm: jnp.ndarray,
        currents_A: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Advance one full hold and return its end and midpoint states."""
        potential_coeff = sample_U_on_grid(box_center_mm, currents_A)
        psi = shift_box(grid, psi=psi, delta_r_m=(box_center_mm - prev_box_center_mm) * 1e-3)
        psi_mid = _evolve_half(psi, potential_coeff)
        return _evolve_half(psi_mid, potential_coeff), psi_mid

    return ground_state, forward_step, forward_half
