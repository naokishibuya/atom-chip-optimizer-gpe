"""
Per-step inverse-optimization solver: tracks a target trajectory by solving
for wire-current updates via the trap's dr/dI Jacobian.
"""
import logging
from dataclasses import dataclass, fields

import jax
import jax.numpy as jnp
import numpy as np
import optimistix as optx

from ..potential import (
    Atom,
    analyze_trap,
    find_trap_minimum,
    tf_quantities,
    trap_potential_energies,
)
from .chip import AtomChip

logger = logging.getLogger("schedule.optimizer")


#-------------------------------------------------------------------------------
# Schedule generation
#-------------------------------------------------------------------------------


def linear(s: float) -> jnp.ndarray:
    return s


def smoothstep_quintic(s: float) -> jnp.ndarray:
    # C² at both ends: f, f', f'' all zero at s=0 and s=1
    return 10 * s**3 - 15 * s**4 + 6 * s**5


def cosine(s: float) -> jnp.ndarray:
    return 0.5 * (1 - jnp.cos(jnp.pi * s))


SCHEDULERS = {
    "linear": linear,
    "smoothstep": smoothstep_quintic,
    "cosine": cosine,
}


DRIFT_TOLERANCE = 0.01  # accept non-success if drift < 1% of half_bounds


# fmt: off
@dataclass
class CurrentSchedule:
    """Per-block transport schedule produced by `optimize_current_schedule`.

    Row t is the chip configuration at schedule block t (currents + the resulting
    trap properties). Persists to/from a single `.npz` via `save`/`load`.
    """
    I_logical_A    : np.ndarray  # (steps + 1, n_logical_wires) logical wire currents [A]
    I_segments_A   : np.ndarray  # (steps + 1, n_segments) per-segment currents [A]
    r_mins_mm      : np.ndarray  # (steps + 1, 3) trap minimum position [mm]
    target_rs_mm   : np.ndarray  # (steps + 1, 3) commanded target position [mm]
    U0s_J          : np.ndarray  # (steps + 1, ) trap-bottom potential energy [J]
    omegas_hz      : np.ndarray  # (steps + 1, 3) trap frequencies [Hz]
    eigvecs        : np.ndarray  # (steps + 1, 3, 3) trap-axis eigenvectors (columns)
    a_ho_axes_m    : np.ndarray  # (steps + 1, 3) per-axis harmonic-oscillator length [m]
    larmor_freqs_hz: np.ndarray  # (steps + 1, ) Larmor frequency [Hz]
    J_conds        : np.ndarray  # (steps + 1, ) Jacobian condition number κ(dr/dI); NaN at init/held steps

    def save(self, path: str) -> None:
        """Dump every field into a compressed .npz."""
        np.savez_compressed(path, **{f.name: getattr(self, f.name) for f in fields(self)})

    @classmethod
    def load(cls, path: str) -> "CurrentSchedule":
        """Reverse of `save`."""
        with np.load(path) as npz:
            return cls(**{k: npz[k] for k in npz.files})

    def slice(self, step_slice: slice) -> "CurrentSchedule":
        """Slice every field along axis 0 (the step axis) and return a new schedule."""
        return CurrentSchedule(**{f.name: getattr(self, f.name)[step_slice] for f in fields(self)})

    def tf_quantities(self, atom: Atom, n_atoms: int) -> tuple[np.ndarray, np.ndarray]:
        """Thomas-Fermi μ and R_TF per schedule step, from `self.omegas_hz`. Returns (mu_J, tf_radii_m), both shaped like ω."""
        angular = 2.0 * np.pi * np.asarray(self.omegas_hz, dtype=np.float64)
        mu, radii = tf_quantities(angular, atom, n_atoms)
        return np.asarray(mu), np.asarray(radii)
# fmt: on


def optimize_current_schedule(
    *,
    chip: AtomChip,
    scheduler_name: str,
    steps: int,
    reg: float,
    I_max_shifting_A: float,
    I_max_guiding_A: float,
    wire_ids: list[int],
    destination_offset_mm: jnp.ndarray,  # target displacement from the initial minimum [mm]
) -> CurrentSchedule:
    """Run the inverse-optimization schedule against `chip`. Raises on any step failure."""

    # optimization bounds
    half_bounds_mm = jnp.asarray(chip.half_bounds_mm, dtype=jnp.float64)
    drift_threshold = float(jnp.max(half_bounds_mm)) * DRIFT_TOLERANCE

    # initial values
    atom = chip.atom
    geometry = chip.geometry
    bias_field = chip.bias_field

    # Initial trap stats
    minimum, geom = chip.analyze_trap_minimum()
    if not minimum.found:
        raise RuntimeError(f"Initial trap minimization failed: {minimum.message}")
    r0_ref_mm = jnp.asarray(minimum.position, dtype=jnp.float64)

    omega_hz = geom.trap.omega_hz
    logger.info("initial trap:  r=%s mm  ω=%s Hz", np.asarray(r0_ref_mm).tolist(), np.asarray(omega_hz).tolist())

    destination_r_mm = r0_ref_mm + jnp.asarray(destination_offset_mm, dtype=jnp.float64)
    logger.info(
        "destination: %s mm (offset %s mm)",
        np.asarray(destination_r_mm).tolist(), np.asarray(destination_offset_mm).tolist())

    # scheduler function and inverse optimization solver for current schedule
    scheduler = SCHEDULERS[scheduler_name]

    @jax.jit
    def distribute_currents(I_logical_A: jnp.ndarray) -> jnp.ndarray:
        """Map per-logical-wire currents to per-segment currents via the lw_indices gather."""
        return I_logical_A[chip.lw_indices]

    @jax.jit
    def r_target(t: int, T: int) -> jnp.ndarray:
        s = t / T
        return r0_ref_mm + scheduler(s) * (destination_r_mm - r0_ref_mm)

    @jax.jit
    def trap_U(r_mm: jnp.ndarray, I_logical_A: jnp.ndarray) -> jnp.ndarray:
        wire_currents = distribute_currents(I_logical_A)
        U, _, _ = trap_potential_energies(
            jnp.atleast_2d(r_mm), atom, geometry, wire_currents, bias_field
        )
        return U[0]

    grad_U_r = jax.grad(trap_U, argnums=0)
    hess_U_r = jax.jacfwd(grad_U_r, argnums=0)
    cross_jac = jax.jacfwd(grad_U_r, argnums=1)

    def calc_dr_dI(r0_mm: jnp.ndarray, I_logical_A: jnp.ndarray) -> jnp.ndarray:
        H = hess_U_r(r0_mm, I_logical_A)
        J = cross_jac(r0_mm, I_logical_A)
        return -jnp.linalg.solve(H, J)

    @jax.jit
    def solve_delta_I(
        I_logical_A: jnp.ndarray,
        r_now_mm: jnp.ndarray,
        r_next_mm: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        delta_r = r_next_mm - r_now_mm
        dr_dI = calc_dr_dI(r_now_mm, I_logical_A)
        cond = jnp.linalg.cond(dr_dI)
        alpha = reg * (1 + cond)
        delta_I = jnp.linalg.solve(
            dr_dI.T @ dr_dI + alpha * jnp.eye(dr_dI.shape[1]),
            dr_dI.T @ delta_r,
        )
        return delta_I, cond

    @jax.jit
    def evaluate_trap(wire_currents_A: jnp.ndarray, trap_position_mm: jnp.ndarray):
        """Evaluate U, trap geometry, σ_HO at a fixed trap position. N-independent."""
        trap_geom = analyze_trap(atom, geometry, wire_currents_A, bias_field, trap_position_mm)
        U_arr, _, _ = trap_potential_energies(
            jnp.atleast_2d(trap_position_mm), atom, geometry, wire_currents_A, bias_field
        )
        a_ho_axes = atom.harmonic_oscillator_length(trap_geom.trap.omega_rad)
        return U_arr[0], trap_geom, a_ho_axes

    @jax.jit
    def refine_trap_minimum(wire_currents_A: jnp.ndarray, guess_mm: jnp.ndarray):
        return find_trap_minimum(
            atom,
            geometry,
            wire_currents_A,
            bias_field,
            guess_mm,
            half_bounds_mm,
        )

    # current limits
    shifting_wires = chip.wire_layout["shifting"]
    guiding_wires = chip.wire_layout["guiding"]
    n_logical_shifting = len(shifting_wires)
    n_logical_guiding = len(guiding_wires)
    n_logical = n_logical_shifting + n_logical_guiding

    I_limits = jnp.concatenate([
        jnp.full(n_logical_shifting, I_max_shifting_A, dtype=jnp.float64),
        jnp.full(n_logical_guiding, I_max_guiding_A, dtype=jnp.float64),
    ])

    # mask: select which wires to optimize
    wire_ids = jnp.array(wire_ids, dtype=jnp.int32)
    mask = jnp.zeros(n_logical, dtype=jnp.float64).at[wire_ids].set(1.0)

    shifting_wire_currents = [lw.current_A for lw in shifting_wires]
    guiding_wire_currents = [lw.current_A for lw in guiding_wires]
    I_logical_A = jnp.array(shifting_wire_currents + guiding_wire_currents, dtype=jnp.float64)
    wire_currents_A = distribute_currents(I_logical_A)
    U0_0_J, trap_geom_0, a_ho_axes_0 = evaluate_trap(wire_currents_A, r0_ref_mm)

    # fmt: off
    I_logs        = [I_logical_A]
    r_mins        = [r0_ref_mm]
    target_rs     = [r0_ref_mm]
    U0s           = [U0_0_J]
    omegas_hz     = [trap_geom_0.trap.omega_hz]
    eigvecs_arr   = [trap_geom_0.hessian.eigenvectors]
    a_ho_axes_arr = [a_ho_axes_0]
    larmor_freqs  = [trap_geom_0.larmor.omega_hz]
    conds         = [float("nan")]
    # fmt: on

    # Generate current schedule by inverse optimization
    for t in range(steps):
        r_now_mm = r_mins[-1]
        r_next_mm = r_target(t + 1, steps)

        cond = float("nan")
        if r_next_mm[0] - r_now_mm[0] > 1e-6:
            delta_I, cond_val = solve_delta_I(I_logical_A, r_now_mm, r_next_mm)
            cond = float(cond_val)
            I_logical_A = I_logical_A + delta_I * mask
            I_logical_A = jnp.clip(I_logical_A, -I_limits, I_limits)

        wire_currents_A = distribute_currents(I_logical_A)
        r_min_mm, _, sol = refine_trap_minimum(wire_currents_A, r_now_mm)
        if sol.result != optx.RESULTS.successful:
            drift = float(jnp.linalg.norm(r_min_mm - r_now_mm))
            if drift > drift_threshold:
                raise RuntimeError(
                    f"Step {t + 1}: Minimization failed: x={r_min_mm} guess={r_now_mm} "
                    f"diff={drift * 1e3:.3g} µm"
                )

        U0_J, trap_geom, a_ho_axes = evaluate_trap(wire_currents_A, r_min_mm)
        omega_hz = trap_geom.trap.omega_hz

        if jnp.any(jnp.isnan(omega_hz)) or jnp.any(trap_geom.hessian.eigenvalues < 0):
            raise RuntimeError(
                f"Step {t + 1}: NaN ω or negative eigenvalues (r={_format_array(r_min_mm)})"
            )

        I_logs.append(I_logical_A)
        r_mins.append(r_min_mm)
        target_rs.append(r_next_mm)
        U0s.append(U0_J)
        omegas_hz.append(omega_hz)
        eigvecs_arr.append(trap_geom.hessian.eigenvectors)
        a_ho_axes_arr.append(a_ho_axes)
        larmor_freqs.append(trap_geom.larmor.omega_hz)
        conds.append(cond)

        if (t + 1) % 100 == 0:
            logger.info("step %4d/%d  r=%s  U=%9.3e  ω_hz=%s",
                        t + 1, steps, _format_array(r_min_mm), float(U0_J), _format_array(omega_hz))

    logger.info("schedule complete (%d steps)", steps)

    # Optimization results
    I_logs_stacked = jnp.stack(I_logs)
    I_segments = jax.vmap(distribute_currents)(I_logs_stacked)

    # fmt: off
    sched = CurrentSchedule(
        I_logical_A     = np.asarray(I_logs_stacked),
        I_segments_A    = np.asarray(I_segments),
        r_mins_mm       = np.asarray(jnp.stack(r_mins)),
        target_rs_mm    = np.asarray(jnp.stack(target_rs)),
        U0s_J           = np.asarray(jnp.array(U0s)),
        omegas_hz       = np.asarray(jnp.stack(omegas_hz)),
        eigvecs         = np.asarray(jnp.stack(eigvecs_arr)),
        a_ho_axes_m     = np.asarray(jnp.stack(a_ho_axes_arr)),
        larmor_freqs_hz = np.asarray(jnp.array(larmor_freqs)),
        J_conds         = np.asarray(conds),
    )
    # fmt: on
    logger.info(_format_summary(sched))
    return sched


def _format_array(x: jnp.ndarray) -> str:
    return np.array2string(
        np.asarray(x),
        formatter={"float_kind": lambda v: f"{v: 10.4g}"},
        separator=" ",
    )


def _format_summary(sched: CurrentSchedule) -> str:
    """Reduce a result to position-tracking and drift statistics,
    in display units (μm, Hz, MHz, %, J)."""
    # fmt: off
    r_mins         = sched.r_mins_mm
    target_rs      = sched.target_rs_mm
    U0s            = sched.U0s_J
    omegas_hz      = sched.omegas_hz
    a_ho_axes      = sched.a_ho_axes_m
    larmor_freqs   = sched.larmor_freqs_hz

    r_err_um       = (target_rs - r_mins) * 1e3
    r_step_diff_um = np.diff(r_mins, axis=0) * 1e3
    omega_err_hz   = omegas_hz - omegas_hz[0]
    a_ho_err_um    = (a_ho_axes - a_ho_axes[0]) * 1e6
    u0_dev         = U0s - U0s[0]

    rmse_um        = np.sqrt(np.mean(r_err_um ** 2, axis=0))
    max_um         = np.max(np.abs(r_err_um), axis=0)
    step_rmse_um   = np.sqrt(np.mean(r_step_diff_um ** 2, axis=0))
    step_max_um    = np.max(np.abs(r_step_diff_um), axis=0)
    # fmt: on

    return f"""Trajectory statistics:

Trajectory tracking (target − r_min):
  axis  RMSE [μm]  max |err| [μm]  step RMSE [μm]  step max [μm]
  x    {rmse_um[0]:10.4f}  {max_um[0]:14.4f}  {step_rmse_um[0]:14.4f}  {step_max_um[0]:13.4f}
  y    {rmse_um[1]:10.4f}  {max_um[1]:14.4f}  {step_rmse_um[1]:14.4f}  {step_max_um[1]:13.4f}
  z    {rmse_um[2]:10.4f}  {max_um[2]:14.4f}  {step_rmse_um[2]:14.4f}  {step_max_um[2]:13.4f}

Trap potential (U0):
  initial:   {float(U0s[0]): .4e} J
  final  :   {float(U0s[-1]): .4e} J  (drift {_drift_pct(U0s):+.4f} %)
  RMSE   :   {float(np.sqrt(np.mean(u0_dev ** 2))): .4e} J
  max Δ  :   {float(np.max(np.abs(u0_dev))): .4e} J

Trap frequencies ω [Hz]:
  initial:   {omegas_hz[0]}
  final  :   {omegas_hz[-1]}
  RMSE   :   {np.sqrt(np.mean(omega_err_hz ** 2, axis=0))}
  max Δ  :   {np.max(np.abs(omega_err_hz), axis=0)}

HO length per axis a_ho [μm]:
  initial:   {a_ho_axes[0] * 1e6}
  final  :   {a_ho_axes[-1] * 1e6}
  RMSE   :   {np.sqrt(np.mean(a_ho_err_um ** 2, axis=0))}
  max Δ  :   {np.max(np.abs(a_ho_err_um), axis=0)}

Larmor frequency ω_L [MHz]:
  initial:   {float(larmor_freqs[0]) * 1e-6:.4f}
  final  :   {float(larmor_freqs[-1]) * 1e-6:.4f}  (drift {_drift_pct(larmor_freqs):+.4f} %)
  min    :   {float(np.min(larmor_freqs)) * 1e-6:.4f}
  max    :   {float(np.max(larmor_freqs)) * 1e-6:.4f}
"""


def _drift_pct(x: np.ndarray) -> float:
    """Percent change from first to last sample; NaN if first sample is zero."""
    return float(x[-1] / x[0] - 1.0) * 100.0 if x[0] != 0.0 else float("nan")
