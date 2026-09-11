"""Atom-chip BEC transport as a time-dependent GPE: run the schedule, measure the delivered cloud."""
import logging
from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np

from ..potential import constants, trap_potential_energies
from ..schedule import AtomChip, CurrentSchedule
from .box import BoxGrid
from .cloud import boundary_mass, cloud_observables, fidelity, k_shell_masses, k_tail_mass
from .fourier import shift_box
from .solver import Stepping, make_solvers

logger = logging.getLogger("tdgpe.transport")


# Real-space and momentum-space diagnostic thresholds.
_BOUNDARY_EDGE_FRAC = 0.85
_K_TAIL_EDGE_FRAC = 0.85

# Near-Nyquist shells; their sum is an occupancy diagnostic, not an error bound.
_K_SHELL_EDGES_FRAC = (0.5, 0.62996, 0.79370)

# Warn on under-relaxed ground states; plotting uses the same threshold.
GS_RESIDUAL_WARN = 1e-4


# fmt: off
@dataclass
class EvalResult:
    F_3D                : float
    delta               : float          # interaction parameter δ = 4π·N·a_s/a_ho (signed); records the run's regime
    com_traj_lab_mm     : np.ndarray     # (T, 3) lab-frame COM
    sigma_traj_m        : np.ndarray     # (T, 3) σ along initial-trap principal axes
    cov_traj_m2         : np.ndarray     # (T, 3, 3) box-local covariance
    boundary_mass_traj  : np.ndarray     # (T, 3) mass fraction in the outer 15% grid-axis slabs (containment canary)
    k_tail_mass_traj    : np.ndarray     # (T, 3) per-axis |ψ|² fraction near Nyquist (resolution canary)
    k_shell_mass_traj   : np.ndarray     # (T, 3, S) per-axis nested near-Nyquist shells (m_shelf canary)
    gs_residual_initial : float          # ‖(Ĥ−μ)ψ‖/|μ| of the relaxed initial GS (stationary-state check)
    gs_residual_target  : float          # same for the target GS (the fidelity reference)
# fmt: on


def evaluate(
    chip: AtomChip,
    schedule: CurrentSchedule,
    grid: BoxGrid,
    stepping: Stepping,
    *,
    n_atoms: int,
    linear: bool,
) -> EvalResult:
    # Trap potential sampled in laboratory coordinates.
    I_segments = jnp.asarray(schedule.I_segments_A)
    def U_func(points_lab_mm: jnp.ndarray, currents_A: jnp.ndarray) -> jnp.ndarray:
        U_flat, _, _ = trap_potential_energies(
            points_lab_mm, chip.atom, chip.geometry, currents_A, chip.bias_field,
        )
        return U_flat

    # Mean-field coupling and dimensionless interaction strength.
    a_s_m = float(chip.atom.scattering_length_m)
    g = float(4.0 * np.pi * constants.hbar ** 2 * a_s_m * n_atoms / chip.atom.mass_kg)
    omega_ref_rad = float(np.prod(2.0 * np.pi * np.asarray(schedule.omegas_hz[0])) ** (1.0 / 3.0))
    a_ho_m = float(np.sqrt(constants.hbar / (chip.atom.mass_kg * omega_ref_rad)))
    delta = float(4.0 * np.pi * n_atoms * a_s_m / a_ho_m)

    _, tf_radii = schedule.tf_quantities(chip.atom, n_atoms)

    # Validate the grid and build the propagators.
    ground_state, forward_step, forward_half = make_solvers(
        grid,
        stepping,
        U_func=U_func,
        mass_kg=float(chip.atom.mass_kg),
        g=g,
        omegas_hz=schedule.omegas_hz,
        linear=linear,
    )

    # fmt: off
    r_mins_mm    = jnp.asarray(schedule.r_mins_mm       , dtype=jnp.float64)  # (T, 3) trap-minimum trajectory [mm]; the box is centered here
    target_r_mm  = jnp.asarray(schedule.target_rs_mm[-1], dtype=jnp.float64)  # (3,) final target position [mm]
    initial_axes = jnp.asarray(schedule.eigvecs[0]      , dtype=jnp.float64)  # initial principal axes; schedule.eigvecs is (T, 3, 3)
    # fmt: on

    # Initial and target ground states, seeded with TF widths σ = R_TF/√7.
    logger.info("relaxing ground states (initial + target)...")
    sigma_seed_m = np.asarray(tf_radii) / np.sqrt(7.0)   # (T, 3) TF-width GS seeds [m]
    psi0, gs_residual_initial = ground_state(r_mins_mm[0], I_segments[0], sigma_seed_m[0])
    psi_target, gs_residual_target = ground_state(target_r_mm, I_segments[-1], sigma_seed_m[-1])

    # Flag under-relaxed states without stopping the run.
    for gs_name, residual in (("initial", gs_residual_initial), ("target", gs_residual_target)):
        if residual > GS_RESIDUAL_WARN:
            logger.warning("GS %s: stationary residual %.2e exceeds %.0e — ground state may be "
                           "under-relaxed; fidelity may be unreliable", gs_name, residual, GS_RESIDUAL_WARN)
        else:
            logger.info("GS %s: stationary residual %.2e", gs_name, residual)

    # Compare relaxed and Thomas–Fermi widths.
    _, _, sigma0 = cloud_observables(grid, psi=psi0, axes=initial_axes)
    sigma_tf0_m  = sigma_seed_m[0]
    gs_rel_dev   = [round(float((sigma0[i] - sigma_tf0_m[i]) / sigma_tf0_m[i]), 4)
                    if sigma_tf0_m[i] > 0 else float("nan") for i in range(3)]
    rel_dev_pct = "[" + " ".join(f"{d * 100:+5.1f}%" for d in gs_rel_dev) + "]"
    logger.info("GS sanity: relaxed σ=%s µm  TF σ=%s µm  rel-dev=%s",
        _format_um(sigma0), _format_um(sigma_tf0_m), rel_dev_pct)

    # Node-centred holds: half intervals at the endpoints; full interior holds return ψ(t_j) for
    # node-time diagnostics.
    n_nodes = int(r_mins_mm.shape[0])

    psi = psi0
    com_traj, cov_traj, sigma_traj, boundary_traj, k_tail_traj, k_shell_traj = [], [], [], [], [], []
    def record(psi: jnp.ndarray, j: int) -> None:
        com, cov, sigma = cloud_observables(grid, psi=psi, axes=initial_axes)
        edge_mass = np.asarray(boundary_mass(grid, psi=psi, edge_frac=_BOUNDARY_EDGE_FRAC))
        k_tail = np.asarray(k_tail_mass(grid, psi=psi, edge_frac=_K_TAIL_EDGE_FRAC))
        k_shells = np.asarray(k_shell_masses(grid, psi=psi, edges_frac=_K_SHELL_EDGES_FRAC))
        for lst, v in ((com_traj, com), (cov_traj, cov), (sigma_traj, sigma),
                       (boundary_traj, edge_mass), (k_tail_traj, k_tail), (k_shell_traj, k_shells)):
            lst.append(np.asarray(v))
        if j % 100 == 0 or j == n_nodes - 1:
            m_shelf = float(k_shells.sum(axis=1).max())
            logger.info("[%4d/%d]  com=%s µm  σ=%s µm  edge=%.1e  ktail=%.1e  mshelf=%.1e",
                        j, n_nodes - 1, _format_um(com), _format_um(sigma),
                        float(edge_mass.max()), float(k_tail.max()), m_shelf)

    logger.info("node-centred holds: %d nodes → %d potential samples, %d×%d = %d Strang substeps",
                n_nodes, n_nodes, n_nodes - 1, stepping.n_sub, (n_nodes - 1) * stepping.n_sub)
    record(psi, 0)
    psi = forward_half(psi, r_mins_mm[0], r_mins_mm[0], I_segments[0])
    for j in range(1, n_nodes - 1):
        psi, psi_at_node = forward_step(psi, r_mins_mm[j], r_mins_mm[j - 1], I_segments[j])
        record(psi_at_node, j)
    psi = forward_half(psi, r_mins_mm[-1], r_mins_mm[-2], I_segments[-1])
    record(psi, n_nodes - 1)
    assert len(com_traj) == n_nodes, f"expected {n_nodes} node records, got {len(com_traj)}"

    # Align the target state with the final computational box.
    psi_target_aligned = shift_box(
        grid,
        psi=psi_target,
        delta_r_m=(r_mins_mm[-1] - target_r_mm) * 1e-3,
    )
    F_3D = float(fidelity(grid, psi_a=psi, psi_b=psi_target_aligned))

    # fmt: off
    com_traj_local_m = np.stack(com_traj)
    cov_traj_m2      = np.stack(cov_traj)
    sigma_traj_m     = np.stack(sigma_traj)

    # Convert the local COM trajectory to laboratory coordinates.
    com_traj_lab_mm   = com_traj_local_m * 1e3 + schedule.r_mins_mm

    return EvalResult(
        F_3D                 = F_3D,
        delta                = delta,
        com_traj_lab_mm      = com_traj_lab_mm,
        sigma_traj_m         = sigma_traj_m,
        cov_traj_m2          = cov_traj_m2,
        boundary_mass_traj   = np.stack(boundary_traj),
        k_tail_mass_traj     = np.stack(k_tail_traj),
        k_shell_mass_traj    = np.stack(k_shell_traj),
        gs_residual_initial  = gs_residual_initial,
        gs_residual_target   = gs_residual_target,
    )
    # fmt: on


def _format_um(val_m: np.ndarray) -> str:
    """Fixed-width µm string for progress logs (input in metres)."""
    return np.array2string(np.asarray(val_m) * 1e6, formatter={"float_kind": lambda x: f"{x:6.2f}"})
