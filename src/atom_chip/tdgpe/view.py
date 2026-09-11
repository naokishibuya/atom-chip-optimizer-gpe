"""Consumer side of an eval: the on-disk format (write/load), the derived view object, and rendering."""
import csv
import json
import logging
import os
from dataclasses import asdict, dataclass
from itertools import permutations
from typing import NamedTuple

import matplotlib

matplotlib.use("Agg")  # headless backend; must be set before the pyplot import

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from matplotlib.ticker import MultipleLocator

from ..schedule import AtomChip, CurrentSchedule
from .box import BoxGrid, tail_margin_m
from .solver import Stepping
from .transport import EvalResult

# The sweep aggregate's columns, shared by the csv file and the terminal table.
# Each entry maps a header to (extractor(eval_gpe, eval_linear) -> value | None, format); either
# EvalResult may be None (that mode not run). gs_resid is the worst GS residual in the cell, across
# both runs and both states (initial/target) — flags a shaky ground state behind the fidelity.
# fmt: off
_SWEEP_COLUMNS = {
    "F_GPE":       (lambda eg, el: eg.F_3D if eg else None, "{:.6f}"),
    "F_linear":    (lambda eg, el: el.F_3D if el else None, "{:.6f}"),
    "F_lin-F_GPE": (lambda eg, el: (el.F_3D - eg.F_3D) if (eg and el) else None, "{:+.4f}"),
    "gs_resid":    (lambda eg, el: max(
        (v for r in (eg, el) if r for v in (r.gs_residual_initial, r.gs_residual_target) if v is not None),
        default=None), "{:.1e}"),
}
# fmt: on


# fmt: off
@dataclass
class CloudDiagnostics:
    """Analysis-ready eval point: raw moments + the views derived from cov + the schedule.

    What the plotters consume; built by `load_diagnostics`. Plotters take individual fields off
    this, never the object itself, so the container can change without touching them.
    """
    F_3D                : float
    T_total_s           : float
    n_atoms             : int
    com_traj_lab_mm     : np.ndarray     # (T, 3) lab-frame COM
    box_center_traj_mm  : np.ndarray     # (T, 3) lab box position (= schedule r_mins)
    cloud_sigma_traj_m  : np.ndarray     # (T, 3) cloud principal widths paired to trap axes
    cloud_misalign_deg  : np.ndarray     # (T, 3) cloud-vs-trap axis angle per paired axis
    sigma_tf_traj_m     : np.ndarray     # (T, 3) instantaneous TF σ, same trap-axis pairing
    tf_extent_lab_m     : np.ndarray     # (T, 3) instantaneous TF half-extents on lab x/y/z
    boundary_mass_traj  : np.ndarray     # (T, 3) per-axis outer-slab |ψ|² fraction (grid x/y/z)
    k_tail_mass_traj    : np.ndarray     # (T, 3) per-axis near-Nyquist |ψ|² fraction (grid x/y/z)
    k_shell_mass_traj   : np.ndarray     # (T, 3, S) nested octave shells (m_shelf)
    obs_extent_grid_m   : np.ndarray     # (3,) peak TF-equivalent extent on grid axes
    obs_com_grid_m      : np.ndarray     # (3,) peak COM offset on grid axes
    budget_extent_m     : np.ndarray     # (3,) sizer TF half-extent per axis (the box budget)
    budget_slosh_m      : np.ndarray     # (3,) sizer predicted peak COM excursion per axis
    budget_shift_margin_m: np.ndarray    # (3,) maximum node jump
    budget_tail_m       : float          # sizer surface-tail margin (isotropic)
    box_half_widths_m   : np.ndarray     # (3,) resolved box half-width per axis (the hard wall)
# fmt: on


#-------------------------------------------------------------------------------
# Figure-group dispatchers (cli calls these; the plot_* below are internal)
#-------------------------------------------------------------------------------


def render_eval_figs(diag: CloudDiagnostics, fig_dir: str, *, reg: str | None = None) -> None:
    """Save the dynamics and grid-occupancy plots for one evaluated schedule.

    Args:
        diag: Simulation results and derived cloud/trap diagnostics for one (N, T).
        fig_dir: Output directory, created if needed.
        reg: Optional regularization label added to saved figures.
    """
    os.makedirs(fig_dir, exist_ok=True)
    t_ms = _time_axis_ms(diag.T_total_s, n_blocks=len(diag.com_traj_lab_mm))
    _save_fig(plot_com(t_ms, diag.com_traj_lab_mm, diag.box_center_traj_mm, F_3D=diag.F_3D),
              os.path.join(fig_dir, "com.png"), reg)
    _save_fig(plot_com_displacement(t_ms, diag.com_traj_lab_mm, diag.box_center_traj_mm,
                                    F_3D=diag.F_3D),
              os.path.join(fig_dir, "com_displacement.png"), reg)
    _save_fig(plot_breathing(t_ms, diag.cloud_sigma_traj_m, diag.sigma_tf_traj_m, F_3D=diag.F_3D),
              os.path.join(fig_dir, "breathing.png"), reg)
    _save_fig(plot_breathing_response(t_ms, diag.cloud_sigma_traj_m, diag.sigma_tf_traj_m),
              os.path.join(fig_dir, "breathing_response.png"), reg)
    _save_fig(plot_extent_occupancy(t_ms, diag.tf_extent_lab_m, diag.boundary_mass_traj,
                                    n_atoms=diag.n_atoms),
              os.path.join(fig_dir, "extent_occupancy.png"), reg)
    _save_fig(plot_edge_occupancy(t_ms, diag.boundary_mass_traj, n_atoms=diag.n_atoms),
              os.path.join(fig_dir, "edge_occupancy.png"), reg)
    _save_fig(plot_spectral_occupancy(t_ms, diag.k_tail_mass_traj, n_atoms=diag.n_atoms),
              os.path.join(fig_dir, "spectral_occupancy.png"), reg)
    _save_fig(plot_ellipticity(t_ms, diag.cloud_sigma_traj_m, F_3D=diag.F_3D),
              os.path.join(fig_dir, "ellipticity.png"), reg)
    _save_fig(plot_axial_misalignment(t_ms, diag.cloud_misalign_deg, F_3D=diag.F_3D),
              os.path.join(fig_dir, "axial_misalignment.png"), reg)
    _save_fig(plot_sizer_adequacy(diag.obs_extent_grid_m, diag.obs_com_grid_m, diag.budget_extent_m,
                                  diag.budget_slosh_m, diag.budget_shift_margin_m, diag.budget_tail_m,
                                  diag.box_half_widths_m, n_atoms=diag.n_atoms),
              os.path.join(fig_dir, "sizer_adequacy.png"), reg)
    logging.getLogger("chip.plot").info("Eval plots written to %s", fig_dir)


def render_sweep_figs(fig_dir: str, *, rows: list, reg: str | None = None) -> None:
    """Save a GPE-versus-linear endpoint-loss comparison across transport durations.

    Args:
        fig_dir: Output directory, created if needed.
        rows: (N, T, eval_gpe, eval_linear) tuples; T is in seconds and missing results are None.
        reg: Optional regularization label added to the saved figure.
    """
    os.makedirs(fig_dir, exist_ok=True)
    f_rows = [(N, T, eg.F_3D if eg else None, el.F_3D if el else None) for N, T, eg, el in rows]
    _save_fig(plot_f_vs_t(f_rows), os.path.join(fig_dir, "f_vs_t.png"), reg)
    logging.getLogger("chip.plot").info("Sweep plots written to %s", fig_dir)


def write_fidelity_csv(csv_path: str, *, rows: list) -> None:
    """Write the sweep aggregate as long-format rows (one per (N, T) cell), N as a column."""
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(["N", "T_s", *_SWEEP_COLUMNS])
        for N, T, eg, el in rows:
            row: list = [N, f"{T:.2f}"]
            for extractor, fmt in _SWEEP_COLUMNS.values():
                v = extractor(eg, el)
                row.append("" if v is None else fmt.format(v).lstrip("+"))
            writer.writerow(row)
    logging.getLogger("chip.plot").info("Sweep table written to %s", csv_path)


def format_fidelity_table(rows: list) -> str:
    """The terminal view of the sweep aggregate: right-justified columns, grouped by N.
    `rows` are (N, T, eval_gpe, eval_linear)."""
    headers = ["T (s)", *_SWEEP_COLUMNS]
    formatted = []
    for N, T, eg, el in rows:
        cells = [f"{T:.2f}"]
        for extractor, fmt in _SWEEP_COLUMNS.values():
            v = extractor(eg, el)
            cells.append("—" if v is None else fmt.format(v))
        formatted.append((N, cells))
    widths = [max(len(headers[i]), *(len(r[1][i]) for r in formatted)) for i in range(len(headers))]

    lines: list[str] = []
    last_N = None
    for N, cells in formatted:
        if last_N != N:
            if last_N is not None:
                lines.append("")
            lines.append(f"N = {N}")
            lines.append("  " + "  ".join(h.rjust(w) for h, w in zip(headers, widths, strict=True)))
            last_N = N
        lines.append("  " + "  ".join(c.rjust(w) for c, w in zip(cells, widths, strict=True)))
    return "\n".join(lines)


def render_overlay_fig(fig_dir: str, *, gpe: EvalResult, lin: EvalResult, T_total_s: float,
                       trap_eigvecs: np.ndarray, axis_index: int = 0, reg: str | None = None) -> None:
    """Save a cloud-width comparison between GPE and linear evolution for one schedule.

    Args:
        fig_dir: Output directory, created if needed.
        gpe: Interacting simulation results at the recorded control nodes.
        lin: Linear results at the same control nodes.
        T_total_s: Total transport duration [s].
        trap_eigvecs: Trap principal axes, shape (n_times, 3, 3), stored as columns.
        axis_index: Paired cloud axis: 0 axial, 1 or 2 transverse.
        reg: Optional regularization label added to the saved figure.
    """
    os.makedirs(fig_dir, exist_ok=True)
    t_ms = _time_axis_ms(T_total_s, n_blocks=len(gpe.com_traj_lab_mm))
    cloud_sigma_gpe, _ = _cloud_axes_paired(gpe.cov_traj_m2, trap_eigvecs)
    cloud_sigma_lin, _ = _cloud_axes_paired(lin.cov_traj_m2, trap_eigvecs)
    _save_fig(
        plot_sigma_overlay(t_ms, cloud_sigma_gpe[:, axis_index],
                           cloud_sigma_lin[:, axis_index], axis_index=axis_index),
        os.path.join(fig_dir, "sigma_overlay.png"), reg,
    )


#-------------------------------------------------------------------------------
# On-disk eval format: write (producer side) + load (consumer side)
#-------------------------------------------------------------------------------


def write_eval(
    json_path: str,
    npz_path: str,
    result: EvalResult,
    *,
    n_atoms: int,
    T_total_s: float,
    grid: BoxGrid,
    stepping: Stepping,
    sizing: dict,
    linear: bool,
) -> None:
    """Serialize one eval point: scalars (run config + F_3D/δ) to json, series to npz.

    `sizing` records the auto-sizer inputs (predicted CloudGeometry) and policy knobs behind
    the resolved half_widths_m / n_cells, so the grid choice is reproducible.
    """
    scalars = {
        "n_atoms": int(n_atoms),
        "T_total_s": float(T_total_s),
        "half_widths_m": list(grid.half_widths_m),
        "n_cells": list(grid.shape),
        "sizing": sizing,
        **asdict(stepping),
        "n_sub": stepping.n_sub,
        "linear": linear,
        "F_3D": float(result.F_3D),
        "delta": float(result.delta),
        "ground_state": {
            "residual_initial": result.gs_residual_initial,
            "residual_target": result.gs_residual_target,
        },
    }
    with open(json_path, "w") as f:
        json.dump(scalars, f, indent=2)
    np.savez_compressed(
        npz_path,
        com_traj_lab_mm=result.com_traj_lab_mm,
        sigma_traj_m=result.sigma_traj_m,
        cov_traj_m2=result.cov_traj_m2,
        boundary_mass_traj=result.boundary_mass_traj,
        k_tail_mass_traj=result.k_tail_mass_traj,
        k_shell_mass_traj=result.k_shell_mass_traj,
    )


def load_eval_result(json_path: str, npz_path: str) -> EvalResult:
    """Read the raw `EvalResult` back (no schedule needed): for sweep aggregation."""
    return _result_from_dict(_load_eval_data(json_path, npz_path))


def load_diagnostics(json_path: str, npz_path: str, *, chip: AtomChip, sched: CurrentSchedule,
                     n_atoms: int) -> CloudDiagnostics:
    """Read an eval point and derive the σ/rotation/TF views from cov + the schedule."""
    data = _load_eval_data(json_path, npz_path)
    result = _result_from_dict(data)
    _, tf_radii = sched.tf_quantities(chip.atom, n_atoms)
    tf_radii = np.asarray(tf_radii)
    tf_extent_lab = np.sqrt(np.einsum("tik,tk->ti", np.asarray(sched.eigvecs) ** 2,
                                      tf_radii ** 2))
    cloud_sigma, cloud_misalign = _cloud_axes_paired(result.cov_traj_m2, sched.eigvecs)
    residual_mm = result.com_traj_lab_mm - np.asarray(sched.r_mins_mm)
    sizing = data["sizing"]
    return CloudDiagnostics(
        F_3D                 = result.F_3D,
        T_total_s            = float(data["T_total_s"]),
        n_atoms              = n_atoms,
        com_traj_lab_mm      = result.com_traj_lab_mm,
        box_center_traj_mm   = np.asarray(sched.r_mins_mm),
        cloud_sigma_traj_m   = cloud_sigma,
        cloud_misalign_deg   = cloud_misalign,
        sigma_tf_traj_m      = tf_radii / np.sqrt(7.0),
        tf_extent_lab_m      = tf_extent_lab,
        boundary_mass_traj   = result.boundary_mass_traj,
        k_tail_mass_traj     = result.k_tail_mass_traj,
        k_shell_mass_traj    = result.k_shell_mass_traj,
        obs_extent_grid_m    = np.sqrt(7.0 * np.asarray(result.cov_traj_m2).diagonal(axis1=1, axis2=2).max(axis=0)),
        obs_com_grid_m       = np.abs(residual_mm).max(axis=0) * 1e-3,
        budget_extent_m      = np.asarray(sizing["extent_m"]),
        budget_slosh_m       = np.asarray(sizing["slosh_m"]),
        budget_shift_margin_m = np.asarray(sizing["box_shift_margin_m"]),
        budget_tail_m        = tail_margin_m(float(sizing["xi_ref_m"]), float(sizing["boundary_tol"])),
        box_half_widths_m    = np.asarray(data["half_widths_m"]),
    )


#-------------------------------------------------------------------------------
# Eval plotters — pure array renderers (no CloudDiagnostics dependency)
#-------------------------------------------------------------------------------


def plot_com(t_ms: np.ndarray, com_traj_lab_mm: np.ndarray, box_center_traj_mm: np.ndarray,
             *, F_3D: float) -> Figure:
    """Compare the cloud centroid with the trap minimum along each laboratory axis.

    Args:
        t_ms: Recorded times [ms], shape (n_times,).
        com_traj_lab_mm: Laboratory COM positions [mm], shape (n_times, 3), ordered x/y/z.
        box_center_traj_mm: Box centers at the trap minima [mm], same shape and axis order.
        F_3D: Endpoint fidelity shown in the title.

    Returns:
        Figure with three position panels sharing the time axis.
    """
    com = np.asarray(com_traj_lab_mm)
    box = np.asarray(box_center_traj_mm)
    fig, axes = plt.subplots(3, 1, figsize=(8, 7), sharex=True)
    for ax, i, label in zip(axes, range(3), "xyz", strict=True):
        ax.plot(t_ms, com[:, i], label=f"COM {label}", linewidth=1.2)
        ax.plot(t_ms, box[:, i], "--", label=rf"$r_\mathrm{{min}}$ {label}", linewidth=1.0, alpha=0.7)
        ax.set_ylabel(f"{label} [mm]")
        ax.legend()
        ax.grid(alpha=0.3)
    axes[-1].set_xlabel("time [ms]")
    fig.suptitle(f"COM vs trap minimum — $F_{{3D}}$ = {F_3D:.4f}")
    fig.tight_layout()
    return fig


def plot_com_displacement(
    t_ms: np.ndarray, com_traj_lab_mm: np.ndarray, box_center_traj_mm: np.ndarray,
    *, F_3D: float, comparison_ylim_um: tuple[float, float] = (-0.65, 0.65),
    ytick_step_um: float = 0.2) -> Figure:
    """Show signed COM displacement from the trap minimum to reveal sloshing.

    Args:
        t_ms: Recorded times [ms], shape (n_times,).
        com_traj_lab_mm: Laboratory COM positions [mm], shape (n_times, 3), ordered x/y/z.
        box_center_traj_mm: Box centers at the trap minima [mm], same shape and axis order.
        F_3D: Endpoint fidelity shown in the title.
        comparison_ylim_um: Common vertical limits [µm], used only when all data fit.
        ytick_step_um: Major vertical tick spacing [µm].

    Returns:
        Figure with x/y/z displacement curves in micrometres.
    """
    com = np.asarray(com_traj_lab_mm)
    box = np.asarray(box_center_traj_mm)
    displacement_um = (com - box) * 1e3
    fig, ax = plt.subplots(figsize=(8, 4))
    for i, label in enumerate("xyz"):
        ax.plot(t_ms, displacement_um[:, i], label=rf"$\delta {label}$", linewidth=1.2)
    ax.set_xlabel("time [ms]")
    ax.set_ylabel(r"$\langle r\rangle-r_{\rm min}$ [$\mu$m]")
    ax.set_title(rf"COM Displacement from Trap Minimum — $F_{{3D}}={F_3D:.6f}$")
    ax.legend()
    ax.grid(alpha=0.3)
    ax.yaxis.set_major_locator(MultipleLocator(ytick_step_um))
    if displacement_um.min() >= comparison_ylim_um[0] and displacement_um.max() <= comparison_ylim_um[1]:
        ax.set_ylim(comparison_ylim_um)
    fig.tight_layout()
    return fig


def plot_breathing(t_ms: np.ndarray, cloud_sigma_traj_m: np.ndarray, sigma_tf_traj_m: np.ndarray,
                   *, F_3D: float) -> Figure:
    """Compare absolute cloud widths with the equilibrium TF prediction over time.

    Args:
        t_ms: Recorded times [ms], shape (n_times,).
        cloud_sigma_traj_m: Cloud principal RMS widths [m], shape (n_times, 3),
            paired to trap axes in axial/trans-1/trans-2 order.
        sigma_tf_traj_m: Equilibrium TF RMS widths [m], same shape and axis order.
        F_3D: Endpoint fidelity shown in the title.

    Returns:
        Figure with axial and geometric-mean transverse widths in micrometres.
    """
    ch = _breathing_ratios(cloud_sigma_traj_m, sigma_tf_traj_m)
    to_um = 1e6

    fig, ax_size = plt.subplots(figsize=(8, 6.5), sharex=True)
    ax_size.plot(t_ms, ch.cloud_axial * to_um, color="C0", linewidth=0.4, label="cloud axial")
    ax_size.plot(t_ms, ch.tf_axial * to_um, color="C0", linewidth=0.8, alpha=0.6, label="TF axial")
    ax_size.plot(t_ms, ch.cloud_trans * to_um, color="C1", linewidth=0.4, label="cloud transverse (geom. mean)")
    ax_size.plot(t_ms, ch.tf_trans * to_um, color="C1", linewidth=0.8, alpha=0.6,
                label="TF transverse (geom. mean)")
    ax_size.set_ylabel("σ [μm]")
    ax_size.set_title(f"Cloud vs instantaneous TF widths — $F_{{3D}}$ = {F_3D:.4f}")
    ax_size.legend(fontsize="small")
    ax_size.grid(alpha=0.3)
    ax_size.set_xlabel("time [ms]")
    fig.tight_layout()
    return fig


def plot_breathing_response(
    t_ms: np.ndarray,
    cloud_sigma_traj_m: np.ndarray,
    sigma_tf_traj_m: np.ndarray,
) -> Figure:
    """Compare fractional cloud-width changes with fractional TF-width changes.

    Args:
        t_ms: Recorded times [ms], shape (n_times,).
        cloud_sigma_traj_m: Cloud principal RMS widths [m], shape (n_times, 3),
            paired to trap axes in axial/trans-1/trans-2 order.
        sigma_tf_traj_m: Equilibrium TF RMS widths [m], same shape and axis order.

    Returns:
        Figure of (σ(t)/σ(0)) / (σ_TF(t)/σ_TF(0)) for axial and geometric-mean
        transverse widths. Unity means equal fractional changes.
    """
    ch = _breathing_ratios(cloud_sigma_traj_m, sigma_tf_traj_m)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.axhline(1.0, color="gray", linewidth=0.8, alpha=0.6)
    ax.plot(t_ms, ch.b_tilde_trans, color="C1", linewidth=0.4, alpha=0.85, label="transverse")
    ax.plot(t_ms, ch.b_tilde_axial, color="C0", linewidth=0.4, alpha=0.95, label="axial")
    ax.set_xticks(np.linspace(t_ms[0], t_ms[-1], 9))
    ax.set_ylim(0.95, 1.05)
    ax.set_xlabel("time [ms]")
    ax.set_ylabel(r"$b_k(t)$")
    ax.set_title("GPE / Thomas–Fermi Width Comparison")
    ax.legend(fontsize="small")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def plot_edge_occupancy(t_ms: np.ndarray, boundary_mass_traj: np.ndarray, *, n_atoms: int) -> Figure:
    """Show how much probability occupies the regions near each spatial grid edge.

    Args:
        t_ms: Recorded times [ms], shape (n_times,).
        boundary_mass_traj: Near-edge probability fractions, shape (n_times, 3),
            ordered grid x/y/z and including both sides of each axis.
        n_atoms: Atom number shown in the title.

    Returns:
        Figure with three occupancy curves on a logarithmic vertical scale.
    """
    boundary = np.clip(np.asarray(boundary_mass_traj), 1e-20, None)   # (T, 3)
    fig, ax = plt.subplots(figsize=(8, 4))
    for i, label in enumerate(("x", "y", "z")):
        ax.semilogy(t_ms, boundary[:, i], color=f"C{i}", linewidth=0.5, alpha=0.8, label=label)
    ax.set_title(f"Near-Edge Spatial Occupancy ($N={n_atoms:,}$)")
    ax.set_xlabel("time [ms]")
    ax.set_ylabel("probability mass")
    ax.set_ylim(1e-16, 1e-4)
    ax.legend(ncol=3, fontsize="small", title="axis")
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    return fig


def plot_extent_occupancy(
    t_ms: np.ndarray,
    tf_extent_lab_m: np.ndarray,
    boundary_mass_traj: np.ndarray,
    *,
    n_atoms: int,
    comparison_ylim_um: tuple[float, float] = (0.0, 7.5),
) -> Figure:
    """Compare trap-dependent TF extents with spatial occupancy on a shared time axis.

    Args:
        t_ms: Recorded times [ms], shape (n_times,).
        tf_extent_lab_m: Equilibrium TF half-extents along laboratory x/y/z [m],
            shape (n_times, 3).
        boundary_mass_traj: Near-edge probability fractions for grid x/y/z,
            shape (n_times, 3), including both sides of each axis.
        n_atoms: Atom number shown in the title.
        comparison_ylim_um: Extent-panel limits [µm], used when the maximum fits below the upper limit.

    Returns:
        Figure with TF extents above and logarithmic near-edge occupancy below.
    """
    extents_um = np.asarray(tf_extent_lab_m) * 1e6
    boundary = np.clip(np.asarray(boundary_mass_traj), 1e-20, None)
    fig, (ax_extent, ax_occupancy) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    for i, label in enumerate(("x", "y", "z")):
        ax_extent.plot(t_ms, extents_um[:, i], color=f"C{i}", linewidth=1.2, label=label)
        ax_occupancy.semilogy(
            t_ms, boundary[:, i], color=f"C{i}", linewidth=0.5, alpha=0.8,
        )

    ax_extent.set_ylabel(r"$E_i^{\rm TF}$ [$\mu$m]")
    ax_extent.set_title(rf"Laboratory-Axis Thomas–Fermi Extents ($N={n_atoms:,}$)")
    if np.max(extents_um) <= comparison_ylim_um[1]:
        ax_extent.set_ylim(comparison_ylim_um)
    ax_extent.grid(alpha=0.3)

    ax_occupancy.set_xlabel("time [ms]")
    ax_occupancy.set_ylabel("probability mass")
    ax_occupancy.set_title("Near-Edge Spatial Occupancy")
    ax_occupancy.set_ylim(1e-16, 1e-4)
    ax_occupancy.grid(alpha=0.3, which="both")

    ax_occupancy.legend(
        handles=ax_extent.lines,
        labels=("x", "y", "z"),
        ncol=3,
        fontsize="small",
        loc="lower right",
        bbox_to_anchor=(1.0, 1.05),
        borderaxespad=0.0,
    )
    fig.subplots_adjust(left=0.12, right=0.99, bottom=0.09, top=0.94, hspace=0.22)
    return fig


def plot_spectral_occupancy(t_ms: np.ndarray, k_tail_mass_traj: np.ndarray, *, n_atoms: int) -> Figure:
    """Show whether the momentum distribution occupies the region near the grid cutoff.

    Args:
        t_ms: Recorded times [ms], shape (n_times,).
        k_tail_mass_traj: Near-Nyquist probability fractions for grid x/y/z,
            shape (n_times, 3), including positive and negative wavevectors.
        n_atoms: Atom number shown in the title.

    Returns:
        Figure with three spectral-occupancy curves on a logarithmic vertical scale.
    """
    k_tail = np.clip(np.asarray(k_tail_mass_traj), 1e-20, None)       # (T, 3)
    fig, ax = plt.subplots(figsize=(8, 4))
    for i, label in enumerate(("x", "y", "z")):
        ax.semilogy(t_ms, k_tail[:, i], color=f"C{i}", linewidth=0.5, alpha=0.8, label=label)
    ax.set_title(f"Near-Nyquist Spectral Occupancy ($N={n_atoms:,}$)")
    ax.set_xlabel("time [ms]")
    ax.set_ylabel("probability mass")
    ax.set_ylim(1e-20, 1e-7)
    ax.legend(ncol=3, fontsize="small", title="axis", loc="lower right")
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    return fig


def plot_ellipticity(t_ms: np.ndarray, cloud_sigma_traj_m: np.ndarray, *, F_3D: float) -> Figure:
    """Show the difference between the two transverse widths as unsigned ellipticity.

    Args:
        t_ms: Recorded times [ms], shape (n_times,).
        cloud_sigma_traj_m: Cloud principal RMS widths [m], shape (n_times, 3),
            paired to trap axes in axial/trans-1/trans-2 order.
        F_3D: Endpoint fidelity shown in the title.

    Returns:
        Figure of 100 |σ_trans1 - σ_trans2| / (σ_trans1 + σ_trans2).
        Swapping transverse labels does not change this percentage.
    """
    eps = _ellipticity_pct(cloud_sigma_traj_m)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(t_ms, eps, color="C3", linewidth=0.1, alpha=0.5, label="ellipticity")
    ax.set_xlabel("time [ms]")
    ax.set_ylabel(r"$|\sigma_{\perp 1}-\sigma_{\perp 2}|\,/\,(\sigma_{\perp 1}+\sigma_{\perp 2})$  [%]")
    ax.set_title(f"Unsigned transverse ellipticity — $F_{{3D}}$ = {F_3D:.4f}")
    ax.legend(fontsize="small")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def plot_axial_misalignment(
    t_ms: np.ndarray, cloud_misalign_deg: np.ndarray, *, F_3D: float) -> Figure:
    """Show how closely the cloud's axial principal direction follows the trap's.

    Args:
        t_ms: Recorded times [ms], shape (n_times,).
        cloud_misalign_deg: Unsigned paired cloud–trap angles [degrees],
            shape (n_times, 3), axial/trans-1/trans-2 order; only column 0 is plotted.
        F_3D: Endpoint fidelity shown in the title.

    Returns:
        Figure of axial misalignment over time; transverse angles are not shown.
    """
    misalign = np.asarray(cloud_misalign_deg)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(t_ms, misalign[:, 0], color="C0", linewidth=0.1, alpha=0.5)
    ax.set_ylim(0.0, 0.05)
    ax.set_xlabel("time [ms]")
    ax.set_ylabel(r"$\theta_1$ [deg]")
    ax.set_title(f"Axial Cloud–Trap Misalignment — $F_{{3D}}={F_3D:.6f}$")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def plot_sizer_adequacy(obs_extent_grid_m: np.ndarray, obs_com_grid_m: np.ndarray,
                        budget_extent_m: np.ndarray, budget_slosh_m: np.ndarray,
                        budget_shift_margin_m: np.ndarray, budget_tail_m: float,
                        box_half_widths_m: np.ndarray, *, n_atoms: int) -> Figure:
    """Compare the surrogate sizing budget with simulated extents and grid boundaries.

    Args:
        obs_extent_grid_m: Maximum sampled TF-equivalent cloud half-extents [m],
            shape (3,), ordered grid x/y/z.
        obs_com_grid_m: Maximum sampled absolute COM displacements from the box center [m], (3,).
        budget_extent_m: Maximum Ermakov half-extents [m], (3,), in the same axis order.
        budget_slosh_m: Predicted maximum absolute COM excursions [m], (3,).
        budget_shift_margin_m: Per-axis translation buffers [m], (3,).
        budget_tail_m: Scalar tail margin [m], applied to each axis.
        box_half_widths_m: Actual grid half-widths [m], (3,).
        n_atoms: Atom number shown in the title.

    Returns:
        Figure with stacked budget and simulation bars per axis, and grid half-width markers.
        Cloud and COM maxima are stacked separately; they need not occur at the same time.
    """
    to_um = 1e6
    obs_extent = np.asarray(obs_extent_grid_m) * to_um
    obs_slosh = np.asarray(obs_com_grid_m) * to_um
    bud_extent = np.asarray(budget_extent_m) * to_um
    bud_slosh = np.asarray(budget_slosh_m) * to_um
    bud_shift = np.asarray(budget_shift_margin_m) * to_um
    bud_tail = np.full(3, float(budget_tail_m) * to_um)
    wall = np.asarray(box_half_widths_m) * to_um
    x = np.arange(3)
    w = 0.36
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.set_xticks(x)
    ax.set_xticklabels(("x", "y", "z"))
    ax.set_ylabel("distance [μm]")
    ax.set_title(f"Surrogate-Assisted Grid Sizing and GPE Simulation Results ($N={n_atoms:,}$)")

    # legend for budget
    wall_line = ax.hlines(
        wall, x - 0.45, x + 0.45,
        color="k", linestyle="--", linewidth=1.2,
        label="grid half-width",
    )
    tail_bar = ax.bar(
        x - w / 2, bud_tail, w,
        bottom=bud_extent + bud_slosh + bud_shift,
        color="C4", alpha=0.5, label="tail margin",
    )
    shift_bar = ax.bar(
        x - w / 2, bud_shift, w, bottom=bud_extent + bud_slosh,
        color="C2", alpha=0.5, label="translation buffer",
    )
    surrogate_com_bar = ax.bar(
        x - w / 2, bud_slosh, w, bottom=bud_extent,
        color="C1", alpha=0.5, label="maximum COM excursion",
    )
    erm_bar = ax.bar(
        x - w / 2, bud_extent, w,
        color="C0", alpha=0.5, label="maximum Ermakov extent",
    )

    ax.grid(alpha=0.3, axis="y")
    sizing_legend = ax.legend(
        handles=[
            wall_line,
            tail_bar,
            shift_bar,
            surrogate_com_bar,
            erm_bar,
        ],
        title="Surrogate-assisted grid sizing",
        fontsize="small",
        loc="upper right",
        bbox_to_anchor=(0.65, 0.98),
        borderaxespad=0.0,
    )
    ax.add_artist(sizing_legend)

    # legend for observed
    gpe_com_bar = ax.bar(
        x + w / 2, obs_slosh, w, bottom=obs_extent,
        color="C1", label="maximum COM excursion",
    )
    gpe_extent_bar = ax.bar(
        x + w / 2, obs_extent, w,
        color="C0", label="maximum TF-equivalent extent",
    )

    ax.legend(
        handles=[gpe_com_bar, gpe_extent_bar],
        title="GPE simulation results",
        fontsize="small",
        loc="upper right",
        bbox_to_anchor=(0.99, 0.98),
        borderaxespad=0.0,
    )

    # keep the common y-axis scale if it fits
    comparison_ylim_um = (0.0, 10.0)
    displayed_max = max(
        np.max(wall),
        np.max(bud_extent + bud_slosh + bud_shift + bud_tail),
        np.max(obs_extent + obs_slosh),
    )
    if displayed_max <= comparison_ylim_um[1]:
        ax.set_ylim(comparison_ylim_um)

    fig.tight_layout()
    return fig


#-------------------------------------------------------------------------------
# Sweep plotters — pure array renderers
#-------------------------------------------------------------------------------


def plot_f_vs_t(rows: list[tuple[int, float, float | None, float | None]]) -> Figure:
    """Compare GPE and linear endpoint losses across durations and atom numbers.

    Args:
        rows: (N, T, F_gpe, F_linear) tuples; T is in seconds, fidelities are dimensionless,
            and None denotes a missing result.

    Returns:
        Figure of 1 - F versus T on a logarithmic vertical scale, with two series per N.
    """
    by_n: dict[int, list[tuple[float, float | None, float | None]]] = {}
    for N, T, f_gpe, f_lin in rows:
        by_n.setdefault(N, []).append((T, f_gpe, f_lin))
    for series in by_n.values():
        series.sort()

    fig, ax = plt.subplots(figsize=(8, 4.5))
    cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    colors = {N: cycle[i % len(cycle)] for i, N in enumerate(sorted(by_n))}
    for N in sorted(by_n):
        ts = [t for t, fg, fl in by_n[N]]
        gpe_y = [(1 - fg) if fg is not None else np.nan for t, fg, fl in by_n[N]]
        lin_y = [(1 - fl) if fl is not None else np.nan for t, fg, fl in by_n[N]]
        ax.plot(ts, gpe_y, "o-",  color=colors[N], label=f"GPE, N = {N:g}", linewidth=1.4)
        ax.plot(ts, lin_y, "s--", color=colors[N], label=f"linear, N = {N:g}", linewidth=1.0, alpha=0.7)

    ax.set_yscale("log")
    ax.set_xlabel("transport time T [s]")
    ax.set_ylabel(r"$1 - F_{3D}$")
    ax.set_title("Fidelity vs transport time across the (N, T) sweep")
    ax.legend(fontsize="small")
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    return fig


def plot_sigma_overlay(t_ms: np.ndarray, sig_gpe: np.ndarray, sig_lin: np.ndarray,
                       *, axis_index: int = 0, xticks: int = 9) -> Figure:
    """Compare GPE and linear cloud sizes and width variations along one paired axis.

    Args:
        t_ms: Shared recorded times [ms], shape (n_times,).
        sig_gpe: GPE RMS widths along the selected cloud principal axis [m], (n_times,).
        sig_lin: Linear RMS widths along the same trap-paired axis [m], (n_times,).
        axis_index: Axis label: 0 axial, 1 trans-1, or 2 trans-2; inputs are already selected.
        xticks: Number of evenly spaced time ticks.

    Returns:
        Figure in micrometres with a broken vertical axis. Both panels use equal vertical
        spans but different offsets to show the width variations on the same scale.
    """
    name = ("axial", "trans-1", "trans-2")[axis_index]
    sig_gpe_um = np.asarray(sig_gpe) * 1e6
    sig_lin_um = np.asarray(sig_lin) * 1e6
    fig, (ax_gpe, ax_lin) = plt.subplots(
        2,
        1,
        sharex=True,
        figsize=(8, 4),
        gridspec_kw={"hspace": 0.05},
    )

    for ax in (ax_gpe, ax_lin):
        ax.plot(
            t_ms,
            sig_gpe_um,
            label="GPE",
            linewidth=0.5,
            color="C0",
        )
        ax.plot(
            t_ms,
            sig_lin_um,
            label="linear",
            linewidth=0.5,
            color="C1",
        )
        ax.set_xticks(np.linspace(t_ms[0], t_ms[-1], xticks))
        ax.yaxis.set_major_locator(MultipleLocator(0.05))
        ax.grid(alpha=0.3)

    # use the common range for the both gpe and linear cases
    gpe_min, gpe_max = np.min(sig_gpe_um), np.max(sig_gpe_um)
    lin_min, lin_max = np.min(sig_lin_um), np.max(sig_lin_um)

    margin = 0.02
    common_span = max(gpe_max - gpe_min, lin_max - lin_min) + 2 * margin

    gpe_lower = gpe_min - margin
    lin_lower = lin_min - margin

    ax_gpe.set_ylim(gpe_lower, gpe_lower + common_span)
    ax_lin.set_ylim(lin_lower, lin_lower + common_span)

    ax_gpe.spines["bottom"].set_visible(False)
    ax_lin.spines["top"].set_visible(False)
    ax_gpe.tick_params(bottom=False, labelbottom=False)
    ax_lin.tick_params(top=False)

    ax_gpe.set_title(f"{name.capitalize()} cloud width: GPE vs linear")
    ax_gpe.legend(fontsize="small")
    ax_lin.set_xlabel("time [ms]")
    fig.supylabel(rf"$\sigma_{{{axis_index + 1}}}(t)$ [$\mu$m]")

    # Diagonal marks indicate the omitted vertical interval.
    d = 0.008
    kwargs = {
        "color": "k",
        "clip_on": False,
        "linewidth": 0.8,
    }
    ax_gpe.plot((-d, d), (-d, d), transform=ax_gpe.transAxes, **kwargs)
    ax_gpe.plot((1 - d, 1 + d), (-d, d), transform=ax_gpe.transAxes, **kwargs)
    ax_lin.plot((-d, d), (1 - d, 1 + d), transform=ax_lin.transAxes, **kwargs)
    ax_lin.plot((1 - d, 1 + d), (1 - d, 1 + d), transform=ax_lin.transAxes, **kwargs)

    fig.subplots_adjust(
        left=0.12,
        right=0.98,
        bottom=0.1,
        top=0.94,
        hspace=0.05,
    )
    return fig


#-------------------------------------------------------------------------------
# Helpers: time axis, deserialization, derived views
#-------------------------------------------------------------------------------


def _time_axis_ms(T_total_s: float, *, n_blocks: int) -> np.ndarray:
    return np.arange(n_blocks) * (float(T_total_s) / (n_blocks - 1)) * 1e3


def _load_eval_data(json_path: str, npz_path: str) -> dict:
    """Read one eval point back: json scalars merged with the npz series arrays."""
    with open(json_path) as f:
        data = json.load(f)
    with np.load(npz_path) as npz:
        data.update({k: npz[k] for k in npz.files})
    return data


def _result_from_dict(data: dict) -> EvalResult:
    return EvalResult(
        F_3D                = float(data["F_3D"]),
        delta               = float(data["delta"]),
        com_traj_lab_mm     = np.asarray(data["com_traj_lab_mm"]),
        sigma_traj_m        = np.asarray(data["sigma_traj_m"]),
        cov_traj_m2         = np.asarray(data["cov_traj_m2"]),
        boundary_mass_traj  = np.asarray(data["boundary_mass_traj"]),
        k_tail_mass_traj    = np.asarray(data["k_tail_mass_traj"]),
        k_shell_mass_traj   = np.asarray(data["k_shell_mass_traj"]),
        gs_residual_initial = data["ground_state"]["residual_initial"],
        gs_residual_target  = data["ground_state"]["residual_target"],
    )


def _cloud_axes_paired(cov_traj_m2: np.ndarray, trap_eigvecs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Diagonalize each block's cov, pair cloud eigenvectors to trap eigenvectors by max overlap.

    Returns (cloud_sigma_m, misalign_deg), both (T, 3), labelled by the trap's k-th axis.
    """
    cov_traj_m2 = np.asarray(cov_traj_m2)
    trap_eigvecs = np.asarray(trap_eigvecs)
    T_blocks = cov_traj_m2.shape[0]
    cloud_sigma = np.zeros((T_blocks, 3))
    misalign_deg = np.zeros((T_blocks, 3))
    for t in range(T_blocks):
        eigvals, vecs = np.linalg.eigh(cov_traj_m2[t])
        overlap = np.abs(trap_eigvecs[t].T @ vecs)
        best_perm = max(permutations(range(3)), key=lambda p: sum(overlap[k, p[k]] for k in range(3)))
        for k in range(3):
            j = best_perm[k]
            cloud_sigma[t, k] = np.sqrt(max(eigvals[j], 0.0))
            misalign_deg[t, k] = np.degrees(np.arccos(np.clip(overlap[k, j], 0.0, 1.0)))
    return cloud_sigma, misalign_deg


# fmt: off
class _BreathingRatios(NamedTuple):
    cloud_axial  : np.ndarray  # (T,) cloud axial principal width [m]
    cloud_trans  : np.ndarray  # (T,) cloud transverse geometric-mean width [m]
    tf_axial     : np.ndarray  # (T,) instantaneous TF axial width [m]
    tf_trans     : np.ndarray  # (T,) instantaneous TF transverse geometric-mean width [m]
    b_axial      : np.ndarray  # (T,) unanchored b_k = sigma/sigma_TF, axial
    b_trans      : np.ndarray  # (T,) unanchored b_k, transverse
    b_tilde_axial: np.ndarray  # (T,) initial-anchored relative response, axial
    b_tilde_trans: np.ndarray  # (T,) initial-anchored relative response, transverse
# fmt: on


def _breathing_ratios(cloud_sigma_traj_m: np.ndarray, sigma_tf_traj_m: np.ndarray) -> _BreathingRatios:
    """Axial/transverse breathing channels shared by the breathing and collective-observable figures.

    Transverse uses the geometric mean of the two near-degenerate widths, invariant to their label order.
    """
    cloud_axial, cloud_t1, cloud_t2 = np.asarray(cloud_sigma_traj_m).T
    tf_axial, tf_t1, tf_t2 = np.asarray(sigma_tf_traj_m).T
    cloud_trans = np.sqrt(cloud_t1 * cloud_t2)
    tf_trans = np.sqrt(tf_t1 * tf_t2)
    return _BreathingRatios(
        cloud_axial=cloud_axial, cloud_trans=cloud_trans, tf_axial=tf_axial, tf_trans=tf_trans,
        b_axial=cloud_axial / tf_axial, b_trans=cloud_trans / tf_trans,
        b_tilde_axial=(cloud_axial / cloud_axial[0]) / (tf_axial / tf_axial[0]),
        b_tilde_trans=(cloud_trans / cloud_trans[0]) / (tf_trans / tf_trans[0]),
    )


def _ellipticity_pct(cloud_sigma_traj_m: np.ndarray) -> np.ndarray:
    """Unsigned transverse ellipticity [%]: |σ_⊥1 − σ_⊥2| / (σ_⊥1 + σ_⊥2) × 100."""
    sig = np.asarray(cloud_sigma_traj_m)                           # (T, 3): axial, trans-1, trans-2
    s1, s2 = sig[:, 1], sig[:, 2]
    return np.abs(s1 - s2) / (s1 + s2) * 100.0


def _save_fig(fig: Figure, path: str, reg: str | None = None) -> None:
    if reg is not None:
        fig.text(0.995, 0.005, f"reg={reg}", ha="right", va="bottom", fontsize=7, color="0.6")
    fig.savefig(path, dpi=150)
    plt.close(fig)
