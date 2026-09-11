"""Rendering library for visualizing one experiment-point's schedule + derived data."""
import logging
import os

import matplotlib

matplotlib.use("Agg")  # headless backend; must be set before the pyplot import

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import MultipleLocator

from .chip import AtomChip
from .optimizer import CurrentSchedule

#-------------------------------------------------------------------------------
# Figure-group dispatchers (cli calls these; the plot_* below are internal)
#-------------------------------------------------------------------------------


def render_schedule_figs(chip: AtomChip, sched: CurrentSchedule, fig_dir: str,
                         *, step_slice: slice = slice(None), step_suffix: str = "",
                         reg: str | None = None) -> None:
    """Render the whole schedule-level fig group (currents, position, omega, …) into `fig_dir`."""
    n_full = sched.r_mins_mm.shape[0]
    s = sched.slice(step_slice)
    abs_steps = np.arange(n_full)[step_slice] if step_slice != slice(None) else None
    xs = shifting_xs(chip)
    os.makedirs(fig_dir, exist_ok=True)

    def save(fig: Figure, name: str) -> None:
        _save_fig(fig, os.path.join(fig_dir, f"{name}{step_suffix}.png"), reg)

    save(plot_currents(s.I_logical_A, [lw.id for lw in chip.wire_layout["shifting"]],
                       "Shifting wire currents", steps=abs_steps), "currents_shifting")
    save(plot_currents(s.I_logical_A, [lw.id for lw in chip.wire_layout["guiding"]],
                       "Guiding wire currents", steps=abs_steps), "currents_guiding")
    save(plot_position_x(s.r_mins_mm, s.target_rs_mm, xs, steps=abs_steps), "position_x")
    save(plot_position_yz(s.r_mins_mm, s.target_rs_mm, steps=abs_steps), "position_yz")
    save(plot_xy_top(s.r_mins_mm, s.target_rs_mm, xs), "xy_top")
    save(plot_xz_side(s.r_mins_mm, s.target_rs_mm, xs), "xz_side")
    save(plot_omega(s.omegas_hz, steps=abs_steps), "omega")
    save(plot_a_ho_axes(s.a_ho_axes_m, steps=abs_steps), "a_ho")
    save(plot_u0(s.U0s_J, steps=abs_steps), "u0")
    save(plot_larmor(s.larmor_freqs_hz, s.omegas_hz, steps=abs_steps), "larmor")
    logging.getLogger("chip.plot").info("Schedule plots written to %s", fig_dir)


def render_derived_figs(chip: AtomChip, sched: CurrentSchedule, n_atoms: int, T_total_s: float,
                        fig_dir: str,
                        *, step_slice: slice = slice(None), step_suffix: str = "",
                        reg: str | None = None) -> None:
    """Render the per-N derived fig group (`tf_radii`, `tf_extents`, `mu`) into `fig_dir`."""
    n_full = sched.omegas_hz.shape[0]
    sliced = sched.slice(step_slice)
    abs_steps = np.arange(n_full)[step_slice] if step_slice != slice(None) else None
    t_ms = np.linspace(0.0, T_total_s * 1e3, n_full)[step_slice]
    mu, tf_radii = sliced.tf_quantities(chip.atom, n_atoms)
    os.makedirs(fig_dir, exist_ok=True)

    def save(fig: Figure, name: str) -> None:
        _save_fig(fig, os.path.join(fig_dir, f"{name}{step_suffix}.png"), reg)

    save(plot_tf_radii(np.asarray(tf_radii), n_atoms, steps=abs_steps), "tf_radii")
    save(plot_tf_extents(np.asarray(tf_radii), sliced.eigvecs, n_atoms, t_ms=t_ms), "tf_extents")
    save(plot_mu(np.asarray(mu), n_atoms, steps=abs_steps), "mu")
    logging.getLogger("chip.plot").info("Derived plots written to %s", fig_dir)


#-------------------------------------------------------------------------------
# Schedule plots (consume CurrentSchedule fields)
#-------------------------------------------------------------------------------


def plot_currents(I_logical_A: np.ndarray, indices: list[int], title: str,
                  *, steps: np.ndarray | None = None) -> Figure:
    if steps is None:
        steps = np.arange(I_logical_A.shape[0])
    fig, ax = plt.subplots(figsize=(8, 4))
    for pos, i in enumerate(indices):
        ax.plot(steps, I_logical_A[:, i], label=f"wire {pos}", linewidth=1.2)
    ax.set_xlabel("step")
    ax.set_ylabel("current [A]")
    ax.set_title(title)
    ax.legend(fontsize="small", ncol=2)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def plot_position_x(r_mins_mm: np.ndarray, target_rs_mm: np.ndarray, shifting_xs_mm: np.ndarray,
                    *, steps: np.ndarray | None = None) -> Figure:
    if steps is None:
        steps = np.arange(r_mins_mm.shape[0])
    x_lo, x_hi = float(r_mins_mm[:, 0].min()), float(r_mins_mm[:, 0].max())
    visible = shifting_xs_mm[(shifting_xs_mm >= x_lo) & (shifting_xs_mm <= x_hi)]
    fig, ax = plt.subplots(figsize=(9, 3.5))
    _draw_wire_markers(ax, visible, axis="x")
    ax.plot(steps, r_mins_mm[:, 0], label=r"$r_\mathrm{min}$ x", linewidth=1.2)
    ax.plot(steps, target_rs_mm[:, 0], "--", label="target x", linewidth=1.0, alpha=0.8)
    ax.set_xlabel("step")
    ax.set_ylabel("x [mm]")
    ax.set_title("Trap x position vs target")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def plot_position_yz(r_mins_mm: np.ndarray, target_rs_mm: np.ndarray,
                     *, steps: np.ndarray | None = None,
                     comparison_ylims_mm: tuple[tuple[float, float], tuple[float, float]] = (
                         (-0.022, 0.001), (0.327, 0.340)),
                     ytick_steps_mm: tuple[float, float] = (0.005, 0.002)) -> Figure:
    if steps is None:
        steps = np.arange(r_mins_mm.shape[0])
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    for ax, i, label, ylim, tick_step in zip(
            axes, (1, 2), "yz", comparison_ylims_mm, ytick_steps_mm, strict=True):
        ax.plot(steps, r_mins_mm[:, i], label=rf"$r_\mathrm{{min}}$ {label}", linewidth=1.2)
        ax.plot(steps, target_rs_mm[:, i], "--", label=f"target {label}", linewidth=1.0, alpha=0.8)
        ax.set_xlabel("step")
        ax.set_ylabel(f"{label} [mm]")
        ax.legend()
        ax.grid(alpha=0.3)
        ax.yaxis.set_major_locator(MultipleLocator(tick_step))
        axis_data = np.concatenate((r_mins_mm[:, i], target_rs_mm[:, i]))
        if axis_data.min() >= ylim[0] and axis_data.max() <= ylim[1]:
            ax.set_ylim(ylim)
    fig.suptitle("Transverse trap positions vs target")
    fig.tight_layout()
    return fig


def plot_xy_top(r_mins_mm: np.ndarray, target_rs_mm: np.ndarray, shifting_xs_mm: np.ndarray) -> Figure:
    x_lo, x_hi = float(r_mins_mm[:, 0].min()), float(r_mins_mm[:, 0].max())
    visible = shifting_xs_mm[(shifting_xs_mm >= x_lo) & (shifting_xs_mm <= x_hi)]
    fig, ax = plt.subplots(figsize=(9, 3.5))
    _draw_wire_markers(ax, visible, axis="y")
    ax.plot(target_rs_mm[:, 0], target_rs_mm[:, 1], "--", label="target", linewidth=1.0, alpha=0.8)
    ax.plot(r_mins_mm[:, 0], r_mins_mm[:, 1], label=r"$r_\mathrm{min}$", linewidth=1.2)
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    ax.set_title("Trap trajectory (top-down x-y)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def plot_xz_side(r_mins_mm: np.ndarray, target_rs_mm: np.ndarray, shifting_xs_mm: np.ndarray) -> Figure:
    x_lo, x_hi = float(r_mins_mm[:, 0].min()), float(r_mins_mm[:, 0].max())
    visible = shifting_xs_mm[(shifting_xs_mm >= x_lo) & (shifting_xs_mm <= x_hi)]
    fig, ax = plt.subplots(figsize=(9, 3.5))
    _draw_wire_markers(ax, visible, axis="y")
    ax.plot(target_rs_mm[:, 0], target_rs_mm[:, 2], "--", label="target", linewidth=1.0, alpha=0.8)
    ax.plot(r_mins_mm[:, 0], r_mins_mm[:, 2], label=r"$r_\mathrm{min}$", linewidth=1.2)
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("z [mm]")
    ax.set_title("Trap trajectory (side x-z)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def plot_omega(omegas_hz: np.ndarray, *, steps: np.ndarray | None = None) -> Figure:
    if steps is None:
        steps = np.arange(omegas_hz.shape[0])
    fig, ax = plt.subplots(figsize=(8, 3.5))
    for i, label in enumerate((r"$\omega_1$", r"$\omega_2$", r"$\omega_3$")):
        ax.plot(steps, omegas_hz[:, i], label=label, linewidth=1.2)
    ax.set_xlabel("step")
    ax.set_ylabel("frequency [Hz]")
    ax.set_title("Trap frequencies")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def plot_a_ho_axes(a_ho_axes_m: np.ndarray, *, steps: np.ndarray | None = None) -> Figure:
    if steps is None:
        steps = np.arange(a_ho_axes_m.shape[0])
    fig, ax = plt.subplots(figsize=(8, 3.5))
    for i, label in enumerate(("axial", "trans-1", "trans-2")):
        ax.plot(steps, a_ho_axes_m[:, i] * 1e6, label=label, linewidth=1.2)
    ax.set_xlabel("step")
    ax.set_ylabel(r"$a_\mathrm{ho}$ [μm]")
    ax.set_title(r"Per-axis harmonic-oscillator length $a_\mathrm{ho}$")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def plot_u0(U0s_J: np.ndarray, *, steps: np.ndarray | None = None) -> Figure:
    if steps is None:
        steps = np.arange(U0s_J.shape[0])
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.plot(steps, U0s_J, linewidth=1.2)
    ax.set_xlabel("step")
    ax.set_ylabel(r"$U_0$ [J]")
    ax.set_title(r"Trap-bottom potential energy $U_0$")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def plot_larmor(larmor_freqs_hz: np.ndarray, omegas_hz: np.ndarray,
                *, steps: np.ndarray | None = None) -> Figure:
    if steps is None:
        steps = np.arange(larmor_freqs_hz.shape[0])
    omega_max_hz = np.max(omegas_hz, axis=1)
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.plot(steps, larmor_freqs_hz * 1e-6, label=r"$\omega_L \, / \, 2\pi$", linewidth=1.2)
    ax.plot(steps, omega_max_hz * 1e-6, "--", label=r"max $\omega_\mathrm{trap} \, / \, 2\pi$", linewidth=1.0, alpha=0.7)
    ax.set_xlabel("step")
    ax.set_ylabel("frequency [MHz]")
    ax.set_title("Larmor frequency vs trap frequency (spin-flip safety margin)")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    return fig


#-------------------------------------------------------------------------------
# Thomas-Fermi plots (consume CurrentSchedule.tf_quantities output)
#-------------------------------------------------------------------------------


def plot_tf_radii(tf_radii_m: np.ndarray, n_atoms: int,
                  *, steps: np.ndarray | None = None) -> Figure:
    if steps is None:
        steps = np.arange(tf_radii_m.shape[0])
    fig, ax = plt.subplots(figsize=(8, 3.5))
    for i, label in enumerate((r"$R_1$", r"$R_2$", r"$R_3$")):
        ax.plot(steps, tf_radii_m[:, i] * 1e6, label=label, linewidth=1.2)
    ax.set_xlabel("step")
    ax.set_ylabel(r"$R_\mathrm{TF}$ [μm]")
    ax.set_title(rf"Thomas-Fermi radii (N={n_atoms})")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def plot_tf_extents(tf_radii_m: np.ndarray, trap_eigvecs: np.ndarray, n_atoms: int,
                    *, t_ms: np.ndarray,
                    comparison_ylim_um: tuple[float, float] = (0.0, 7.5)) -> Figure:
    """Project the principal-axis TF radii onto the laboratory axes."""
    radii = np.asarray(tf_radii_m)
    eigvecs = np.asarray(trap_eigvecs)
    extents_m = np.sqrt(np.einsum("tik,tk->ti", eigvecs ** 2, radii ** 2))

    fig, ax = plt.subplots(figsize=(8, 3.5))
    for i, label in enumerate((r"$E_x^{\rm TF}$", r"$E_y^{\rm TF}$", r"$E_z^{\rm TF}$")):
        ax.plot(t_ms, extents_m[:, i] * 1e6, color=f"C{i}", label=label, linewidth=1.2)
    ax.set_xlabel("time [ms]")
    ax.set_ylabel(r"$E_i^{\rm TF}$ [$\mu$m]")
    ax.set_title(rf"Laboratory-Axis Thomas–Fermi Extents ($N={n_atoms:,}$)")
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5))
    ax.grid(alpha=0.3)
    if np.max(extents_m) * 1e6 <= comparison_ylim_um[1]:
        ax.set_ylim(comparison_ylim_um)
    fig.tight_layout()
    return fig


def plot_mu(mu_J: np.ndarray, n_atoms: int, *, steps: np.ndarray | None = None) -> Figure:
    if steps is None:
        steps = np.arange(mu_J.shape[0])
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.plot(steps, mu_J, linewidth=1.2)
    ax.set_xlabel("step")
    ax.set_ylabel(r"$\mu$ [J]")
    ax.set_title(rf"Chemical potential $\mu$ (N={n_atoms})")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


#-------------------------------------------------------------------------------
# Chip-data helpers (data prep for the plotters)
#-------------------------------------------------------------------------------


def shifting_xs(chip: AtomChip) -> np.ndarray:
    """Unique x-coordinates of the shifting wires (sorted)."""
    xs = [float(lw.segments.starts_mm[0, 0]) for lw in chip.wire_layout["shifting"]]
    return np.asarray(sorted(set(xs)), dtype=np.float64)


def _save_fig(fig: Figure, path: str, reg: str | None = None) -> None:
    if reg is not None:
        fig.text(0.995, 0.005, f"reg={reg}", ha="right", va="bottom", fontsize=7, color="0.6")
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _draw_wire_markers(ax: Axes, xs: np.ndarray, axis: str = "x") -> None:
    for x in xs:
        if axis == "x":
            ax.axhline(x, linestyle="--", color="gray", linewidth=0.5, alpha=0.5)
        else:
            ax.axvline(x, linestyle="--", color="gray", linewidth=0.5, alpha=0.5)
