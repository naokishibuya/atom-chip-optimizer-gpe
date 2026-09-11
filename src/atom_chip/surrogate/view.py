"""Plot predicted delivery quality and COM excursions to compare transport schedules."""
import math
import os
from decimal import Decimal

import matplotlib

matplotlib.use("Agg")  # headless backend; must be set before the pyplot import

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

from .scan import HarmonicGrid, NScan, _reg_value


def render_cliff_fig(harmonic: HarmonicGrid, scans_by_n: dict[int, NScan], fig_dir: str) -> None:
    """Save the COM-excursion comparison as cliff.png.

    Args:
        harmonic: Shared maximum COM excursions over regularizations and durations.
        scans_by_n: Atom number to NScan mapping, supplying each initial TF RMS width.
            Scans must use the same regularizations and durations as harmonic.
        fig_dir: Output directory, created if needed.
    """
    os.makedirs(fig_dir, exist_ok=True)
    _save_fig(_plot_cliff(harmonic, scans_by_n), os.path.join(fig_dir, "cliff.png"))


def render_f_com_loss_fig(nscan: NScan, fig_dir: str, *, comparisons: list[dict]) -> None:
    """Save the endpoint-loss comparison for one atom number as f_com_loss_vs_t.png.

    Args:
        nscan: F_COM scores for one atom number over regularizations and durations.
        fig_dir: Output directory, created if needed.
        comparisons: CellComparison dictionaries for this scan, with one entry per duration.
    """
    os.makedirs(fig_dir, exist_ok=True)
    _save_fig(_plot_f_com_loss(nscan, comparisons), os.path.join(fig_dir, "f_com_loss_vs_t.png"))


def render_f_com_summary_fig(summary: dict, fig_dir: str) -> None:
    """Save the schedule-selection summary as f_com_summary_by_n_t.png.

    Args:
        summary: Combined scan summary with n_atoms, T_s [s], and comparisons_by_n.
            Comparisons are keyed by str(N), with entries in T_s order.
        fig_dir: Output directory, created if needed.
    """
    os.makedirs(fig_dir, exist_ok=True)
    _save_fig(_plot_f_com_summary(summary), os.path.join(fig_dir, "f_com_summary_by_n_t.png"))


def _plot_cliff(harmonic: HarmonicGrid, scans_by_n: dict[int, NScan]) -> Figure:
    """Compare maximum axial COM excursions with the initial cloud width across schedules.

    Args:
        harmonic: Shared results containing max_abs_com_m, shape (n_reg, n_T, 3) [m],
            in laboratory x/y/z order, and transport durations T_grid_s [s].
        scans_by_n: Atom number to NScan mapping. Each supplies sigma_x0_m, the initial
            laboratory-x TF RMS width [m], and shares the harmonic parameter grid.

    Returns:
        Figure with one panel per N and one curve per regularization, showing
        max-node |COM_x - r_min,x| / sigma_x0_m versus T. Values above one are shaded.
        The plot uses the excursion ratios directly, not the stored cliff_T_s values.
    """
    n_list = sorted(scans_by_n)
    exc_m = harmonic.max_abs_com_m[..., 0]                                    # (n_reg, n_T) lab-x
    lambdas = np.asarray(harmonic.reg_values, dtype=float)
    reg_colors = _reg_color_map(harmonic.reg_labels)
    fig, axes = plt.subplots(len(n_list), 1, figsize=(9.5, 2.6 * len(n_list)),
                             sharex=True, squeeze=False, layout="constrained")
    for i, n in enumerate(n_list):
        ax_c = axes[i, 0]
        ratio = exc_m / scans_by_n[n].sigma_x0_m                             # (n_reg, n_T) dimensionless
        for r, label in enumerate(harmonic.reg_labels):
            ax_c.semilogy(harmonic.T_grid_s, ratio[r], "o-", color=reg_colors[label],
                          linewidth=1.4, markersize=4, label=_fmt_lambda(float(lambdas[r])))
        ax_c.set_ylabel(rf"$C_{{\lambda,N}}$   (N={_fmt_n(n)})")
        lo, hi = ax_c.get_ylim()
        ax_c.axhline(1.0, color="0.3", linestyle="--", linewidth=1.0, zorder=1)
        if hi > 1.0:  # Shade excursions exceeding the initial RMS width.
            ax_c.axhspan(1.0, hi, facecolor="tab:red", alpha=0.06, zorder=0)
        ax_c.grid(alpha=0.3, which="both")
        ax_c.set_ylim(lo, hi)
    if axes[0, 0].get_ylim()[1] > 1.0:  # Label the shaded region only in the top panel.
        # Horizontal position is an axes fraction; vertical position is the excursion ratio.
        axes[0, 0].text(0.98, 1.1, "excursion exceeds initial cloud width", fontsize=8, color="darkred",
                        ha="right", va="bottom", transform=axes[0, 0].get_yaxis_transform())
    axes[-1, 0].set_xlabel("transport time T [s]")
    fig.suptitle("Normalized center-of-mass excursion", fontsize=13)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncols=len(labels),
               fontsize="small", title=r"$\lambda$")
    return fig


def _plot_f_com_loss(nscan: NScan, comparisons: list[dict]) -> Figure:
    """Compare estimated endpoint losses and mark the best schedules for one atom number.

    Args:
        nscan: F_COM scores, shape (n_reg, n_T), and transport durations T_grid_s [s].
        comparisons: CellComparison dictionaries for this scan, including T_s,
            best_F_COM, best_reg, and tied_regs for each duration.

    Returns:
        Figure of 1 - F_COM versus T on a logarithmic vertical scale, one curve per reg.
        Stars mark unique winners; open circles mark exact ties. Exact zero losses are
        omitted and counted in the title, not replaced by a positive plotting floor.
    """
    fig, ax = plt.subplots(figsize=(8, 4.5))
    reg_colors = _reg_color_map(nscan.reg_labels)
    n_exact = 0
    for r, label in enumerate(nscan.reg_labels):
        # NaN leaves a gap at zero loss, which cannot appear on a logarithmic axis.
        f_com = nscan.F_COM[r]
        loss = np.where(f_com < 1.0, 1.0 - f_com, np.nan)
        n_exact += int(np.sum(f_com >= 1.0))
        ax.semilogy(nscan.T_grid_s, loss, "o-", label=f"λ = {_fmt_lambda(_reg_value(label))}",
                    color=reg_colors[label], linewidth=1.2, markersize=4)
    winners = [(c["T_s"], 1.0 - c["best_F_COM"]) for c in comparisons
               if c["best_reg"] is not None and c["best_F_COM"] < 1.0]
    ties = [(c["T_s"], 1.0 - c["best_F_COM"]) for c in comparisons
            if c["tied_regs"] and c["best_F_COM"] < 1.0]
    if winners:
        ax.plot(*zip(*winners, strict=True), "*", color="black", markersize=12, linestyle="none",
                zorder=4, label="best reg at this T")
    if ties:
        ax.plot(*zip(*ties, strict=True), "o", markerfacecolor="none", markeredgecolor="crimson",
                markersize=11, linestyle="none", zorder=4, label="exact tie")
    title = f"Surrogate COM-overlap loss vs T (estimated, not GPE fidelity; N={_fmt_n(nscan.n_atoms)})"
    if n_exact:
        title += f"\n{n_exact} point(s) at exact $F_{{\\rm COM}}=1$ (zero loss) omitted from the log scale"
    ax.set_xlabel("transport time T [s]")
    ax.set_ylabel(r"surrogate loss $1-F_{\rm COM}$")
    ax.set_title(title)
    ax.legend(fontsize="small")
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    return fig


def _plot_f_com_summary(summary: dict) -> Figure:
    """Summarize the winning regularizations and their score separation across N and T.

    Args:
        summary: Mapping with n_atoms (row order), T_s [s] (column order), and
            comparisons_by_n[str(N)] containing CellComparison dictionaries in T_s order.

    Returns:
        Figure with one cell per (N, T), listing the winning λ, estimated loss 1-F_COM,
        and runner-up separation ΔF when available. Brighter cells indicate lower loss.
        Exact ties are hatched and list all winners; a lone candidate is marked 'only'.
        Exact zero loss is labeled explicitly and mapped to the brightest color.
    """
    n_list, T_list = summary["n_atoms"], summary["T_s"]
    cells = {(i, j): summary["comparisons_by_n"][str(n)][j]
             for i, n in enumerate(n_list) for j in range(len(T_list))}
    losses = {ij: 1.0 - c["best_F_COM"] for ij, c in cells.items()}
    positive = sorted(v for v in losses.values() if v > 0.0)
    cmap = plt.get_cmap("viridis_r")
    # Logarithmic colors need a nonzero range; equal positive losses use the midpoint color.
    norm = matplotlib.colors.LogNorm(vmin=positive[0], vmax=positive[-1]) \
        if positive and positive[0] < positive[-1] else None

    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    rgba = np.zeros((len(n_list), len(T_list), 4))
    for (i, j), cell in cells.items():
        loss = losses[i, j]
        # Map zero loss directly to the brightest color, outside logarithmic normalization.
        frac = 0.0 if loss == 0.0 else (norm(loss) if norm else 0.5)
        rgba[i, j] = cmap(frac)
        r, g, b, _ = rgba[i, j]
        # Choose light or dark text for contrast against the cell color.
        text_color = "black" if 0.299 * r + 0.587 * g + 0.114 * b > 0.5 else "white"
        if cell["tied_regs"]:
            ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, hatch="//",
                                   edgecolor=text_color, linewidth=0))
            rows = [("tie", ", ".join(_fmt_lambda(_reg_value(label)) for label in cell["tied_regs"]))]
        else:
            only = " (only)" if cell["runner_up_reg"] is None else ""
            rows = [("λ*", _fmt_lambda(_reg_value(cell["best_reg"])) + only)]
        rows.append(("1-F*", "0 (exact)" if loss == 0.0 else f"{loss:.1e}"))
        delta = cell["delta_F"]
        if delta is not None:
            rows.append(("ΔF", "0 (tie)" if delta == 0.0 else f"{delta:.1e}"))
        # Align labels and values at a shared anchor inside each cell.
        kw = {"va": "center", "fontsize": 8, "color": text_color, "linespacing": 1.3}
        ax.text(j - 0.28, i, "\n".join(label for label, _ in rows), ha="right", **kw)
        ax.text(j - 0.28, i, "\n".join(f" = {value}" for _, value in rows), ha="left", **kw)
    ax.imshow(rgba, aspect="auto", origin="upper", zorder=0)
    _label_nt_axes(ax, n_list, T_list)
    ax.set_title("Surrogate schedule comparison", fontsize=11)
    fig.tight_layout()
    return fig


def _reg_color_map(labels: list[str]) -> dict[str, tuple]:
    """Assign consistent regularization colors to excursion and loss curves.

    Args:
        labels: Nonempty list of 'reg=<value>' labels with positive values.

    Returns:
        Label-to-RGBA mapping, spaced by log10(λ) over the first 80% of plasma.
        The bright end is excluded for contrast on white; a single λ uses the dark end.
    """
    logs = {label: math.log10(_reg_value(label)) for label in labels}
    lo, hi = min(logs.values()), max(logs.values())
    span = (hi - lo) or 1.0
    cmap = plt.get_cmap("plasma")
    return {label: cmap(0.8 * (logs[label] - lo) / span) for label in labels}


def _label_nt_axes(ax: plt.Axes, n_list: list[int], T_list: list[float]) -> None:
    """Label summary-map rows by atom number and columns by transport duration.

    Args:
        ax: Axes to label in place.
        n_list: Atom numbers in row order.
        T_list: Transport durations [s] in column order.
    """
    # Show 1.0 rather than 1, but retain finer durations such as 0.25.
    ax.set_xticks(range(len(T_list)),
                  [f"{T:.1f}" if round(T, 1) == T else f"{T:g}" for T in T_list], fontsize=9)
    ax.set_yticks(range(len(n_list)), [_fmt_n(n) for n in n_list], fontsize=9)
    ax.set_xlabel("transport time T [s]", fontsize=10)
    ax.set_ylabel("atom number N", fontsize=10)


def _fmt_lambda(value: float) -> str:
    r"""Format positive λ with a mantissa in (0.1, 1], e.g. 5e-3 as $0.5 \times 10^{-2}$."""
    d = Decimal(repr(float(value)))  # Use the float's shortest round-trip decimal representation.
    exponent = d.adjusted()          # Initial mantissa is in [1, 10).
    if d.scaleb(-exponent) > 1:
        exponent += 1  # Move the mantissa into (0.1, 1].
    digits = format(d.scaleb(-exponent).normalize(), "f")
    if "." not in digits:  # a bare power of ten renders as 1.0
        digits += ".0"
    return rf"${digits} \times 10^{{{exponent}}}$"


def _fmt_n(n_atoms: int) -> str:
    """Format powers of ten as math text; use general numeric formatting otherwise."""
    exponent = math.log10(n_atoms)
    return rf"$10^{{{round(exponent)}}}$" if exponent == round(exponent) else f"{n_atoms:g}"


def _save_fig(fig: Figure, path: str) -> None:
    """Save a figure at 150 dpi, then close it.

    Args:
        fig: Figure to save and close.
        path: Output file path; its parent directory must already exist.
    """
    fig.savefig(path, dpi=150)
    plt.close(fig)
