"""Render figures from saved experiment data within the requested directory.

Selecting any part of scan/ regenerates the shared scan figures for all saved atom numbers.
"""
import argparse
import glob
import os
import sys

import atom_chip as ac

from . import common
from .config import Config


def register(sub: argparse._SubParsersAction) -> None:
    """Register the plotting directory and optional control-step range."""
    p = sub.add_parser("plot", help="Render figures from saved data")
    p.add_argument("path", nargs="?", default=None,
                   help="Directory to render (default: current dir); renders every figure at or under it")
    p.add_argument("--step-range", type=str, default="",
                   help="Restrict step-indexed plots to [a:b] (e.g. ':60', '2400:', '500:1500'). "
                        "Applies to schedule and derived figs; eval figs unaffected. "
                        "Output filename gets a '_steps_<a>-<b>' suffix")
    p.set_defaults(func=run)


def run(args: argparse.Namespace) -> int:
    """Render scan, schedule, and evaluation figures selected by the starting directory.

    At a sweep root, also regenerate the fidelity CSV and printed summary.
    """
    start = os.path.abspath(args.path or args.work_dir)
    if not os.path.isdir(start):
        print(f"chip plot: no such directory '{args.path}'.", file=sys.stderr)
        return 1
    try:
        step_slice = _parse_step_range(args.step_range)
    except ValueError as e:
        print(f"chip plot: {e}", file=sys.stderr)
        return 1
    step_suffix = _step_range_suffix(step_slice)

    chip_root = Config.find_root(start)
    if chip_root is None:
        print("chip plot: no chip.json at or above the given path. Run `chip init` first.", file=sys.stderr)
        return 1
    ep = Config(chip_root)
    chip = common.load_chip(ep)
    rendered = False

    # Validate all saved scan pairs before rendering their shared comparisons.
    # Scan selection includes all atom numbers, even when start is inside one N directory.
    scan_dir = ep.path("scan_dir")
    if (_overlaps(scan_dir, start)
            and (os.path.exists(ep.path("harmonic_npz")) or os.path.exists(ep.path("harmonic_json")))):
        try:
            harmonic = ac.surrogate.load_harmonic_pair(ep.path("harmonic_npz"), ep.path("harmonic_json"))
            scans_by_n, summaries = {}, {}
            for n_dir in sorted(glob.glob(os.path.join(scan_dir, "N-*"))):
                n = Config.n_from_dirname(os.path.basename(n_dir))
                if n is None:
                    continue
                npz, meta = ep.path("scan_n_npz", n=n), ep.path("scan_n_json", n=n)
                # Ignore directories with no scan data; a lone NPZ or JSON must still be validated.
                if not (os.path.exists(npz) or os.path.exists(meta)):
                    continue
                scans_by_n[n], summaries[n] = ac.surrogate.load_n_scan_pair(npz, meta, harmonic, chip)
            if not summaries:
                raise ValueError("scan is incomplete: no N-*/scan.npz + scan.json pairs found")
        except ValueError as e:
            print(f"chip plot: {e}", file=sys.stderr)
        else:
            fig_root = ep.figures_dir(scan_dir)
            summary = {
                "n_atoms": sorted(summaries),
                "T_s": next(iter(summaries.values()))["T_s"],
                "comparisons_by_n": {str(n): summaries[n]["comparisons"] for n in summaries},
            }
            ac.surrogate.render_f_com_summary_fig(summary, fig_root)
            for n, nscan in scans_by_n.items():
                ac.surrogate.render_f_com_loss_fig(nscan, ep.figures_dir(ep.path("scan_n_dir", n=n)),
                                                   comparisons=summaries[n]["comparisons"])
            ac.surrogate.render_cliff_fig(harmonic, scans_by_n, fig_root)
            rendered = True

    # Render planned schedules at or below the selected directory.
    for reg in ep.iter_plan_regs():
        for t in ep.iter_plan_ts(reg):
            plan_dir = ep.path("plan_dir", reg=reg, t=Config.t_label(t))
            if _under(plan_dir, start):
                try:
                    sched = common.load_schedule(ep, reg, t)
                except ValueError as e:
                    print(f"chip plot: {e}", file=sys.stderr)
                    continue
                ac.schedule.render_schedule_figs(chip, sched, ep.figures_dir(plan_dir),
                                                 step_slice=step_slice, step_suffix=step_suffix,
                                                 reg=f"{reg} T={t:g}")
                rendered = True

    # Render derived schedule quantities, mode diagnostics, and GPE/linear comparisons.
    for reg in ep.iter_sweep_regs():
        sweep_dir = ep.path("sweep_dir", reg=reg)
        if not _overlaps(sweep_dir, start):
            continue
        for n, _n_dir in ep.iter_n_dirs(reg):
            for t, t_dir in ep.iter_t_dirs(reg, n):
                try:
                    sched = common.load_schedule(ep, reg, t)
                except ValueError as e:
                    print(f"chip plot: {e}", file=sys.stderr)
                    continue
                # Duration-level figures are excluded when only a gpe/linear subdirectory is selected.
                if _under(t_dir, start):
                    ac.schedule.render_derived_figs(chip, sched, n, t, ep.figures_dir(t_dir),
                                                    step_slice=step_slice, step_suffix=step_suffix,
                                                    reg=f"{reg} T={t:g}")
                    rendered = True
                for mode_dir in Config.iter_mode_dirs_in(t_dir):
                    if not _under(mode_dir, start):
                        continue
                    mode = os.path.basename(mode_dir)
                    try:
                        diag = ac.tdgpe.load_diagnostics(ep.path("eval_json", reg=reg, n=n, t=t, mode=mode),
                                                         ep.path("eval_npz", reg=reg, n=n, t=t, mode=mode),
                                                         chip=chip, sched=sched, n_atoms=n)
                    except (KeyError, FileNotFoundError) as e:
                        print(
                            f"Skipping {os.path.relpath(mode_dir, chip_root)} "
                            f"(incomplete eval: {e}; re-run `chip eval --force`)",
                            file=sys.stderr,
                        )
                        continue
                    ac.tdgpe.render_eval_figs(diag, ep.figures_dir(mode_dir), reg=reg)
                    rendered = True
                if _under(t_dir, start):
                    gpe = common.load_eval(ep, reg, n, t, Config.GPE_SUBDIR)
                    lin = common.load_eval(ep, reg, n, t, Config.LINEAR_SUBDIR)
                    if gpe is not None and lin is not None:
                        ac.tdgpe.render_overlay_fig(ep.figures_dir(t_dir), gpe=gpe, lin=lin, T_total_s=t,
                                                    trap_eigvecs=sched.eigvecs, reg=reg)
                        rendered = True
        # Aggregate across cases only when the entire regularization directory is selected.
        if _under(sweep_dir, start):
            rows = common.collect_rows(ep, reg)
            if rows:
                ac.tdgpe.render_sweep_figs(ep.figures_dir(sweep_dir), rows=rows, reg=reg)
                ac.tdgpe.write_fidelity_csv(ep.path("fidelity_csv", reg=reg), rows=rows)
                print(f"reg = {reg}\n{ac.tdgpe.format_fidelity_table(rows)}\n")
                # Repeat the heuristic under-relaxation warning when reporting saved fidelities.
                for N, T, eg, el in rows:
                    for run, res in (("gpe", eg), ("linear", el)):
                        worst = max((v for v in (res.gs_residual_initial, res.gs_residual_target)
                                     if v is not None), default=None) if res else None
                        if worst is not None and worst > ac.tdgpe.GS_RESIDUAL_WARN:
                            print(f"chip plot: ⚠ reg={reg} N={N} T={T:.2f} {run}: GS residual {worst:.1e} "
                                  f"> {ac.tdgpe.GS_RESIDUAL_WARN:.0e} (ground state may be under-relaxed)",
                                  file=sys.stderr)
                rendered = True

    if not rendered:
        print(f"chip plot: nothing to render under {os.path.relpath(start, chip_root)}.", file=sys.stderr)
        return 1
    return 0


def _under(path: str, start: str) -> bool:
    """Return whether path equals start or lies beneath it."""
    path, start = os.path.abspath(path), os.path.abspath(start)
    return path == start or path.startswith(start + os.sep)


def _overlaps(path: str, start: str) -> bool:
    """Return whether either path equals or contains the other."""
    return _under(path, start) or _under(start, path)


def _parse_step_range(s: str) -> slice:
    """Parse a Python-style start:stop slice; an empty string selects all steps."""
    if not s:
        return slice(None)
    if ":" not in s:
        raise ValueError(f"--step-range must contain ':' (got {s!r})")
    a_str, b_str = s.split(":", 1)
    a = int(a_str) if a_str.strip() else None
    b = int(b_str) if b_str.strip() else None
    return slice(a, b)


def _step_range_suffix(s: slice) -> str:
    """Label step-restricted output filenames; return no suffix for an unrestricted slice."""
    if s == slice(None):
        return ""
    a = s.start if s.start is not None else 0
    b = s.stop if s.stop is not None else "end"
    return f"_steps_{a}-{b}"
