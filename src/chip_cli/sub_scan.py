"""Compare saved schedules using shared Gaussian predictions and atom-number-dependent COM scores."""

import argparse
import glob
import json
import math
import os
import sys
import time
from dataclasses import asdict

import numpy as np

import atom_chip as ac

from . import common
from .config import Config


def register(sub: argparse._SubParsersAction) -> None:
    """Register the atom numbers and optional subset of planned durations to compare."""
    # fmt: off
    p = sub.add_parser("scan", help="Gaussian phase-space surrogate comparison of the saved (reg, T) plans")
    p.add_argument("--N", type=str, nargs="+", default=["1e3", "1e4", "1e5"], metavar="N",
                   help="Atom numbers: positive integers, scientific notation ok (default: 1e3 1e4 1e5)")
    p.add_argument("--T", type=str, nargs="+", default=None, metavar="T",
                   help="Subset of planned transport times [s]: values and/or START:STOP:STEP "
                        "(default: every duration planned for all regularizations)")
    p.set_defaults(func=run)
    # fmt: on


def run(args: argparse.Namespace) -> int:
    """Run the Gaussian model for each selected plan, then derive and save per-N comparisons.

    This command scores existing schedules; it does not optimize them or run the GPE.
    """
    try:
        n_atoms_list = common.parse_n_args(args.N)
    except ValueError as e:
        print(f"chip scan: {e}", file=sys.stderr)
        return 1

    cfg = common.resolve_root("scan", args.work_dir)
    if cfg is None:
        return 1
    chip = common.load_chip(cfg)

    try:
        reg_labels, T_grid = _planned_grid(cfg, args.T)
    except ValueError as e:
        print(f"chip scan: {e}", file=sys.stderr)
        return 1

    scan_dir = cfg.path("scan_dir")
    os.makedirs(scan_dir, exist_ok=True)
    logger = common.configure_file_logging("chip.scan", cfg.path("scan_log"))
    logger.info("scan: %d reg(s) × %d transport times (Gaussian phase-space surrogate)",
                len(reg_labels), len(T_grid))

    # Run once per (reg, T); the Gaussian predictions are shared across atom numbers.
    results, schedules, dts = {}, {}, set()
    for label in reg_labels:
        reg = _bare(label)
        t0 = time.time()
        for T in T_grid:
            try:
                sched = common.load_schedule(cfg, reg, float(T))
            except ValueError as e:
                print(f"chip scan: {e}", file=sys.stderr)
                return 1
            with open(cfg.path("trajectory_json", reg=reg, t=float(T))) as f:
                dts.add(float(json.load(f)["control_dt_s"]))
            schedules[(label, float(T))] = sched
            results[(label, float(T))] = ac.surrogate.run_gaussian(sched, float(T), chip.atom.mass_kg)
        logger.info("    %-12s Gaussian sweep over %d T (%.1fs)", label, len(T_grid), time.time() - t0)
    if len(dts) != 1:
        print(f"chip scan: planned trajectories mix control cadences {sorted(dts)}; re-plan consistently",
              file=sys.stderr)
        return 1

    # Save shared predictions with references to their source schedules.
    harmonic = ac.surrogate.HarmonicGrid.from_results(reg_labels, T_grid, results, schedules)
    plan_refs = {label: {repr(float(T)): os.path.relpath(
        cfg.path("trajectory_npz", reg=_bare(label), t=float(T)), cfg.chip_root)
        for T in T_grid} for label in reg_labels}
    harmonic.save(cfg.path("harmonic_npz"), cfg.path("harmonic_json"), plan_refs, dts.pop())

    # Combine shared COM predictions with equilibrium TF widths for each atom number.
    summaries = {}
    for n in n_atoms_list:
        nscan = ac.surrogate.derive_n_scan(harmonic, chip, n)
        os.makedirs(cfg.path("scan_n_dir", n=n), exist_ok=True)
        nscan.save(cfg.path("scan_n_npz", n=n))
        summaries[n] = _n_summary(nscan)
        with open(cfg.path("scan_n_json", n=n), "w") as f:
            json.dump(summaries[n], f, indent=2)

    logger.info("per-(N, T) comparison -> %s  (`chip plot` renders the figures)",
                os.path.relpath(scan_dir, cfg.chip_root))
    for n in n_atoms_list:
        logger.info("")
        logger.info("N = %g", n)
        for label in reg_labels:
            logger.info("  %-12s cliff T* %s", label, _fmt_cliff(summaries[n]["cliff_T_s"][label]))
        logger.info("  %-8s %-12s %-14s %-12s %s", "T [s]", "best reg", "F_COM", "runner-up", "ΔF")
        for cell in summaries[n]["comparisons"]:
            logger.info("  %-8g %s", cell["T_s"], _format_cell(cell))
    logger.info("")
    logger.info("estimated F_COM only (surrogate, not delivered GPE fidelity); the covariance "
                "channel in harmonic.npz is an interaction-free harmonic diagnostic. Inspect the "
                "per-(N, T) comparisons and choose the (reg, N, T) cases for `chip eval`.")
    return 0


def _planned_grid(cfg: Config, t_tokens: list[str] | None) -> tuple[list[str], np.ndarray]:
    """Find the regularization labels and duration grid in the saved plans.

    Require valid directory names, both trajectory files, and the same duration set
    for every regularization. Only then apply an optional --T subset. File contents
    are validated later when the selected schedules are loaded.
    """
    all_reg_dirs = sorted(glob.glob(cfg.path("plan_reg_dir", reg="*")))
    regs = []
    for d in all_reg_dirs:
        reg = Config.reg_from_dirname(d)
        if reg is None:
            raise ValueError(f"plan/{os.path.basename(d)} is not a valid regularization "
                             f"directory (canonical form is the positional decimal, "
                             f"e.g. reg-0.00001)")
        regs.append(reg)
    if not regs:
        raise ValueError("no planned (reg, T) trajectories under plan/; run `chip plan` first")

    t_sets = {}
    for reg in regs:
        ts = []
        for d in sorted(glob.glob(os.path.join(cfg.path("plan_reg_dir", reg=reg), "T-*"))):
            name = os.path.basename(d)
            t = Config.t_from_dirname(name)
            if t is None:
                raise ValueError(f"plan/reg-{reg}/{name} is not a valid duration directory")
            for kind in ("trajectory_npz", "trajectory_json"):
                if not os.path.exists(cfg.path(kind, reg=reg, t=t)):
                    raise ValueError(f"plan/reg-{reg}/{name} is incomplete (missing "
                                     f"{os.path.basename(cfg.path(kind, reg=reg, t=t))}); "
                                     "re-plan this case or remove the directory")
            ts.append(t)
        if not ts:
            raise ValueError(f"plan/reg-{reg} contains no duration directories; re-plan or remove it")
        t_sets[reg] = tuple(sorted(ts))
    first = t_sets[regs[0]]
    mismatched = {reg: ts for reg, ts in t_sets.items() if ts != first}
    if mismatched:
        detail = "; ".join(f"reg-{reg}: {[f'{t:g}' for t in ts]}" for reg, ts in
                           {regs[0]: first, **mismatched}.items())
        raise ValueError(f"regularizations carry inconsistent duration sets ({detail}); "
                         "complete the plan before scanning")

    T_grid = np.array(first, dtype=np.float64)
    if t_tokens is not None:
        requested = common.parse_t_args(t_tokens)
        missing = [f"{t:g}" for t in requested if not np.any(T_grid == t)]
        if missing:
            raise ValueError(f"requested T not planned: {missing} (planned: {[f'{t:g}' for t in T_grid]})")
        T_grid = requested
    # Order from strongest to weakest regularization.
    labels = [f"reg={reg}" for reg in sorted(regs, key=float, reverse=True)]
    return labels, T_grid


def _n_summary(nscan: "ac.surrogate.NScan") -> dict:
    """Build scan.json for one atom number, with per-duration rankings and cliff durations.

    Rankings include the winner, runner-up, score gap, and any ties. No single
    regularization is selected across durations.
    """
    return {
        "schema": ac.surrogate.N_SCAN_SCHEMA,
        "n_atoms": nscan.n_atoms,
        "T_s": [float(T) for T in nscan.T_grid_s],
        "regs": list(nscan.reg_labels),
        "sigma_x0_m": nscan.sigma_x0_m,
        "comparisons": [asdict(nscan.compare(i)) for i in range(len(nscan.T_grid_s))],
        "cliff_T_s": {label: None if math.isnan(t) else float(t)
                      for label, t in zip(nscan.reg_labels, nscan.cliff_T_s, strict=True)},
    }


def _format_cell(cell: dict) -> str:
    """Format a comparison row after the duration column, including ties or a lone candidate."""
    if cell["tied_regs"]:
        return f"ambiguous tie ({', '.join(_bare(label) for label in cell['tied_regs'])})  F_COM={cell['best_F_COM']:.9f}"
    if cell["runner_up_reg"] is None:
        return f"{_bare(cell['best_reg']):<12} {cell['best_F_COM']:<14.9f} only candidate"
    return (f"{_bare(cell['best_reg']):<12} {cell['best_F_COM']:<14.9f} "
            f"{_bare(cell['runner_up_reg']):<12} {cell['delta_F']:.3e}")


def _bare(label: str) -> str:
    """Extract the value from a scan label, e.g. 'reg=0.001' becomes '0.001'."""
    return label.split("=", 1)[1]


def _fmt_cliff(cliff_T_s: float | None) -> str:
    """Format the longest tested T whose maximum x COM excursion exceeds the initial TF RMS width.

    None or NaN means no tested duration exceeded that width.
    """
    if cliff_T_s is None or math.isnan(cliff_T_s):
        return "none in requested T grid"
    return f"≈{cliff_T_s:.2f} s"
