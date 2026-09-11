"""Generate and save an inverse-optimized current schedule for each (regularization, duration)."""

import argparse
import os
import sys
import time

import atom_chip as ac

from . import common
from .config import Config

_DEFAULT_REGS = [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2]


def register(sub: argparse._SubParsersAction) -> None:
    """Register the candidate grid, control cadence, and current-optimization options."""
    # fmt: off
    p = sub.add_parser("plan", help="Inverse-optimize one held transport schedule per (reg, T) at a fixed control cadence")
    p.add_argument("--reg",            type=float, nargs="+", default=_DEFAULT_REGS, metavar="REG", help="Regularization parameters (default: 1e-5..1e-2, 1 and 5 per decade)")
    p.add_argument("--T",              type=str,   nargs="+", default=["0.5:3.0:0.5"], metavar="T", help="Transport times [s]: values and/or START:STOP:STEP (default 0.5:3.0:0.5)")
    p.add_argument("--dt",             type=float, default=5e-4, help="Physical control-update interval [s] (default 5e-4); every T must be an integer multiple")
    p.add_argument("--scheduler",      choices=list(ac.schedule.SCHEDULERS), default="smoothstep", help="Target path shape")
    p.add_argument("--I-max-shifting", type=float, default=3.5,  help="Shifting wire current cap (A)")
    p.add_argument("--I-max-guiding",  type=float, default=70.0, help="Guiding wire current cap (A)")
    p.add_argument("--wire-ids",       type=int, nargs="*", default=ac.schedule.SHIFTING_WIRE_IDS, help="Wires available for current updates during the schedule")
    p.add_argument("--num-shifts",     type=int,   default=6,    help="Number of shifts to apply to the trap")
    p.add_argument("--force",          action="store_true", help="Overwrite existing plan/reg-<value>/T-<value>/ folders")
    p.set_defaults(func=run)
    # fmt: on


def run(args: argparse.Namespace) -> int:
    """Validate the requested cases, optimize each independently, and save their schedules.

    Without --force, any existing destination stops the entire batch before optimization.
    """
    try:
        cases = _workload(args.reg, args.T, args.dt)
    except ValueError as e:
        print(f"chip plan: {e}", file=sys.stderr)
        return 1

    # Show every requested case and its control-node count before starting work.
    print(f"chip plan: {len(cases)} case(s), dt={args.dt:g} s")
    print(f"  {'reg':>8s} {'T[s]':>6s} {'intervals':>9s} {'nodes':>6s}")
    for _, lbl, t, n_int in cases:
        print(f"  {lbl:>8s} {Config.t_label(t):>6s} {n_int:9d} {n_int + 1:6d}")

    if not common.ensure_gpu("plan"):
        return 1
    ep = common.resolve_root("plan", args.work_dir)
    if ep is None:
        return 1
    chip = common.load_chip(ep)

    # Check all destinations first to avoid starting a batch with an overwrite conflict.
    if not args.force:
        existing = [f"reg-{lbl}/T-{Config.t_label(t)}" for _, lbl, t, _ in cases
                    if os.path.exists(ep.path("plan_dir", reg=lbl, t=Config.t_label(t)))]
        if existing:
            print("chip plan: destination(s) already exist under plan/: " + ", ".join(existing) +
                  ".\nRemove them or pass --force to overwrite.", file=sys.stderr)
            return 1

    # Each shift advances the target by one wire spacing along laboratory x.
    x_slide_mm = ac.schedule.SHIFTING_WIRE_X_SLIDE
    destination_offset_mm = [args.num_shifts * x_slide_mm, 0.0, 0.0]
    for reg, lbl, t, n_int in cases:
        t_lbl = Config.t_label(t)
        plan_dir = ep.path("plan_dir", reg=lbl, t=t_lbl)
        os.makedirs(plan_dir, exist_ok=True)
        logger = common.configure_file_logging("chip.plan", ep.path("trajectory_log", reg=lbl, t=t_lbl))
        logger.info("chip plan: reg=%.17g  T=%g s  dt=%g s  ->  %s  (%d intervals, %d nodes)",
                    reg, t, args.dt, os.path.relpath(plan_dir, ep.chip_root), n_int, n_int + 1)
        t0 = time.time()
        # fmt: off
        result = ac.schedule.optimize_current_schedule(
            chip           = chip,
            scheduler_name = args.scheduler,
            steps          = n_int,
            reg            = reg,
            I_max_shifting_A = args.I_max_shifting,
            I_max_guiding_A  = args.I_max_guiding,
            wire_ids       = args.wire_ids,
            destination_offset_mm = destination_offset_mm,
        )
        # fmt: on
        # Keep the planning inputs and control timing alongside the schedule arrays.
        common.write_trajectory_pair(ep, lbl, t, result, meta={
            "reg": reg,
            "reg_label": lbl,
            "T_s": t,
            "control_dt_s": args.dt,
            "n_intervals": n_int,
            "n_nodes": n_int + 1,
            "scheduler": args.scheduler,
            "optimizer_args": {
                "I_max_shifting_A": args.I_max_shifting,
                "I_max_guiding_A": args.I_max_guiding,
                "wire_ids": list(args.wire_ids),
                "num_shifts": args.num_shifts,
                "destination_offset_mm": destination_offset_mm,
            },
        })
        logger.info("Wrote %s (runtime %.1fs)",
                    os.path.relpath(ep.path("trajectory_npz", reg=lbl, t=t_lbl), ep.chip_root),
                    time.time() - t0)
    return 0


def _workload(regs: list[float], t_tokens: list[str], dt: float) -> list[tuple[float, str, float, int]]:
    """Return validated (reg, reg_label, T, n_intervals) cases, sorted by reg then T.

    Remove exact duplicates and check that folder labels are distinct. Each duration
    must be an integer multiple of dt within the shared floating-point tolerance;
    the requested duration is not changed.
    """
    import math

    if not (math.isfinite(dt) and dt > 0):
        raise ValueError(f"--dt must be positive and finite (got {dt!r})")
    for r in regs:
        if not (math.isfinite(r) and r > 0):
            raise ValueError(f"--reg values must be positive and finite (got {r!r})")
    uniq_regs = sorted(set(regs))
    ts = common.parse_t_args(t_tokens)

    labels = [Config.reg_value_label(r) for r in uniq_regs]
    if len(set(labels)) != len(labels):
        raise ValueError(f"regularization folder labels collide: {labels}")
    t_labels = [Config.t_label(t) for t in ts]
    if len(set(t_labels)) != len(t_labels):
        raise ValueError(f"duration folder labels collide: {t_labels}")

    cases = []
    for r, lbl in zip(uniq_regs, labels, strict=True):
        for t in ts:
            n = float(t) / dt
            if abs(n - round(n)) > common._RANGE_RTOL * max(1.0, n):
                raise ValueError(f"T={t:g} is not an integer multiple of dt={dt:g} "
                                 f"(T/dt = {n:g}); adjust --T or --dt")
            cases.append((r, lbl, float(t), round(n)))
    return cases
