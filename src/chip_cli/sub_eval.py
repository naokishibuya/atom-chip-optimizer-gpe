"""Size the simulation grid and evaluate selected schedules with the GPE or linear baseline."""

import argparse
import os
import sys
import time

import atom_chip as ac

from . import common
from .config import Config


def register(sub: argparse._SubParsersAction) -> None:
    """Register evaluation options and require explicit atom numbers and durations."""
    p = sub.add_parser("eval", help="Run the 3D GPE forward evaluation for selected (N, T) cases")
    # fmt: off
    p.add_argument("plan_dir",                    metavar="PLAN_DIR", help="Regularization plan folder, e.g. plan/reg-0.001")
    p.add_argument("--N",                         type=str, nargs="+", required=True, metavar="N",
                   help="Atom numbers: positive integers, scientific notation ok (required; no implicit sweep)")
    p.add_argument("--T",                         type=str, nargs="+", required=True, metavar="T",
                   help="Transport times [s]: values and/or START:STOP:STEP; each must be planned (required)")
    p.add_argument("--cells-per-healing-length",  type=float, default=ac.tdgpe.CELLS_PER_HEALING_LENGTH,
                   help="Grid resolution: dx = ξ_min / this. Raise to tighten resolution (convergence check)")
    p.add_argument("--boundary-tol",              type=float, default=ac.tdgpe.BOUNDARY_TOL,
                   help="Heuristic tail-sizing parameter")
    p.add_argument("--linear",                    action="store_true",
                   help="Linear-Schrödinger comparison run: g_eff=0 throughout. Output lands under linear/ instead of gpe/")
    p.add_argument("--dry-run",                   action="store_true", help="Log config only")
    p.add_argument("--force",                     action="store_true", help="Overwrite existing eval files")
    # fmt: on
    p.set_defaults(func=run)


def run(args: argparse.Namespace) -> int:
    """Evaluate requested (N, T) cases using saved schedules for one regularization.

    Existing eval.json files are skipped unless --force is set. A dry run still
    computes and logs the grid and stepping settings, but skips GPE propagation.
    """
    if not common.ensure_gpu("eval"):
        return 1
    plan_dir = os.path.abspath(args.plan_dir)
    if not os.path.isdir(plan_dir):
        print(f"chip eval: no such directory '{args.plan_dir}'. Pass a plan folder, e.g. plan/reg-0.001.",
              file=sys.stderr)
        return 1
    chip_root = Config.find_root(plan_dir)
    if chip_root is None:
        print(f"chip eval: no chip.json above '{args.plan_dir}'. Run `chip init` first.", file=sys.stderr)
        return 1
    ep = Config(chip_root)
    reg = Config.reg_label(plan_dir)

    # Validate all requested schedules before starting any evaluations.
    try:
        n_list = common.parse_n_args(args.N)
        t_list = [float(t) for t in common.parse_t_args(args.T)]
        schedules = {t: common.load_schedule(ep, reg, t) for t in t_list}
    except ValueError as e:
        print(f"chip eval: {e}", file=sys.stderr)
        return 1

    chip = common.load_chip(ep)
    mode = ep.LINEAR_SUBDIR if args.linear else ep.GPE_SUBDIR
    # Restore the shared trap report if it is missing.
    if not os.path.exists(ep.path("info")):
        with open(ep.path("info"), "w") as f:
            f.write(ac.schedule.format_trap(chip))

    for n_atoms in n_list:
        os.makedirs(ep.path("n_dir", reg=reg, n=n_atoms), exist_ok=True)
        with open(ep.path("n_info", reg=reg, n=n_atoms), "w") as f:
            f.write(ac.schedule.format_cloud(chip, n_atoms))

        for T_total_s in t_list:
            sched = schedules[T_total_s]
            n_nodes = int(sched.r_mins_mm.shape[0])
            mode_d = ep.path("mode_dir", reg=reg, n=n_atoms, t=T_total_s, mode=mode)
            eval_path = ep.path("eval_json", reg=reg, n=n_atoms, t=T_total_s, mode=mode)
            npz_path = ep.path("eval_npz", reg=reg, n=n_atoms, t=T_total_s, mode=mode)
            if os.path.exists(eval_path) and not args.force:
                print(f"Skipping {os.path.relpath(eval_path, ep.chip_root)} (use --force to overwrite).",
                      file=sys.stderr)
                continue

            os.makedirs(mode_d, exist_ok=True)
            logger = common.configure_file_logging(
                "chip.eval", ep.path("eval_log", reg=reg, n=n_atoms, t=T_total_s, mode=mode))
            logger.info("chip eval  reg=%s  N=%d  T=%.2fs  mode=%s  →  %s",
                        reg, n_atoms, T_total_s, mode, os.path.relpath(mode_d, ep.chip_root))

            # Combine predicted cloud extents and healing lengths with the box-shift allowance.
            geom = ac.surrogate.predict_cloud_geometry(sched, chip, n_atoms, T_total_s)
            shift_margin_m = ac.tdgpe.box_shift_margin(sched.r_mins_mm)
            grid = ac.tdgpe.size_box(
                geom.extent_m, geom.slosh_m, geom.xi_ref_m, geom.xi_min_m,
                shift_margin_m=shift_margin_m,
                cells_per_xi=args.cells_per_healing_length,
                boundary_tol=args.boundary_tol,
            )
            # Choose propagation and relaxation settings from the grid and trap frequencies.
            real_sub_s, imag_step_s, imag_iters = ac.tdgpe.derive_stepping(
                grid, chip.atom.mass_kg, sched.omegas_hz,
            )
            # Adjust the real-time step to divide each control half-interval exactly.
            stepping = ac.tdgpe.Stepping(
                real_step_s=T_total_s / (n_nodes - 1),
                real_sub_s=real_sub_s,
                imag_step_s=imag_step_s,
                imag_iters=imag_iters,
            )
            # Save the sizing inputs alongside the evaluation for traceability.
            # fmt: off
            sizing = {
                "cells_per_healing_length": float(args.cells_per_healing_length),
                "boundary_tol"            : float(args.boundary_tol),
                "extent_m"                : list(geom.extent_m),
                "slosh_m"                 : list(geom.slosh_m),
                "box_shift_margin_m"      : list(shift_margin_m),
                "xi_ref_m"                : float(geom.xi_ref_m),
                "xi_min_m"                : float(geom.xi_min_m),
            }
            # fmt: on
            nx, ny, nz = grid.shape
            logger.info("\n".join([
                "sizing",
                (
                    f"  resolution : cells/ξ = {args.cells_per_healing_length:g}   "
                    f"boundary_tol = {args.boundary_tol:g}"
                ),
                f"  cloud (ROM): extent {_um(geom.extent_m)} µm   slosh {_um(geom.slosh_m)} µm",
                f"  box shift  : {_um(shift_margin_m)} µm (max node jump; schedule geometry, not ROM)",
                (
                    f"  healing ξ  : {geom.xi_min_m * 1e6:.3f} µm compressed → sets grid dx   /   "
                    f"{geom.xi_ref_m * 1e6:.3f} µm initial → sets box margin"
                ),
                (
                    f"  box        : half-widths {_um(grid.half_widths_m)} µm   n_cells {nx}×{ny}×{nz}   "
                    f"dx {_um(grid.dx_m)} µm"
                ),
                (
                    f"  stepping   : hold {stepping.real_step_s:.3g} s × {n_nodes - 1}   "
                    f"substep {stepping.real_sub_realized_s:.3g} s × {stepping.n_sub}/hold   "
                    f"imag {stepping.imag_step_s:.3g} s × {stepping.imag_iters}"
                ),
            ]))

            if args.dry_run:
                logger.info("dry-run: skipping evaluation.")
                continue

            t0 = time.time()
            result = ac.tdgpe.evaluate(
                chip, sched, grid, stepping,
                n_atoms=n_atoms,
                linear=args.linear,
            )
            runtime_s = time.time() - t0

            ac.tdgpe.write_eval(eval_path, npz_path, result, n_atoms=n_atoms, T_total_s=T_total_s,
                                grid=grid, stepping=stepping, sizing=sizing, linear=args.linear)
            logger.info("F_3D=%.5f  ·  %.1fs  →  %s", result.F_3D, runtime_s, os.path.basename(eval_path))

    return 0


def _um(vals: tuple[float, float, float]) -> str:
    """Format three lengths in metres as a micrometre tuple for the log."""
    return "(" + ", ".join(f"{v * 1e6:.2f}" for v in vals) + ")"
