"""Shared CLI setup, argument parsing, and saved-result loading."""

import json
import logging
import math
import os
import sys

import numpy as np

import atom_chip as ac

from .config import Config


def configure_file_logging(name: str, log_path: str) -> logging.Logger:
    """Log to the terminal and a freshly overwritten file, with timestamps."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(), logging.FileHandler(log_path, mode="w")],
        force=True,
    )
    return logging.getLogger(name)


def configure_stream_logging(name: str) -> logging.Logger:
    """Log plain messages to the terminal."""
    logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)
    return logging.getLogger(name)


def ensure_gpu(command: str) -> bool:
    """Reject the CPU JAX backend with a CLI error message; return whether to proceed."""
    import jax
    if jax.default_backend() != "cpu":
        return True
    print(f"chip {command}: requires a GPU JAX backend (found cpu). "
          "scan/plot work on CPU; init/plan/eval do not.", file=sys.stderr)
    return False


def resolve_root(command: str, work_dir: str) -> Config | None:
    """Find chip.json at or above work_dir; return its Config, or report failure."""
    chip_root = Config.find_root(work_dir)
    if chip_root is None:
        print(f"chip {command}: no chip.json at or above the current directory. Run `chip init` first.",
              file=sys.stderr)
        return None
    return Config(chip_root)


def load_chip(ep: Config) -> ac.schedule.AtomChip:
    return ac.schedule.AtomChip.load(ep.path("chip"))


TRAJECTORY_SCHEMA = "trajectory-v1"

# Units recorded in trajectory.json; shapes and dtypes come from the saved arrays.
TRAJECTORY_UNITS = {
    "I_logical_A": "A", "I_segments_A": "A", "r_mins_mm": "mm", "target_rs_mm": "mm",
    "U0s_J": "J", "omegas_hz": "Hz", "eigvecs": "1", "a_ho_axes_m": "m",
    "larmor_freqs_hz": "Hz", "J_conds": "1",
}

_RANGE_RTOL = 1e-9  # Floating-point tolerance for range division and schedule-duration checks.


def parse_t_args(tokens: list[str]) -> np.ndarray:
    """Parse transport times in seconds, returning sorted, unique values.

    Each token is a positive value or an inclusive START:STOP:STEP range.
    The step must reach STOP within _RANGE_RTOL; ranges are not truncated.
    """
    values: list[float] = []
    for token in tokens:
        parts = token.split(":")
        if len(parts) == 1:
            values.append(_positive_float(token, "T"))
            continue
        if len(parts) != 3:
            raise ValueError(f"T token must be VALUE or START:STOP:STEP (got {token!r})")
        start, stop = _positive_float(parts[0], "T"), _positive_float(parts[1], "T")
        try:
            step = float(parts[2])
        except ValueError:
            raise ValueError(f"T range {token!r}: STEP is not a number") from None
        if stop < start:
            raise ValueError(f"T range {token!r}: STOP must be >= START")
        if not math.isfinite(step) or step <= 0:
            raise ValueError(f"T range {token!r}: STEP must be positive and finite")
        n_steps = (stop - start) / step
        if abs(n_steps - round(n_steps)) > _RANGE_RTOL * max(1.0, n_steps):
            raise ValueError(f"T range {token!r} does not divide: (STOP-START)/STEP = {n_steps:g}")
        grid = start + step * np.arange(round(n_steps) + 1)
        # Use the requested endpoint rather than its floating-point reconstruction.
        grid[-1] = stop
        values.extend(float(v) for v in grid)
    unique = sorted(set(values))
    return np.array(unique, dtype=np.float64)


def parse_n_args(tokens: list[str]) -> list[int]:
    """Parse positive integer atom numbers, accepting scientific notation but rejecting duplicates."""
    values = []
    for token in tokens:
        v = _positive_float(token, "N")
        if v != round(v):
            raise ValueError(f"N {token!r} must be integer-valued")
        values.append(round(v))
    if len(set(values)) != len(values):
        raise ValueError(f"duplicate N values in {tokens}")
    return values


def _positive_float(token: str, what: str) -> float:
    try:
        v = float(token)
    except ValueError:
        raise ValueError(f"{what} value {token.strip()!r} is not a number") from None
    if not math.isfinite(v) or v <= 0:
        raise ValueError(f"{what} values must be positive and finite (got {token.strip()!r})")
    return v


def write_trajectory_pair(ep: Config, reg_label: str, t: float, sched: ac.schedule.CurrentSchedule,
                          meta: dict) -> None:
    """Save one (reg, T) schedule as trajectory.npz and trajectory.json.

    The NPZ stores arrays; the JSON records their layout, units, and hold convention,
    together with the planning metadata supplied in meta.
    """
    t_lbl = Config.t_label(t)
    sched.save(ep.path("trajectory_npz", reg=reg_label, t=t_lbl))
    from dataclasses import fields
    arrays = {f.name: {"shape": list(np.asarray(getattr(sched, f.name)).shape),
                       "dtype": str(np.asarray(getattr(sched, f.name)).dtype),
                       "units": TRAJECTORY_UNITS[f.name]}
              for f in fields(sched)}
    payload = {
        "schema": TRAJECTORY_SCHEMA,
        "hold_convention": "node-centred half-first/full-interior/half-last held waveform",
        "arrays": arrays,
        **meta,
    }
    with open(ep.path("trajectory_json", reg=reg_label, t=t_lbl), "w") as f:
        json.dump(payload, f, indent=2)


def load_schedule(ep: Config, reg: str, t: float) -> ac.schedule.CurrentSchedule:
    """Load a schedule and check its companion JSON metadata.

    Check the schema, regularization, hold convention, node counts, duration,
    and each array's shape, dtype, and units.
    """
    t_lbl = Config.t_label(t)
    npz_path = ep.path("trajectory_npz", reg=reg, t=t_lbl)
    json_path = ep.path("trajectory_json", reg=reg, t=t_lbl)
    for path in (npz_path, json_path):
        if not os.path.exists(path):
            raise ValueError(f"missing {os.path.relpath(path, ep.chip_root)}; run `chip plan` for this case")
    with open(json_path) as f:
        meta = json.load(f)
    if meta.get("schema") != TRAJECTORY_SCHEMA:
        raise ValueError(f"{json_path}: schema {meta.get('schema')!r} != {TRAJECTORY_SCHEMA!r}; re-plan")
    if meta.get("reg_label") != reg or Config.reg_value_label(float(meta["reg"])) != reg:
        raise ValueError(f"{json_path}: regularization {meta.get('reg')!r}/{meta.get('reg_label')!r} "
                         f"does not match folder reg-{reg}")
    if "node-centred" not in str(meta.get("hold_convention", "")):
        raise ValueError(f"{json_path}: unexpected hold convention {meta.get('hold_convention')!r}")
    sched = ac.schedule.CurrentSchedule.load(npz_path)
    n_nodes = int(np.asarray(sched.r_mins_mm).shape[0])
    if meta["n_nodes"] != n_nodes or meta["n_intervals"] != n_nodes - 1:
        raise ValueError(f"{json_path}: node counts disagree with {npz_path}")
    if float(meta["T_s"]) != float(t):
        raise ValueError(f"{json_path}: T={meta['T_s']} != requested {t}")
    dt = float(meta["control_dt_s"])
    if not (dt > 0 and abs(float(t) - meta["n_intervals"] * dt) <= _RANGE_RTOL * float(t)):
        raise ValueError(f"{json_path}: T={t} != n_intervals*dt = {meta['n_intervals']}*{dt}")
    from dataclasses import fields
    for f_ in fields(sched):
        arr = np.asarray(getattr(sched, f_.name))
        declared = meta["arrays"].get(f_.name)
        if (declared is None or list(arr.shape) != declared["shape"]
                or str(arr.dtype) != declared["dtype"]
                or declared.get("units") != TRAJECTORY_UNITS[f_.name]):
            raise ValueError(f"{json_path}: array {f_.name} disagrees with {npz_path}")
    return sched



def load_eval(ep: Config, reg: str, n: int, t: float, mode: str) -> ac.tdgpe.EvalResult | None:
    """Load an evaluation, returning None for missing files or missing result keys."""
    path = ep.path("eval_json", reg=reg, n=n, t=t, mode=mode)
    if not os.path.exists(path):
        return None
    try:
        return ac.tdgpe.load_eval_result(path, ep.path("eval_npz", reg=reg, n=n, t=t, mode=mode))
    except (KeyError, FileNotFoundError):
        return None  # Skip incomplete results when collecting evaluations.


def collect_rows(ep: Config, reg: str) -> list[tuple[int, float, ac.tdgpe.EvalResult | None, ac.tdgpe.EvalResult | None]]:
    """Collect (N, T, GPE, linear) rows for one regularization, sorted by N then T.

    A missing mode is None; cases with neither result are omitted.
    """
    by_nt: dict[tuple[int, float], dict[str, ac.tdgpe.EvalResult]] = {}
    for n, _ in ep.iter_n_dirs(reg):
        for t, _ in ep.iter_t_dirs(reg, n):
            for mode in ep.MODES:
                result = load_eval(ep, reg, n, t, mode)
                if result is not None:
                    by_nt.setdefault((n, t), {})[mode] = result
    rows = []
    for (n, t) in sorted(by_nt):
        cell = by_nt[(n, t)]
        rows.append((n, t, cell.get(ep.GPE_SUBDIR), cell.get(ep.LINEAR_SUBDIR)))
    return rows
