"""Locate experiment files and interpret their directory names.

Paths are relative to the experiment root, identified by chip.json:

    exp-root/
        chip.json  info.txt
        plan/
            reg-{reg}/
                T-{t}/   trajectory.npz  trajectory.json  trajectory.log  figures/(schedule)
        scan/
            scan.log  harmonic.npz  harmonic.json  figures/(f_com_summary_by_n_t, cliff)
            N-{n}/   scan.npz  scan.json  figures/(f_com_loss_vs_t)
        sweep/
            reg-{reg}/   fidelity.csv  figures/(f_vs_t)
                N-{n}/   info.txt(cloud)
                    T-{t}/   figures/(comparison)
                        gpe/     eval.json  eval.npz  eval.log  figures/(eval)
                        linear/  eval.json  eval.npz  eval.log  figures/(eval)

find_root searches upward from a starting directory. Use path(key, reg=..., n=..., t=...,
mode=...) to fill a PATHS template without creating files or directories.
"""
import glob
import math
import os
import re
from collections.abc import Iterator
from decimal import Decimal
from typing import ClassVar


class Config:
    """Resolve paths and discover saved cases within one experiment."""

    # Shared directory names; full paths are defined below.
    REG_PREFIX      = "reg-"
    PLAN_SUBDIR     = "plan"
    SCAN_SUBDIR     = "scan"
    SWEEP_SUBDIR    = "sweep"
    GPE_SUBDIR      = "gpe"
    LINEAR_SUBDIR   = "linear"
    FIGURES_DIRNAME = "figures"
    MODES           = (GPE_SUBDIR, LINEAR_SUBDIR)

    # Path templates filled by path() using the root and supplied case labels.
    PATHS: ClassVar[dict[str, str]] = {
        "chip":            "{chip_root}/chip.json",
        "info":            "{chip_root}/info.txt",
        "plan_reg_dir":    "{chip_root}/plan/reg-{reg}",
        "plan_dir":        "{chip_root}/plan/reg-{reg}/T-{t}",
        "trajectory_npz":  "{chip_root}/plan/reg-{reg}/T-{t}/trajectory.npz",
        "trajectory_json": "{chip_root}/plan/reg-{reg}/T-{t}/trajectory.json",
        "trajectory_log":  "{chip_root}/plan/reg-{reg}/T-{t}/trajectory.log",
        "scan_dir":        "{chip_root}/scan",
        "scan_log":        "{chip_root}/scan/scan.log",
        "harmonic_npz":    "{chip_root}/scan/harmonic.npz",
        "harmonic_json":   "{chip_root}/scan/harmonic.json",
        "scan_n_dir":     "{chip_root}/scan/N-{n}",
        "scan_n_npz":     "{chip_root}/scan/N-{n}/scan.npz",
        "scan_n_json":    "{chip_root}/scan/N-{n}/scan.json",
        "sweep_dir":      "{chip_root}/sweep/reg-{reg}",
        "fidelity_csv":   "{chip_root}/sweep/reg-{reg}/fidelity.csv",
        "n_dir":          "{chip_root}/sweep/reg-{reg}/N-{n}",
        "n_info":         "{chip_root}/sweep/reg-{reg}/N-{n}/info.txt",
        "t_dir":          "{chip_root}/sweep/reg-{reg}/N-{n}/T-{t}",
        "mode_dir":       "{chip_root}/sweep/reg-{reg}/N-{n}/T-{t}/{mode}",
        "eval_json":      "{chip_root}/sweep/reg-{reg}/N-{n}/T-{t}/{mode}/eval.json",
        "eval_npz":       "{chip_root}/sweep/reg-{reg}/N-{n}/T-{t}/{mode}/eval.npz",
        "eval_log":       "{chip_root}/sweep/reg-{reg}/N-{n}/T-{t}/{mode}/eval.log",
    }

    def __init__(self, chip_root: str):
        self.chip_root = chip_root

    def path(self, key: str, **kw: object) -> str:
        """Fill a path template with the experiment root and supplied case labels."""
        return self.PATHS[key].format(chip_root=self.chip_root, **kw)

    def figures_dir(self, level_dir: str) -> str:
        """Return `<level_dir>/figures`, creating it if needed."""
        path = os.path.join(level_dir, self.FIGURES_DIRNAME)
        os.makedirs(path, exist_ok=True)
        return path

    def iter_plan_regs(self) -> Iterator[str]:
        """Yield regularization labels with a trajectory NPZ, sorted by directory name."""
        for d in sorted(glob.glob(self.path("plan_reg_dir", reg="*"))):
            reg = self.reg_from_dirname(d)
            if reg is not None and list(self.iter_plan_ts(reg)):
                yield reg

    def iter_plan_ts(self, reg: str) -> Iterator[float]:
        """Yield durations with a trajectory NPZ under plan/reg-<reg>, in numerical order."""
        ts = []
        for d in glob.glob(os.path.join(self.path("plan_reg_dir", reg=reg), "T-*")):
            t = self.t_from_dirname(os.path.basename(d))
            if t is not None and os.path.exists(self.path("trajectory_npz", reg=reg, t=t)):
                ts.append(t)
        yield from sorted(ts)

    @staticmethod
    def t_label(t: float) -> str:
        """Format a duration without losing float precision, e.g. 2.0 becomes '2.0'."""
        return repr(float(t))

    @classmethod
    def t_from_dirname(cls, name: str) -> float | None:
        """Parse a positive, finite duration named using t_label(); otherwise return None."""
        if not name.startswith("T-"):
            return None
        label = name[2:]
        try:
            t = float(label)
        except ValueError:
            return None
        return t if math.isfinite(t) and t > 0.0 and cls.t_label(t) == label else None

    def iter_sweep_regs(self) -> Iterator[str]:
        """Yield regularization labels under sweep/, sorted by directory name."""
        for d in sorted(glob.glob(self.path("sweep_dir", reg="*"))):
            reg = self.reg_from_dirname(d)
            if reg is not None and os.path.isdir(d):
                yield reg

    def iter_n_dirs(self, reg: str) -> Iterator[tuple[int, str]]:
        """Yield (atom number, path) pairs under a sweep, sorted by directory name."""
        for n_path in sorted(glob.glob(os.path.join(self.path("sweep_dir", reg=reg), "N-*"))):
            n = self.n_from_dirname(os.path.basename(n_path))
            if n is not None and os.path.isdir(n_path):
                yield n, n_path

    def iter_t_dirs(self, reg: str, n: int) -> Iterator[tuple[float, str]]:
        """Yield (duration, path) pairs for one (reg, N), sorted by directory name."""
        for t_path in sorted(glob.glob(os.path.join(self.path("n_dir", reg=reg, n=n), "T-*"))):
            t = self.t_from_dirname(os.path.basename(t_path))
            if t is not None and os.path.isdir(t_path):
                yield t, t_path

    @classmethod
    def iter_mode_dirs_in(cls, t_dir: str) -> Iterator[str]:
        """Yield existing gpe and linear directory paths beneath a duration directory."""
        for mode in cls.MODES:
            p = os.path.join(t_dir, mode)
            if os.path.isdir(p):
                yield p

    # These helpers check names, not whether directories exist.
    @classmethod
    def is_t_dir(cls, d: str) -> bool:
        return cls.t_from_dirname(os.path.basename(d)) is not None

    @classmethod
    def is_mode_dir(cls, d: str) -> bool:
        return os.path.basename(d) in cls.MODES

    @classmethod
    def reg_label(cls, arg: str) -> str:
        """Extract a regularization label from a CLI argument, without validating it.

        '0.01', 'reg-0.01', and 'plan/reg-0.01/' all give '0.01'.
        """
        return os.path.basename(arg.rstrip("/")).removeprefix(cls.REG_PREFIX)

    @staticmethod
    def reg_value_label(value: float) -> str:
        """Format a regularization without scientific notation or loss of float precision.

        For example, 1e-05 becomes '0.00001'.
        """
        label = format(Decimal(repr(float(value))), "f")
        assert float(label) == float(value), f"label {label!r} does not round-trip {value!r}"
        return label

    @classmethod
    def reg_from_dirname(cls, name: str) -> str | None:
        """Return a positive, finite regularization label in the standard directory format.

        Accept reg-0.00001, but return None for alternatives such as reg-1e-05 or reg-00.1.
        """
        base = os.path.basename(name.rstrip("/"))
        if not base.startswith(cls.REG_PREFIX):
            return None
        label = base[len(cls.REG_PREFIX):]
        try:
            value = float(label)
        except ValueError:
            return None
        if not (math.isfinite(value) and value > 0.0):
            return None
        return label if cls.reg_value_label(value) == label else None

    @staticmethod
    def n_from_dirname(name: str) -> int | None:
        """Parse an N-<digits> name; return None if the name does not match."""
        m = re.fullmatch(r"N-(\d+)", name)
        return int(m.group(1)) if m else None



    @classmethod
    def find_root(cls, start_dir: str) -> str | None:
        """Return the nearest directory containing chip.json at or above start_dir, or None."""
        d = os.path.abspath(start_dir)
        while True:
            if os.path.exists(cls.PATHS["chip"].format(chip_root=d)):
                return d
            parent = os.path.dirname(d)
            if parent == d:
                return None
            d = parent
