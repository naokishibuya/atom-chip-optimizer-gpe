"""Create the chip configuration and trap report for an experiment."""

import argparse
import os
import sys

import atom_chip as ac

from . import common
from .config import Config


def register(sub: argparse._SubParsersAction) -> None:
    """Register the init command and its overwrite option."""
    p = sub.add_parser("init", help="Build the atom chip")
    p.add_argument("--force", action="store_true", help="Overwrite existing chip.json / info.txt")
    p.set_defaults(func=run)


def run(args: argparse.Namespace) -> int:
    """Build the chip and write chip.json and info.txt in the invocation directory.

    Refuse to overwrite either file unless --force is set.
    """
    if not common.ensure_gpu("init"):
        return 1
    logger = common.configure_stream_logging("chip.init")
    # Initialize here rather than searching parent directories for an existing experiment.
    ep = Config(args.work_dir)

    if not args.force:
        existing = [p for p in [ep.path("chip"), ep.path("info")] if os.path.exists(p)]
        if existing:
            print("The following already exists:", file=sys.stderr)
            for p in existing:
                print(f"- {os.path.relpath(p, ep.chip_root)}", file=sys.stderr)
            print("use --force to overwrite.", file=sys.stderr)
            return 1

    chip = ac.schedule.build_chip()
    chip.save(ep.path("chip"))
    logger.info("Wrote %s", os.path.relpath(ep.path("chip"), ep.chip_root))
    with open(ep.path("info"), "w") as f:
        f.write(ac.schedule.format_trap(chip))
    logger.info("Wrote %s", os.path.relpath(ep.path("info"), ep.chip_root))
    return 0
