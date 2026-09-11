"""Display the initial trap in an interactive application or web viewer."""

import argparse
import sys

import atom_chip as ac

from . import common
from .config import Config


def register(sub: argparse._SubParsersAction) -> None:
    """Register visualization settings and the optional web viewer."""
    # fmt: off
    p = sub.add_parser("show", help="View the initial trap interactively")
    p.add_argument("--config", type=str, default=None, help="Visualization yaml config")
    p.add_argument("--web",    action="store_true",    help="Serve the figures over HTTP")
    p.add_argument("--port",   type=int, default=8988, help="Port for --web")
    p.set_defaults(func=run)
    # fmt: on


def run(args: argparse.Namespace) -> int:
    """Load the experiment's chip, locate its initial trap minimum, and launch the viewer."""
    logger = common.configure_stream_logging("chip.show")

    chip_root = Config.find_root(args.work_dir)
    if chip_root is None:
        print("chip show: no chip.json found at or above the current "
              "directory. Run `chip init` first.", file=sys.stderr)
        return 1

    config = Config(chip_root)
    chip = ac.schedule.AtomChip.load(config.path("chip"))

    logger.info("Finding initial trap minimum...")
    minimum = chip.search_trap_minimum()

    if args.web:
        ac.visualization.show_web(chip, minimum, args.config, port=args.port)
    else:
        ac.visualization.show_app(chip, minimum, args.config)

    return 0
