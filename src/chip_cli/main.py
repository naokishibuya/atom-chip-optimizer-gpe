"""Dispatch chip CLI commands to their handlers."""

import argparse
import functools
import os
import sys

from . import sub_eval, sub_init, sub_plan, sub_plot, sub_scan, sub_show


def main(argv: list[str] | None = None) -> int:
    """Parse arguments and run the selected command, returning its exit status.

    If argv is None, read arguments from the command line.
    """
    parser = argparse.ArgumentParser(prog="chip", description="Optimize and validate atom chip BEC transport")
    sub = parser.add_subparsers(
        dest="cmd",
        required=True,
        metavar="<command>",
        parser_class=functools.partial(
            argparse.ArgumentParser,
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        ),
    )
    # Each command registers its arguments and handler as args.func.
    sub_init.register(sub)
    sub_show.register(sub)
    sub_plan.register(sub)
    sub_scan.register(sub)
    sub_eval.register(sub)
    sub_plot.register(sub)
    args = parser.parse_args(argv)
    # Commands use the invocation directory to locate the experiment and scope their work.
    args.work_dir = os.getcwd()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
