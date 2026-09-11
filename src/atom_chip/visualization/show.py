import importlib
import logging
import os
import sys
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import yaml
from matplotlib.backend_bases import CloseEvent
from matplotlib.figure import Figure

from ..schedule.chip import AtomChip, Minimum

logger = logging.getLogger("visualization.show")

DEFAULT_YAML = "visualization.yaml"


def show_web(
    atom_chip: AtomChip,
    minimum: Minimum,
    yaml_path: str | None = None,
    *,
    port: int = 8988,
) -> None:
    """Remote interactive viewer: serve the figures over HTTP (webagg), viewable in a browser."""
    if "--no-show" in sys.argv:
        return

    matplotlib.use("webagg")
    matplotlib.rcParams["savefig.bbox"] = "tight"
    matplotlib.rcParams["savefig.pad_inches"] = 0.02
    matplotlib.rcParams["webagg.port"] = port
    matplotlib.rcParams["webagg.open_in_browser"] = False  # headless host

    _build_figures(atom_chip, minimum, yaml_path)

    # Prints the URL rather than opening a browser. Good for headless host and VS Code Remote.
    logger.info("Serving figures at http://localhost:%d/  (Ctrl-C to stop)", port)
    logger.info("VS Code Remote forwards the port automatically; otherwise forward %d in the Ports panel.", port)

    plt.show()


def show_app(
    atom_chip: AtomChip,
    minimum: Minimum,
    yaml_path: str | None = None,
) -> None:
    """Local interactive viewer: tiled Qt windows (needs a display)."""
    if "--no-show" in sys.argv:
        return

    # NOTE: the interactive Qt backend and PyQt5 are imported lazily so batch CLI stays headless-safe
    matplotlib.use("Qt5Agg")
    matplotlib.rcParams["savefig.bbox"] = "tight"
    matplotlib.rcParams["savefig.pad_inches"] = 0.02

    if yaml_path is None:
        yaml_path = os.path.join(os.path.dirname(__file__), DEFAULT_YAML)

    visualizer = Visualizer(yaml_path)
    visualizer.update(atom_chip, minimum)

    input("\nPress Enter to close the figures...\n\n")


class Visualizer:
    def __init__(self, config_path: str):
        self._plot_windows = {}
        self._yaml_path = config_path
        self._config = _load_config(config_path)

        self._global_top = self._config.get("top", 0)
        self._global_left = self._config.get("left", 0)
        self._top = self._global_top
        self._left = self._global_left
        self._bottom = 0

    def update(self, atom_chip: AtomChip, minimum: Minimum):
        plt.ioff()
        for name, fig in _build_figures(atom_chip, minimum, self._yaml_path).items():
            self._initialize_position(fig)
            self._plot_windows[name] = fig
            fig.canvas.mpl_connect("close_event", self._close_handler)
            plt.pause(0.01)
        plt.ion()

    def _initialize_position(self, fig: plt.Figure) -> None:
        from PyQt5.QtWidgets import QApplication  # local import: keeps the Qt dependency out of module load
        screen_width, screen_height = QApplication.desktop().screenGeometry().getRect()[2:]
        window = fig.canvas.manager.window
        width, height = window.geometry().getRect()[2:]
        top, left = self._top, self._left
        if left + width > screen_width:
            top = self._bottom + 40
            left = self._global_left
            if top + height > screen_height:
                top = self._global_top
        window.setGeometry(left, top, width, height)
        self._bottom = max(self._bottom, top + height)
        self._top = top
        self._left = left + width + 10

    def _close_handler(self, event: CloseEvent) -> None:
        title = event.canvas.manager.get_window_title()
        logger.info(f"Closed: '{title}'")
        for key, fig in list(self._plot_windows.items()):
            if fig.canvas.manager.get_window_title() == title:
                del self._plot_windows[key]
                break
        if not self._plot_windows:
            logger.info("All figures closed.")
            sys.exit(0)


def _build_figures(
    atom_chip: AtomChip,
    minimum: Minimum,
    yaml_path: str | None = None,
) -> dict[str, Figure]:
    """Build every figure in the yaml spec, backend-agnostic (no windowing). Returns {name: Figure}
    in spec order. The Qt viewer (`show`) and the web viewer (`show_web`) both render these, and a
    notebook can call it directly under `%matplotlib widget`.
    """
    if yaml_path is None:
        yaml_path = os.path.join(os.path.dirname(__file__), DEFAULT_YAML)
    config = _load_config(yaml_path)
    logger.info("Processing figures...")
    figures: dict[str, Figure] = {}
    for name in config["plots"]:
        plot_config = config[name]
        function = _resolve_function(plot_config["function"])
        figures[name] = function(atom_chip, minimum, **plot_config.get("params", {}))
        logger.info("- %s", name)
    return figures


def _resolve_function(fq_name: str):
    """Resolve a fully-qualified `pkg.module.func` name to the callable."""
    module_path, func_name = fq_name.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, func_name)


def _load_config(yaml_path: str):
    with open(yaml_path, encoding="utf-8") as file:
        config = yaml.safe_load(file)
    return _convert_scientific_notation(config)


def _convert_scientific_notation(data: Any) -> Any:
    if isinstance(data, dict):
        return {k: _convert_scientific_notation(v) for k, v in data.items()}
    if isinstance(data, list):
        return [_convert_scientific_notation(i) for i in data]
    if isinstance(data, str):
        try:
            return float(data)
        except ValueError:
            return data
    return data
