"""Time-dependent Gross-Pitaevskii solver via Strang split-step Fourier."""
from .box import (
    BOUNDARY_TOL,
    CELLS_PER_HEALING_LENGTH,
    BoxGrid,
    box_shift_margin,
    build_box,
    size_box,
    tail_margin_m,
)
from .solver import Stepping, derive_stepping
from .transport import GS_RESIDUAL_WARN, EvalResult, evaluate
from .view import (
    CloudDiagnostics,
    format_fidelity_table,
    load_diagnostics,
    load_eval_result,
    render_eval_figs,
    render_overlay_fig,
    render_sweep_figs,
    write_eval,
    write_fidelity_csv,
)

__all__ = [
    "BOUNDARY_TOL",
    "CELLS_PER_HEALING_LENGTH",
    "GS_RESIDUAL_WARN",
    "BoxGrid",
    "CloudDiagnostics",
    "EvalResult",
    "Stepping",
    "box_shift_margin",
    "build_box",
    "derive_stepping",
    "evaluate",
    "format_fidelity_table",
    "load_diagnostics",
    "load_eval_result",
    "render_eval_figs",
    "render_overlay_fig",
    "render_sweep_figs",
    "size_box",
    "tail_margin_m",
    "write_eval",
    "write_fidelity_csv",
]
