"""Gaussian and Thomas–Fermi surrogate models for schedule screening and grid sizing."""
from .gaussian import run_gaussian
from .scan import (
    N_SCAN_SCHEMA,
    HarmonicGrid,
    NScan,
    derive_n_scan,
    load_harmonic_pair,
    load_n_scan_pair,
)
from .sizing import predict_cloud_geometry
from .view import (
    render_cliff_fig,
    render_f_com_loss_fig,
    render_f_com_summary_fig,
)

__all__ = [
    "N_SCAN_SCHEMA",
    "HarmonicGrid",
    "NScan",
    "derive_n_scan",
    "load_harmonic_pair",
    "load_n_scan_pair",
    "predict_cloud_geometry",
    "render_cliff_fig",
    "render_f_com_loss_fig",
    "render_f_com_summary_fig",
    "run_gaussian",
]
