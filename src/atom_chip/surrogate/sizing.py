"""Surrogate estimates of cloud extents, COM excursions, and healing lengths for grid sizing."""
from typing import NamedTuple

import numpy as np

from ..potential import constants
from ..schedule import AtomChip, CurrentSchedule
from .ermakov import run_ermakov
from .gaussian import run_gaussian


class CloudGeometry(NamedTuple):
    # fmt: off
    extent_m: tuple[float, float, float]  # maximum sampled laboratory-axis half-extents [m]
    slosh_m : tuple[float, float, float]  # maximum control-node COM excursions per laboratory axis [m]
    xi_ref_m: float                       # initial central healing length [m]
    xi_min_m: float                       # minimum sampled central healing length [m]
    # fmt: on


def predict_cloud_geometry(sched: CurrentSchedule, chip: AtomChip, n_atoms: int,
                           T_s: float) -> CloudGeometry:
    """Predict spatial scales for one schedule and atom number; size_box applies the grid policy."""
    mu, tf_radii = sched.tf_quantities(chip.atom, n_atoms)                # (n_nodes,) [J], (n_nodes, 3) [m]
    R0 = np.asarray(tf_radii)[0]                                          # (3,) initial TF radii [m]
    mu0 = float(np.asarray(mu)[0])
    xi0 = float(constants.hbar / np.sqrt(2.0 * chip.atom.mass_kg * mu0))  # initial central healing length [m]
    V0 = np.asarray(sched.eigvecs)[0]                                     # (3, 3) initial axes (columns)

    breathing = run_ermakov(sched, T_s)
    cloud = np.einsum("tij,jk,k->tik", breathing.B_traj, V0, R0)          # (n_fine, 3, 3): B(t) V0 diag(R0)
    extent = np.sqrt((cloud ** 2).sum(axis=2)).max(axis=0)                # row norms give laboratory-axis half-extents

    slosh = run_gaussian(sched, T_s, chip.atom.mass_kg).max_abs_com_m
    xi_min = xi0 * np.sqrt(breathing.min_det_B)                           # ξ ∝ n^(-1/2), n ∝ 1/det(B)

    return CloudGeometry(
        extent_m=tuple(float(x) for x in extent),
        slosh_m=tuple(float(x) for x in slosh),
        xi_ref_m=xi0,
        xi_min_m=float(xi_min),
    )
