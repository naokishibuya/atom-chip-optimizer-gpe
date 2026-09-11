"""The chip's reference-trap + cloud characterization, rendered as the `info.txt` text report."""
import jax.numpy as jnp
import numpy as np

from ..potential import GaussianAnalysis, TFAnalysis, TrapGeometry, analyze_gaussian, analyze_tf
from .chip import AtomChip, Minimum

_RULE = "-" * 76


def format_trap(chip: AtomChip) -> str:
    """The chip's reference-trap characterization, N-independent: field/trap minimum, frequencies, Hessian.

    Written once at the init level (`chip init`); the per-N cloud is `format_cloud`.
    """
    field_min, field_geom = chip.analyze_field_minimum()
    trap_min, trap_geom = chip.analyze_trap_minimum()
    return _format_trap_analysis(chip, field_min, field_geom, trap_min, trap_geom)


def format_cloud(chip: AtomChip, n_atoms: int) -> str:
    """The predicted BEC + Thomas-Fermi cloud for `n_atoms` in the chip's reference trap (N-dependent)."""
    trap_min, trap_geom = chip.analyze_trap_minimum()
    if not trap_min.found:
        return ""
    # T=0 assumption: all atoms condensed, so total_atoms = condensed_atoms = n_atoms.
    bec = analyze_gaussian(trap_geom.trap, chip.atom, total_atoms=n_atoms)
    tf = analyze_tf(trap_geom.trap, chip.atom, condensed_atoms=n_atoms)
    return _format_cloud_analysis(n_atoms, bec, tf)


def _format_trap_analysis(
    atom_chip: AtomChip,
    field_min: Minimum,
    field_geom: TrapGeometry | None,
    trap_min: Minimum,
    trap_geom: TrapGeometry | None,
) -> str:
    atom = atom_chip.atom
    bias = atom_chip.bias_field
    return f"""\
Trap Analysis ({atom.name} on '{atom_chip.name}'; N-independent)

Atom Species
{_RULE}
Name                              : {atom.name}
Mass                         [kg] : {_format_value(atom.mass_kg)}
Landé g-factor                    : {atom.g_F}
Magnetic quantum number mF        : {atom.m_F}
s-wave scattering length      [m] : {_format_value(atom.scattering_length_m)}

Bias Field Parameters
{_RULE}
Coil factors                [G/A] : {_format_array(bias.coil_factors_G_per_A)}
Coil currents                 [A] : {_format_array(bias.coil_currents_A)}
Stray fields                  [G] : {_format_array(bias.stray_field_G)}

{_format_field_section(field_min, field_geom)}
{_format_trap_section(trap_min, trap_geom)}"""


def _format_cloud_analysis(n_atoms: int, bec: GaussianAnalysis, tf: TFAnalysis) -> str:
    return f"""\
BEC Parameters [N = {_format_count(n_atoms)}] (Harmonic Oscillator Approximation)
{_RULE}
HO Length a_ho               [μm] : {_format_value(bec.a_ho_m, unit=1e6)}
Trap Frequency G-Avg ω_ho [rad/s] : {_format_value(bec.omega_ho_rad)}

Non-interacting           [atoms] : {_format_count(bec.total_atoms)}
Chemical Potential μ0         [J] : {_format_value(bec.mu_0_J)}
HO Length per axis           [μm] : {_format_array(bec.a_ho_axes_m, unit=1e6)}
Critical Temperature         [nK] : {_format_value(bec.critical_temperature_K, unit=1e9)}

Thomas-Fermi              [atoms] : {_format_count(tf.condensed_atoms)}
Chemical Potential μ          [J] : {_format_value(tf.mu_J)}
Thomas-Fermi Radii           [μm] : {_format_array(tf.radii_m, unit=1e6)}
"""


def _format_field_section(minimum: Minimum, geometry: TrapGeometry | None) -> str:
    if geometry is None:
        return f"Magnetic Field Minimum\n{_RULE}\nN/A, {minimum.message}\n"
    return f"""\
Magnetic Field Minimum
{_RULE}
Field Minimum                 [G] : {_format_value(minimum.value)}
Minimum Location             [mm] : {_format_array(minimum.position)}
Larmor frequency            [MHz] : {_format_value(geometry.larmor.omega_hz, unit=1e-6)}
Trap frequencies             [Hz] : {_format_array(geometry.trap.omega_hz)}

Hessian Eigenvalues and Eigenvectors:
{_format_array(geometry.hessian.eigenvalues)}
{_format_matrix(geometry.hessian.eigenvectors)}
"""


def _format_trap_section(minimum: Minimum, geometry: TrapGeometry | None) -> str:
    if geometry is None:
        return f"Trap Potential Minimum\n{_RULE}\nN/A, {minimum.message}\n"
    return f"""\
Trap Potential Minimum
{_RULE}
Potential Minimum             [J] : {_format_value(minimum.value)}
Minimum Location             [mm] : {_format_array(minimum.position)}
Larmor frequency            [MHz] : {_format_value(geometry.larmor.omega_hz, unit=1e-6)}
Trap frequencies             [Hz] : {_format_array(geometry.trap.omega_hz)}

Hessian Eigenvalues and Eigenvectors:
{_format_array(geometry.hessian.eigenvalues)}
{_format_matrix(geometry.hessian.eigenvectors)}
"""


def _format_count(count: int) -> str:
    return f"{int(count):,d}"


def _format_value(value: float, precision: int = 4, unit: float = 1.0) -> str:
    if unit != 1.0:
        value = value * unit
    if abs(value) < 1e-3 or abs(value) >= 1e4:
        return f"{value:.{precision}e}"
    return f"{value:.{precision}f}"


def _format_matrix(matrix: jnp.ndarray, precision: int = 4) -> str:
    return "\n".join(
        _format_array(matrix[i], precision=precision).replace("[", "|").replace("]", "|")
        for i in range(len(matrix))
    )


def _format_array(array: jnp.ndarray, precision: int = 4, unit: float = 1.0) -> str:
    arr = np.asarray(array)
    if unit != 1.0:
        arr = arr * unit
    return np.array2string(
        arr,
        formatter={"float_kind": lambda x: f"{x: {precision + 6}.{precision}g}"},
        separator=" ",
    )
