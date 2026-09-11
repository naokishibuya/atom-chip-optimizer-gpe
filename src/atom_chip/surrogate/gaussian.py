"""Gaussian mean and covariance propagation in a held quadratic trap."""
from dataclasses import dataclass

import numpy as np
from scipy.linalg import expm

from ..potential import constants
from ..schedule import CurrentSchedule

J6 = np.block([[np.zeros((3, 3)), np.eye(3)], [-np.eye(3), np.zeros((3, 3))]])


# fmt: off
@dataclass
class GaussianResult:
    T_total_s     : float          # transport time [s]
    l0_m          : float          # canonical coordinate scale [m]
    means         : np.ndarray     # (n, 6) canonical phase-space mean per node
    gammas        : np.ndarray     # (n, 6, 6) canonical covariance per node
    end_com_m     : np.ndarray     # (3,) endpoint ⟨r⟩-r_min per laboratory axis [m]
    end_vel_ms    : np.ndarray     # (3,) endpoint laboratory velocity [m/s]
    max_abs_com_m : np.ndarray     # (3,) control-node maxima of |⟨r⟩-r_min| per axis [m]
    max_abs_vel_ms: np.ndarray     # (3,) control-node maxima of |⟨p⟩/m| per axis [m/s]
    mean_loss     : float          # 1 - mean factor of F_G vs the harmonic target
    cov_loss      : float          # 1 - covariance factor of F_G vs the harmonic target
    chirp_tr_end  : float          # endpoint transverse chirp [sym C_rp]/ħ
    chirp_tr_max  : float          # maximum absolute chirp over control nodes
    l_int_end     : float          # endpoint |internal angular momentum|/ħ (antisym C_rp)
    l_int_max     : float          # maximum over control nodes
    sympl_defect  : float          # maximum entrywise |SᵀJS - J| over half-holds
# fmt: on


def run_gaussian(
    sched: CurrentSchedule,
    T_total_s: float,
    mass_kg: float,
) -> GaussianResult:
    """Propagate the initial harmonic ground state and compare with the final harmonic ground state."""
    omegas = 2.0 * np.pi * np.asarray(sched.omegas_hz, dtype=np.float64)
    V = np.asarray(sched.eigvecs, dtype=np.float64)
    # H has eigenvalues mω², so K = H/m = V diag(ω²) Vᵀ.
    K_nodes = np.einsum("tai,ti,tbi->tab", V, omegas ** 2, V)
    r_nodes = np.asarray(sched.r_mins_mm, dtype=np.float64) * 1e-3
    n = K_nodes.shape[0]
    dt_ctrl = T_total_s / (n - 1)

    # Default l0 uses the initial geometric-mean frequency; physical results are scale-independent.
    k0_evals = np.linalg.eigvalsh(K_nodes[0])  # computes only the eigenvalues of a symmetric/Hermitian matrix
    if np.any(k0_evals <= 0.0):
        raise ValueError(f"non-positive curvature eigenvalue {k0_evals.min():.3e}; model domain violated")
    wbar = float(np.prod(k0_evals) ** (1.0 / 6.0))
    l0 = float(np.sqrt(constants.hbar / (mass_kg * wbar)))  # harmonic oscillator length
    halves = [expm(_hold_generator(K_nodes[j], r_nodes[j], l0, mass_kg) * (dt_ctrl / 2.0))
              for j in range(n)]
    sympl_defect = max(np.max(np.abs(E[:6, :6].T @ J6 @ E[:6, :6] - J6)) for E in halves)

    means = np.empty((n, 6))      # average position and momentum
    gammas = np.empty((n, 6, 6))  # covariance
    z_aug = np.append(np.concatenate([r_nodes[0] / l0, np.zeros(3)]), 1.0)
    gam = ground_gamma_bar(K_nodes[0], l0, mass_kg)
    means[0], gammas[0] = z_aug[:6], gam
    for j in range(n - 1):
        for E in (halves[j], halves[j + 1]):
            z_aug = E @ z_aug
            S = E[:6, :6]
            gam = S @ gam @ S.T
        means[j + 1], gammas[j + 1] = z_aug[:6], gam

    target_mean = np.concatenate([r_nodes[-1] / l0, np.zeros(3)])
    target_gam = ground_gamma_bar(K_nodes[-1], l0, mass_kg)
    _, mean_f, cov_f = gaussian_fidelity(means[-1], gammas[-1], target_mean, target_gam)

    xi = means[:, :3] * l0 - r_nodes
    vel = means[:, 3:] * constants.hbar / (mass_kg * l0)
    chirp_tr, l_int = _channel_series(gammas, K_nodes, l0)
    return GaussianResult(
        T_total_s=T_total_s, l0_m=l0, means=means, gammas=gammas,
        end_com_m=xi[-1], end_vel_ms=vel[-1],
        max_abs_com_m=np.max(np.abs(xi), axis=0), max_abs_vel_ms=np.max(np.abs(vel), axis=0),
        mean_loss=1.0 - mean_f, cov_loss=1.0 - cov_f,
        chirp_tr_end=float(chirp_tr[-1]), chirp_tr_max=float(np.max(np.abs(chirp_tr))),
        l_int_end=float(l_int[-1]), l_int_max=float(np.max(l_int)),
        sympl_defect=float(sympl_defect),
    )


def ground_gamma_bar(K: np.ndarray, l0: float, mass_kg: float) -> np.ndarray:
    """Harmonic ground-state covariance in canonical coordinates (C_rp = 0)."""

    # K^(-1/2) = V diag(1/ω_k) Vᵀ
    # K^(+1/2) = V diag( ω_k ) Vᵀ
    gam = np.zeros((6, 6))
    gam[:3, :3] = constants.hbar / (2.0 * mass_kg) * _sym_pow(K, -0.5) / l0 ** 2
    gam[3:, 3:] = constants.hbar * mass_kg / 2.0 * _sym_pow(K, +0.5) * (l0 / constants.hbar) ** 2
    return gam


def gaussian_fidelity(mean1: np.ndarray, gam1: np.ndarray, mean2: np.ndarray,
                      gam2: np.ndarray) -> tuple[float, float, float]:
    """Return (squared overlap, mean factor, covariance factor) for pure canonical Gaussians."""
    total = gam1 + gam2
    sign, logdet = np.linalg.slogdet(total)
    if sign <= 0.0:
        raise ValueError("covariance sum is not positive definite")
    cov_factor = float(np.exp(-0.5 * logdet))
    delta = mean1 - mean2
    mean_factor = float(np.exp(-0.5 * delta @ np.linalg.solve(total, delta)))
    return cov_factor * mean_factor, mean_factor, cov_factor


def symplectic_eigenvalues(gam: np.ndarray) -> np.ndarray:
    """The three symplectic eigenvalues (1/2 each for a pure Gaussian)."""
    return np.sort(np.abs(np.linalg.eigvals(J6 @ gam).imag))[::2]


def _sym_pow(K: np.ndarray, power: float) -> np.ndarray:
    evals, evecs = np.linalg.eigh(K)
    if np.any(evals <= 0.0):
        raise ValueError(f"non-positive curvature eigenvalue {evals.min():.3e}; model domain violated")
    return (evecs * evals ** power) @ evecs.T


def _hold_generator(K: np.ndarray, r_min: np.ndarray, l0: float, mass_kg: float) -> np.ndarray:
    """Constant-hold generator for the augmented canonical mean (q, π, 1)."""
    wbar = constants.hbar / (mass_kg * l0 ** 2)
    M = np.zeros((7, 7))
    M[:3, 3:6] = wbar * np.eye(3)
    M[3:6, :3] = -K / wbar
    M[3:6, 6] = K @ r_min / (wbar * l0)
    return M


def _channel_series(gammas: np.ndarray, K_nodes: np.ndarray, l0: float) -> tuple[np.ndarray, np.ndarray]:
    """Per-node transverse chirp (largest magnitude, signed) and |internal L|/ħ.

    Chirp uses the cloud principal axes; the axial axis best aligns with the weakest trap axis.
    """
    n = gammas.shape[0]
    chirp_tr = np.empty(n)
    l_int = np.empty(n)
    for i in range(n):
        c_rr = gammas[i, :3, :3] * l0 ** 2
        c_rp = gammas[i, :3, 3:] * constants.hbar
        kevals, V = np.linalg.eigh(K_nodes[i])
        _, evecs = np.linalg.eigh(c_rr)
        k_ax = int(np.argmax(np.abs(evecs.T @ V[:, np.argmin(kevals)])))
        sym_rp = (c_rp + c_rp.T) / 2.0
        chirps = np.array([evecs[:, k] @ sym_rp @ evecs[:, k] for k in range(3)]) / constants.hbar
        trans = np.delete(chirps, k_ax)
        chirp_tr[i] = trans[np.argmax(np.abs(trans))]
        l_vec = np.array([c_rp[1, 2] - c_rp[2, 1], c_rp[2, 0] - c_rp[0, 2], c_rp[0, 1] - c_rp[1, 0]])
        l_int[i] = np.linalg.norm(l_vec) / constants.hbar
    return chirp_tr, l_int
