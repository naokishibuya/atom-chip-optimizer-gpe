"""Compare transport schedules using predicted delivery quality at different atom numbers."""
import json
import math
import os
from dataclasses import asdict, dataclass

import numpy as np

from ..potential import constants
from ..potential.trap import tf_quantities
from ..schedule import AtomChip, CurrentSchedule
from .gaussian import GaussianResult

# Identify the saved file formats and model so loaders can check compatibility.
HARMONIC_SCHEMA = "harmonic-scan-v2"
HARMONIC_MODEL = "gaussian-phase-space"
N_SCAN_SCHEMA = "n-scan-v3"

# Saved numerical fields and their metadata.
# Format: "field_name": (dimension_names, units, description)
_HARMONIC_ARRAYS = {
    "reg_values":      (("reg",), "1", "numeric Tikhonov λ"),
    "T_grid_s":        (("T",), "s", "requested transport times"),
    "end_com_m":       (("reg", "T", "xyz"), "m", "signed endpoint COM displacement ξ(T)"),
    "end_vel_ms":      (("reg", "T", "xyz"), "m/s", "endpoint velocity ⟨p⟩(T)/m (lab; trap held at rest)"),
    "max_abs_com_m":   (("reg", "T", "xyz"), "m", "max_j |ξ(t_j)| per axis over recorded control-node times"),
    "max_abs_vel_ms":  (("reg", "T", "xyz"), "m/s", "max_j |⟨p⟩(t_j)/m| per axis over recorded control-node times"),
    "cov_loss":        (("reg", "T"), "1", "1 - covariance factor of F_G (harmonic, interaction-free)"),
    "chirp_tr_end":    (("reg", "T"), "1", "endpoint transverse chirp [sym C_rp]/ħ"),
    "chirp_tr_max":    (("reg", "T"), "1", "node-sampled max |transverse chirp|"),
    "l_int_max":       (("reg", "T"), "1", "node-sampled max |internal angular momentum|/ħ"),
    "initial_axes":    (("xyz", "principal"), "1", "common initial trap axes V(0), columns"),
    "initial_omegas_hz": (("principal",), "Hz", "common initial trap frequencies"),
    "final_axes":      (("reg", "T", "xyz", "principal"), "1", "final trap axes V(T), columns"),
    "final_omegas_hz": (("reg", "T", "principal"), "Hz", "final trap frequencies (for TF widths)"),
}


# fmt: off
@dataclass
class HarmonicGrid:
    """Shared, atom-number-independent COM predictions and diagnostics from the Gaussian harmonic model."""
    reg_labels      : list[str]      # (n_reg,) "reg=<value>" labels, λ descending
    reg_values      : np.ndarray     # (n_reg) numeric λ
    T_grid_s        : np.ndarray     # (n_T) requested transport times [s]
    end_com_m       : np.ndarray     # (n_reg, n_T, 3)
    end_vel_ms      : np.ndarray     # (n_reg, n_T, 3)
    max_abs_com_m   : np.ndarray     # (n_reg, n_T, 3)
    max_abs_vel_ms  : np.ndarray     # (n_reg, n_T, 3)
    cov_loss        : np.ndarray     # (n_reg, n_T) harmonic covariance-channel loss
    chirp_tr_end    : np.ndarray     # (n_reg, n_T)
    chirp_tr_max    : np.ndarray     # (n_reg, n_T)
    l_int_max       : np.ndarray     # (n_reg, n_T)
    initial_axes    : np.ndarray     # (3, 3) common initial axes, columns in lab coordinates
    initial_omegas_hz: np.ndarray    # (3,) common initial frequencies [Hz]
    final_axes      : np.ndarray     # (n_reg, n_T, 3, 3) final axes, columns in lab coordinates
    final_omegas_hz : np.ndarray     # (n_reg, n_T, 3) final frequencies [Hz]
    n_nodes         : dict           # {label: {T-label: node count}}, saved in JSON
# fmt: on

    def save(self, npz_path: str, json_path: str, plan_refs: dict[str, dict[str, str]],
             control_dt_s: float) -> None:
        """Save harmonic arrays to NPZ and schema, units, and provenance to JSON."""
        np.savez_compressed(npz_path, reg_labels=np.array(self.reg_labels),
                            **{name: getattr(self, name) for name in _HARMONIC_ARRAYS})
        meta = {
            "schema": HARMONIC_SCHEMA,
            "model": HARMONIC_MODEL,
            "scope": "interaction-free harmonic surrogate; the mean sector is the Kohn COM "
                     "dynamics, the covariance channel is a cadence/schedule diagnostic and is "
                     "not an interacting GPE fidelity",
            "hold_convention": "node-centred half-first/full-interior/half-last held waveform",
            "control_dt_s": control_dt_s,
            "arrays": {
                name: {"shape": list(getattr(self, name).shape), "dims": list(dims),
                       "units": units, "description": description}
                for name, (dims, units, description) in _HARMONIC_ARRAYS.items()
            },
            "reg_labels": self.reg_labels,
            "reg_values": [float(v) for v in self.reg_values],
            "T_s": [float(T) for T in self.T_grid_s],
            "n_nodes": self.n_nodes,
            "plan_refs": plan_refs,
        }
        with open(json_path, "w") as f:
            json.dump(meta, f, indent=2)

    @classmethod
    def from_results(cls, reg_labels: list[str], T_grid_s: np.ndarray,
                     results: dict[tuple[str, float], GaussianResult],
                     schedules: dict[tuple[str, float], CurrentSchedule]) -> "HarmonicGrid":
        """Assemble summaries from one Gaussian result per (reg, T)."""
        def stack(get: "callable") -> np.ndarray:
            return np.array([[get(results[(label, float(T))], schedules[(label, float(T))])
                              for T in T_grid_s] for label in reg_labels])

        first = schedules[(reg_labels[0], float(T_grid_s[0]))]
        initial_axes = np.asarray(first.eigvecs, dtype=np.float64)[0]
        initial_omegas_hz = np.asarray(first.omegas_hz, dtype=np.float64)[0]
        for label in reg_labels:
            for T in T_grid_s:
                schedule = schedules[(label, float(T))]
                if (not np.array_equal(np.asarray(schedule.eigvecs)[0], initial_axes)
                        or not np.array_equal(np.asarray(schedule.omegas_hz)[0], initial_omegas_hz)):
                    raise ValueError("all schedules in a scan must share the same initial trap")

        return cls(
            reg_labels=list(reg_labels),
            reg_values=np.array([_reg_value(label) for label in reg_labels]),
            T_grid_s=np.asarray(T_grid_s, dtype=np.float64),
            end_com_m=stack(lambda r, s: r.end_com_m),
            end_vel_ms=stack(lambda r, s: r.end_vel_ms),
            max_abs_com_m=stack(lambda r, s: r.max_abs_com_m),
            max_abs_vel_ms=stack(lambda r, s: r.max_abs_vel_ms),
            cov_loss=stack(lambda r, s: r.cov_loss),
            chirp_tr_end=stack(lambda r, s: r.chirp_tr_end),
            chirp_tr_max=stack(lambda r, s: r.chirp_tr_max),
            l_int_max=stack(lambda r, s: r.l_int_max),
            initial_axes=initial_axes,
            initial_omegas_hz=initial_omegas_hz,
            final_axes=stack(lambda r, s: np.asarray(s.eigvecs)[-1]),
            final_omegas_hz=stack(lambda r, s: np.asarray(s.omegas_hz)[-1]),
            n_nodes={label: {repr(float(T)): int(np.asarray(schedules[(label, float(T))].r_mins_mm).shape[0])
                             for T in T_grid_s} for label in reg_labels},
        )


def load_harmonic_pair(npz_path: str, json_path: str) -> HarmonicGrid:
    """Load harmonic arrays and validate their metadata, shapes, dtypes, and control cadence."""
    for path in (npz_path, json_path):
        if not os.path.exists(path):
            raise ValueError(f"scan is incomplete: missing {path}")
    with open(json_path) as f:
        meta = json.load(f)
    if meta.get("schema") != HARMONIC_SCHEMA or meta.get("model") != HARMONIC_MODEL:
        raise ValueError(f"{json_path}: schema/model {meta.get('schema')!r}/{meta.get('model')!r} "
                         f"!= {HARMONIC_SCHEMA!r}/{HARMONIC_MODEL!r}; rerun `chip scan`")
    with np.load(npz_path) as d:
        missing = [name for name in ("reg_labels", *_HARMONIC_ARRAYS) if name not in d.files]
        if missing:
            raise ValueError(f"{npz_path}: missing array(s) {missing}")
        labels = [str(label) for label in d["reg_labels"]]
        arrays = {name: d[name] for name in _HARMONIC_ARRAYS}
    if labels != meta["reg_labels"]:
        raise ValueError(f"{npz_path} labels {labels} != {json_path} labels {meta['reg_labels']}")
    n_reg, n_T = len(labels), len(meta["T_s"])
    implied = {"reg_values": (n_reg,), "T_grid_s": (n_T,), "cov_loss": (n_reg, n_T),
               "chirp_tr_end": (n_reg, n_T), "chirp_tr_max": (n_reg, n_T), "l_int_max": (n_reg, n_T),
               "end_com_m": (n_reg, n_T, 3), "end_vel_ms": (n_reg, n_T, 3),
               "max_abs_com_m": (n_reg, n_T, 3), "max_abs_vel_ms": (n_reg, n_T, 3),
               "initial_axes": (3, 3), "initial_omegas_hz": (3,),
               "final_axes": (n_reg, n_T, 3, 3), "final_omegas_hz": (n_reg, n_T, 3)}
    for name, arr in arrays.items():
        declared = meta["arrays"].get(name, {}).get("shape")
        if list(arr.shape) != declared or arr.shape != implied[name]:
            raise ValueError(f"{npz_path}: {name} shape {list(arr.shape)} != declared {declared} "
                             f"/ implied {list(implied[name])}")
        if arr.dtype != np.float64:
            raise ValueError(f"{npz_path}: {name} dtype {arr.dtype} != float64")
    if not np.array_equal(arrays["reg_values"], np.array(meta["reg_values"])):
        raise ValueError(f"{npz_path}/{json_path}: reg_values disagree")
    if not np.array_equal(arrays["T_grid_s"], np.array(meta["T_s"])):
        raise ValueError(f"{npz_path}/{json_path}: T grids disagree")
    try:
        control_dt = float(meta["control_dt_s"])
        n_nodes = meta["n_nodes"]
        plan_refs = meta["plan_refs"]
    except (KeyError, TypeError, ValueError) as e:
        raise ValueError(f"{json_path}: incomplete cadence provenance") from e
    if not math.isfinite(control_dt) or control_dt <= 0.0:
        raise ValueError(f"{json_path}: invalid control_dt_s {control_dt!r}")
    expected_labels = set(labels)
    if set(n_nodes) != expected_labels or set(plan_refs) != expected_labels:
        raise ValueError(f"{json_path}: provenance regularizations disagree with the saved grid")
    t_keys = [repr(float(T)) for T in arrays["T_grid_s"]]
    for label in labels:
        if set(n_nodes[label]) != set(t_keys) or set(plan_refs[label]) != set(t_keys):
            raise ValueError(f"{json_path}: provenance durations for {label} disagree with the saved grid")
        for T, key in zip(arrays["T_grid_s"], t_keys, strict=True):
            count = n_nodes[label][key]
            n_float = float(T) / control_dt
            if (not isinstance(count, int) or count < 2
                    or abs(n_float - round(n_float)) > 1e-9 * max(1.0, n_float)
                    or count != round(n_float) + 1):
                raise ValueError(f"{json_path}: node count for {label}, T={key} disagrees with "
                                 "control_dt_s")
    return HarmonicGrid(reg_labels=labels, n_nodes=n_nodes, **arrays)


# fmt: off
@dataclass
class CellComparison:
    T_s            : float          # transport duration [s]
    best_reg       : str | None     # argmax_λ estimated F_COM; None on an exact tie
    best_F_COM     : float          # maximum estimated F_COM, including ties
    runner_up_reg  : str | None     # second-ranked λ; None on a tie or a single candidate
    runner_up_F_COM: float | None   # its estimated F_COM; None on a tie or a single candidate
    delta_F        : float | None   # best - runner-up; 0 on a tie, None for one candidate
    tied_regs      : list[str]      # exact co-winners, or [] for a unique winner
# fmt: on


# fmt: off
@dataclass
class NScan:
    """Atom-number-dependent F_COM scores derived from shared COM predictions and equilibrium TF widths."""
    n_atoms        : int             # atom number
    reg_labels     : list[str]       # (n_reg) same order as the source HarmonicGrid
    T_grid_s       : np.ndarray      # (n_T) requested transport times [s]
    sigma_k_m      : np.ndarray      # (n_reg, n_T, 3) final principal TF widths σ_k = R_TF/√7 [m]
    sigma_x_m      : np.ndarray      # (n_reg, n_T) final lab-x TF rms width [m]
    sigma_x0_m     : float           # common initial lab-x TF rms width [m]
    E_displacement : np.ndarray      # (n_reg, n_T) endpoint exponent Σ_k (ξ_k/2σ_k)²
    E_momentum     : np.ndarray      # (n_reg, n_T) endpoint exponent Σ_k (mσ_k v_k/ħ)²
    F_COM          : np.ndarray      # (n_reg, n_T) exp(-(E_displacement + E_momentum))
    cliff_T_s      : np.ndarray      # (n_reg,) largest sampled T with max-node |ξ_x| > σ_x0 [s]; NaN if none
# fmt: on

    def compare(self, t_index: int) -> "CellComparison":
        """Compare all regularizations at one transport duration, using their F_COM scores.

        It returns a CellComparison containing the winner, runner-up, their scores, and the score difference.

        If several candidates share exactly the highest score, it lists them in tied_regs and leaves best_reg=None.
        It does not use a tolerance to decide ties.
        """
        scores = {label: float(self.F_COM[r, t_index]) for r, label in enumerate(self.reg_labels)}
        ranked = sorted(scores, key=lambda label: (-scores[label], -_reg_value(label)))
        T_s = float(self.T_grid_s[t_index])
        best = ranked[0]
        winners = [label for label in ranked if scores[label] == scores[best]]
        if len(winners) > 1:
            return CellComparison(T_s, None, scores[best], None, None, 0.0, winners)
        if len(ranked) == 1:
            return CellComparison(T_s, best, scores[best], None, None, None, [])
        runner_up = ranked[1]
        return CellComparison(T_s, best, scores[best], runner_up, scores[runner_up],
                              scores[best] - scores[runner_up], [])

    def save(self, npz_path: str) -> None:
        np.savez_compressed(
            npz_path,
            n_atoms=np.array(self.n_atoms),
            reg_labels=np.array(self.reg_labels),
            T_grid_s=self.T_grid_s,
            sigma_k_m=self.sigma_k_m,
            sigma_x_m=self.sigma_x_m,
            sigma_x0_m=np.array(self.sigma_x0_m, dtype=np.float64),
            E_displacement=self.E_displacement,
            E_momentum=self.E_momentum,
            F_COM=self.F_COM,
            cliff_T_s=self.cliff_T_s,
        )

    @classmethod
    def load(cls, npz_path: str) -> "NScan":
        with np.load(npz_path) as d:
            return cls(
                n_atoms=int(d["n_atoms"]),
                reg_labels=[str(label) for label in d["reg_labels"]],
                T_grid_s=d["T_grid_s"],
                sigma_k_m=d["sigma_k_m"],
                sigma_x_m=d["sigma_x_m"],
                sigma_x0_m=float(d["sigma_x0_m"]),
                E_displacement=d["E_displacement"],
                E_momentum=d["E_momentum"],
                F_COM=d["F_COM"],
                cliff_T_s=d["cliff_T_s"],
            )


def load_n_scan_pair(npz_path: str, json_path: str, harmonic: HarmonicGrid,
                     chip: AtomChip) -> tuple[NScan, dict]:
    """Load one atom number’s scan.npz and scan.json, then check that:

    - The arrays and JSON metadata agree.
    - The regularizations and durations match the supplied HarmonicGrid.
    - Recalculating the scores from that HarmonicGrid and chip configuration reproduces the saved results.

    Returns the validated NScan and JSON metadata.
    Unlike NScan.load(), it checks consistency with both the companion JSON file and the shared harmonic results.
    """
    for path in (npz_path, json_path):
        if not os.path.exists(path):
            raise ValueError(f"scan is incomplete: missing {path}")
    with open(json_path) as f:
        payload = json.load(f)
    if payload.get("schema") != N_SCAN_SCHEMA:
        raise ValueError(f"{json_path}: schema {payload.get('schema')!r} != {N_SCAN_SCHEMA!r}; rerun `chip scan`")
    nscan = NScan.load(npz_path)

    n_reg, n_T = len(nscan.reg_labels), len(nscan.T_grid_s)
    expected_shapes = {
        "T_grid_s": (n_T,),
        "sigma_k_m": (n_reg, n_T, 3),
        "sigma_x_m": (n_reg, n_T),
        "E_displacement": (n_reg, n_T),
        "E_momentum": (n_reg, n_T),
        "F_COM": (n_reg, n_T),
        "cliff_T_s": (n_reg,),
    }
    for name, shape in expected_shapes.items():
        arr = getattr(nscan, name)
        if arr.shape != shape:
            raise ValueError(f"{npz_path}: {name} shape {arr.shape} != expected {shape}")
        if arr.dtype != np.float64:
            raise ValueError(f"{npz_path}: {name} dtype {arr.dtype} != float64")
    if not math.isfinite(nscan.sigma_x0_m) or nscan.sigma_x0_m <= 0.0:
        raise ValueError(f"{npz_path}: sigma_x0_m must be finite and positive")

    if payload["n_atoms"] != nscan.n_atoms:
        raise ValueError(f"{json_path} N={payload['n_atoms']} != {npz_path} N={nscan.n_atoms}")
    if payload["regs"] != nscan.reg_labels:
        raise ValueError(f"{json_path}/{npz_path}: reg labels disagree")
    if not np.array_equal(np.array(payload["T_s"]), nscan.T_grid_s):
        raise ValueError(f"{json_path}/{npz_path}: T grids disagree")
    if payload.get("sigma_x0_m") != nscan.sigma_x0_m:
        raise ValueError(f"{json_path}/{npz_path}: sigma_x0_m disagrees")
    if len(payload["comparisons"]) != n_T:
        raise ValueError(f"{json_path}: {len(payload['comparisons'])} comparisons for {n_T} T samples")
    for i, cell in enumerate(payload["comparisons"]):
        if cell != asdict(nscan.compare(i)):
            raise ValueError(f"{json_path}: comparison at T={float(nscan.T_grid_s[i]):g} disagrees "
                             "with the saved arrays")
    if list(payload["cliff_T_s"]) != nscan.reg_labels:
        raise ValueError(f"{json_path}: cliff_T_s labels disagree with reg labels")
    for label, cliff in zip(nscan.reg_labels, nscan.cliff_T_s, strict=True):
        json_cliff = payload["cliff_T_s"][label]
        if (json_cliff is None) != bool(np.isnan(cliff)) or (json_cliff is not None and json_cliff != cliff):
            raise ValueError(f"{json_path}: cliff_T_s[{label!r}] disagrees with the saved arrays")
    if nscan.reg_labels != harmonic.reg_labels:
        raise ValueError(f"{npz_path}: reg labels disagree with the harmonic artifact")
    if not np.array_equal(nscan.T_grid_s, harmonic.T_grid_s):
        raise ValueError(f"{npz_path}: T grid disagrees with the harmonic artifact")
    # Recompute to check consistency with the source; allow small relative rounding differences.
    fresh = derive_n_scan(harmonic, chip, nscan.n_atoms)
    for name in ("sigma_k_m", "sigma_x_m", "E_displacement", "E_momentum", "F_COM"):
        if not np.allclose(getattr(nscan, name), getattr(fresh, name), rtol=1e-12, atol=0.0):
            raise ValueError(f"{npz_path}: {name} does not rederive from the harmonic artifact; "
                             "the per-N scan belongs to a different run")
    if not math.isclose(nscan.sigma_x0_m, fresh.sigma_x0_m, rel_tol=1e-12, abs_tol=0.0):
        raise ValueError(f"{npz_path}: sigma_x0_m does not rederive from the harmonic artifact; "
                         "the per-N scan belongs to a different run")
    if not np.array_equal(np.isnan(nscan.cliff_T_s), np.isnan(fresh.cliff_T_s)) or \
            not np.allclose(np.nan_to_num(nscan.cliff_T_s), np.nan_to_num(fresh.cliff_T_s),
                            rtol=1e-12, atol=0.0):
        raise ValueError(f"{npz_path}: cliff_T_s does not rederive from the harmonic artifact")
    return nscan, payload


def derive_n_scan(harmonic: HarmonicGrid, chip: AtomChip, n_atoms: int) -> NScan:
    """Estimate F_COM from endpoint displacement and velocity using final TF rms widths.

    The overlap assumes equal Gaussian profiles with σ_k = R_TF,k/√7, not evolved covariances.
    Node-sampled maximum excursions are used only for the separate cliff diagnostic.
    """
    mass = float(chip.atom.mass_kg)
    angular = 2.0 * np.pi * harmonic.final_omegas_hz
    _, tf_radii = tf_quantities(angular, chip.atom, n_atoms)
    sigma_k = np.asarray(tf_radii) / np.sqrt(7.0)                            # (n_reg, n_T, 3)
    V = harmonic.final_axes
    # Rotate the principal-axis variances into the laboratory frame: V diag(σ²) Vᵀ.
    Sigma = np.einsum("rtak,rtk,rtbk->rtab", V, sigma_k ** 2, V)  # (n_reg, n_T, 3, 3)
    sigma_x = np.sqrt(Sigma[:, :, 0, 0])  # the x-axis RMS width for every (reg, T) combo

    _, initial_tf_radii = tf_quantities(
        2.0 * np.pi * harmonic.initial_omegas_hz, chip.atom, n_atoms)
    initial_sigma_k = np.asarray(initial_tf_radii) / np.sqrt(7.0)
    initial_sigma = harmonic.initial_axes @ np.diag(initial_sigma_k ** 2) @ harmonic.initial_axes.T
    sigma_x0 = float(np.sqrt(initial_sigma[0, 0]))

    # Express the final COM displacement and velocity along the final trap principal axes.
    xi_q = np.einsum("rtak,rta->rtk", V, harmonic.end_com_m)  # displacement in m (n_reg, n_T, 3)
    v_q = np.einsum("rtak,rta->rtk", V, harmonic.end_vel_ms)  # velocity in m/s (n_reg, n_T, 3)
    E_disp = np.sum((xi_q / (2.0 * sigma_k)) ** 2, axis=-1)                # F_COM contribution from the final displacement.
    E_mom = np.sum((mass * sigma_k * v_q / constants.hbar) ** 2, axis=-1)  # F_COM contribution from the final momentum.

    # Find the longest tested duration where the maximum x-axis COM excursion exceeds the initial TF RMS width.
    over = harmonic.max_abs_com_m[..., 0] > sigma_x0
    cliff = np.array([float(harmonic.T_grid_s[row].max()) if row.any() else float("nan")
                      for row in over])
    return NScan(
        n_atoms=n_atoms,
        reg_labels=list(harmonic.reg_labels),
        T_grid_s=harmonic.T_grid_s,
        sigma_k_m=sigma_k,
        sigma_x_m=sigma_x,
        sigma_x0_m=sigma_x0,
        E_displacement=E_disp,
        E_momentum=E_mom,
        F_COM=np.exp(-(E_disp + E_mom)),
        cliff_T_s=cliff,
    )


def _reg_value(label: str) -> float:
    """Parse a label such as 'reg=0.001' into its numeric value."""
    return float(label.split("=", 1)[1])
