import json
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import optimistix as optx

from ..field import BiasField, WireSegments
from ..potential import (
    Atom,
    TrapGeometry,
    analyze_field,
    analyze_trap,
    find_field_minimum,
    find_trap_minimum,
    trap_magnetic_fields,
    trap_potential_energies,
)


@dataclass
class Minimum:
    """Plain dataclass (not a pytree) so JIT misuse fails fast."""
    found: bool
    value: jnp.ndarray     # unit-agnostic
    position: jnp.ndarray  # unit-agnostic
    message: str

    @staticmethod
    def from_solution(sol: optx.Solution, x: jnp.ndarray, value: jnp.ndarray) -> "Minimum":
        found = bool(sol.result == optx.RESULTS.successful)
        return Minimum(
            found=found,
            value=value,
            position=x,
            message="" if found else str(sol.result),
        )


@dataclass(frozen=True)
class LogicalWire:
    id: int
    material: str
    current_A: float
    segments: WireSegments

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "material": self.material,
            "current_A": self.current_A,
            "segments": self.segments.to_records(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "LogicalWire":
        return cls(
            id=data["id"],
            material=data["material"],
            current_A=data["current_A"],
            segments=WireSegments.from_records(data["segments"]),
        )


class AtomChip:
    """Atom + wires + bias + initial guess. `search_*`/`analyze_*` are NOT JIT-safe."""

    def __init__(
        self,
        name: str,
        atom: Atom,
        wire_layout: dict[str, list[LogicalWire]],
        bias_field: BiasField,
        x0_mm: jnp.ndarray,           # initial guess for the trap minimum (3,) [mm]
        half_bounds_mm: jnp.ndarray,  # half-bounds for the trap minimum search (3,) [mm]
    ):
        self.name = name
        self.atom = atom
        self.wire_layout = wire_layout
        self.bias_field = bias_field
        self.x0_mm = jnp.asarray(x0_mm, dtype=jnp.float64)
        self.half_bounds_mm = jnp.asarray(half_bounds_mm, dtype=jnp.float64)
        self.geometry, self.currents_A, self.lw_indices = _flatten_wires(wire_layout)

    def get_fields(self, points_mm: jnp.ndarray) -> jnp.ndarray:
        points_mm = jnp.atleast_2d(points_mm).astype(jnp.float64)
        return jax.jit(trap_magnetic_fields)(points_mm, self.geometry, self.currents_A, self.bias_field)

    def get_potentials(self, points_mm: jnp.ndarray) -> jnp.ndarray:
        points_mm = jnp.atleast_2d(points_mm).astype(jnp.float64)
        return jax.jit(trap_potential_energies)(points_mm, self.atom, self.geometry, self.currents_A, self.bias_field)

    def search_field_minimum(self) -> Minimum:
        x, val, sol = find_field_minimum(
            self.geometry,
            self.currents_A,
            self.bias_field,
            self.x0_mm,
            self.half_bounds_mm,
        )
        return Minimum.from_solution(sol, x, val)

    def search_trap_minimum(self) -> Minimum:
        # Seed from the field minimum (gravity sag is small).
        field_min = self.search_field_minimum()
        seed = field_min.position if field_min.found else self.x0_mm
        x, val, sol = find_trap_minimum(
            self.atom,
            self.geometry,
            self.currents_A,
            self.bias_field,
            seed,
            self.half_bounds_mm,
        )
        return Minimum.from_solution(sol, x, val)

    def analyze_field_minimum(self) -> tuple[Minimum, TrapGeometry | None]:
        minimum = self.search_field_minimum()
        if not minimum.found:
            return minimum, None
        geom = analyze_field(self.atom, self.geometry, self.currents_A, self.bias_field, minimum.position)
        return minimum, geom

    def analyze_trap_minimum(self) -> tuple[Minimum, TrapGeometry | None]:
        minimum = self.search_trap_minimum()
        if not minimum.found:
            return minimum, None
        geom = analyze_trap(self.atom, self.geometry, self.currents_A, self.bias_field, minimum.position)
        return minimum, geom

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "atom": self.atom._asdict(),
            "x0": self.x0_mm.tolist(),
            "half_bounds": self.half_bounds_mm.tolist(),
            "wire_layout": {name: [lw.to_dict() for lw in group]
                            for name, group in self.wire_layout.items()},
            "bias_field": self.bias_field.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "AtomChip":
        return cls(
            name=data["name"],
            atom=Atom(**data["atom"]),
            wire_layout={name: [LogicalWire.from_dict(d) for d in group]
                         for name, group in data["wire_layout"].items()},
            bias_field=BiasField.from_dict(data["bias_field"]),
            x0_mm=data["x0"],
            half_bounds_mm=data["half_bounds"],
        )

    @classmethod
    def load(cls, path: str) -> "AtomChip":
        with open(path) as f:
            return cls.from_dict(json.load(f))

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            f.write(_format_json(self.to_dict()))
            f.write("\n")


def _flatten_wires(
    wire_layout: dict[str, list[LogicalWire]],
) -> tuple[WireSegments, jnp.ndarray, jnp.ndarray]:
    """Lower the logical wire groups to chip-wide per-segment (geometry, currents, lw_indices)."""
    all_starts, all_ends, all_widths, all_heights = [], [], [], []
    I_logical_A, all_lw_indices = [], []
    for group in wire_layout.values():
        for logical_wire in group:
            segs = logical_wire.segments
            all_starts.append(segs.starts_mm)
            all_ends.append(segs.ends_mm)
            all_widths.append(segs.widths_mm)
            all_heights.append(segs.heights_mm)
            I_logical_A.append(logical_wire.current_A)
            n_segs = segs.starts_mm.shape[0]
            all_lw_indices.append(jnp.full(n_segs, logical_wire.id, dtype=jnp.int32))
    geometry = WireSegments(
        starts_mm=jnp.concatenate(all_starts),
        ends_mm=jnp.concatenate(all_ends),
        widths_mm=jnp.concatenate(all_widths),
        heights_mm=jnp.concatenate(all_heights),
    )
    I_logical_A = jnp.array(I_logical_A, dtype=jnp.float64)
    lw_indices = jnp.concatenate(all_lw_indices)
    return geometry, I_logical_A[lw_indices], lw_indices


def _format_json(data: Any, indent: int = 2, level: int = 0) -> str:
    pad = " " * (indent * level)
    inner = " " * (indent * (level + 1))
    if isinstance(data, dict):
        if not data:
            return "{}"
        parts = [f"{inner}{json.dumps(k, ensure_ascii=False)}: {_format_json(v, indent, level + 1)}"
                 for k, v in data.items()]
        return "{\n" + ",\n".join(parts) + "\n" + pad + "}"
    if isinstance(data, list):
        if not data:
            return "[]"
        if all(not isinstance(x, (dict, list)) for x in data):
            return json.dumps(data, ensure_ascii=False)
        if all(_is_simple_dict(x) for x in data):
            parts = [inner + json.dumps(x, ensure_ascii=False) for x in data]
            return "[\n" + ",\n".join(parts) + "\n" + pad + "]"
        parts = [inner + _format_json(x, indent, level + 1) for x in data]
        return "[\n" + ",\n".join(parts) + "\n" + pad + "]"
    return json.dumps(data, ensure_ascii=False)


def _is_simple_dict(x: Any) -> bool:
    if not isinstance(x, dict):
        return False
    for v in x.values():
        if isinstance(v, dict):
            return False
        if isinstance(v, list) and any(isinstance(item, (dict, list)) for item in v):
            return False
    return True
