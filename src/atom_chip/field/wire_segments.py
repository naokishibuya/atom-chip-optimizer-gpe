"""The Biot-Savart wire primitive: a batch of straight rectangular segments."""
from typing import NamedTuple

import jax.numpy as jnp

from .biot_savart import batched_biot_savart_rectangular


class WireSegments(NamedTuple):
    # fmt: off
    starts_mm : jnp.ndarray  # (N, 3)
    ends_mm   : jnp.ndarray  # (N, 3)
    widths_mm : jnp.ndarray  # (N,)
    heights_mm: jnp.ndarray  # (N,)
    # fmt: on

    def get_fields(self, points_mm: jnp.ndarray, currents_A: jnp.ndarray) -> jnp.ndarray:
        return batched_biot_savart_rectangular(
            points_mm, self.starts_mm, self.ends_mm, self.widths_mm, self.heights_mm, currents_A
        )

    def to_records(self) -> list[dict]:
        return [
            {"start_mm": s.tolist(), "end_mm": e.tolist(), "width_mm": float(w), "height_mm": float(h)}
            for s, e, w, h in zip(self.starts_mm, self.ends_mm, self.widths_mm, self.heights_mm, strict=True)
        ]

    @classmethod
    def from_records(cls, segments: list[dict]) -> "WireSegments":
        # fmt: off
        return cls(
            starts_mm =jnp.array([s["start_mm"] for s in segments] , dtype=jnp.float64),
            ends_mm   =jnp.array([s["end_mm"] for s in segments]   , dtype=jnp.float64),
            widths_mm =jnp.array([s["width_mm"] for s in segments] , dtype=jnp.float64),
            heights_mm=jnp.array([s["height_mm"] for s in segments], dtype=jnp.float64),
        )
        # fmt: on
