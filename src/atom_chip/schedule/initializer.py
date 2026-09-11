from itertools import pairwise

import jax.numpy as jnp

from ..field import BiasField, WireSegments
from ..potential import rb87
from .chip import AtomChip, LogicalWire

# fmt: off
#-------------------------------------------------------------------------------
# PCB
#-------------------------------------------------------------------------------
PCB_GLASS_HEIGHT        : float = 0.0   # Half-wave plate thickness
PCB_LAYER_TOP_HEIGHT    : float = 0.07  # PCB Top Layer tickness
PCB_CORE_HEIGHT         : float = 0.38  # PCB Core Layer tickness
PCB_LAYER_BOTTOM_HEIGHT : float = 0.07  # PCB Bottom Layer tickness
PCB_Y_OFFSET            : float = 0.0
PCB_X_OFFSET            : float = 0.0
PCB_TOP_OFFSET          : float = 0.0   # PCB top layer vertical offset from reflecting surface (after QWP fell)

#-------------------------------------------------------------------------------
# Shifting wires on chip for Transport
#-------------------------------------------------------------------------------
SHIFTING_WIRE_LENGTH    : float = 16.0  # Shifting wire length
SHIFTING_WIRE_WIDTH     : float = 0.3   # Shifting wire width
SHIFTING_WIRE_GAP       : float = 0.1   # gap between adjacent shifting wires
SHIFTING_WIRE_X_SLIDE   : float = SHIFTING_WIRE_WIDTH + SHIFTING_WIRE_GAP  # 0.4, center-to-center x-slide per wire
SHIFTING_WIRE_Y_SLIDE   : float = 0.5   # periodic y-slide per wire within a period (thesis Fig. 5)
SHIFTING_WIRE_X0        : float = -5.6  # leftmost wire x
SHIFTING_WIRE_Y0        : float = -1.2  # first y-slide offset within a period
N_SHIFTING_PERIODS      : int   = 5     # five periods of six shifting wires (thesis)
# Shifting wire currents (A) for the PCB: T1, T2, T3, T4, T5, T6
SHIFTING_WIRE_CURRENTS  : list[float] = [0.6, 1.05, -0.9, 1.05, 0.6, 0.0]

#-------------------------------------------------------------------------------
# Guiding wires for Quadrupole fields
#-------------------------------------------------------------------------------
GUIDING_WIRE_LENGTH     : float = 62.0  # Guiding wire length
PCB_PIN_LENGTH          : float = 50.0  # PCB pin length (leg length)
# Guiding wire currents (A) for the PCB: Q4, Q3, Q2, Q1, Q0, Q1', Q2', Q3', Q4'
GUIDING_WIRE_CURRENTS   : list[float] = [0.0, 13.79, 13.76, -3.78, -3.78, -3.78, 13.76, 13.79, 0.0]

#-------------------------------------------------------------------------------
# Bias fields
#-------------------------------------------------------------------------------
# Coil Current [A] to Field [G] Conversion:
BIAS_X_COIL_FACTOR  : float = -1.068  # [G/A]
BIAS_Y_COIL_FACTOR  : float =  1.8    # [G/A]
BIAS_Z_COIL_FACTOR  : float =  3.0    # [G/A]

# Coil currents [A] to be applied to the external coils
BIAS_X_COIL_CURRENT : float = 0.0
BIAS_Y_COIL_CURRENT : float = 0.0
BIAS_Z_COIL_CURRENT : float = 0.0

# Stray fields (G)
BIAS_X_STRAY_FIELD  : float = 0.0
BIAS_Y_STRAY_FIELD  : float = 0.0
BIAS_Z_STRAY_FIELD  : float = 0.0

# Bias field properties
COIL_FACTORS  = jnp.array([BIAS_X_COIL_FACTOR , BIAS_Y_COIL_FACTOR , BIAS_Z_COIL_FACTOR ], dtype=jnp.float64)
COIL_CURRENTS = jnp.array([BIAS_X_COIL_CURRENT, BIAS_Y_COIL_CURRENT, BIAS_Z_COIL_CURRENT], dtype=jnp.float64)
STRAY_FIELDS  = jnp.array([BIAS_X_STRAY_FIELD , BIAS_Y_STRAY_FIELD , BIAS_Z_STRAY_FIELD ], dtype=jnp.float64)

N_LOGICAL_SHIFTING = len(SHIFTING_WIRE_CURRENTS)
N_LOGICAL_GUIDING  = len(GUIDING_WIRE_CURRENTS)
N_LOGICAL          = N_LOGICAL_SHIFTING + N_LOGICAL_GUIDING
# fmt: on

# The shifting wires are the schedule's actuators (guiding wires stay fixed); default --wire-ids.
SHIFTING_WIRE_IDS: tuple[int, ...] = tuple(range(N_LOGICAL_SHIFTING))


def build_chip() -> AtomChip:
    """Construct the BEC transport AtomChip."""
    return AtomChip(
        name="BEC Transport",
        atom=rb87,
        wire_layout=_setup_wire_layout(),
        bias_field=_setup_bias_config(),
        x0_mm=[0.0, 0.0, 0.5],
        half_bounds_mm=[0.5, 0.5, 0.5],
    )


# fmt: off
def _setup_wire_layout():
    """Construct the wire layout for the BEC transport chip."""

    def place(segment: list, dz: float) -> None:
        """Shift a segment's two endpoints by the PCB placement offset (x_offset, y_offset, dz)."""
        for pt in (segment[0], segment[1]):
            pt[0] += PCB_X_OFFSET
            pt[1] += PCB_Y_OFFSET
            pt[2] += dz

    #------------------------------------------------------------------------------------------
    # Shifting wires for transport: 6 logical shifting wires (T1..T6 = "Transport")
    #------------------------------------------------------------------------------------------
    SL = SHIFTING_WIRE_LENGTH / 2
    SW = SHIFTING_WIRE_WIDTH
    SH = PCB_LAYER_TOP_HEIGHT

    shifting_wires = []
    for lw_idx in range(N_LOGICAL_SHIFTING):
        segments = []
        for period in range(N_SHIFTING_PERIODS):
            i  = period * N_LOGICAL_SHIFTING + lw_idx
            x  = SHIFTING_WIRE_X0 + i * SHIFTING_WIRE_X_SLIDE
            dy = SHIFTING_WIRE_Y0 + lw_idx * SHIFTING_WIRE_Y_SLIDE
            start, end = [x, -SL + dy, 0], [x, SL + dy, 0]
            if period % 2 == 1:
                start, end = end, start
            segments.append([start, end, SW, SH])
        shifting_wires.append(segments)

    top_offset_mm = -(
        PCB_GLASS_HEIGHT +
        PCB_TOP_OFFSET   +
        PCB_LAYER_TOP_HEIGHT / 2
    )

    for wire in shifting_wires:
        for segment in wire:
            place(segment, top_offset_mm)

    #------------------------------------------------------------------------------------------
    # Guiding wires for quadrupole trap: 9 logical guiding wires (Q4..Q0, Q1'..Q4')
    #------------------------------------------------------------------------------------------
    GL = GUIDING_WIRE_LENGTH / 2
    GH = PCB_LAYER_BOTTOM_HEIGHT
    PL = PCB_PIN_LENGTH

    def guiding_wire(waypoints: list[tuple[float, float]], width: float, y_leg: float | None = None) -> list:
        """One quadrupole guiding wire from its LEFT half-waypoints (x, y at z=0), outer to
        central-left. Right half is the x-mirror; PCB legs (z: -PL..0) at x=+-40.2 if y_leg given."""
        pts = list(waypoints) + [(-x, y) for x, y in reversed(waypoints)]
        body = [[[x0, y0, 0], [x1, y1, 0], width, GH] for (x0, y0), (x1, y1) in pairwise(pts)]
        if y_leg is None:
            return body
        left_leg  = [[-40.2, y_leg, -PL], [-40.2, y_leg,  0 ], 1.0, GH]
        right_leg = [[ 40.2, y_leg,  0 ], [ 40.2, y_leg, -PL], 1.0, GH]
        return [left_leg, *body, right_leg]

    def mirror_y(wire: list) -> list:
        """Reflect a wire across y=0 (a Qn -> Qn' prime)."""
        return [[[s[0], -s[1], s[2]], [e[0], -e[1], e[2]], w, h] for s, e, w, h in wire]

    # "Q" = quadrupole: they form the static quadrupole trap for lateral confinement.
    Q4 = guiding_wire([(-GL, 4.9)], 1.5)
    Q3 = guiding_wire([
            (-GL-2.03-1.8-6.6, 3.05+1.78+2.6+6.6),
            (-GL-2.03-1.8, 3.05+1.78+2.6),
            (-GL-2.03-1.8, 3.05+1.78),
            (-GL-2.03, 3.05),
        ], 2.0, y_leg=14)
    Q2 = guiding_wire([
            (-GL-3.2-6.4, 1.45+6.65),
            (-GL-3.2, 1.45),
        ], 1.0, y_leg=8.47)
    Q1 = guiding_wire([
            (-GL-3.875-5, 0.6+1.8),
            (-GL-3.875, 0.6),
        ], 0.5, y_leg=3.7)
    Q0 = guiding_wire([(-72/2-2.9, 0.0)], 0.5, y_leg=0.0)

    guiding_wires = [Q4, Q3, Q2, Q1, Q0, mirror_y(Q1), mirror_y(Q2), mirror_y(Q3), mirror_y(Q4)]

    bottom_offset_mm = -(
        PCB_GLASS_HEIGHT     +
        PCB_TOP_OFFSET       +
        PCB_LAYER_TOP_HEIGHT +
        PCB_CORE_HEIGHT      +
        PCB_LAYER_BOTTOM_HEIGHT / 2
    )

    for wire in guiding_wires:
        for segment in wire:
            place(segment, bottom_offset_mm)

    # Both families are lists of wires (each wire a list of segments).
    def make_lws(wires: list, currents_A: list[float], material: str, id0: int) -> list[LogicalWire]:
        logical_wires = []
        for k, wire in enumerate(wires):
            starts, ends, widths, heights = zip(*wire, strict=True)
            wire_segments = WireSegments(
                starts_mm  = jnp.array(starts, dtype=jnp.float64),
                ends_mm    = jnp.array(ends, dtype=jnp.float64),
                widths_mm  = jnp.array(widths, dtype=jnp.float64),
                heights_mm = jnp.array(heights, dtype=jnp.float64),
            )
            logical_wires.append(LogicalWire(
                id        = id0 + k,
                material  = material,
                current_A = currents_A[k],
                segments  = wire_segments,
            ))
        return logical_wires

    return {
        "shifting": make_lws(shifting_wires, SHIFTING_WIRE_CURRENTS, "gold",   0),
        "guiding":  make_lws(guiding_wires,  GUIDING_WIRE_CURRENTS,  "copper", N_LOGICAL_SHIFTING),
    }


def _setup_bias_config(
    coil_currents_A     : jnp.ndarray = COIL_CURRENTS,
    coil_factors_G_per_A: jnp.ndarray = COIL_FACTORS,
    stray_fields_G      : jnp.ndarray = STRAY_FIELDS,
) -> BiasField:
    return BiasField(
        coil_currents_A      = coil_currents_A,
        coil_factors_G_per_A = coil_factors_G_per_A,
        stray_field_G        = stray_fields_G,
    )
# fmt: on
