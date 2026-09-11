from typing import Final

mu_B: Final[float] = 9.274009994e-24  # Bohr magneton [J/T]
g: Final[float] = 9.81  # gravitational acceleration [m/s²]
k_B: Final[float] = 1.38065e-23  # Boltzmann constant [J/K]
h: Final[float] = 6.62607015e-34  # Planck constant [J·s]
hbar: Final[float] = 1.0545718e-34  # reduced Planck constant [J·s]


def joule_to_microKelvin(energy_J: float) -> float:
    return energy_J * 1e6 / k_B


def microKelvin_to_joule(temperature_uK: float) -> float:
    return temperature_uK * k_B / 1e6
