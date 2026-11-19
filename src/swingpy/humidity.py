"""
Water vapour: saturation pressure, relative humidity and dew point.

Flue gases and air carry water, which zeolites take up far more strongly
than CO2. These helpers turn a relative humidity into a mole fraction for
the feed composition, and back.

The saturation pressure follows the Buck equation over liquid water,
accurate to about 0.05 % between 0 and 100 C.
"""

from __future__ import annotations

import numpy as np


def saturation_pressure(temperature):
    """Saturation pressure of water over a flat liquid surface, bar, for a temperature in K (Buck equation)."""
    t = np.asarray(temperature, dtype=float) - 273.15
    return 0.0061121 * np.exp((18.678 - t / 234.5) * (t / (257.14 + t)))


def mole_fraction(relative_humidity, temperature, pressure):
    """
    Mole fraction of water in a gas of a given relative humidity.

    Args:
        relative_humidity: Between 0 and 1.
        temperature: K.
        pressure: Total pressure, bar.
    """
    if not 0.0 <= relative_humidity <= 1.0:
        raise ValueError(f"Relative humidity must lie between 0 and 1, got {relative_humidity}.")
    return float(relative_humidity * saturation_pressure(temperature) / pressure)


def relative_humidity(water_mole_fraction, temperature, pressure):
    """Relative humidity (0 to 1, above 1 if supersaturated) of a gas with a given water mole fraction."""
    return float(water_mole_fraction * pressure / saturation_pressure(temperature))


def dew_point(water_mole_fraction, pressure):
    """Temperature at which the gas becomes saturated with water, K (inverse of the Buck equation, by Newton)."""
    target = water_mole_fraction * pressure
    if target <= 0.0:
        raise ValueError("The gas holds no water, so it has no dew point.")
    T = 300.0
    for _ in range(50):
        f = float(saturation_pressure(T)) - target
        slope = (float(saturation_pressure(T + 1e-3)) - float(saturation_pressure(T - 1e-3))) / 2e-3
        step = f / slope
        T -= step
        if abs(step) < 1e-9:
            break
    return T


def humid_composition(dry_composition, relative_humidity_value, temperature, pressure):
    """
    Composition of a gas made humid: the dry mole fractions scaled down and
    water added at the given relative humidity.

    Args:
        dry_composition: Mole fractions of the dry gas, by component.
        relative_humidity_value: Between 0 and 1.
        temperature: K.
        pressure: bar.
    """
    water = mole_fraction(relative_humidity_value, temperature, pressure)
    total = sum(dry_composition.values())
    humid = {k: v / total * (1.0 - water) for k, v in dry_composition.items()}
    humid["H2O"] = humid.get("H2O", 0.0) + water
    return humid
