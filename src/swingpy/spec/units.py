"""
Quantities with units for problem specifications.

A value is either a plain number, taken to be in the base unit of its kind
(the unit SwingPy uses internally, listed in ``BASE_UNITS``), or a string
"<number> <unit>", for example "1.2 m", "25 C" or "36.6 kJ/mol". The unit
must belong to the kind of quantity expected; anything else is an error.
"""

from __future__ import annotations

import re

from ..boundaries import STANDARD_MOLAR_VOLUME

"""J/(mol K), the gas constant used throughout SwingPy."""
R = 8.314

""": kind: {unit: (factor, offset)}, value_in_base = factor * value + offset."""
"""SLPM is the standard litre per minute at 0 C and 1 atm, as in swingpy.boundaries.standard_litres_per_minute."""
UNITS = {
    "length": {"m": (1.0, 0.0), "cm": (1e-2, 0.0), "mm": (1e-3, 0.0), "um": (1e-6, 0.0), "in": (0.0254, 0.0),
               "ft": (0.3048, 0.0)},
    "area": {"m2": (1.0, 0.0), "cm2": (1e-4, 0.0)},
    "volume": {"m3": (1.0, 0.0), "L": (1e-3, 0.0), "cm3": (1e-6, 0.0)},
    "pressure": {"bar": (1.0, 0.0), "Pa": (1e-5, 0.0), "kPa": (1e-2, 0.0), "MPa": (10.0, 0.0), "mbar": (1e-3, 0.0),
                 "atm": (1.01325, 0.0), "psi": (0.0689476, 0.0), "Torr": (1.01325 / 760.0, 0.0)},
    "temperature": {"K": (1.0, 0.0), "C": (1.0, 273.15), "degC": (1.0, 273.15)},
    "time": {"s": (1.0, 0.0), "min": (60.0, 0.0), "h": (3600.0, 0.0)},
    "rate": {"1/s": (1.0, 0.0), "1/min": (1 / 60.0, 0.0)},
    "velocity": {"m/s": (1.0, 0.0), "cm/s": (1e-2, 0.0)},
    "volumetric_flow": {"m3/h": (1.0, 0.0), "m3/s": (3600.0, 0.0), "L/min": (0.06, 0.0), "L/s": (3.6, 0.0)},
    "molar_flow": {"mol/s": (1.0, 0.0), "mmol/s": (1e-3, 0.0), "kmol/h": (1000.0 / 3600.0, 0.0),
                   "SLPM": (1.0 / 1000.0 / STANDARD_MOLAR_VOLUME / 60.0, 0.0)},
    "molar_energy": {"J/mol": (1.0, 0.0), "kJ/mol": (1e3, 0.0), "kcal/mol": (4184.0, 0.0)},
    "mass_density": {"kg/m3": (1.0, 0.0), "g/cm3": (1e3, 0.0), "g/L": (1.0, 0.0)},
    "specific_heat": {"J/(kg K)": (1.0, 0.0), "kJ/(kg K)": (1e3, 0.0), "J/(g K)": (1e3, 0.0)},
    "molar_heat_capacity": {"J/(mol K)": (1.0, 0.0), "kJ/(mol K)": (1e3, 0.0)},
    "viscosity": {"Pa s": (1.0, 0.0), "mPa s": (1e-3, 0.0), "cP": (1e-3, 0.0), "uPa s": (1e-6, 0.0)},
    "diffusivity": {"m2/s": (1.0, 0.0), "cm2/s": (1e-4, 0.0)},
    "molar_mass": {"kg/mol": (1.0, 0.0), "g/mol": (1e-3, 0.0)},
    "loading": {"mol/kg": (1.0, 0.0), "mmol/g": (1.0, 0.0)},
    "inverse_concentration": {"m3/mol": (1.0, 0.0), "L/mol": (1e-3, 0.0), "cm3/mol": (1e-6, 0.0)},
    "inverse_pressure": {"1/bar": (1.0, 0.0), "1/Pa": (1e5, 0.0), "1/kPa": (100.0, 0.0), "1/MPa": (0.1, 0.0),
                         "1/atm": (1 / 1.01325, 0.0)},
    "henry_constant": {"mol/(kg bar)": (1.0, 0.0), "mol/(kg Pa)": (1e5, 0.0), "mmol/(g bar)": (1.0, 0.0),
                       "mol/(kg kPa)": (100.0, 0.0)},
    "specific_area": {"m2/kg": (1.0, 0.0), "m2/g": (1e3, 0.0)},
    "virial_a": {"m2/mol": (1.0, 0.0)},
    "virial_a_temperature": {"m2 K/mol": (1.0, 0.0)},
    "virial_b": {"m4/mol2": (1.0, 0.0)},
    "virial_b_temperature": {"m4 K/mol2": (1.0, 0.0)},
    "heat_transfer": {"W/(m2 K)": (1.0, 0.0)},
    "conductivity": {"W/(m K)": (1.0, 0.0)},
    "dimensionless": {"": (1.0, 0.0), "-": (1.0, 0.0), "%": (0.01, 0.0)},
}
BASE_UNITS = {kind: next(iter(units)) for kind, units in UNITS.items()}

_ALIASES = {"J/mol/K": "J/(mol K)", "J/kg/K": "J/(kg K)", "kJ/kg/K": "kJ/(kg K)", "J/g/K": "J/(g K)",
            "kJ/mol/K": "kJ/(mol K)", "W/m2/K": "W/(m2 K)", "W/m/K": "W/(m K)", "Pa.s": "Pa s", "Pas": "Pa s",
            "°C": "C", "m^2/s": "m2/s", "m^3/mol": "m3/mol", "kg/m^3": "kg/m3", "m^3/h": "m3/h",
            "mol/kg/bar": "mol/(kg bar)", "mol/kg/Pa": "mol/(kg Pa)", "m^2/g": "m2/g", "m^2/kg": "m2/kg"}
_NUMBER = re.compile(r"^\s*([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)\s*(.*?)\s*$")


class UnitError(ValueError):
    """A value whose unit is missing, unknown or of the wrong kind."""


def quantity(value, kind):
    """
    The value in the base unit of ``kind``.

    Args:
        value: A number (already in the base unit) or a string "<number> <unit>".
        kind: A key of ``UNITS``.
    """
    if isinstance(value, bool):
        raise UnitError(f"expected {_article(kind)}, got {value!r}")
    if isinstance(value, (int, float)):
        return float(value)
    if not isinstance(value, str):
        raise UnitError(f"expected {_article(kind)}, got {value!r}")
    match = _NUMBER.match(value)
    if not match:
        raise UnitError(f"cannot read {value!r} as a number with a unit")
    number, unit = float(match.group(1)), _ALIASES.get(match.group(2), match.group(2))
    table = UNITS[kind]
    if unit not in table:
        found = [k for k, units in UNITS.items() if unit in units]
        hint = f" (that is {_article(found[0])} unit)" if found else ""
        raise UnitError(f"unit {match.group(2)!r} is not {_article(kind)} unit{hint}; "
                        f"use one of {', '.join(u for u in table if u)}")
    factor, offset = table[unit]
    return factor * number + offset


def _article(kind):
    words = kind.replace("_", " ")
    return ("an " if words[0] in "aeiou" else "a ") + words


def kind_of(value, kinds):
    """The first of ``kinds`` whose units include the unit of ``value`` (a string), or None."""
    if not isinstance(value, str):
        return None
    match = _NUMBER.match(value)
    unit = _ALIASES.get(match.group(2), match.group(2)) if match else None
    return next((kind for kind in kinds if unit in UNITS[kind]), None)
