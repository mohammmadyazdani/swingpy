"""
Pure-component gas properties and mixture property estimates.

A bank of pure-component properties (molar mass, ideal-gas heat capacity,
viscosity and Fuller diffusion volume, each kind with its source, ``gas_sources()``), and estimates for mixtures:
Fuller molecular diffusivity and Wilke viscosity. Your own components can be
added with ``register_gas``.

    from swingpy import data

    data.list_gases()
    data.gas_properties("CO2")
    gas = data.gas(["CO2", "N2"], {"CO2": 0.15, "N2": 0.85}, temperature=298.15, pressure=1.0)
"""

from __future__ import annotations

import json
import math
from importlib import resources
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np

from ..gas import GasMixture

_gases: Dict[str, dict] = {}


def _load_package_file(name):
    return json.loads(resources.files(__name__).joinpath(name).read_text(encoding="utf-8"))


def _ensure_loaded():
    if not _gases:
        _gases.update(_load_package_file("gases.json")["gases"])


def list_gases() -> List[str]:
    """Names of the gases in the bank."""
    _ensure_loaded()
    return sorted(_gases)


def gas_sources() -> dict:
    """Where every kind of gas property in the bank comes from."""
    return dict(_load_package_file("gases.json")["sources"])


def gas_properties(name: str) -> dict:
    """Molar mass (kg/mol), heat capacity (J/(mol K)), viscosity (Pa s) and Fuller diffusion volume of a gas."""
    _ensure_loaded()
    if name not in _gases:
        raise KeyError(f"Unknown gas '{name}'. Available: {', '.join(sorted(_gases))}.")
    return dict(_gases[name])


def register_gas(name: str, molar_mass: float, heat_capacity: float, viscosity: float,
                 diffusion_volume: float) -> None:
    """Add a gas to the bank for this session (units as in ``gas_properties``)."""
    _ensure_loaded()
    for label, value in (("molar_mass", molar_mass), ("heat_capacity", heat_capacity), ("viscosity", viscosity),
                         ("diffusion_volume", diffusion_volume)):
        if not value > 0.0:
            raise ValueError(f"{label} of '{name}' must be positive.")
    _gases[name] = dict(molar_mass=molar_mass, heat_capacity=heat_capacity, viscosity=viscosity,
                        diffusion_volume=diffusion_volume)


def fuller_diffusivity(first: str, second: str, temperature: float = 298.15, pressure: float = 1.0) -> float:
    """
    Binary molecular diffusivity by the method of Fuller, Schettler and Giddings, m2/s.

    Args:
        first: Name of the first gas.
        second: Name of the second gas.
        temperature: K.
        pressure: bar.
    """
    a, b = gas_properties(first), gas_properties(second)
    reduced_mass = 2.0 / (1.0 / (a["molar_mass"] * 1e3) + 1.0 / (b["molar_mass"] * 1e3))
    volumes = a["diffusion_volume"] ** (1.0 / 3.0) + b["diffusion_volume"] ** (1.0 / 3.0)
    centimetres = 0.00143 * temperature ** 1.75 / (pressure * math.sqrt(reduced_mass) * volumes ** 2)
    return centimetres * 1.0e-4


def wilke_viscosity(components: Sequence[str], mole_fractions: Sequence[float], temperature: float = 298.15) -> float:
    """Viscosity of a gas mixture by Wilke's mixing rule, Pa s (pure viscosities scaled as T^0.7)."""
    props = [gas_properties(c) for c in components]
    mu = np.array([p["viscosity"] for p in props]) * (temperature / 298.15) ** 0.7
    mass = np.array([p["molar_mass"] for p in props])
    y = np.asarray(mole_fractions, dtype=float)
    y = y / y.sum()
    phi = (1.0 + np.sqrt(mu[:, None] / mu[None, :]) * (mass[None, :] / mass[:, None]) ** 0.25) ** 2 / \
        np.sqrt(8.0 * (1.0 + mass[:, None] / mass[None, :]))
    return float(np.sum(y * mu / (phi @ y)))


def gas(components: Iterable[str], composition: Optional[Mapping[str, float]] = None, temperature: float = 298.15,
        pressure: float = 1.0, **overrides) -> GasMixture:
    """
    A GasMixture with properties evaluated for a representative composition.

    The heat capacity is the mole-fraction average, the viscosity follows
    Wilke's mixing rule, and the molecular diffusivity is the Fuller binary
    diffusivity of the first two components (the key pair) at the given
    temperature and pressure. These are constants of the model, so evaluate
    them at the feed conditions.

    Args:
        components: Gas components, in the order used throughout the model.
        composition: Mole fractions by component; equal fractions if omitted.
        temperature: K.
        pressure: bar.
        **overrides: Any GasMixture field to set directly, for example
            ``molecular_diffusivity=1.3e-5`` or ``adiabatic_constant=...``.

    Returns:
        GasMixture.
    """
    components = tuple(components)
    if len(set(components)) != len(components):
        raise ValueError(f"Components must be distinct: {components}.")
    composition = dict(composition or {c: 1.0 / len(components) for c in components})
    unknown = set(composition) - set(components)
    if unknown:
        raise ValueError(f"Composition names components that are not in the gas: {sorted(unknown)}.")
    y = np.array([composition.get(c, 0.0) for c in components], dtype=float)
    if y.sum() <= 0.0 or np.any(y < 0.0):
        raise ValueError("Mole fractions must be non-negative and not all zero.")
    y = y / y.sum()
    props = [gas_properties(c) for c in components]
    fields = dict(
        components=components,
        heat_capacity=float(np.sum(y * [p["heat_capacity"] for p in props])),
        viscosity=wilke_viscosity(components, y, temperature),
        molecular_diffusivity=(fuller_diffusivity(components[0], components[1], temperature, pressure)
                               if len(components) > 1 else 1.0e-5),
        molar_masses=[p["molar_mass"] for p in props],
    )
    fields.update(overrides)
    return GasMixture(**fields)
