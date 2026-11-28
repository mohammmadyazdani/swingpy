"""
Adsorbents for the examples: isotherms, heats of adsorption and particle
properties of materials reported in the literature (``adsorbents.json``,
each entry with its sources and assumptions).

This is example data, not part of the SwingPy library: a problem takes its
adsorbent from the user (see ``swingpy.spec``).

    from setups import adsorbents
    adsorbents.list_adsorbents()
    sorbent = adsorbents.adsorbent("zeolite-13x", particle_radius=0.75e-3)
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence

from swingpy.adsorbent import Adsorbent
from swingpy.isotherms import DualSiteLangmuir, Sips, Toth

_ISOTHERMS = {"DualSiteLangmuir": DualSiteLangmuir, "Sips": Sips, "Toth": Toth}
_PER_COMPONENT = {"qsb", "qsd", "b0", "d0", "dUb", "dUd", "qs", "dU", "n", "t", "alpha", "chi"}
_ADSORBENT_FIELDS = {"density", "particle_radius", "particle_voidage", "tortuosity", "heat_capacity",
                     "adsorbed_phase_heat_capacity"}

_adsorbents: Dict[str, dict] = {}
_FILE = Path(__file__).with_name("adsorbents.json")


def _bank() -> dict:
    """The entries of adsorbents.json."""
    return json.loads(_FILE.read_text(encoding="utf-8"))


def _ensure_loaded():
    if not _adsorbents:
        _adsorbents.update(_bank())


def list_adsorbents() -> List[str]:
    """Names of the adsorbents in the bank."""
    _ensure_loaded()
    return sorted(_adsorbents)


def describe_adsorbent(name: str) -> str:
    """One-line description of an adsorbent of the bank and the components it covers."""
    entry = _adsorbent_entry(name)
    return f"{name}: {entry.get('description', '')} (components: {', '.join(entry['components'])})"


def _adsorbent_entry(name: str) -> dict:
    _ensure_loaded()
    if name not in _adsorbents:
        raise KeyError(f"Unknown adsorbent '{name}'. Available: {', '.join(sorted(_adsorbents))}.")
    return _adsorbents[name]


def adsorbent(name: str, components: Optional[Sequence[str]] = None, **overrides) -> Adsorbent:
    """
    An adsorbent from the bank.

    Args:
        name: Name in the bank (see ``list_adsorbents``).
        components: Gas components, in the order of the GasMixture the
            adsorbent will be used with. They must be among the components of
            the entry; the isotherm parameters are reordered (and the others
            dropped) to match. By default the order of the entry.
        **overrides: Particle properties to change, for example
            ``particle_radius=0.75e-3`` or ``kinetics=ConstantLDF(...)``.

    Returns:
        Adsorbent.
    """
    entry = copy.deepcopy(_adsorbent_entry(name))
    available = list(entry["components"])
    components = list(components) if components is not None else available
    missing = [c for c in components if c not in available]
    if missing:
        raise ValueError(f"Adsorbent '{name}' has no parameters for {missing}; it covers {available}.")
    order = [available.index(c) for c in components]

    isotherm_entry = dict(entry["isotherm"])
    isotherm_type = _ISOTHERMS[isotherm_entry.pop("type")]
    for key, value in isotherm_entry.items():
        if key in _PER_COMPONENT and value is not None:
            isotherm_entry[key] = [value[i] for i in order]
    isotherm = isotherm_type(**isotherm_entry)

    unknown = set(overrides) - _ADSORBENT_FIELDS - {"kinetics", "notes", "heat_of_adsorption", "isotherm"}
    if unknown:
        raise TypeError(f"Unknown adsorbent properties: {sorted(unknown)}.")
    fields = {key: entry[key] for key in _ADSORBENT_FIELDS if key in entry}
    fields.update(isotherm=isotherm, heat_of_adsorption=[entry["heat_of_adsorption"][i] for i in order],
                  notes=entry.get("description", ""))
    fields.update(overrides)
    return Adsorbent(name=name, **fields)


def register_adsorbent(name: str, entry: Mapping) -> None:
    """
    Add an adsorbent to the bank for this session.

    ``entry`` has the layout of the entries of ``adsorbents.json``:
    "components", "isotherm" (with "type" and its parameters, one value per
    component), "heat_of_adsorption" (J/mol, positive), "density" (kg/m3),
    "particle_radius" (m) and optionally "particle_voidage", "tortuosity",
    "heat_capacity", "adsorbed_phase_heat_capacity" and "description".
    """
    _ensure_loaded()
    required = {"components", "isotherm", "heat_of_adsorption", "density", "particle_radius"}
    missing = required - set(entry)
    if missing:
        raise ValueError(f"Adsorbent entry '{name}' lacks {sorted(missing)}.")
    if entry["isotherm"].get("type") not in _ISOTHERMS:
        raise ValueError(f"Isotherm type of '{name}' must be one of {sorted(_ISOTHERMS)}.")
    n = len(entry["components"])
    for key, value in entry["isotherm"].items():
        if key in _PER_COMPONENT and value is not None and len(value) != n:
            raise ValueError(f"Isotherm parameter '{key}' of '{name}' needs {n} values, one per component.")
    _adsorbents[name] = dict(entry)
    """Fail now, not at first use, if the entry cannot build an Adsorbent."""
    adsorbent(name)


def load_adsorbents(path) -> List[str]:
    """Add every adsorbent of a JSON file (same layout as the bank) and return their names."""
    with open(path, encoding="utf-8") as handle:
        entries = json.load(handle)
    for name, entry in entries.items():
        register_adsorbent(name, entry)
    return list(entries)



def adsorbent_sources(name: str) -> dict:
    """
    References (citation and DOI) and assumptions behind an adsorbent of the
    bank; every built-in entry has them, registered entries may add a
    ``"sources"`` field of the same form.
    """
    entry = _adsorbent_entry(name)
    return dict(entry.get("sources", {"references": [], "assumptions": []}))
