"""
Save a problem built in Python as a specification (the inverse of ``load_problem``).

``to_spec(model, cycle, initial)`` returns a dict that ``load_problem``
turns back into an identical model and cycle; ``save_problem`` writes it as
YAML. Quantities are written in base units with their unit, so the file is
readable and editable. Objects a specification cannot express (functions of
the user's own, layered beds, valves) raise ``SpecError`` naming them.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..adsorbent import ConstantLDF, MacroporeLDF, ParticleDiffusion, PressureScaledLDF, SeriesLDF
from ..boundaries import (BufferInlet, Closed, Constant, ExponentialRamp, FeedInlet, FlowInlet, MixingInlet,
                          PressureOutlet,
                          PressurizationInlet, TankInlet, VacuumPump, falling_vacuum_efficiency)
from ..gas import LinearCO2AdiabaticConstant
from ..isotherms import DualSiteLangmuir, Sips, Toth, MultisiteLangmuir, Virial
from .builder import SPEC_VERSION, SpecError


def _q(value, unit):
    """A quantity as "<number> <unit>", the number with full precision."""
    return f"{float(value)!r} {unit}" if unit else float(value)


class _Exporter:
    def __init__(self, components):
        self.components = list(components)
        self.streams = {}
        self.unsupported = []

    def per(self, values, unit):
        return {c: _q(v, unit) for c, v in zip(self.components, np.atleast_1d(values))}

    def stream(self, composition, temperature):
        comp = {c: float(composition.get(c, 0.0)) for c in self.components}
        for name, s in self.streams.items():
            if s["composition"] == comp and s["temperature"] == _q(temperature, "K"):
                return name
        name = f"stream_{len(self.streams) + 1}"
        self.streams[name] = {"composition": comp, "temperature": _q(temperature, "K")}
        return name

    def efficiency(self, value, where):
        if value is falling_vacuum_efficiency:
            return "falling_vacuum"
        if callable(value):
            self.unsupported.append(f"{where}: an efficiency function of your own")
            return None
        return float(value)

    def pressure(self, history, where):
        if isinstance(history, Constant):
            return _q(history.value, "bar")
        if isinstance(history, ExponentialRamp):
            return {"ramp": "exponential", "from": _q(history.start, "bar"), "to": _q(history.end, "bar"),
                    "rate": _q(history.rate, "1/s")}
        if isinstance(history, (int, float)):
            return _q(history, "bar")
        self.unsupported.append(f"{where}: a pressure history of type {type(history).__name__}")
        return None

    def boundary(self, b, where):
        if b is None or isinstance(b, Closed):
            return {"type": "closed"}
        if isinstance(b, FeedInlet):
            if b.buffer:
                self.unsupported.append(f"{where}: a feed inlet fed from a recorded stream")
                return None
            return {"type": "feed", "stream": self.stream(b.composition, b.temperature),
                    "velocity": _q(b.velocity, "m/s"), "supply_pressure": _q(b.supply_pressure, "bar"),
                    "efficiency": self.efficiency(b.efficiency, where), "label": b.stream}
        if isinstance(b, FlowInlet):
            out = {"type": "flow_inlet", "molar_flow": _q(b.molar_flow, "mol/s"),
                   "supply_pressure": _q(b.supply_pressure, "bar") if b.supply_pressure is not None else "none",
                   "efficiency": self.efficiency(b.efficiency, where), "label": b.stream}
            if b.buffer is not None:
                out["record"] = b.buffer
                if b.default_composition is not None:
                    out["stream"] = self.stream(b.default_composition, b.temperature)
                else:
                    self.unsupported.append(f"{where}: a recorded inlet without a default composition")
            else:
                out["stream"] = self.stream(b.composition, b.temperature)
            if b.max_pressure is not None:
                out["max_pressure"] = _q(b.max_pressure, "bar")
            if b.record_as:
                out["record_as"] = b.record_as
            return out
        if isinstance(b, PressurizationInlet):
            out = {"type": "pressure_inlet",
                   "stream": self.stream(b.default_composition if b.buffer else b.composition, b.temperature),
                   "pressure": self.pressure(b.pressure, where),
                   "supply_pressure": _q(b.supply_pressure, "bar") if b.supply_pressure is not None else "none",
                   "efficiency": self.efficiency(b.efficiency, where), "label": b.stream}
            if b.buffer:
                out["record"] = b.buffer
            return out
        if isinstance(b, PressureOutlet):
            out = {"type": "outlet", "pressure": self.pressure(b.pressure, where), "label": b.stream,
                   "discharge_pressure": _q(b.discharge_pressure, "bar") if b.discharge_pressure is not None else "none",
                   "efficiency": self.efficiency(b.efficiency, where), "check_valve": bool(b.check_valve)}
            if b.record_as:
                out["record_as"] = b.record_as
            return out
        if isinstance(b, VacuumPump):
            out = {"type": "vacuum_pump", "label": b.stream, "discharge_pressure": _q(b.discharge_pressure, "bar"),
                   "efficiency": self.efficiency(b.efficiency, where)}
            if b.velocity is not None:
                out["velocity"] = _q(b.velocity, "m/s")
            else:
                out["volumetric_flow"] = _q(b.volumetric_flow, "m3/h")
            if b.record_as:
                out["record_as"] = b.record_as
            return out
        if isinstance(b, BufferInlet):
            out = {"type": "replay", "record": b.buffer, "label": b.stream, "offset": _q(b.offset, "s")}
            if b.composition is not None:
                out["stream"] = self.stream(b.composition, 298.15)
            if b.max_pressure is not None:
                out["max_pressure"] = _q(b.max_pressure, "bar")
            return out
        if isinstance(b, MixingInlet):
            out = {"type": "mixing_inlet", "stream": self.stream(b.composition, b.temperature),
                   "record": b.buffer, "molar_flow": _q(b.molar_flow, "mol/s"), "fraction": float(b.fraction),
                   "label": b.stream, "recycle_label": b.recycle_stream,
                   "supply_pressure": _q(b.supply_pressure, "bar") if b.supply_pressure is not None else "none",
                   "efficiency": self.efficiency(b.efficiency, where)}
            return out
        if isinstance(b, TankInlet):
            return {"type": "tank", "record": b.buffer, "fraction": float(b.fraction),
                    "temperature": _q(b.temperature, "K"), "label": b.stream,
                    "supply_pressure": _q(b.supply_pressure, "bar"), "efficiency": self.efficiency(b.efficiency, where)}
        self.unsupported.append(f"{where}: a boundary of type {type(b).__name__}")
        return None

    def isotherm(self, iso, where):
        def block(fields):
            return {c: {name: _q(np.atleast_1d(getattr(iso, attr))[i], unit) for name, (attr, unit) in fields.items()}
                    for i, c in enumerate(self.components)}
        if isinstance(iso, DualSiteLangmuir):
            return dict(model="dual_site_langmuir", **block(dict(
                qsb=("qsb", "mol/kg"), qsd=("qsd", "mol/kg"), b0=("b0", "m3/mol"), d0=("d0", "m3/mol"),
                dUb=("dUb", "J/mol"), dUd=("dUd", "J/mol"))))
        if isinstance(iso, (Sips, Toth)):
            if getattr(iso, "saturation_function", None) or getattr(iso, "exponent_function", None):
                self.unsupported.append(f"{where}: temperature functions of your own")
                return None
            shape = ("n", "n") if isinstance(iso, Sips) else ("t", "t")
            out = dict(model="sips" if isinstance(iso, Sips) else "toth", **block({
                "qs": ("qs", "mol/kg"), "b0": ("b0", "1/bar" if iso.pressure_based else "m3/mol"),
                "dU": ("dU", "J/mol"), shape[0]: (shape[1], "")}))
            for i, c in enumerate(self.components):
                for name in ("alpha", "chi"):
                    value = getattr(iso, name)
                    if value is not None and np.any(np.atleast_1d(value) != 0.0):
                        out[c][name] = float(np.atleast_1d(value)[i])
                out[c]["reference_temperature"] = _q(iso.reference_temperature, "K")
            if isinstance(iso, Toth):
                out["competitive"] = bool(iso.competitive)
            if isinstance(iso, Sips) and iso.exponent_form != "inverse":
                out["exponent_form"] = iso.exponent_form
            return out
        if isinstance(iso, MultisiteLangmuir):
            return dict(model="multisite_langmuir", **block(dict(
                qmax=("qmax", "mol/kg"), a=("a", ""), K_inf=("K_inf", "1/Pa"), dH=("dH", "J/mol"))))
        if isinstance(iso, Virial):
            out = dict(model="virial", specific_area=_q(iso.specific_area, "m2/kg"), **block(dict(
                K_inf=("K_inf", "mol/(kg Pa)"), dH=("dH", "J/mol"), A0=("A0", "m2/mol"), A1=("A1", "m2 K/mol"),
                B0=("B0", "m4/mol2"), B1=("B1", "m4 K/mol2"))))
            return out
        self.unsupported.append(f"{where}: an isotherm of type {type(iso).__name__}")
        return None

    def kinetics(self, k, where):
        if isinstance(k, MacroporeLDF):
            return {"model": "macropore_ldf", "max_rate": _q(k.max_rate, "1/s")}
        if isinstance(k, ConstantLDF):
            return {"model": "constant_ldf", "rates": self.per(k.rates, "1/s")}
        if isinstance(k, PressureScaledLDF):
            return {"model": "pressure_scaled_ldf", "rates": self.per(k.rates, "1/s"),
                    "reference_pressure": _q(k.reference_pressure, "bar")}
        if isinstance(k, SeriesLDF):
            return {"model": "series_ldf", "micropore_rates": self.per(k.micropore_rates, "1/s"),
                    "activation_energies": self.per(k.activation_energies, "J/mol"),
                    "macropore_diffusivities": self.per(k.macropore_diffusivities, "m2/s"),
                    "film_coefficient": _q(k.film_coefficient, "m/s"), "micropore_factor": float(k.micropore_factor),
                    "macropore_factor": float(k.macropore_factor), "area_factor": float(k.area_factor),
                    "max_rate": _q(k.max_rate, "1/s")}
        if isinstance(k, ParticleDiffusion):
            return {"model": "particle_diffusion", "rates": self.per(k.rates, "1/s"), "shells": int(k.shells)}
        self.unsupported.append(f"{where}: kinetics of type {type(k).__name__}")
        return None


def to_spec(model, cycle=None, initial=None, eos=None) -> dict:
    """
    The specification of a model, and optionally its cycle and initial state.

    Args:
        model: A ColumnModel.
        cycle: A Cycle, or None.
        initial: dict(pressure=bar, temperature=K, composition={component: fraction}), or None.
        eos: A ``PengRobinson`` for ``gas.compressibility_check``, or None.
    """
    gas, ads, col = model.gas, model.adsorbent, model.column
    x = _Exporter(gas.components)
    if getattr(ads, "layers", None) is not None:
        raise SpecError(["adsorbent: layered beds cannot be written as a specification yet"])
    gamma = gas.adiabatic_constant
    if isinstance(gamma, LinearCO2AdiabaticConstant):
        gamma = "linear_co2"
    elif callable(gamma):
        x.unsupported.append("gas.adiabatic_constant: a function of your own")
    else:
        gamma = float(gamma)
    check = None
    if eos is not None:
        if list(eos.components) != list(gas.components):
            raise SpecError([f"gas.compressibility_check: the equation of state has the components {eos.components}, "
                             f"the gas {list(gas.components)}"])
        Tc, pc, w = zip(*eos.constants)
        check = {"model": "peng_robinson", "critical_temperature": x.per(Tc, "K"),
                 "critical_pressure": x.per(pc, "bar"),
                 "acentric_factor": {c: float(v) for c, v in zip(gas.components, w)}}
        if eos.interaction:
            check["binary_interaction"] = {f"{a}-{b}": float(v) for (a, b), v in eos.interaction.items()}
        if eos.name:
            check["source"] = eos.name
    spec = {
        "version": SPEC_VERSION,
        "components": list(gas.components),
        "gas": {"equation_of_state": "ideal", "molar_masses": x.per(gas.molar_masses, "kg/mol"),
                "heat_capacity": _q(gas.heat_capacity, "J/(mol K)"), "viscosity": _q(gas.viscosity, "Pa s"),
                "molecular_diffusivity": _q(gas.molecular_diffusivity, "m2/s"), "adiabatic_constant": gamma,
                **({"compressibility_check": check} if check else {})},
        "adsorbent": {"name": ads.name, "particle_density": _q(ads.density, "kg/m3"),
                      "particle_radius": _q(ads.particle_radius, "m"),
                      "particle_voidage": float(ads.particle_voidage), "tortuosity": float(ads.tortuosity),
                      "heat_capacity": _q(ads.heat_capacity, "J/(kg K)"),
                      "adsorbed_phase_heat_capacity": _q(ads.adsorbed_phase_heat_capacity, "J/(mol K)"),
                      "heat_of_adsorption": x.per(ads.heat_of_adsorption, "J/mol"),
                      "isotherm": x.isotherm(ads.isotherm, "adsorbent.isotherm"),
                      "kinetics": x.kinetics(ads.kinetics, "adsorbent.kinetics")},
        "bed": {"length": _q(col.length, "m"), "inner_radius": _q(col.inner_radius, "m"),
                "outer_radius": _q(col.outer_radius, "m"), "voidage": float(col.bed_voidage),
                "pressure_drop": col.pressure_drop,
                "axial_thermal_conductivity": _q(col.axial_thermal_conductivity, "W/(m K)"),
                "include_pore_gas": bool(col.include_pore_gas),
                "wall": {"density": _q(col.wall_density, "kg/m3"), "heat_capacity": _q(col.wall_heat_capacity, "J/(kg K)"),
                         "thermal_conductivity": _q(col.wall_thermal_conductivity, "W/(m K)")},
                "heat_transfer": {"inner": _q(col.inner_heat_transfer, "W/(m2 K)"),
                                  "outer": _q(col.outer_heat_transfer, "W/(m2 K)"),
                                  "ambient_temperature": _q(col.ambient_temperature, "K")},
                "dead_volume": {"feed": _q(col.dead_volume_feed, "m3"), "product": _q(col.dead_volume_product, "m3")}},
        "model": {"cells": int(model.n_cells), "scheme": model.scheme,
                  "backend": model.options.get("backend", "auto"),
                  "reference": {"pressure": _q(model.P0 / 1.0e5, "bar"), "temperature": _q(model.T0, "K"),
                                "velocity": _q(model.v0, "m/s")}},
    }
    if model.options.get("axial_dispersion"):
        spec["model"]["axial_dispersion"] = _q(model.options["axial_dispersion"], "m2/s")
    if model.options.get("reference_loading") is not None:
        x.unsupported.append("model.reference_loading")
    if cycle is not None:
        steps = []
        for i, step in enumerate(cycle.steps):
            where = f"cycle.steps[{i}]"
            out = {"name": step.name}
            if step.stop is not None:
                out["until"] = {"pressure": _q(step.stop.pressure, "bar"), "at": step.stop.location,
                                "direction": "rising" if step.stop.rising else "falling"}
                out["max_duration"] = _q(step.duration, "s")
            else:
                out["duration"] = _q(step.duration, "s")
            if step.ambient_temperature is not None:
                out["ambient_temperature"] = _q(step.ambient_temperature, "K")
            out["feed_end"] = x.boundary(step.feed_end, f"{where}.feed_end")
            out["product_end"] = x.boundary(step.product_end, f"{where}.product_end")
            steps.append(out)
        key = cycle.key_component
        spec["cycle"] = {"key_component": gas.components[key] if isinstance(key, int) else key,
                         "product_stream": cycle.product_stream, "feed_stream": cycle.feed_stream, "steps": steps}
    if initial is not None:
        spec["initial"] = {"pressure": _q(initial["pressure"], "bar"), "temperature": _q(initial["temperature"], "K"),
                           "composition": {c: float(initial["composition"].get(c, 0.0)) for c in gas.components}}
    if x.streams:
        spec["streams"] = x.streams
    if x.unsupported:
        raise SpecError(["cannot be written as a specification: " + u for u in x.unsupported])
    return spec


def save_problem(path, model, cycle=None, initial=None, header=None, eos=None):
    """Write ``to_spec(model, cycle, initial, eos)`` as a YAML file, with an optional comment header."""
    import yaml
    text = yaml.safe_dump(to_spec(model, cycle, initial, eos), sort_keys=False, width=110)
    if header:
        text = "".join(f"# {line}\n" if line else "#\n" for line in header.splitlines()) + text
    Path(path).write_text(text)
