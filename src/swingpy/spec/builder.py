"""
Build a SwingPy problem from a specification (a dict, or a YAML or JSON file).

Every physical quantity is taken from the specification: nothing about the
gas, the adsorbent, the bed or the feed has a hidden default. Numerical
options (cells, scheme, backend, solver settings) have documented defaults.
All errors are collected and reported together, each with its path in the
specification. See docs/guide/specification.md for the full layout.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


from ..adsorbent import Adsorbent, ConstantLDF, MacroporeLDF, ParticleDiffusion, PressureScaledLDF, SeriesLDF
from ..boundaries import (BufferInlet, Closed, ExponentialRamp, FeedInlet, FlowInlet, MixingInlet, PressureOutlet,
                          PressurizationInlet, TankInlet, VacuumPump, falling_vacuum_efficiency)
from ..column import Column
from ..cycle import Cycle
from ..gas import GasMixture, LinearCO2AdiabaticConstant
from ..isotherms import DualSiteLangmuir, Sips, Toth
from ..model import ColumnModel
from ..steps import PressureReached, Step
from .units import UnitError, kind_of, quantity

SPEC_VERSION = 1


class SpecError(ValueError):
    """Errors in a specification, each as "path: message"."""

    def __init__(self, errors):
        self.errors = list(errors)
        super().__init__("the specification has %d error(s):\n  " % len(self.errors) + "\n  ".join(self.errors))


class _Reader:
    """Reads values from a nested specification and records errors with their paths."""

    def __init__(self):
        self.errors = []

    def error(self, path, message):
        self.errors.append(f"{path}: {message}")

    def section(self, spec, key, path, required=True):
        value = spec.get(key) if isinstance(spec, dict) else None
        if value is None:
            if required:
                self.error(f"{path}.{key}" if path else key, "missing")
            return None
        if not isinstance(value, dict):
            self.error(f"{path}.{key}" if path else key, "expected a section (key: value pairs)")
            return None
        return value

    def get(self, spec, key, path, kind=None, required=True, default=None, choices=None):
        where = f"{path}.{key}" if path else key
        if not isinstance(spec, dict) or key not in spec or spec[key] is None:
            if required:
                self.error(where, "missing")
            return default
        value = spec[key]
        if choices is not None:
            if value not in choices:
                self.error(where, f"{value!r} is not one of {', '.join(map(str, choices))}")
                return default
            return value
        if kind is None:
            return value
        try:
            return quantity(value, kind)
        except UnitError as exc:
            self.error(where, str(exc))
            return default

    def per_component(self, spec, key, path, components, kind, required=True):
        where = f"{path}.{key}" if path else key
        block = spec.get(key) if isinstance(spec, dict) else None
        if block is None:
            if required:
                self.error(where, "missing")
            return None
        if not isinstance(block, dict):
            self.error(where, "expected one value per component, as component: value")
            return None
        for extra in sorted(set(block) - set(components)):
            self.error(f"{where}.{extra}", "not one of the components")
        return [self.get(block, c, where, kind) for c in components]

    def unknown(self, spec, allowed, path):
        if isinstance(spec, dict):
            for key in sorted(set(spec) - set(allowed)):
                self.error(f"{path}.{key}" if path else key, "unknown key")


"""The isotherm models a specification can name, with the fields and units each one takes."""

_ISOTHERMS = {
    "dual_site_langmuir": dict(cls=DualSiteLangmuir, fields=dict(
        qsb="loading", qsd="loading", b0="inverse_concentration", d0="inverse_concentration",
        dUb="molar_energy", dUd="molar_energy")),
    "langmuir": dict(cls=DualSiteLangmuir, fields=dict(qs="loading", b0="inverse_concentration", dU="molar_energy")),
    "sips": dict(cls=Sips, fields=dict(qs="loading", b0=("inverse_concentration", "inverse_pressure"),
                                       dU="molar_energy", n="dimensionless"),
                 optional=dict(alpha="dimensionless", chi="dimensionless", reference_temperature="temperature")),
    "toth": dict(cls=Toth, fields=dict(qs="loading", b0=("inverse_concentration", "inverse_pressure"),
                                       dU="molar_energy", t="dimensionless"),
                 optional=dict(alpha="dimensionless", chi="dimensionless", reference_temperature="temperature")),
}


def _isotherm(r, spec, path, components):
    model = r.get(spec, "model", path, choices=sorted(_ISOTHERMS) + ["python"])
    if model is None:
        return None
    if model == "python":
        target = r.get(spec, "object", path)
        if target is None:
            return None
        module, _, name = str(target).partition(":")
        try:
            import importlib
            return getattr(importlib.import_module(module), name)
        except Exception as exc:  # noqa: BLE001 - reported as a specification error
            r.error(f"{path}.object", f"cannot import {target!r} ({exc})")
            return None
    info = _ISOTHERMS[model]
    extra = {"toth": ["competitive"], "sips": ["exponent_form"]}.get(model, [])
    r.unknown(spec, ["model"] + extra + components, path)
    fields, optional = info["fields"], info.get("optional", {})
    values = {name: [] for name in list(fields) + list(optional)}
    basis = set()
    for c in components:
        where = f"{path}.{c}"
        block = r.section(spec, c, path)
        if block is None:
            continue
        r.unknown(block, list(fields) + list(optional), where)
        for name, kind in fields.items():
            if isinstance(kind, tuple):
                found = kind_of(block.get(name), kind)
                if block.get(name) is not None and not isinstance(block.get(name), str):
                    r.error(f"{where}.{name}", "give a unit: m3/mol (concentration basis) or 1/bar (pressure basis)")
                    continue
                basis.add(found)
                kind = found or kind[0]
            values[name].append(r.get(block, name, where, kind))
        for name, kind in optional.items():
            values[name].append(r.get(block, name, where, kind, required=False))
    if len(basis - {None}) > 1:
        r.error(path, "b0 must be on the same basis (concentration or pressure) for every component")
    if r.errors:
        return None
    if model == "langmuir":
        zeros = (0.0,) * len(components)
        return DualSiteLangmuir(qsb=tuple(values["qs"]), qsd=zeros, b0=tuple(values["b0"]), d0=zeros,
                                dUb=tuple(values["dU"]), dUd=zeros)
    if model == "dual_site_langmuir":
        return DualSiteLangmuir(**{k: tuple(v) for k, v in values.items()})
    kwargs = {k: tuple(v) for k, v in values.items() if k in fields}
    for name in ("alpha", "chi"):
        if any(v is not None for v in values[name]):
            kwargs[name] = tuple(0.0 if v is None else v for v in values[name])
    temperatures = {v for v in values["reference_temperature"] if v is not None}
    if len(temperatures) > 1:
        r.error(path, "reference_temperature must be the same for every component")
        return None
    if temperatures:
        kwargs["reference_temperature"] = temperatures.pop()
    if "inverse_pressure" in basis:
        kwargs["pressure_based"] = True
    if model == "sips" and "exponent_form" in spec:
        if spec["exponent_form"] not in ("inverse", "linear"):
            r.error(f"{path}.exponent_form", "must be 'inverse' (n0 + alpha (1 - Tref/T)) or 'linear' (n0 + alpha (T/Tref - 1))")
            return None
        kwargs["exponent_form"] = spec["exponent_form"]
    if model == "toth" and "competitive" in spec:
        kwargs["competitive"] = bool(spec["competitive"])
    return info["cls"](**kwargs)


"""The functions below read one section of a specification each."""

def _gas(r, spec, components):
    s = r.section(spec, "gas", "")
    if s is None:
        return None
    r.unknown(s, ["equation_of_state", "molar_masses", "heat_capacity", "viscosity", "molecular_diffusivity",
                  "adiabatic_constant"], "gas")
    gamma = s.get("adiabatic_constant")
    if gamma == "linear_co2":
        if "CO2" not in components:
            r.error("gas.adiabatic_constant", "linear_co2 needs a component named CO2")
            gamma = None
        else:
            gamma = LinearCO2AdiabaticConstant(co2_index=components.index("CO2"))
    else:
        gamma = r.get(s, "adiabatic_constant", "gas", "dimensionless")
    r.get(s, "equation_of_state", "gas", choices=["ideal"])
    masses = r.per_component(s, "molar_masses", "gas", components, "molar_mass")
    values = dict(heat_capacity=r.get(s, "heat_capacity", "gas", "molar_heat_capacity"),
                  viscosity=r.get(s, "viscosity", "gas", "viscosity"),
                  molecular_diffusivity=r.get(s, "molecular_diffusivity", "gas", "diffusivity"),
                  adiabatic_constant=gamma)
    if r.errors or masses is None:
        return None
    return GasMixture(components=tuple(components), molar_masses=tuple(masses), **values)


def _adsorbent(r, spec, components):
    s = r.section(spec, "adsorbent", "")
    if s is None:
        return None
    p = "adsorbent"
    r.unknown(s, ["name", "particle_density", "particle_radius", "particle_voidage", "tortuosity", "heat_capacity",
                  "adsorbed_phase_heat_capacity", "heat_of_adsorption", "isotherm", "kinetics"], p)
    isotherm = _isotherm(r, r.section(s, "isotherm", p) or {}, f"{p}.isotherm", components) \
        if s.get("isotherm") is not None else r.section(s, "isotherm", p)
    heats = r.per_component(s, "heat_of_adsorption", p, components, "molar_energy")
    values = dict(density=r.get(s, "particle_density", p, "mass_density"),
                  particle_radius=r.get(s, "particle_radius", p, "length"),
                  particle_voidage=r.get(s, "particle_voidage", p, "dimensionless"),
                  tortuosity=r.get(s, "tortuosity", p, "dimensionless"),
                  heat_capacity=r.get(s, "heat_capacity", p, "specific_heat"),
                  adsorbed_phase_heat_capacity=r.get(s, "adsorbed_phase_heat_capacity", p, "molar_heat_capacity"))
    k = r.section(s, "kinetics", p)
    kinetics = None
    if k is not None:
        model = r.get(k, "model", f"{p}.kinetics",
                      choices=["macropore_ldf", "constant_ldf", "pressure_scaled_ldf", "particle_diffusion",
                               "series_ldf"])
        if model == "series_ldf":
            where = f"{p}.kinetics"
            r.unknown(k, ["model", "micropore_rates", "activation_energies", "macropore_diffusivities",
                          "film_coefficient", "micropore_factor", "macropore_factor", "area_factor", "max_rate"], where)
            micro = r.per_component(k, "micropore_rates", where, components, "rate")
            energies = r.per_component(k, "activation_energies", where, components, "molar_energy")
            macro = r.per_component(k, "macropore_diffusivities", where, components, "diffusivity")
            scalars = dict(film_coefficient=r.get(k, "film_coefficient", where, "velocity"),
                           micropore_factor=r.get(k, "micropore_factor", where, "dimensionless"),
                           macropore_factor=r.get(k, "macropore_factor", where, "dimensionless"),
                           area_factor=r.get(k, "area_factor", where, "dimensionless"),
                           max_rate=r.get(k, "max_rate", where, "rate", required=False, default=1.0e3))
            if not r.errors:
                try:
                    kinetics = SeriesLDF(tuple(micro), tuple(energies), tuple(macro), **scalars)
                except ValueError as exc:
                    r.error(where, str(exc))
        elif model == "macropore_ldf":
            r.unknown(k, ["model", "max_rate"], f"{p}.kinetics")
            kinetics = MacroporeLDF(max_rate=r.get(k, "max_rate", f"{p}.kinetics", "rate", required=False,
                                                   default=1.0e3))
        elif model in ("constant_ldf", "pressure_scaled_ldf", "particle_diffusion"):
            rates = r.per_component(k, "rates", f"{p}.kinetics", components, "rate")
            if model == "constant_ldf":
                r.unknown(k, ["model", "rates"], f"{p}.kinetics")
                kinetics = ConstantLDF(tuple(rates)) if rates and None not in rates else None
            elif model == "pressure_scaled_ldf":
                r.unknown(k, ["model", "rates", "reference_pressure"], f"{p}.kinetics")
                ref = r.get(k, "reference_pressure", f"{p}.kinetics", "pressure")
                kinetics = PressureScaledLDF(tuple(rates), ref) if rates and None not in rates else None
            else:
                r.unknown(k, ["model", "rates", "shells"], f"{p}.kinetics")
                shells = int(r.get(k, "shells", f"{p}.kinetics", required=False, default=12))
                kinetics = ParticleDiffusion(tuple(rates), shells) if rates and None not in rates else None
    if r.errors:
        return None
    return Adsorbent(name=str(s.get("name", "adsorbent")), isotherm=isotherm, heat_of_adsorption=tuple(heats),
                     kinetics=kinetics, **values)


def _bed(r, spec):
    s = r.section(spec, "bed", "")
    if s is None:
        return None
    p = "bed"
    r.unknown(s, ["length", "inner_radius", "outer_radius", "voidage", "pressure_drop", "axial_thermal_conductivity",
                  "include_pore_gas", "wall", "heat_transfer", "dead_volume"], p)
    wall = r.section(s, "wall", p) or {}
    heat = r.section(s, "heat_transfer", p) or {}
    dead = r.section(s, "dead_volume", p, required=False) or {}
    r.unknown(wall, ["density", "heat_capacity", "thermal_conductivity"], f"{p}.wall")
    r.unknown(heat, ["inner", "outer", "ambient_temperature"], f"{p}.heat_transfer")
    r.unknown(dead, ["feed", "product"], f"{p}.dead_volume")
    values = dict(
        length=r.get(s, "length", p, "length"), inner_radius=r.get(s, "inner_radius", p, "length"),
        outer_radius=r.get(s, "outer_radius", p, "length"), bed_voidage=r.get(s, "voidage", p, "dimensionless"),
        pressure_drop=r.get(s, "pressure_drop", p, choices=["darcy", "ergun"]),
        axial_thermal_conductivity=r.get(s, "axial_thermal_conductivity", p, "conductivity"),
        include_pore_gas=bool(r.get(s, "include_pore_gas", p, required=False, default=False)),
        wall_density=r.get(wall, "density", f"{p}.wall", "mass_density"),
        wall_heat_capacity=r.get(wall, "heat_capacity", f"{p}.wall", "specific_heat"),
        wall_thermal_conductivity=r.get(wall, "thermal_conductivity", f"{p}.wall", "conductivity"),
        inner_heat_transfer=r.get(heat, "inner", f"{p}.heat_transfer", "heat_transfer"),
        outer_heat_transfer=r.get(heat, "outer", f"{p}.heat_transfer", "heat_transfer"),
        ambient_temperature=r.get(heat, "ambient_temperature", f"{p}.heat_transfer", "temperature"),
        dead_volume_feed=r.get(dead, "feed", f"{p}.dead_volume", "volume", required=False, default=0.0),
        dead_volume_product=r.get(dead, "product", f"{p}.dead_volume", "volume", required=False, default=0.0))
    return None if r.errors else Column(**values)


def _streams(r, spec, components):
    s = r.section(spec, "streams", "")
    out = {}
    for name, block in (s or {}).items():
        where = f"streams.{name}"
        if not isinstance(block, dict):
            r.error(where, "expected a section with composition and temperature")
            continue
        r.unknown(block, ["composition", "temperature"], where)
        comp = r.per_component(block, "composition", where, components, "dimensionless")
        T = r.get(block, "temperature", where, "temperature")
        if comp is not None and None not in comp:
            if abs(sum(comp) - 1.0) > 1e-6:
                r.error(f"{where}.composition", f"mole fractions add up to {sum(comp):.6g}, not 1")
            out[name] = dict(composition=dict(zip(components, comp)), temperature=T)
    return out


def _pressure_history(r, value, where):
    if isinstance(value, dict):
        r.unknown(value, ["ramp", "from", "to", "rate"], where)
        r.get(value, "ramp", where, choices=["exponential"])
        start, end = r.get(value, "from", where, "pressure"), r.get(value, "to", where, "pressure")
        rate = r.get(value, "rate", where, "rate")
        return None if None in (start, end, rate) else ExponentialRamp(start, end, rate)
    try:
        return quantity(value, "pressure")
    except UnitError as exc:
        r.error(where, str(exc))
        return None


def _efficiency(r, block, where, machines):
    value = block.get("efficiency", machines.get("efficiency"))
    if value is None:
        r.error(f"{where}.efficiency", "missing (give it here or in the machines section)")
        return None
    if value == "falling_vacuum":
        return falling_vacuum_efficiency
    try:
        return quantity(value, "dimensionless")
    except UnitError as exc:
        r.error(f"{where}.efficiency", str(exc))
        return None


_NO_MACHINE = object()


def _machine_pressure(r, block, key, where, machines):
    value = block.get(key, machines.get(key))
    if value == "none":
        """No machine on this stream: no work counted."""
        return _NO_MACHINE
    if value is None:
        r.error(f"{where}.{key}", "missing (give it here or in the machines section)")
        return None
    try:
        return quantity(value, "pressure")
    except UnitError as exc:
        r.error(f"{where}.{key}", str(exc))
        return None


def _boundary(r, block, where, streams, machines):
    if block is None:
        return None
    if not isinstance(block, dict):
        r.error(where, "expected a section with a type")
        return None
    kind = r.get(block, "type", where,
                 choices=["closed", "feed", "flow_inlet", "pressure_inlet", "outlet", "vacuum_pump", "replay", "tank",
                          "mixing_inlet"])
    if kind is None:
        return None

    def stream_of(key="stream"):
        name = r.get(block, key, where)
        if name is not None and name not in streams:
            r.error(f"{where}.{key}", f"no stream {name!r} in the streams section")
            return None
        return streams.get(name)

    if kind == "closed":
        r.unknown(block, ["type"], where)
        return Closed()
    if kind == "feed":
        r.unknown(block, ["type", "stream", "velocity", "supply_pressure", "efficiency", "label"], where)
        st, v = stream_of(), r.get(block, "velocity", where, "velocity")
        sp, eff = _machine_pressure(r, block, "supply_pressure", where, machines), _efficiency(r, block, where, machines)
        if None in (st, v, sp, eff):
            return None
        return FeedInlet(v, st["composition"], st["temperature"], supply_pressure=sp, efficiency=eff,
                         stream=block.get("label", "feed"))
    if kind == "flow_inlet":
        r.unknown(block, ["type", "stream", "record", "molar_flow", "max_pressure", "supply_pressure", "efficiency",
                          "label", "record_as"], where)
        flow = r.get(block, "molar_flow", where, "molar_flow")
        top = r.get(block, "max_pressure", where, "pressure", required=False)
        sp, eff = _machine_pressure(r, block, "supply_pressure", where, machines), _efficiency(r, block, where, machines)
        if block.get("record") is not None:
            default = stream_of() if block.get("stream") is not None else None
            composition, temperature = block["record"], (default or {}).get("temperature")
            if temperature is None:
                r.error(f"{where}.stream", "give the stream used until the record exists (its temperature too)")
                return None
        else:
            st = stream_of()
            if st is None:
                return None
            composition, temperature, default = st["composition"], st["temperature"], None
        if None in (flow, sp, eff):
            return None
        return FlowInlet(flow, composition, temperature, max_pressure=top,
                         supply_pressure=None if sp is _NO_MACHINE else sp, efficiency=eff,
                         stream=block.get("label", "feed"),
                         default_composition=(default or {}).get("composition"), record_as=block.get("record_as"))
    if kind == "pressure_inlet":
        r.unknown(block, ["type", "stream", "record", "pressure", "supply_pressure", "efficiency", "label"], where)
        st = stream_of()
        pressure = _pressure_history(r, block.get("pressure"), f"{where}.pressure") if "pressure" in block \
            else r.get(block, "pressure", where)
        sp, eff = _machine_pressure(r, block, "supply_pressure", where, machines), _efficiency(r, block, where, machines)
        if None in (st, pressure, sp, eff):
            return None
        sp = None if sp is _NO_MACHINE else sp
        """The mean gas of a record; the stream is used until it exists."""
        if block.get("record") is not None:
            return PressurizationInlet(pressure, block["record"], st["temperature"], supply_pressure=sp,
                                       efficiency=eff, stream=block.get("label", "feed"),
                                       default_composition=st["composition"])
        return PressurizationInlet(pressure, st["composition"], st["temperature"], supply_pressure=sp, efficiency=eff,
                                   stream=block.get("label", "feed"))
    if kind == "outlet":
        r.unknown(block, ["type", "pressure", "label", "record_as", "check_valve", "discharge_pressure",
                          "efficiency"], where)
        pressure = _pressure_history(r, block.get("pressure"), f"{where}.pressure") if "pressure" in block \
            else r.get(block, "pressure", where)
        label = r.get(block, "label", where)
        dp = _machine_pressure(r, block, "discharge_pressure", where, machines)
        eff = _efficiency(r, block, where, machines)
        if None in (pressure, label, dp, eff):
            return None
        dp = None if dp is _NO_MACHINE else dp
        return PressureOutlet(pressure, discharge_pressure=dp, efficiency=eff, stream=label,
                              record_as=block.get("record_as"), check_valve=bool(block.get("check_valve", False)))
    if kind == "vacuum_pump":
        r.unknown(block, ["type", "volumetric_flow", "velocity", "label", "record_as", "discharge_pressure",
                          "efficiency"], where)
        flow = r.get(block, "volumetric_flow", where, "volumetric_flow", required=False)
        velocity = r.get(block, "velocity", where, "velocity", required=False)
        if (flow is None) == (velocity is None):
            r.error(where, "give either volumetric_flow or velocity")
        label = r.get(block, "label", where)
        dp = _machine_pressure(r, block, "discharge_pressure", where, machines)
        eff = _efficiency(r, block, where, machines)
        if None in (label, dp, eff):
            return None
        return VacuumPump(flow, discharge_pressure=dp, efficiency=eff, stream=label,
                          record_as=block.get("record_as"), velocity=velocity)
    if kind == "mixing_inlet":
        r.unknown(block, ["type", "stream", "record", "molar_flow", "fraction", "supply_pressure", "efficiency",
                          "label", "recycle_label"], where)
        st, flow = stream_of(), r.get(block, "molar_flow", where, "molar_flow")
        record = r.get(block, "record", where)
        fraction = r.get(block, "fraction", where, "dimensionless", required=False, default=1.0)
        sp, eff = _machine_pressure(r, block, "supply_pressure", where, machines), _efficiency(r, block, where, machines)
        if None in (st, flow, record, fraction, sp, eff):
            return None
        return MixingInlet(flow, st["composition"], record, fraction=fraction, temperature=st["temperature"],
                           supply_pressure=None if sp is _NO_MACHINE else sp, efficiency=eff,
                           stream=block.get("label", "feed"), recycle_stream=block.get("recycle_label", "recycle"))
    if kind == "replay":
        r.unknown(block, ["type", "record", "offset", "label", "max_pressure", "stream"], where)
        record, label = r.get(block, "record", where), r.get(block, "label", where)
        offset = r.get(block, "offset", where, "time", required=False, default=0.0)
        top = r.get(block, "max_pressure", where, "pressure", required=False)
        """Replay the flow only, with this gas."""
        gas = stream_of() if block.get("stream") is not None else None
        return None if None in (record, label) else BufferInlet(record, offset=offset, stream=label, max_pressure=top,
                                                                composition=(gas or {}).get("composition"))
    """A tank fed by a recorded stream, which returns a fraction of it to the column."""
    r.unknown(block, ["type", "record", "fraction", "temperature", "label", "supply_pressure", "efficiency"], where)
    record, label = r.get(block, "record", where), r.get(block, "label", where)
    fraction = r.get(block, "fraction", where, "dimensionless")
    T = r.get(block, "temperature", where, "temperature")
    sp, eff = _machine_pressure(r, block, "supply_pressure", where, machines), _efficiency(r, block, where, machines)
    if None in (record, label, fraction, T, sp, eff):
        return None
    return TankInlet(record, fraction, temperature=T, supply_pressure=sp, efficiency=eff, stream=label)


def _cycle(r, spec, components, streams):
    s = r.section(spec, "cycle", "")
    if s is None:
        return None
    r.unknown(s, ["key_component", "product_stream", "feed_stream", "machines", "steps"], "cycle")
    key = r.get(s, "key_component", "cycle", choices=components)
    machines = r.section(s, "machines", "cycle", required=False) or {}
    r.unknown(machines, ["efficiency", "supply_pressure", "discharge_pressure"], "cycle.machines")
    steps_spec = s.get("steps")
    if not isinstance(steps_spec, list) or not steps_spec:
        r.error("cycle.steps", "missing or empty (a list of steps)")
        return None
    steps = []
    for i, st in enumerate(steps_spec):
        where = f"cycle.steps[{i}]"
        if not isinstance(st, dict):
            r.error(where, "expected a section")
            continue
        r.unknown(st, ["name", "duration", "until", "max_duration", "feed_end", "product_end"], where)
        name = r.get(st, "name", where)
        stop = None
        if st.get("until") is not None:
            u = st["until"]
            if not isinstance(u, dict):
                r.error(f"{where}.until", "expected pressure, at and direction")
            else:
                r.unknown(u, ["pressure", "at", "direction"], f"{where}.until")
                target = r.get(u, "pressure", f"{where}.until", "pressure")
                at = r.get(u, "at", f"{where}.until", choices=["feed", "product", "mean"])
                direction = r.get(u, "direction", f"{where}.until", choices=["rising", "falling"])
                if None not in (target, at, direction):
                    stop = PressureReached(target, location=at, rising=direction == "rising")
            duration = r.get(st, "max_duration", where, "time")
        else:
            duration = r.get(st, "duration", where, "time")
        feed_end = _boundary(r, st.get("feed_end", {"type": "closed"}), f"{where}.feed_end", streams, machines)
        product_end = _boundary(r, st.get("product_end", {"type": "closed"}), f"{where}.product_end", streams,
                                machines)
        if None not in (name, duration, feed_end, product_end):
            steps.append(Step(name, duration, feed_end=feed_end, product_end=product_end, stop=stop))
    if r.errors:
        return None
    return Cycle(steps, key_component=key, product_stream=s.get("product_stream", "product"),
                 feed_stream=s.get("feed_stream", "feed"))


@dataclass
class Problem:
    """A problem built from a specification: the column model, the cycle and the initial state."""

    spec: dict
    model: ColumnModel
    cycle: Cycle = None
    initial: dict = field(default_factory=dict)

    @property
    def components(self):
        """The components in the order of the model; results (profiles, flows) follow this order."""
        return self.model.gas.components

    def initial_state(self):
        """The column state at the start, from the ``initial`` section."""
        i = self.initial
        return self.model.initial_state(i["pressure"], i["composition"], i["temperature"])

    def report(self, run, folder, **options):
        """Write the full report of a run (``swingpy.report.report_run``) to ``folder``."""
        from ..report import report_run
        return report_run(self.model, run, folder, **options)

    def run(self, **options):
        """Run the cycle to cyclic steady state (options of ``Cycle.run``)."""
        if self.cycle is None:
            raise ValueError("the specification has no cycle section")
        return self.cycle.run(self.model, self.initial_state(), **options)


def load_problem(source) -> Problem:
    """
    Build a problem from a specification.

    Args:
        source: A dict, or the path of a YAML (.yaml, .yml) or JSON file.

    Raises:
        SpecError: listing every error found, each with its path.
    """
    if isinstance(source, (str, Path)):
        path = Path(source)
        text = path.read_text()
        if path.suffix in (".yaml", ".yml"):
            import yaml
            spec = yaml.safe_load(text)
        else:
            spec = json.loads(text)
    else:
        spec = source
    r = _Reader()
    if not isinstance(spec, dict):
        raise SpecError(["the specification must be a section of key: value pairs"])
    r.unknown(spec, ["version", "components", "gas", "adsorbent", "bed", "model", "streams", "initial", "cycle"], "")
    if spec.get("version", SPEC_VERSION) != SPEC_VERSION:
        r.error("version", f"this SwingPy reads specification version {SPEC_VERSION}")
    components = spec.get("components")
    if not isinstance(components, list) or len(components) < 2 or len(set(components)) != len(components):
        raise SpecError(r.errors + ["components: missing, or not a list of two or more different names"])
    components = [str(c) for c in components]
    gas, adsorbent, bed = _gas(r, spec, components), _adsorbent(r, spec, components), _bed(r, spec)
    streams = _streams(r, spec, components)
    m = r.section(spec, "model", "", required=False) or {}
    r.unknown(m, ["cells", "scheme", "backend", "axial_dispersion", "reference"], "model")
    ref = r.section(m, "reference", "model")
    if ref is not None:
        r.unknown(ref, ["pressure", "temperature", "velocity"], "model.reference")
    options = dict(n_cells=int(m.get("cells", 30)), scheme=m.get("scheme", "weno"), backend=m.get("backend", "auto"),
                   reference_pressure=r.get(ref or {}, "pressure", "model.reference", "pressure"),
                   reference_temperature=r.get(ref or {}, "temperature", "model.reference", "temperature"),
                   reference_velocity=r.get(ref or {}, "velocity", "model.reference", "velocity"))
    dispersion = m.get("axial_dispersion", "correlation")
    if dispersion != "correlation":
        options["axial_dispersion"] = r.get(m, "axial_dispersion", "model", "diffusivity")
    initial = {}
    i = r.section(spec, "initial", "", required=False)
    if i is not None:
        r.unknown(i, ["pressure", "temperature", "composition"], "initial")
        comp = r.per_component(i, "composition", "initial", components, "dimensionless")
        initial = dict(pressure=r.get(i, "pressure", "initial", "pressure"),
                       temperature=r.get(i, "temperature", "initial", "temperature"),
                       composition=dict(zip(components, comp)) if comp else None)
    cycle = _cycle(r, spec, components, streams) if spec.get("cycle") is not None else None
    if r.errors:
        raise SpecError(r.errors)
    model = ColumnModel(bed, adsorbent, gas, **options)
    return Problem(spec=spec, model=model, cycle=cycle, initial=initial)
