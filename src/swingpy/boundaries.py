"""
Boundary conditions at the two ends of the column.

A cycle step is defined by what happens at the feed end (Z = 0) and at the
product end (Z = 1). Each end is described by one of the classes below, so
a new step is built by combining two of them rather than by editing the
model. The same class works at either end: for example ``PressureOutlet``
at the product end is a blowdown, and at the feed end it is an evacuation.

User inputs are in practical units: pressures in bar, velocities in m/s
(interstitial), temperatures in K, compositions as dicts or sequences of
mole fractions. The model converts them internally.

Every boundary also says what kind of stream it is (``stream``), which the
cycle uses to compute purity and recovery, and which machine moves the gas
(a compressor fed from ``supply_pressure``, or a vacuum pump discharging to
``discharge_pressure``), which the cycle uses to compute energy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, NamedTuple, Union

import numpy as np

Efficiency = Union[float, Callable[[float], float]]
"""A machine efficiency: a number, or a function of the suction pressure in bar."""


class Constant:
    """A pressure that does not change during the step, in bar."""

    def __init__(self, value: float):
        self.value = float(value)

    def __call__(self, time_s: float) -> float:
        return self.value


@dataclass
class ExponentialRamp:
    """
    Pressure relaxing exponentially from ``start`` to ``end`` (bar):

        P(t) = end + (start - end) exp(-rate t)

    a common way of prescribing the pressure history of pressurization,
    blowdown and evacuation steps, with ``rate`` the constant lambda in 1/s.
    """

    start: float
    end: float
    rate: float = 0.5

    def __call__(self, time_s: float) -> float:
        return self.end + (self.start - self.end) * np.exp(-self.rate * time_s)


@dataclass
class LinearRamp:
    """
    Pressure changing linearly from ``start`` to ``end`` (bar) over
    ``duration`` (s), then held at ``end``:

        P(t) = start + (end - start) min(t / duration, 1)

    as used for the feed pressurization of Nguyen et al. (Chem. Eng. J. 452
    (2023) 139550, Eq. 9).
    """

    start: float
    end: float
    duration: float

    def __post_init__(self):
        if self.duration <= 0.0:
            raise ValueError("LinearRamp: duration must be positive.")

    def __call__(self, time_s: float) -> float:
        return self.start + (self.end - self.start) * min(max(time_s, 0.0) / self.duration, 1.0)


def as_history(pressure):
    """Accept a number (constant pressure) or any callable of time."""
    return pressure if callable(pressure) else Constant(pressure)


def falling_vacuum_efficiency(pressure_bar: float) -> float:
    """
    Vacuum pump efficiency that falls at deep vacuum,
    eta = 15.84 P / (1 + 19.80 P) with P the suction pressure in bar: about
    0.72 at 0.1 bar or more, and much lower below 0.05 bar, as real pumps do.
    """
    return 15.84 * pressure_bar / (1.0 + 19.80 * pressure_bar)


class FaceState(NamedTuple):
    """Dimensionless velocity, pressure, independent mole fractions and temperature at one end face."""

    velocity: float
    pressure: float
    mole_fractions: np.ndarray
    temperature: float


class Boundary:
    """
    Base class of all end conditions.

    Attributes:
        role: "closed", "inlet" or "outlet". Fixes the flow direction of the
            step and how the stream is counted in the cycle balances.
        stream: Label of the stream, for example "feed", "product",
            "raffinate" or "waste". Purity and recovery are computed from
            these labels.
        supply_pressure: For inlets, the pressure the gas is available at
            (bar); a compressor is needed above it.
        discharge_pressure: For outlets, the pressure the gas is delivered
            to (bar); a vacuum pump is needed below it.
        efficiency: Efficiency of that machine (number or function of the
            suction pressure in bar).
    """

    role = "closed"
    stream = None
    supply_pressure = None
    discharge_pressure = None
    efficiency: Efficiency = 1.0
    record_as = None

    def face(self, model, end: str, time_s: float, cell) -> FaceState:
        raise NotImplementedError

    def machine_efficiency(self, pressure_bar: float) -> float:
        efficiency = self.efficiency
        return float(efficiency(pressure_bar)) if callable(efficiency) else float(efficiency)


class Closed(Boundary):
    """A closed valve: no flow, and zero gradients of pressure, composition and temperature."""

    role = "closed"

    def face(self, model, end, time_s, cell):
        return FaceState(0.0, cell.pressure, cell.mole_fractions, cell.temperature)

    def __repr__(self):
        return "Closed()"


class InstantFill(Boundary):
    """
    Feed end of an instantaneous pressurization (``steps.instant_pressurization``):
    the gas in the column is replaced at once by a gas of the given
    composition and pressure, the solid (loadings and temperature) frozen.
    Not a boundary of a simulated step: ``ColumnModel.simulate`` applies it
    directly.
    """

    role = "inlet"
    instantaneous = True

    def __init__(self, composition, pressure, stream="feed"):
        self.composition, self.pressure, self.stream = composition, float(pressure), stream

    def face(self, model, end, time_s, cell):
        raise RuntimeError("An instantaneous fill has no face condition; ColumnModel.simulate applies it at once.")

    def __repr__(self):
        return f"InstantFill({self.composition!r}, {self.pressure!r}, stream={self.stream!r})"


class FeedInlet(Boundary):
    """
    Gas entering at a fixed interstitial velocity (flow controlled feed).

    The face pressure follows from the pressure drop over half a cell, and
    composition and temperature from the Danckwerts conditions
    at the inlet.

    Args:
        velocity: Interstitial velocity at the face, m/s.
        composition: Mole fractions of the entering gas (dict or sequence),
            or the name of a recorded stream (``record_as``): the gas is then
            its mean composition, as from a well mixed tank.
        temperature: Temperature of the entering gas, K.
        supply_pressure: Pressure the gas is available at, bar. Energy is
            charged for compressing it up to the face pressure.
        efficiency: Compressor efficiency.
        stream: Stream label, "feed" by default.
        default_composition: With a recorded stream, the gas used until the
            record exists (the first cycle).
    """

    role = "inlet"

    def __init__(self, velocity, composition, temperature=298.15, supply_pressure=1.0,
                 efficiency: Efficiency = 0.72, stream="feed", default_composition=None):
        self.velocity, self.temperature = velocity, temperature
        self.supply_pressure, self.efficiency, self.stream = supply_pressure, efficiency, stream
        """
        A recorded stream (its name) as the gas: the mean composition of the record, for example light product
        pressurization from a well mixed tank of the adsorption outlet.
        """
        self.buffer = composition if isinstance(composition, str) else None
        self.default_composition = default_composition
        self.composition = default_composition if self.buffer else composition
        if self.buffer and default_composition is None:
            raise ValueError("FeedInlet: give default_composition, used until the record exists.")

    def attach(self, record, step_duration):
        moles = np.asarray(record.total_moles(), dtype=float)
        if moles.sum() > 0.0:
            self.composition = moles / moles.sum()

    def face(self, model, end, time_s, cell):
        v = model.to_velocity(self.velocity) * (1.0 if end == "feed" else -1.0)
        pressure = model.pressure_for_velocity(v, cell, end)
        y, t = model.danckwerts(cell, model.to_mole_fractions(self.composition),
                                model.to_temperature(self.temperature), abs(v), end)
        return FaceState(v, pressure, y, t)


class PressurizationInlet(Boundary):
    """
    Gas entering while the face pressure follows a prescribed history
    (feed pressurization).

    Args:
        pressure: Face pressure in bar, a number or a history such as
            ``ExponentialRamp``.
        composition: As for ``FeedInlet``; it can also be the name of a
            recorded stream (its mean composition), for example
            pressurization with the product of another column.
        temperature: As for ``FeedInlet``.
        supply_pressure: As for ``FeedInlet``.
        efficiency: As for ``FeedInlet``.
        stream: As for ``FeedInlet``.
        default_composition: As for ``FeedInlet``.
    """

    role = "inlet"

    def __init__(self, pressure, composition, temperature=298.15, supply_pressure=1.0,
                 efficiency: Efficiency = 0.72, stream="feed", default_composition=None):
        self.pressure, self.temperature = as_history(pressure), temperature
        self.supply_pressure, self.efficiency, self.stream = supply_pressure, efficiency, stream
        self.buffer = composition if isinstance(composition, str) else None
        self.default_composition = default_composition
        self.composition = default_composition if self.buffer else composition
        if self.buffer and default_composition is None:
            raise ValueError("PressurizationInlet: give default_composition, used until the record exists.")

    def attach(self, record, step_duration):
        moles = np.asarray(record.total_moles(), dtype=float)
        if moles.sum() > 0.0:
            self.composition = moles / moles.sum()

    def face(self, model, end, time_s, cell):
        pressure = model.to_pressure(self.pressure(time_s))
        v = model.velocity_for_pressure(pressure, cell, end)
        inflow = max(v if end == "feed" else -v, 0.0)
        y, t = model.danckwerts(cell, model.to_mole_fractions(self.composition),
                                model.to_temperature(self.temperature), inflow, end)
        return FaceState(v, pressure, y, t)


class PressureOutlet(Boundary):
    """
    Gas leaving while the face pressure is held or ramped: a back pressure
    regulator (constant pressure) or a vacuum pump following a prescribed
    pressure history (blowdown and evacuation).

    Args:
        pressure: Face pressure in bar, a number or a history.
        discharge_pressure: Pressure the gas is delivered to, bar. Energy is
            charged for pumping from the face pressure up to it.
        efficiency: Vacuum pump efficiency, a number or a function of the
            suction pressure such as ``falling_vacuum_efficiency``.
        stream: Stream label, "waste" by default; use "product" for the
            stream whose purity and recovery are reported.
        record_as: Name under which the outgoing stream is recorded, so that
            a later step can use it (``BufferInlet``, ``TankInlet``).
        check_valve: Let gas only leave: while the column is below the set
            pressure the face stays closed (for example while the feed brings
            a column back up from sub-atmospheric pressure).
    """

    role = "outlet"

    def __init__(self, pressure, discharge_pressure=1.0, efficiency: Efficiency = 0.72, stream="waste",
                 record_as=None, check_valve=False):
        self.pressure, self.discharge_pressure = as_history(pressure), discharge_pressure
        self.efficiency, self.stream, self.record_as = efficiency, stream, record_as
        self.check_valve = check_valve

    def face(self, model, end, time_s, cell):
        pressure = model.to_pressure(self.pressure(time_s))
        v = model.velocity_for_pressure(pressure, cell, end)
        outward = v > 0.0 if end == "product" else v < 0.0
        if self.check_valve and not outward:
            """The column is below the set pressure: the valve stays shut instead of letting gas back in."""
            return FaceState(0.0, cell.pressure, cell.mole_fractions, cell.temperature)
        return FaceState(v, pressure, cell.mole_fractions, cell.temperature)


class BufferInlet(Boundary):
    """
    Gas entering from a recorded stream, replayed in time: the molar flow,
    composition and temperature follow the record of another step. Used for
    light product pressurization and light reflux, where the raffinate of
    the adsorption step enters the column at the product end.

    The molar flow is imposed, so the velocity follows from v P / T at the
    face (equivalently v = v_source P_source / P). When the record is exhausted or not yet available
    (first cycle), no gas enters.

    Args:
        buffer: Name of the record (``record_as`` of the source outlet).
        offset: Time into the record at which the replay starts, s. Lets two
            steps share one record, for example light reflux using the first
            t_LR seconds of the raffinate and pressurization the rest.
        stream: Stream label of the recycled gas.
        max_pressure: Pressure of the source, bar. Gas cannot enter a column
            at a higher pressure than the one it is supplied at, so when the
            recorded flow would need a higher face pressure, the face is held
            at ``max_pressure`` and the flow follows from the pressure drop
            instead (it falls to zero as the column approaches that
            pressure). Without it the recorded flow is imposed whatever the
            pressure, which over-pressurizes the column near the end of a
            light product pressurization.
        composition: Replay only the recorded molar flow, with this gas
            instead of the recorded one (mole fractions, dict or sequence),
            for example a single column simulation of pressure equalization
            with pure product gas (Ribeiro et al., Chem. Eng. Sci. 63 (2008)
            5258). None (default) replays the recorded composition.
    """

    role = "inlet"

    def __init__(self, buffer, offset=0.0, stream="raffinate", max_pressure=None, composition=None):
        self.buffer, self.offset, self.stream, self.max_pressure = buffer, offset, stream, max_pressure
        self.composition = composition
        self.record = None

    def attach(self, record, step_duration):
        self.record = record

    def face(self, model, end, time_s, cell):
        if self.record is None:
            return FaceState(0.0, cell.pressure, cell.mole_fractions, cell.temperature)
        flow, y_full, t_kelvin = self.record.at(time_s + self.offset)
        if self.composition is not None:
            y_full = model.gas.composition(self.composition)
        face = model.molar_inflow_face(flow, y_full, t_kelvin, cell, end)
        if self.max_pressure is None:
            return face
        cap = model.to_pressure(self.max_pressure)
        if face.pressure <= cap:
            return face
        """
        The source cannot push the recorded flow: hold the face at the source pressure and let the pressure difference
        set the flow.
        """
        v = model.velocity_for_pressure(cap, cell, end)
        inflow = max(v if end == "feed" else -v, 0.0)
        y, t = model.danckwerts(cell, np.asarray(y_full)[: model.n_independent], model.to_temperature(t_kelvin),
                                inflow, end)
        return FaceState(inflow if end == "feed" else -inflow, cap, y, t)


class TankInlet(Boundary):
    """
    Gas entering at a constant molar flow from a well mixed tank, used for
    heavy reflux: a fraction of the extract of a recorded step is sent back
    to the feed end over the duration of this step. The composition is the mean composition of the
    recorded stream.

    Args:
        buffer: Name of the record that fills the tank.
        fraction: Fraction of the recorded moles fed back (reflux ratio).
        temperature: Temperature of the gas leaving the tank, K.
        supply_pressure: Tank pressure, bar; a compressor is charged above it.
        efficiency: Compressor efficiency.
        stream: Stream label, "product" by default, so that the refluxed
            gas is subtracted from the product.
    """

    role = "inlet"

    def __init__(self, buffer, fraction, temperature=298.15, supply_pressure=1.0,
                 efficiency: Efficiency = 0.72, stream="product"):
        self.buffer, self.fraction, self.temperature = buffer, fraction, temperature
        self.supply_pressure, self.efficiency, self.stream = supply_pressure, efficiency, stream
        self.flow, self.composition = 0.0, None

    def attach(self, record, step_duration):
        moles = record.total_moles()
        self.flow = self.fraction * moles.sum() / step_duration
        self.composition = moles / moles.sum()

    def face(self, model, end, time_s, cell):
        if self.composition is None or self.flow <= 0.0:
            return FaceState(0.0, cell.pressure, cell.mole_fractions, cell.temperature)
        return model.molar_inflow_face(self.flow, self.composition, self.temperature, cell, end)


class VacuumPump(Boundary):
    """
    A vacuum pump of constant volumetric flow drawing gas out of the column.

    The pump fixes the actual volumetric flow at the column end, and hence
    the interstitial velocity v = S / (eps A), while the face pressure
    follows from the pressure drop over half a cell. How long it takes to
    reach a given vacuum is then a result of the simulation rather than an
    input, so the step is usually ended with ``stop=PressureReached(...)``.
    This gives more realistic step times and pump sizes than prescribed
    pressure histories.

    The pump is given either by its capacity or by the interstitial velocity
    it draws. The
    velocity keeps its meaning when an optimizer changes the column
    diameter; the capacity is then computed from it.

    Args:
        volumetric_flow: Pump capacity at suction conditions, m3/h.
        velocity: Interstitial velocity at the column end, m/s (instead of
            ``volumetric_flow``).
        discharge_pressure: Pressure the gas is delivered to, bar.
        efficiency: Pump efficiency, a number or a function of the suction
            pressure such as ``falling_vacuum_efficiency``.
        stream: Stream label.
        record_as: Name under which the pumped stream is recorded.
    """

    role = "outlet"

    def __init__(self, volumetric_flow=None, discharge_pressure=1.01325, efficiency: Efficiency = 0.72,
                 stream="waste", record_as=None, velocity=None):
        if (volumetric_flow is None) == (velocity is None):
            raise ValueError("Give either volumetric_flow (m3/h) or velocity (m/s), not both.")
        self.volumetric_flow, self.velocity = volumetric_flow, velocity
        self.discharge_pressure, self.efficiency = discharge_pressure, efficiency
        self.stream, self.record_as = stream, record_as

    def interstitial_velocity(self, model) -> float:
        """Velocity drawn at the column end, m/s."""
        if self.velocity is not None:
            return float(self.velocity)
        return self.volumetric_flow / 3600.0 / (model.column.bed_voidage * model.column.cross_section)

    def capacity(self, model) -> float:
        """Pump capacity at suction conditions, m3/h."""
        if self.volumetric_flow is not None:
            return float(self.volumetric_flow)
        return self.velocity * model.column.bed_voidage * model.column.cross_section * 3600.0

    def face(self, model, end, time_s, cell):
        speed = model.to_velocity(self.interstitial_velocity(model))
        v = -speed if end == "feed" else speed
        pressure = model.pressure_for_velocity(v, cell, end)
        return FaceState(v, pressure, cell.mole_fractions, cell.temperature)


STANDARD_MOLAR_VOLUME = 22.414e-3
"""Molar volume of an ideal gas at 0 C and 1 atm, m3/mol, for flows given in standard litres."""


def standard_litres_per_minute(flow):
    """Molar flow in mol/s of a gas flow given in standard litres per minute (0 C, 1 atm)."""
    return flow / 1000.0 / STANDARD_MOLAR_VOLUME / 60.0


class FlowInlet(Boundary):
    """
    Gas entering at a fixed molar flow, as set by a mass flow controller.

    The velocity follows from the flow and the face pressure; composition
    and temperature from the Danckwerts conditions. With ``max_pressure`` the
    face cannot rise above the supply pressure: once the flow would need a
    higher face pressure, the face is held there and the flow follows from
    the pressure drop (a pressurization that stops at the supply pressure).

    Args:
        molar_flow: mol/s (use ``standard_litres_per_minute`` for flows in L/min STP).
        composition: Mole fractions of the entering gas, or the name of a
            recorded stream whose mean composition enters.
        default_composition: Composition used while that record does not exist yet.
        temperature: K.
        max_pressure: Supply pressure, bar, or None for no limit.
        supply_pressure: Pressure of the supply, bar; a compressor is charged above it.
        efficiency: Efficiency of that compressor.
        stream: Stream label.
    """

    role = "inlet"

    def __init__(self, molar_flow, composition, temperature=298.15, max_pressure=None, supply_pressure=None,
                 efficiency: Efficiency = 0.72, stream="feed", default_composition=None, record_as=None):
        self.molar_flow, self.composition, self.temperature = molar_flow, composition, temperature
        self.max_pressure, self.supply_pressure, self.efficiency, self.stream = (max_pressure, supply_pressure,
                                                                                efficiency, stream)
        """
        A composition given as the name of a recorded stream is its mean composition (``default_composition`` until the
        record exists).
        """
        self.buffer = composition if isinstance(composition, str) else None
        self.default_composition, self.record, self.record_as = default_composition, None, record_as

    def attach(self, record, duration):
        self.record = record

    def face(self, model, end, time_s, cell):
        if self.buffer is None:
            y_full = model.gas.composition(self.composition)
        elif self.record is not None:
            y_full = self.record.mean_composition()
        else:
            y_full = model.gas.composition(self.default_composition)
        face = model.molar_inflow_face(self.molar_flow, y_full, self.temperature, cell, end)
        if self.max_pressure is None or face.pressure <= model.to_pressure(self.max_pressure):
            return face
        cap = model.to_pressure(self.max_pressure)
        v = model.velocity_for_pressure(cap, cell, end)
        inflow = max(v if end == "feed" else -v, 0.0)
        y, t = model.danckwerts(cell, y_full[: model.n_independent], model.to_temperature(self.temperature), inflow, end)
        return FaceState(inflow if end == "feed" else -inflow, cap, y, t)


class MixingInlet(Boundary):
    """
    Fresh gas at a fixed molar flow mixed with gas recycled from a well mixed
    tank, for a recycle into the feed: the outlet of an earlier step is
    collected (``record_as``) and sent back, spread evenly over this step,
    together with the fresh feed.

    The recycled gas enters at ``temperature`` like the fresh gas (it is
    cooled to the feed temperature). The fresh moles count as the stream
    ``stream`` and the recycled ones as ``recycle_stream``, so that recovery
    is computed on the fresh feed only and the recycle nets out against the
    step it came from (label that outlet with the same ``recycle_stream``).

    Args:
        molar_flow: Fresh gas, mol/s.
        composition: Mole fractions of the fresh gas.
        buffer: Name of the record that fills the tank.
        fraction: Fraction of the recorded moles sent back.
        temperature: Temperature of the mixture, K.
        supply_pressure: Pressure of the supply, bar; a compressor is charged above it.
        efficiency: Efficiency of that compressor.
        stream: Label of the fresh gas ("feed").
        recycle_stream: Label of the recycled gas ("recycle").
    """

    role = "inlet"

    def __init__(self, molar_flow, composition, buffer, fraction=1.0, temperature=298.15, supply_pressure=None,
                 efficiency: Efficiency = 0.72, stream="feed", recycle_stream="recycle"):
        self.molar_flow, self.composition, self.buffer, self.fraction = molar_flow, composition, buffer, fraction
        self.temperature, self.supply_pressure, self.efficiency = temperature, supply_pressure, efficiency
        self.stream, self.recycle_stream = stream, recycle_stream
        self.tank_flow, self.tank_moles = 0.0, None

    def attach(self, record, step_duration):
        moles = np.asarray(record.total_moles(), dtype=float)
        self.tank_moles = self.fraction * moles
        self.tank_flow = float(self.tank_moles.sum()) / step_duration

    def face(self, model, end, time_s, cell):
        fresh = np.asarray(model.gas.composition(self.composition), dtype=float)
        total = self.molar_flow + self.tank_flow
        y_full = fresh if self.tank_flow <= 0.0 else (
            (self.molar_flow * fresh + self.tank_moles / self.tank_moles.sum() * self.tank_flow) / total)
        return model.molar_inflow_face(total, y_full, self.temperature, cell, end)

    def stream_moles(self, model, moles_in, duration):
        """Moles that entered, split into the fresh feed and the recycle."""
        fresh = self.molar_flow * duration * np.asarray(model.gas.composition(self.composition), dtype=float)
        return {self.stream: fresh, self.recycle_stream: np.asarray(moles_in, dtype=float) - fresh}


class RecordedPressure:
    """
    A reservoir pressure taken from the face pressure that another step
    recorded (``record_as``), typically in the previous cycle.

    In a multi-bed plant a bed that equalizes or supplies purge vents into
    another bed, whose pressure changes during the step. A single-column
    simulation records the face pressure of the receiving step and uses it
    as the downstream pressure of the donor step one cycle later; at cyclic
    steady state the two are consistent.

    Args:
        buffer: Name of the record.
        default: Pressure in bar before any record exists (first cycle), a
            number or a function of time in s.
    """

    def __init__(self, buffer, default):
        self.buffer, self.default, self.record = buffer, default, None

    def attach(self, record, duration):
        self.record = record if record is not None and record.pressure is not None else None

    def __call__(self, time_s):
        if self.record is None:
            return float(self.default(time_s) if callable(self.default) else self.default)
        return self.record.pressure_at(min(time_s, self.record.duration))


def _reservoir_pressure(pressure, time_s):
    return float(pressure(time_s) if callable(pressure) else pressure)


class Valve(Boundary):
    """
    A linear valve between the column end and a reservoir,

        molar flow = cv * (upstream pressure - downstream pressure),

    the flow law of the valves of most cycle simulators. The face pressure
    and flow are solved together with the pressure drop over half a cell.

    Args:
        cv: Valve coefficient, mol/(s bar).
        pressure: Reservoir pressure in bar: a number, a function of time
            in s, or a ``RecordedPressure`` (the pressure of another step).
        direction: "out" for gas leaving the column (the reservoir is
            downstream), "in" for gas entering it from the reservoir.
        composition: Gas of the reservoir, for "in".
        temperature: Temperature of the reservoir gas, K, for "in".
        extra_outflow: A fixed molar flow (mol/s) leaving through the same end
            in addition to the valve flow, for "out" (for example product
            sent to repressurize another bed).
        reversible: Allow flow against ``direction`` (the reservoir gas then
            enters, or the column gas leaves). Off by default.
        stream: Stream label.
        record_as: Name under which the flow is recorded.
    """

    def __init__(self, cv, pressure, direction="out", composition=None, temperature=298.15, extra_outflow=0.0,
                 reversible=False, stream=None, record_as=None):
        if direction not in ("out", "in"):
            raise ValueError("Valve direction must be 'out' or 'in'.")
        if not cv >= 0.0:
            raise ValueError(f"Valve cv must be zero or positive (mol/(s bar)), got {cv!r}.")
        if direction == "in" and composition is None:
            raise ValueError("A valve letting gas in needs the composition of the reservoir.")
        self.cv, self.pressure, self.direction = cv, pressure, direction
        self.composition, self.temperature, self.extra_outflow = composition, temperature, extra_outflow
        self.reversible, self.record_as = reversible, record_as
        self.role = "outlet" if direction == "out" else "inlet"
        self.stream = stream or ("product" if direction == "out" else "feed")
        self.buffer = getattr(pressure, "buffer", None)

    def attach(self, record, duration):
        if hasattr(self.pressure, "attach"):
            self.pressure.attach(record, duration)

    def face(self, model, end, time_s, cell):
        from scipy.optimize import brentq
        c = model.column
        to_flux = 1.0 / (c.bed_voidage * c.cross_section * model.v0 * model.c0)
        """to_flux: mol/s -> dimensionless flux."""
        reservoir = model.to_pressure(_reservoir_pressure(self.pressure, time_s))
        cv = self.cv * model.P0 / 1.0e5 * to_flux
        """cv: per dimensionless pressure."""
        extra = self.extra_outflow * to_flux
        """Velocity sign of an outflow."""
        sign_out = -1.0 if end == "feed" else 1.0
        if self.direction == "in":
            y_full = model.gas.composition(self.composition)
            t_in = model.to_temperature(self.temperature)
        else:
            y_full, t_in = None, cell.temperature

        def outflow(face_pressure):
            """Net dimensionless molar flux leaving the column for a face pressure."""
            driving = face_pressure - reservoir if self.direction == "out" else -(reservoir - face_pressure)
            if not self.reversible:
                driving = max(driving, 0.0) if self.direction == "out" else min(driving, 0.0)
            return cv * driving + extra

        inertial = model.end_inertial(cell, end)

        def mismatch(face_pressure):
            """Face pressure minus the one needed to drive the valve flow through half a cell (no clipping)."""
            flux = outflow(face_pressure)
            temperature = cell.temperature if flux >= 0.0 else t_in
            velocity = sign_out * flux * temperature / face_pressure
            drop = model._half_cell_drop(velocity, inertial, end)
            needed = cell.pressure + drop if end == "feed" else cell.pressure - drop
            return face_pressure - needed

        """
        The physical root is the one nearest the cell pressure: search outward from it (down for an outflow, up for an
        inflow) for a sign change. A fixed side draw also gives a spurious root at a vanishing face pressure, which this
        search never reaches.
        """
        at_cell = mismatch(cell.pressure)
        face_pressure = cell.pressure
        solved = at_cell == 0.0
        if not solved:
            """
            Secant iterations from the cell pressure converge to the nearest, physical root in a few evaluations; the
            outward search below is the fallback when they do not.
            """
            x0, f0 = cell.pressure, at_cell
            x1 = cell.pressure * (1.0 - 1.0e-4) if at_cell > 0.0 else cell.pressure * (1.0 + 1.0e-4)
            f1 = mismatch(x1)
            for _ in range(30):
                if f1 == f0:
                    break
                x2 = x1 - f1 * (x1 - x0) / (f1 - f0)
                if not 0.5 * cell.pressure < x2 < 2.0 * max(cell.pressure, reservoir) + 1.0e-3:
                    break
                x0, f0, x1 = x1, f1, x2
                f1 = mismatch(x1)
                if abs(x1 - x0) < 1.0e-13 * max(1.0, x1):
                    face_pressure, solved = x1, True
                    break
        if not solved and at_cell != 0.0:
            step = 1.0e-7 * cell.pressure
            inner = cell.pressure
            while step < 1.0e3 * cell.pressure:
                outer = cell.pressure - step if at_cell > 0.0 else cell.pressure + step
                if outer <= 0.0:
                    break
                value = mismatch(outer)
                if (value > 0.0) != (at_cell > 0.0):
                    low, high = sorted((inner, outer))
                    face_pressure = brentq(mismatch, low, high, xtol=1.0e-14, rtol=1.0e-12)
                    break
                inner, step = outer, step * 2.0
        flux = outflow(face_pressure)
        if flux >= 0.0:
            return FaceState(sign_out * flux * cell.temperature / face_pressure, face_pressure, cell.mole_fractions,
                             cell.temperature)
        speed = -flux * t_in / face_pressure
        y_in = y_full if y_full is not None else model.full_composition(cell.mole_fractions[:, None])[:, 0]
        y, t = model.danckwerts(cell, np.asarray(y_in)[: model.n_independent], t_in, speed, end)
        return FaceState(-sign_out * speed, face_pressure, y, t)
