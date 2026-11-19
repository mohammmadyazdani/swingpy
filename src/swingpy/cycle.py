"""
Cycles, cyclic steady state and key performance indicators.

A cycle is a list of steps run one after the other on a single column; the
state at the end of a step is the initial state of the next one. The key
performance indicators are computed from the stream labels of the boundary
conditions, so they work for any cycle: the purity and recovery of the
``key_component`` are taken in the stream labelled ``product`` against the
streams labelled ``feed``.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import List

import numpy as np

from .model import StepTooStiff


@dataclass
class CyclePerformance:
    """
    Key performance indicators of one cycle.

    Attributes:
        purity: Mole fraction of the key component in the product stream, %.
        recovery: Key component in the product over key component fed, %.
        mass_balance_error: |in - out| / in of the key component over the cycle, %.
        specific_energy: Machine work per tonne of key component in the product, kWh/t.
        productivity: Key component in the product per adsorbent volume and
            cycle time, mol/(m3 s). The adsorbent volume is (1 - eps) A L.
        cycle_time: Sum of the actual step durations, s (steps with a stop
            condition may end early or be skipped).
        energy_by_step: Machine work of each step, J.
        streams: Net moles of each component in each labelled stream per
            cycle, positive out of the column.
        heat_by_step: Heat supplied to the column through its wall in each
            step, J (negative when the column is cooled).
        gas_heating_by_step: Heat needed in each step to bring the gas that
            enters from ``Cycle.gas_supply_temperature`` up to its inlet
            temperature (a hot purge or regeneration gas), J.
        thermal_energy: Heat supplied over the cycle, through the wall in the
            heating steps and to hot inlet gas, per tonne of key component in
            the product, GJ/t; the main energy demand of temperature swing
            cycles.
        residual: Normalized residual error of Subraveti, Karimi and Gazzani
            (2026, eq 2): over the steps, the largest 100 x sum of squares of
            the change of the dimensionless profiles at the end of the step
            from the previous cycle; filled in by ``Cycle.run``.
        profile_change: Largest change of the column profiles over the cycle
            (dimensionless, see ``ColumnModel.profile_change``); filled in by
            ``Cycle.run``.
    """

    purity: float
    recovery: float
    mass_balance_error: float
    specific_energy: float
    productivity: float
    cycle_time: float
    energy_by_step: dict = field(default_factory=dict)
    streams: dict = field(default_factory=dict)
    profile_change: float = float("nan")
    residual: float = float("nan")
    heat_by_step: dict = field(default_factory=dict)
    gas_heating_by_step: dict = field(default_factory=dict)
    thermal_energy: float = 0.0


@dataclass
class CycleRun:
    """
    Result of ``Cycle.run``.

    Attributes:
        results: Step results of the last cycle.
        performance: KPIs of the last cycle.
        history: KPIs of every cycle.
        converged: Whether cyclic steady state was reached.
        recent: Step results of the last few cycles (oldest first), kept for
            plots of the approach to steady state; see ``Cycle.run``.
    """

    results: list
    performance: CyclePerformance
    history: List[CyclePerformance]
    converged: bool
    recent: list = field(default_factory=list)

    @property
    def final_state(self):
        return self.results[-1].final_state


class Cycle:
    """
    A sequence of steps and the definition of what it produces.

    Args:
        steps: The steps, in order.
        key_component: Component whose purity and recovery are reported, for example "CO2".
        product_stream: Label of the product stream.
        feed_stream: Label of the feed stream(s).
    """

    def __init__(self, steps, key_component, product_stream="product", feed_stream="feed",
                 gas_supply_temperature=298.15):
        self.steps = list(steps)
        self.key_component = key_component
        """Gas entering hotter than this has to be heated: its heat counts in the thermal energy."""
        self.gas_supply_temperature = gas_supply_temperature
        self.product_stream = product_stream
        self.feed_stream = feed_stream
        self.buffers = {}

    @property
    def cycle_time(self) -> float:
        return sum(step.duration for step in self.steps)

    def evaluate(self, model, results) -> CyclePerformance:
        """KPIs of one cycle from the step results."""
        key = model.gas.index(self.key_component)
        """
        Net amount of each labelled stream leaving the column (negative for streams that enter it). A recycled stream
        that leaves in one step and re-enters in another under the same label nets out.
        """
        streams = defaultdict(lambda: np.zeros(model.n_components))
        inflow = np.zeros(model.n_components)
        outflow = np.zeros(model.n_components)
        for result in results:
            for end, boundary in (("feed", result.step.feed_end), ("product", result.step.product_end)):
                moles_in = result.moles_in[end]
                if boundary.role == "closed":
                    continue
                if hasattr(boundary, "stream_moles"):
                    for label, moles in boundary.stream_moles(model, moles_in, result.time[-1]).items():
                        streams[label] -= moles
                else:
                    streams[boundary.stream] -= moles_in
                inflow += np.maximum(moles_in, 0.0)
                outflow += np.maximum(-moles_in, 0.0)
        product = streams[self.product_stream]
        fed = -streams[self.feed_stream]
        cycle_time = sum(result.time[-1] for result in results)
        energy = {result.step.name: sum(result.work.values()) for result in results}
        heat = {r.step.name: model.wall_heat(r) for r in results}
        gas_heating = {}
        for r in results:
            duty = 0.0
            for end, boundary in (("feed", r.step.feed_end), ("product", r.step.product_end)):
                temperature = getattr(boundary, "temperature", None)
                entering = float(np.sum(np.maximum(r.moles_in[end], 0.0)))
                if temperature is not None and temperature > self.gas_supply_temperature and entering > 0.0:
                    duty += entering * model.gas.heat_capacity * (temperature - self.gas_supply_temperature)
            gas_heating[r.step.name] = duty
        product_tonnes = product[key] * model.gas.molar_masses[key] / 1000.0
        return CyclePerformance(
            purity=100.0 * product[key] / product.sum(),
            recovery=100.0 * product[key] / fed[key],
            mass_balance_error=100.0 * abs(inflow[key] - outflow[key]) / inflow[key],
            specific_energy=sum(energy.values()) / 3.6e6 / product_tonnes,
            productivity=product[key] / (model.column.adsorbent_volume * cycle_time),
            cycle_time=cycle_time,
            energy_by_step=energy,
            heat_by_step=heat,
            gas_heating_by_step=gas_heating,
            thermal_energy=(sum(q for q in heat.values() if q > 0.0) + sum(gas_heating.values())) / 1.0e9
            / product_tonnes,
            streams=dict(streams),
        )

    def run_once(self, model, state, buffers=None, n_output=50, record_points=200, fallback=True):
        """
        Simulate the steps once and return the list of StepResult.

        ``buffers`` holds the recorded streams (name to StreamRecord). Steps
        that read a record get it before they run, and outlets with a
        ``record_as`` name store their stream after they run, so a record
        made in one cycle is still available to steps earlier in the next
        cycle (for example the heavy reflux tank).
        """
        buffers = self.buffers if buffers is None else buffers
        results = []
        for step in self.steps:
            for boundary in (step.feed_end, step.product_end):
                if hasattr(boundary, "attach") and boundary.buffer in buffers:
                    boundary.attach(buffers[boundary.buffer], step.duration)
            records = [(end, b.record_as) for end, b in (("feed", step.feed_end), ("product", step.product_end))
                       if b.record_as]
            result = model.simulate(step, state, n_output=record_points if records else n_output, fallback=fallback)
            for end, name in records:
                buffers[name] = model.outflow_record(result, end)
            results.append(result)
            state = result.final_state
        return results

    def run(self, model, initial_state, max_cycles=600, tolerance=0.5, consecutive=5,
            report_every=25, verbose=True, keep_cycles=10, criterion="both", profile_tolerance=1.0e-3,
            accelerate=False, acceleration_memory=5, acceleration_start=None, acceleration_every=3,
            acceleration_stop=1.0e-4, residual_tolerance=1.0e-5, acceleration_max_jump=100.0) -> CycleRun:
        """
        Repeat the cycle until cyclic steady state.

        Two tests of steady state are available, both checked over
        ``consecutive`` cycles in a row:

            "mass_balance"  the key component mass balance error over a cycle
                            is below ``tolerance`` (%), the usual criterion in
                            the literature
            "profiles"      no profile of the column changes by more than
                            ``profile_tolerance`` over a cycle (dimensionless)
            "both"          both of the above (default)
            "residual"      the normalized residual error (eq 2 of Subraveti,
                            Karimi and Gazzani 2026, ``CyclePerformance.residual``)
                            is below ``residual_tolerance`` and the mass
                            balance error below ``tolerance``, the criterion
                            of that paper (use with ``consecutive=1``)

        The mass balance alone can be met while the column is still slowly
        warming or loading, so "both" is the safer choice; use
        "mass_balance" to apply only the usual mass balance rule.

        Acceleration (see ``swingpy.acceleration``) starts some cycles from
        an extrapolated state instead of the end of the last cycle:

            accelerate="graves_morris"   or another vector Steffensen method of
                                         ``acceleration.STEFFENSEN_METHODS``
                                         (Subraveti, Karimi and Gazzani 2026):
                                         ``acceleration_start`` plain cycles
                                         (3 by default), then an extrapolation
                                         every ``acceleration_every`` cycles
                                         until the residual is below
                                         ``acceleration_stop``; an extrapolation
                                         that jumps more than
                                         ``acceleration_max_jump`` times the last
                                         cycle's change is discarded
            accelerate=True or "anderson"  Anderson mixing of the last
                                         ``acceleration_memory`` cycles, after
                                         ``acceleration_start`` (5 by default)

        The KPIs and the
        steady state tests are still those of simulated cycles. The step
        results of the last ``keep_cycles`` cycles are kept in
        ``CycleRun.recent``.
        """
        if criterion not in ("mass_balance", "profiles", "both", "residual"):
            raise ValueError("criterion must be 'mass_balance', 'profiles', 'both' or 'residual'.")
        state, history, streak, recent = initial_state, [], 0, []
        accelerator, previous_steps = None, None
        if accelerate is True or accelerate == "anderson":
            from .acceleration import AndersonAccelerator
            accelerator = AndersonAccelerator(model, memory=acceleration_memory,
                                              start=5 if acceleration_start is None else acceleration_start)
        elif accelerate:
            from .acceleration import SteffensenAccelerator
            accelerator = SteffensenAccelerator(model, method=accelerate,
                                                start=3 if acceleration_start is None else acceleration_start,
                                                every=acceleration_every, stop_residual=acceleration_stop,
                                                max_jump=acceleration_max_jump)
        n_profiles = model.state_size - 2 * (model.n_components + 2)
        if verbose:
            print("  Cycle    Purity [%]   Recovery [%]   Mass balance error [%]   Profile change [-]")
        """The simulated end state that an extrapolation replaced, if any."""
        plain = None
        for number in range(1, max_cycles + 1):
            previous = state
            if plain is None:
                results = self.run_once(model, state)
            else:
                """
                An extrapolated start can be a state the stiff solver cannot integrate in reasonable time (seen on
                adiabatic pilot-scale columns): such an extrapolation is discarded and the cycle is run from the
                simulated state instead. The records of a discarded attempt are restored.
                """
                saved = dict(self.buffers)
                try:
                    results = self.run_once(model, state, fallback=False)
                except StepTooStiff:
                    self.buffers.clear()
                    self.buffers.update(saved)
                    accelerator.rejected = getattr(accelerator, "rejected", 0) + 1
                    previous = state = plain
                    results = self.run_once(model, state)
            start_state, state = state, results[-1].final_state
            performance = self.evaluate(model, results)
            performance.profile_change = model.profile_change(previous, state)
            steps = [np.asarray(r.final_state, dtype=float)[:n_profiles] for r in results]
            if previous_steps is not None and len(previous_steps) == len(steps):
                performance.residual = max(100.0 * float(np.sum((a - b) ** 2)) for a, b in zip(steps, previous_steps))
            previous_steps = steps
            history.append(performance)
            recent = (recent + [results])[-keep_cycles:] if keep_cycles else []
            balance_ok = performance.mass_balance_error < tolerance
            profiles_ok = performance.profile_change < profile_tolerance
            residual_ok = performance.residual < residual_tolerance
            passed = {"mass_balance": balance_ok, "profiles": profiles_ok, "both": balance_ok and profiles_ok,
                      "residual": balance_ok and residual_ok}[criterion]
            streak = streak + 1 if passed else 0
            done = streak >= consecutive
            if verbose and (number == 1 or number % report_every == 0 or done):
                print(f"  {number:5d}   {performance.purity:11.2f}   {performance.recovery:12.2f}   "
                      f"{performance.mass_balance_error:22.4f}   {performance.profile_change:18.2e}")
            plain = None
            if accelerator is not None and not done:
                simulated = state
                state = accelerator.next_state(start_state, state, performance.residual)
                if not np.array_equal(state, simulated):
                    plain = simulated
            if done:
                return CycleRun(results, performance, history, True, recent)
        return CycleRun(results, performance, history, False, recent)
