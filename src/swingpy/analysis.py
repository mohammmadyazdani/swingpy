"""
Numbers of a simulation, without plotting: breakthrough metrics, the tables
of a cycle, and export of profiles and histories to CSV files.

    from swingpy import analysis

    metrics = analysis.breakthrough_metrics(model, result, "CO2")
    rows = analysis.step_table(model, run.results)
    analysis.export_run(model, run, "Output/Case one")   # CSV files

Flows are molar flows in mol/s, positive out of the column; pressures in
bar, temperatures in K, loadings in mol/kg, positions Z from 0 (feed end)
to 1 (product end).
"""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

GAS_CONSTANT = 8.314


"""Histories at the column ends."""

def end_flows(model, result, end="product"):
    """
    Molar flow (mol/s, positive out of the column), composition, temperature
    and pressure of the gas crossing one end during a step.

    Returns:
        dict with "time" (s), "flow" (total, mol/s), "component_flows"
        (components x times, mol/s), "mole_fractions", "temperature" (K),
        "pressure" (bar).
    """
    h = model.end_history(result, end)
    c = model.column
    sign = -1.0 if end == "feed" else 1.0
    flow = sign * h["velocity"] * c.bed_voidage * c.cross_section * h["pressure"] * 1.0e5 \
        / (GAS_CONSTANT * h["temperature"])
    return {"time": np.asarray(result.time), "flow": flow, "component_flows": h["mole_fractions"] * flow,
            "mole_fractions": h["mole_fractions"], "temperature": h["temperature"], "pressure": h["pressure"]}


def column_profiles(model, state):
    """
    Profiles along the column of one state: position Z, pressure (bar),
    temperature (K), wall temperature (K), mole fractions and loadings
    (mol/kg, components x cells).
    """
    p = model.unpack(state)
    return {"Z": (np.arange(model.n_cells) + 0.5) * model.cell_width,
            "P": p["P"] * model.P0 / 1.0e5, "T": p["T"] * model.T0, "T_wall": p["T_wall"] * model.T0,
            "y": model.full_composition(p["y"]), "q": p["x"] * model.qs0}


"""Ideal gas check."""

def compressibility_range(model, results, eos, tolerance=0.05):
    """
    How far the gas is from ideal over a set of step results, the check of
    Casas et al. (Adsorption 18 (2012) 143) for a column model that uses the
    ideal gas law: the compressibility factor Z of an equation of state
    (``swingpy.eos``) at every output time and cell.

    Args:
        model: The ColumnModel.
        results: StepResults (a cycle, or one breakthrough step).
        eos: An equation of state with the components of the model, in order.
        tolerance: Largest |Z - 1| reported as acceptable.

    Returns:
        Dict with the smallest and largest Z, where they occur (step, time
        in s, position Z, pressure in bar, temperature in K) and ``ideal``:
        whether |Z - 1| stays within ``tolerance``.
    """
    if hasattr(results, "states"):
        results = [results]
    if getattr(eos, "n_components", model.n_components) != model.n_components:
        raise ValueError(f"The equation of state has {eos.n_components} components, the model {model.n_components}.")
    lowest, highest = None, None
    for r in results:
        for k in range(r.states.shape[1]):
            prof = column_profiles(model, r.states[:, k])
            Z = eos.compressibility(prof["T"], prof["P"], np.clip(prof["y"], 0.0, 1.0))
            for j, pick in ((int(np.argmin(Z)), "low"), (int(np.argmax(Z)), "high")):
                entry = dict(Z=float(Z[j]), step=r.step.name, time=float(r.time[k]), position=float(prof["Z"][j]),
                             pressure=float(prof["P"][j]), temperature=float(prof["T"][j]))
                if pick == "low" and (lowest is None or entry["Z"] < lowest["Z"]):
                    lowest = entry
                if pick == "high" and (highest is None or entry["Z"] > highest["Z"]):
                    highest = entry
    deviation = max(abs(lowest["Z"] - 1.0), abs(highest["Z"] - 1.0))
    return {"Z_min": lowest, "Z_max": highest, "max_deviation": deviation, "tolerance": tolerance,
            "ideal": deviation <= tolerance}


"""The functions below measure a single feed step: times, capacity and the mass transfer zone."""

@dataclass
class BreakthroughMetrics:
    """
    Characteristic numbers of a breakthrough of one component.

    Attributes:
        component: The component.
        feed_mole_fraction: Its mole fraction in the feed.
        breakthrough_times: {level: time, s} at which the outlet mole fraction
            first reaches ``level`` times the feed value (NaN if never).
        stoichiometric_time: Time at which the amount fed equals the amount
            retained by the column at the end of the step, s: the time a
            front without dispersion would break through.
        retained: Moles of the component retained by the column (adsorbed
            and in the gas) over the step, mol.
        dynamic_capacity: Change of the mean loading over the step, mol/kg.
        mass_transfer_zone: Length of the zone between the lowest and the
            highest of ``levels``, estimated for a front of constant pattern
            as the front speed (column length over stoichiometric time)
            times the time between them, m; NaN if the outlet does not reach
            both. A value longer than the column means the front is not of
            constant pattern (for example two transitions in an adiabatic
            column).
        peak_outlet_temperature: Highest outlet temperature, K.
        peak_time: Time of that maximum, s.
    """

    component: str
    feed_mole_fraction: float
    breakthrough_times: dict
    stoichiometric_time: float
    retained: float
    dynamic_capacity: float
    mass_transfer_zone: float
    peak_outlet_temperature: float
    peak_time: float


def _first_crossing(time, values, level):
    above = np.nonzero(values >= level)[0]
    if above.size == 0:
        return float("nan")
    i = above[0]
    if i == 0:
        return float(time[0])
    t0, t1, v0, v1 = time[i - 1], time[i], values[i - 1], values[i]
    return float(t0 + (level - v0) * (t1 - t0) / (v1 - v0)) if v1 != v0 else float(t1)


def breakthrough_metrics(model, result, component, levels=(0.05, 0.5, 0.95)) -> BreakthroughMetrics:
    """
    Breakthrough times, stoichiometric time, retained amount, dynamic
    capacity and mass transfer zone of one component in a feed step.

    Args:
        model: The ColumnModel.
        result: The StepResult of a step fed at the feed end.
        component: Name of the component.
        levels: Outlet mole fractions, as fractions of the feed value.
    """
    k = model.gas.index(component)
    feed = end_flows(model, result, "feed")
    out = end_flows(model, result, "product")
    fed_flow = np.maximum(-feed["component_flows"][k], 0.0)
    if not np.any(fed_flow > 0.0):
        raise ValueError(f"no {component} enters at the feed end in step '{result.step.name}'")
    y_feed = float(np.max(feed["mole_fractions"][k][-feed["time"].size // 4:]))
    ratio = out["mole_fractions"][k] / y_feed if y_feed > 0.0 else np.zeros_like(out["time"])
    times = {level: _first_crossing(out["time"], ratio, level) for level in levels}
    """Exact amounts integrated by the model, not the sampled flows."""
    retained = float(result.moles_in["feed"][k] + result.moles_in["product"][k])
    stoichiometric = retained / float(np.mean(fed_flow[-max(feed["time"].size // 4, 2):]))
    first, last = column_profiles(model, result.initial_state), column_profiles(model, result.final_state)
    capacity = float(np.mean(last["q"][k] - first["q"][k]))
    low, high = times[min(levels)], times[max(levels)]
    zone = model.column.length / stoichiometric * (high - low) if np.isfinite(low) and np.isfinite(high) \
        else float("nan")
    peak = int(np.argmax(out["temperature"]))
    return BreakthroughMetrics(component, y_feed, times, stoichiometric, retained, capacity, zone,
                               float(out["temperature"][peak]), float(out["time"][peak]))


"""Tables of a cycle."""

def step_table(model, results):
    """
    One row per step of a cycle: duration, pressures at both ends at the end
    of the step, moles of each component leaving through each end (negative
    when entering), the stream each end belongs to, machine work and heat
    through the wall.
    """
    rows = []
    for r in results:
        if r.time[-1] <= 0.0:
            continue
        end_state = column_profiles(model, r.final_state)
        row = {"step": r.step.name, "duration [s]": float(r.time[-1]),
               "P feed end [bar]": float(end_state["P"][0]), "P product end [bar]": float(end_state["P"][-1]),
               "T mean [K]": float(np.mean(end_state["T"]))}
        for end, boundary in (("feed", r.step.feed_end), ("product", r.step.product_end)):
            row[f"{end} end stream"] = getattr(boundary, "stream", "") or ""
            moles = -np.asarray(r.moles_in[end])
            for name, value in zip(model.gas.components, moles):
                row[f"{name} out of {end} end [mol]"] = float(value)
        row["work [kJ]"] = sum(r.work.values()) / 1000.0
        row["wall heat [kJ]"] = model.wall_heat(r) / 1000.0
        rows.append(row)
    return rows


def cycle_summary(model, run):
    """
    Performance of the last cycle of a run, the composition of each labelled
    stream and the convergence state.
    """
    p = run.performance
    summary = {"converged": bool(run.converged), "cycles": len(run.history), "purity [%]": p.purity,
               "recovery [%]": p.recovery, "specific energy [kWh/t]": p.specific_energy,
               "productivity [mol/(m3 s)]": p.productivity, "cycle time [s]": p.cycle_time,
               "mass balance error [%]": p.mass_balance_error,
               "profile change [-]": getattr(p, "profile_change", float("nan"))}
    for stream, moles in p.streams.items():
        moles = np.asarray(moles, dtype=float)
        total = moles.sum()
        summary[f"{stream} [mol per cycle]"] = float(total)
        for name, value in zip(model.gas.components, moles):
            summary[f"{stream} y {name} [-]"] = float(value / total) if total else float("nan")
    return summary


def stream_table(model, run):
    """Net moles of each component in each labelled stream per cycle (out of the column positive)."""
    return [{"stream": stream, **{f"{name} [mol]": float(v) for name, v in zip(model.gas.components, moles)},
             "total [mol]": float(np.sum(moles))} for stream, moles in run.performance.streams.items()]


def convergence_table(history):
    """KPIs of every cycle of a run."""
    return [{"cycle": i, "purity [%]": h.purity, "recovery [%]": h.recovery,
             "mass balance error [%]": h.mass_balance_error, "specific energy [kWh/t]": h.specific_energy,
             "productivity [mol/(m3 s)]": h.productivity,
             "profile change [-]": getattr(h, "profile_change", float("nan"))}
            for i, h in enumerate(history, start=1)]


def profile_table(model, results):
    """Profiles along the column at the end of every step (one row per step and cell)."""
    rows = []
    for r in results:
        if r.time[-1] <= 0.0:
            continue
        prof = column_profiles(model, r.final_state)
        for j in range(model.n_cells):
            row = {"step": r.step.name, "Z [-]": prof["Z"][j], "z [m]": prof["Z"][j] * model.column.length,
                   "P [bar]": prof["P"][j], "T [K]": prof["T"][j], "T wall [K]": prof["T_wall"][j]}
            for i, name in enumerate(model.gas.components):
                row[f"y {name} [-]"] = prof["y"][i][j]
            for i, name in enumerate(model.gas.components):
                row[f"q {name} [mol/kg]"] = prof["q"][i][j]
            rows.append(row)
    return rows


def cycle_history_table(model, results, positions=(0.0, 0.25, 0.5, 0.75, 1.0)):
    """
    Time series of one cycle: pressures, temperatures and compositions at a
    few positions, and the flows through both ends (one row per output time).
    """
    cells = [min(int(z * model.n_cells), model.n_cells - 1) for z in positions]
    rows, offset = [], 0.0
    for r in results:
        if r.time[-1] <= 0.0:
            continue
        ends = {end: end_flows(model, r, end) for end in ("feed", "product")}
        for n, (t, state) in enumerate(zip(r.time, r.states.T)):
            prof = column_profiles(model, state)
            row = {"time [s]": offset + t, "step": r.step.name}
            for z, cell in zip(positions, cells):
                row[f"P at Z={z:g} [bar]"] = prof["P"][cell]
                row[f"T at Z={z:g} [K]"] = prof["T"][cell]
                for i, name in enumerate(model.gas.components):
                    row[f"y {name} at Z={z:g} [-]"] = prof["y"][i][cell]
            for end, h in ends.items():
                for i, name in enumerate(model.gas.components):
                    row[f"{name} out of {end} end [mol/s]"] = h["component_flows"][i][n]
            rows.append(row)
        offset += r.time[-1]
    return rows


def step_history_table(model, result, positions=(0.0, 0.25, 0.5, 0.75, 1.0)):
    """Time series of one step (for example a breakthrough), as ``cycle_history_table``."""
    return cycle_history_table(model, [result], positions)


"""The functions below write the tables above as CSV files."""

def write_csv(path, rows):
    """Write a list of dicts (same keys) as a CSV file; a dict is written as key,value rows."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(rows, dict):
        rows = [{"quantity": k, "value": v} for k, v in rows.items()]
    with path.open("w", newline="", encoding="utf-8") as handle:
        if not rows:
            return path
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: (f"{v:.10g}" if isinstance(v, (float, np.floating)) else v) for k, v in row.items()})
    return path


def export_run(model, run, folder, positions=(0.0, 0.25, 0.5, 0.75, 1.0)):
    """
    Write the numbers of a cycle run as CSV files in ``folder``: summary,
    steps, streams, convergence, profiles at the end of every step and the
    time series of the last cycle.

    Returns:
        List of the files written.
    """
    folder = Path(folder)
    return [write_csv(folder / "Summary.csv", cycle_summary(model, run)),
            write_csv(folder / "Steps.csv", step_table(model, run.results)),
            write_csv(folder / "Streams.csv", stream_table(model, run)),
            write_csv(folder / "Convergence.csv", convergence_table(run.history)),
            write_csv(folder / "Profiles.csv", profile_table(model, run.results)),
            write_csv(folder / "Cycle history.csv", cycle_history_table(model, run.results, positions))]


def export_breakthrough(model, result, folder, components=None, positions=(0.0, 0.25, 0.5, 0.75, 1.0)):
    """
    Write a breakthrough step as CSV files: metrics of each component fed,
    the time series and the profiles at the end.

    Returns:
        (list of files, list of BreakthroughMetrics)
    """
    folder = Path(folder)
    if components is None:
        y_feed = end_flows(model, result, "feed")["mole_fractions"][:, -1]
        components = [name for name, y in zip(model.gas.components, y_feed) if y > 0.0]
    metrics = []
    for name in components:
        try:
            metrics.append(breakthrough_metrics(model, result, name))
        except ValueError:
            continue
    rows = []
    for m in metrics:
        row = {k: v for k, v in asdict(m).items() if k != "breakthrough_times"}
        row.update({f"t at {level:g} of feed [s]": t for level, t in m.breakthrough_times.items()})
        rows.append(row)
    files = [write_csv(folder / "Breakthrough metrics.csv", rows),
             write_csv(folder / "Breakthrough history.csv", step_history_table(model, result, positions)),
             write_csv(folder / "Final profiles.csv", profile_table(model, [result]))]
    return files, metrics
