"""
Everything a simulation produces, written to one folder.

    from swingpy.report import report_run, report_breakthrough

    report_run(model, run, "Results/Case one")             # a cycle run to steady state
    report_breakthrough(model, result, "Results/Breakthrough")   # a single feed step

Each writes the CSV files of ``swingpy.analysis``, the figures of
``swingpy.plotting`` (when matplotlib is installed) and ``Summary.md``, a
readable summary with the performance, the tables and the list of files.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from . import analysis


def _matplotlib():
    try:
        import matplotlib
        matplotlib.use("Agg", force=False)
        return True
    except ImportError:
        return False


def _markdown_table(rows, columns=None, digits=4):
    if not rows:
        return "(none)\n"
    columns = columns or list(rows[0])

    def cell(v):
        if isinstance(v, (float, np.floating)):
            return f"{v:.{digits}g}"
        return str(v)
    lines = ["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    lines += ["| " + " | ".join(cell(row.get(c, "")) for c in columns) + " |" for row in rows]
    return "\n".join(lines) + "\n"


def _feed_composition(model, results):
    for r in results:
        composition = getattr(r.step.feed_end, "composition", None)
        if isinstance(composition, dict):
            return composition
    return None


def _isotherm_figure(model, results, path):
    from .plotting import plot_isotherms
    states = [r.final_state for r in results if r.time[-1] > 0.0] or [results[-1].final_state]
    top = max(float(np.max(analysis.column_profiles(model, s)["P"])) for s in states)
    T0 = model.T0
    plot_isotherms(model.adsorbent, model.gas.components, (T0, T0 + 25.0, T0 + 50.0), max(1.2 * top, 0.1), path,
                   composition=_feed_composition(model, results))


def report_run(model, run, folder, component=None, positions=(0.0, 0.5, 1.0), tolerance=None,
               profile_tolerance=None, figures=True):
    """
    Write the full report of a cycle run.

    Args:
        model: The ColumnModel.
        run: The CycleRun (``Cycle.run``); its last cycle is reported.
        folder: Output folder (created).
        component: Component shown in the composition and loading figures
            (the first by default).
        positions: Positions Z of the time histories.
        tolerance: Mass balance tolerance of the steady state, %, drawn on
            the convergence figure.
        profile_tolerance: Profile change tolerance of the steady state,
            drawn on the convergence figure.
        figures: False to write the numbers only.

    Returns:
        List of the files written.
    """
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    files = analysis.export_run(model, run, folder, positions=tuple(sorted(set(positions) | {0.25, 0.75})))
    figure_note = ""
    if figures and _matplotlib():
        from .plotting import plot_all
        cycles = run.recent or [run.results]
        files += [Path(f) for f in plot_all(model, cycles, run.history, str(folder) + "/", component, positions,
                                            tolerance, profile_tolerance)]
        _isotherm_figure(model, run.results, str(folder / "Isotherms.png"))
        files.append(folder / "Isotherms.png")
    elif figures:
        figure_note = "\nFigures were not drawn: matplotlib is not installed (pip install matplotlib).\n"
    summary = analysis.cycle_summary(model, run)
    kpis = [{"quantity": k, "value": v} for k, v in summary.items()]
    steps = analysis.step_table(model, run.results)
    text = ["# Cycle report", "",
            f"Components: {', '.join(model.gas.components)}; {model.n_cells} cells, {model.scheme} scheme.",
            f"{'Cyclic steady state reached' if run.converged else 'Cyclic steady state NOT reached'} after "
            f"{len(run.history)} cycles; the last cycle is reported.", "", "## Performance", "",
            _markdown_table(kpis), "## Steps", "",
            _markdown_table(steps, ["step", "duration [s]", "P feed end [bar]", "P product end [bar]", "T mean [K]",
                                    "feed end stream", "product end stream", "work [kJ]", "wall heat [kJ]"]),
            "## Streams (net moles per cycle, out of the column positive)", "",
            _markdown_table(analysis.stream_table(model, run)), figure_note]
    text += ["## Files", ""]
    text += [f"- `{Path(f).name}`" for f in files] + [""]
    (folder / "Summary.md").write_text("\n".join(text), encoding="utf-8")
    return files + [folder / "Summary.md"]


def report_breakthrough(model, result, folder, component=None, positions=(0.25, 0.5, 0.75), figures=True):
    """
    Write the full report of a breakthrough (a single feed step): metrics of
    every component fed, time series, final profiles and, with matplotlib,
    the breakthrough figure, the movement of the fronts and the isotherms.

    Returns:
        List of the files written.
    """
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    files, metrics = analysis.export_breakthrough(model, result, folder,
                                                  positions=tuple(sorted(set(positions) | {0.0, 1.0})))
    figure_note = ""
    if figures and _matplotlib():
        from .plotting import plot_breakthrough, plot_step_profiles
        plot_breakthrough(model, result, str(folder / "Breakthrough.png"), positions)
        plot_step_profiles(model, result, str(folder / "Front profiles.png"), times=7, component=component)
        _isotherm_figure(model, [result], str(folder / "Isotherms.png"))
        files += [folder / "Breakthrough.png", folder / "Front profiles.png", folder / "Isotherms.png"]
    elif figures:
        figure_note = "\nFigures were not drawn: matplotlib is not installed (pip install matplotlib).\n"
    rows = []
    for m in metrics:
        row = {"component": m.component, "feed y [-]": m.feed_mole_fraction}
        row.update({f"t at {level:g} [s]": t for level, t in m.breakthrough_times.items()})
        row.update({"stoichiometric time [s]": m.stoichiometric_time, "retained [mol]": m.retained,
                    "dynamic capacity [mol/kg]": m.dynamic_capacity, "MTZ [m]": m.mass_transfer_zone,
                    "peak outlet T [K]": m.peak_outlet_temperature})
        rows.append(row)
    text = ["# Breakthrough report", "",
            f"Step '{result.step.name}', {result.time[-1]:.0f} s; components {', '.join(model.gas.components)}.",
            "", "## Metrics", "", _markdown_table(rows), figure_note]
    text += ["## Files", ""]
    text += [f"- `{Path(f).name}`" for f in files] + [""]
    (folder / "Summary.md").write_text("\n".join(text), encoding="utf-8")
    return files + [folder / "Summary.md"]
