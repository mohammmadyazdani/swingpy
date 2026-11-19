"""Figures and reports, on short runs."""

import matplotlib
import numpy as np
import pytest

import setups
from swingpy.reporting import print_heading, print_table
import setups as h

matplotlib.use("Agg")


@pytest.fixture(scope="module")
def short_run():
    model = h.model(n_cells=10)
    cycle = h.four_step_vsa()
    run = cycle.run(model, model.initial_state(0.1, {"N2": 1.0}), max_cycles=3, verbose=False, keep_cycles=3)
    return model, cycle, run


def test_plot_all_writes_every_figure(short_run, tmp_path):
    from swingpy.plotting import plot_all
    model, _, run = short_run
    files = plot_all(model, run.recent, run.history, str(tmp_path / "Case"), tolerance=0.5, profile_tolerance=1e-3)
    names = {f.rsplit("Case ", 1)[1] for f in files}
    assert {"profiles.png", "cycle pressure.png", "end flows.png", "end compositions.png", "step balance.png",
            "feed step profiles.png", "maps.png", "last cycle.png", "last three cycles.png", "convergence.png",
            "energy.png"} <= names
    for f in files:
        assert (tmp_path / f.split("/")[-1]).stat().st_size > 10_000


def test_reports_print_values_and_differences(capsys):
    print_heading("Results")
    print_table([("Purity", (95.0, 1), "%", 94.0), ("Columns", "4", "-", "4"), ("Energy", 1.23e-4, "kWh", None)])
    out = capsys.readouterr().out
    assert "Results" in out and "95.0" in out and "+1.1 %" in out


def test_every_built_in_adsorbent_names_its_sources():
    from swingpy import data
    """Not those other tests register."""
    built_in = setups.adsorbents._bank()
    for name in built_in:
        references = setups.adsorbent_sources(name)["references"]
        assert references, name
        assert all(r["doi"].startswith("10.") and r["citation"] for r in references), name
    sources = data.gas_sources()
    """Every property of every gas has a source."""
    for name in data.list_gases():
        assert set(data.gas_properties(name)) <= set(sources), name


@pytest.fixture(scope="module")
def breakthrough():
    model = h.model(adiabatic=True, n_cells=20)
    result = model.simulate(h.breakthrough_step(3000.0), model.initial_state(1.0, {"N2": 1.0}), n_output=201)
    return model, result


def test_breakthrough_metrics_use_the_exact_amounts(breakthrough):
    from swingpy.analysis import breakthrough_metrics
    model, result = breakthrough
    m = breakthrough_metrics(model, result, "CO2")
    stored = model.component_inventory(result.final_state)[0] - model.component_inventory(result.initial_state)[0]
    assert m.retained == pytest.approx(stored, rel=1e-5)
    assert m.feed_mole_fraction == pytest.approx(0.15)
    """The CO2 front of the 13X breakthrough."""
    assert 450.0 < m.breakthrough_times[0.5] < 600.0
    assert m.breakthrough_times[0.05] < m.breakthrough_times[0.5]
    assert m.peak_outlet_temperature > 340.0
    """
    The thermal front has not left the column at 3000 s: the mean loading lies between the equilibrium loadings at 15 %
    CO2 and 1 bar at the peak temperature and at the feed temperature.
    """
    def equilibrium(T):
        c = np.array([[0.15], [0.85]]) * 1.0e5 / (8.314 * T)
        return model.adsorbent.isotherm.loadings(c, np.array([T]))[0, 0]
    assert equilibrium(m.peak_outlet_temperature) < m.dynamic_capacity < equilibrium(298.15)


def test_cycle_tables_balance(short_run):
    from swingpy import analysis
    model, _, run = short_run
    steps = analysis.step_table(model, run.results)
    assert [r["step"] for r in steps] == [r.step.name for r in run.results if r.time[-1] > 0.0]
    co2_out = sum(r["CO2 out of feed end [mol]"] + r["CO2 out of product end [mol]"] for r in steps)
    streams = run.performance.streams
    assert co2_out == pytest.approx(sum(v[0] for v in streams.values()), rel=1e-9, abs=1e-12)
    summary = analysis.cycle_summary(model, run)
    assert summary["purity [%]"] == run.performance.purity and summary["cycles"] == len(run.history)


def test_reports_write_every_file(short_run, breakthrough, tmp_path):
    from swingpy.report import report_breakthrough, report_run
    model, _, run = short_run
    files = {p.name for p in report_run(model, run, tmp_path / "Cycle")}
    assert {"Summary.csv", "Steps.csv", "Streams.csv", "Convergence.csv", "Profiles.csv", "Cycle history.csv",
            "Summary.md", "Isotherms.png", "Profiles.png", "End compositions.png"} <= files
    text = (tmp_path / "Cycle" / "Summary.md").read_text()
    assert "## Performance" in text and "purity" in text
    numbers_only = {p.name for p in report_run(model, run, tmp_path / "Numbers", figures=False)}
    assert not any(n.endswith(".png") for n in numbers_only)
    files = {p.name for p in report_breakthrough(*breakthrough, tmp_path / "Breakthrough")}
    assert {"Breakthrough metrics.csv", "Breakthrough history.csv", "Final profiles.csv", "Breakthrough.png",
            "Front profiles.png", "Isotherms.png", "Summary.md"} <= files
    for name in files:
        assert (tmp_path / "Breakthrough" / name).stat().st_size > 100

