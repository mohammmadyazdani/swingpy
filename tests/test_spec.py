"""Problem specifications: units, error reporting, and identity with the model built in Python."""

from pathlib import Path

import json

import numpy as np
import pytest
import yaml

import setups as presets
from swingpy.spec import SpecError, UnitError, load_problem, quantity

SPEC = Path(__file__).resolve().parents[1] / "examples" / "specs" / "VSA zeolite 13X flue gas.yaml"


def test_units_convert_and_refuse_the_wrong_kind():
    assert quantity("25 C", "temperature") == pytest.approx(298.15)
    assert quantity("1.5 mm", "length") == pytest.approx(1.5e-3)
    assert quantity("101325 Pa", "pressure") == pytest.approx(1.01325)
    assert quantity("16.8e-9 1/Pa", "inverse_pressure") == pytest.approx(1.68e-3)
    """Plain numbers are in the base unit."""
    assert quantity(2.0, "pressure") == 2.0
    with pytest.raises(UnitError, match="is not a pressure unit"):
        quantity("1 m", "pressure")
    with pytest.raises(UnitError):
        quantity("abc", "length")


def test_specification_builds_exactly_the_model_built_in_python():
    problem = load_problem(SPEC)
    reference, cycle = presets.model(), presets.four_step_vsa()
    state = reference.initial_state(0.1, {"N2": 1.0})
    assert np.array_equal(problem.initial_state(), state)
    perturbed = state * (1.0 + 0.05 * np.random.default_rng(0).random(state.size))
    for a, b in zip(problem.cycle.steps, cycle.steps):
        assert np.array_equal(problem.model.rhs(1.0, perturbed, a), reference.rhs(1.0, perturbed, b)), a.name


def test_every_error_is_reported_with_its_path():
    spec = yaml.safe_load(SPEC.read_text())
    spec["bed"]["length"] = "1 bar"
    del spec["adsorbent"]["heat_capacity"]
    spec["streams"]["flue_gas"]["composition"]["N2"] = 0.80
    spec["bed"]["colour"] = "blue"
    spec["cycle"]["steps"][1]["feed_end"]["stream"] = "syngas"
    with pytest.raises(SpecError) as info:
        load_problem(spec)
    errors = "\n".join(info.value.errors)
    assert len(info.value.errors) == 5
    for path in ("bed.length", "adsorbent.heat_capacity", "streams.flue_gas.composition", "bed.colour",
                 "cycle.steps[1].feed_end.stream"):
        assert path in errors


def test_no_hidden_physical_defaults():
    spec = yaml.safe_load(SPEC.read_text())
    for section, key in (("gas", "viscosity"), ("adsorbent", "particle_voidage"), ("bed", "voidage"),
                         ("bed", "axial_thermal_conductivity")):
        broken = yaml.safe_load(SPEC.read_text())
        del broken[section][key]
        with pytest.raises(SpecError, match=f"{section}.{key}: missing"):
            load_problem(broken)
    del spec["cycle"]["machines"]
    with pytest.raises(SpecError, match="efficiency: missing"):
        load_problem(spec)


def test_isotherm_basis_follows_the_unit_of_b0():
    spec = yaml.safe_load(SPEC.read_text())
    spec["adsorbent"]["isotherm"] = {
        "model": "sips",
        "CO2": {"qs": "7.27 mol/kg", "b0": "1.13e-4 1/bar", "dU": "-28.4 kJ/mol", "n": 0.42},
        "N2": {"qs": "4.05 mol/kg", "b0": "5.85e-5 1/bar", "dU": "-18.5 kJ/mol", "n": 0.99}}
    isotherm = load_problem(spec).model.adsorbent.isotherm
    assert isotherm.pressure_based and isotherm.b0[0] == pytest.approx(1.13e-4)
    spec["adsorbent"]["isotherm"]["N2"]["b0"] = "2.5e-6 m3/mol"
    with pytest.raises(SpecError, match="same basis"):
        load_problem(spec)


def test_a_specified_cycle_runs():
    run = load_problem(SPEC).run(max_cycles=2, verbose=False)
    assert 0.0 < run.performance.purity < 100.0


def _roundtrip_identical(model, cycle, initial):
    from swingpy.spec import to_spec
    problem = load_problem(yaml.safe_load(yaml.safe_dump(to_spec(model, cycle, initial), sort_keys=False)))
    state = problem.initial_state()
    x = state * (1.0 + 0.05 * np.random.default_rng(1).random(state.size))
    for a, b in zip(problem.cycle.steps, cycle.steps):
        assert np.array_equal(problem.model.rhs(0.7, x, a), model.rhs(0.7, x, b)), a.name


def test_every_ready_made_case_survives_the_round_trip_through_a_specification():
    from setups import cycles, skarstrom
    _roundtrip_identical(presets.model(), presets.four_step_vsa(), dict(pressure=0.1, temperature=298.15,
                                                                         composition={"N2": 1.0}))
    for name in cycles.EXAMPLE_POINTS:
        model, cycle, _ = cycles.example_case(name)
        _roundtrip_identical(model, cycle, dict(pressure=1.0, temperature=298.15, composition={"N2": 1.0}))
    model, cycle, _ = skarstrom.example_case(1)
    _roundtrip_identical(model, cycle, dict(pressure=skarstrom.PURGE_PRESSURE, temperature=323.15,
                                            composition=skarstrom.NITROGEN))
    model, cycle, _ = presets.smr_example_case("IISERP MOF2")
    _roundtrip_identical(model, cycle, dict(pressure=presets.SMR_HIGH_PRESSURE, temperature=298.15,
                                            composition=presets.SMR_FEED))


def test_every_example_specification_loads():
    files = sorted(SPEC.parent.glob("*.yaml"))
    assert len(files) >= 12
    for path in files:
        problem = load_problem(path)
        assert problem.cycle is not None and problem.initial_state().size == problem.model.state_size, path.name


def test_what_a_specification_cannot_express_is_named():
    from swingpy import Valve, Step
    from swingpy.spec import to_spec
    model = presets.model()
    cycle = presets.four_step_vsa()
    cycle.steps[0] = Step("VALV", 10.0, product_end=Valve(1.0, 0.5))
    with pytest.raises(SpecError, match="a boundary of type Valve"):
        to_spec(model, cycle)


@pytest.mark.parametrize("dependent", ["N2", "CO2"])
def test_a_trace_component_is_accurate_whether_it_is_independent_or_dependent(dependent):
    """1 ppm of CO2 in N2: the last component's mole fraction is 1 minus the others, yet stays accurate."""
    from swingpy import Step
    spec = yaml.safe_load(SPEC.read_text())
    spec["components"] = ["CO2", "N2"] if dependent == "N2" else ["N2", "CO2"]
    spec["streams"]["flue_gas"]["composition"] = {"CO2": 1e-6, "N2": 1 - 1e-6}
    spec["initial"] = {"pressure": "1 bar", "temperature": "298.15 K", "composition": {"CO2": 0, "N2": 1}}
    problem = load_problem(spec)
    model, adsorption = problem.model, problem.cycle.steps[1]
    assert problem.components[-1] == dependent
    result = model.simulate(Step("FEED", 2000.0, feed_end=adsorption.feed_end, product_end=adsorption.product_end),
                            problem.initial_state())
    co2 = model.gas.index("CO2")
    stored = model.component_inventory(result.final_state)[co2] - model.component_inventory(result.initial_state)[co2]
    entered = result.moles_in["feed"][co2] + result.moles_in["product"][co2]
    """Measured: 8e-5 and 3e-4."""
    assert abs(entered - stored) / stored < 1e-3
    """
    Undershoot ahead of the front stays far below the solver's absolute tolerance on y (1e-7) and below 1 % of the 1 ppm
    feed (measured: about -2e-9).
    """
    assert model.full_composition(model.unpack(result.final_state)["y"])[co2].min() > -1e-8


def test_compressibility_check_from_a_specification(tmp_path):
    spec = yaml.safe_load(Path("examples/specs/VSA zeolite 13X flue gas.yaml").read_text())
    spec["gas"]["compressibility_check"] = {
        "model": "peng_robinson", "source": "test constants",
        "critical_temperature": {"CO2": "304.1282 K", "N2": "126.192 K"},
        "critical_pressure": {"CO2": "73.773 bar", "N2": "33.958 bar"},
        "acentric_factor": {"CO2": 0.22394, "N2": 0.0372},
        "binary_interaction": {"CO2-N2": -0.02}}
    problem = load_problem(spec)
    assert problem.eos.interaction == {("CO2", "N2"): -0.02}
    assert problem.eos.compressibility(298.15, 35.0, [0.5, 0.5]) < 0.95
    from swingpy.spec import to_spec
    again = load_problem(to_spec(problem.model, problem.cycle, problem.initial, eos=problem.eos))
    assert np.allclose(np.asarray(again.eos.constants), np.asarray(problem.eos.constants))
    assert again.eos.interaction == problem.eos.interaction
    """A breakthrough of flue gas near 1 bar: the gas is ideal to well within 1 %."""
    from swingpy import Step
    from swingpy.analysis import compressibility_range
    from swingpy.report import report_breakthrough
    feed = problem.cycle.steps[1]
    step = Step("FEED", 300.0, feed_end=feed.feed_end, product_end=feed.product_end)
    result = problem.model.simulate(step, problem.model.initial_state(1.0, {"N2": 1.0}))
    check = compressibility_range(problem.model, result, problem.eos, tolerance=0.01)
    assert check["ideal"] and 0.99 < check["Z_min"]["Z"] <= check["Z_max"]["Z"] < 1.01
    report_breakthrough(problem.model, result, tmp_path, figures=False, eos=problem.eos)
    assert "Ideal gas check" in (tmp_path / "Summary.md").read_text()
    bad = json.loads(json.dumps(spec))
    bad["gas"]["compressibility_check"]["binary_interaction"] = {"CO2-H2": 0.1}
    with pytest.raises(SpecError, match="binary_interaction"):
        load_problem(bad)


def test_mixing_inlet_from_a_specification_and_back():
    from swingpy import MixingInlet
    from swingpy.spec import to_spec
    spec = yaml.safe_load(Path("examples/specs/VSA zeolite 13X flue gas.yaml").read_text())
    ads = next(s for s in spec["cycle"]["steps"] if s["name"] == "FEED")
    feed_stream = ads["feed_end"]["stream"]
    ads["feed_end"] = {"type": "mixing_inlet", "stream": feed_stream, "molar_flow": "0.01 mol/s",
                       "record": "evacuated", "fraction": 0.5, "supply_pressure": "1 bar", "efficiency": 0.72}
    ev = next(s for s in spec["cycle"]["steps"] if s["name"] == "EVAC")
    ev["feed_end"]["record_as"] = "evacuated"
    ev["feed_end"]["label"] = "recycle"
    problem = load_problem(spec)
    inlet = next(s for s in problem.cycle.steps if s.name == "FEED").feed_end
    assert isinstance(inlet, MixingInlet) and inlet.molar_flow == pytest.approx(0.01) and inlet.fraction == 0.5
    again = load_problem(to_spec(problem.model, problem.cycle, problem.initial))
    back = next(s for s in again.cycle.steps if s.name == "FEED").feed_end
    assert isinstance(back, MixingInlet) and back.buffer == "evacuated" and back.fraction == 0.5
    model = problem.model
    assert model.gas.composition(back.composition) == pytest.approx(model.gas.composition(inlet.composition))
    assert back.temperature == pytest.approx(inlet.temperature) and back.recycle_stream == "recycle"
