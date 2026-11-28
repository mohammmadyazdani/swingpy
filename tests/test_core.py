"""
Tests of SwingPy: physics checks, conservation, and regression values for
the standard 13X flue gas setup of the example setups (examples/setups).

Run with:  pytest -q
"""

import numpy as np
import pytest

from swingpy import (Adsorbent, Column, ColumnModel, DualSiteLangmuir, FeedInlet, GasMixture,
                      PressureOutlet, Step)
import setups as h

"""
Regression values: the time at which the breakthrough front of the standard setup leaves the column, and the
performance of the first cycle of the four-step cycle started from a column at 0.1 bar. That first cycle contains brief
local flow reversals, so it also checks that face values follow the local flow direction.
"""
REFERENCE_BREAKTHROUGH_FRONT_S = 535.3893
REFERENCE_FIRST_CYCLE = {"purity": 12.9231, "recovery": 5.07614, "specific_energy": 1432.79,
                         "productivity": 0.0547797}


def crossing_time(time, values, level):
    i = int(np.argmax(values >= level))
    return time[i - 1] + (level - values[i - 1]) * (time[i] - time[i - 1]) / (values[i] - values[i - 1])


@pytest.fixture(scope="module")
def breakthrough():
    model = h.model(adiabatic=True)
    result = model.simulate(h.breakthrough_step(700.0), model.initial_state(1.0, {"N2": 1.0}), n_output=701)
    return model, result


def test_mass_and_energy_are_conserved(breakthrough):
    model, r = breakthrough
    co2 = model.gas.index("CO2")
    accumulated = model.component_inventory(r.final_state)[co2] - model.component_inventory(r.initial_state)[co2]
    entered = r.moles_in["feed"][co2] + r.moles_in["product"][co2]
    assert abs(entered - accumulated) / accumulated < 1e-6
    heat_in = r.enthalpy_in["feed"] + r.enthalpy_in["product"]
    released = model.heat_of_adsorption_released(r.initial_state, r.final_state)
    stored = model.enthalpy_inventory(r.final_state) - model.enthalpy_inventory(r.initial_state)
    assert abs(heat_in + released - stored) / abs(stored) < 1e-5


def test_breakthrough_front_regression(breakthrough):
    model, r = breakthrough
    y = model.end_history(r, "product")["mole_fractions"][0]
    assert crossing_time(r.time, y, 0.075) == pytest.approx(REFERENCE_BREAKTHROUGH_FRONT_S, rel=1e-4)


def test_first_cycle_regression():
    model, cycle = h.model(), h.four_step_vsa()
    results = cycle.run_once(model, model.initial_state(0.1, {"N2": 1.0}))
    performance = cycle.evaluate(model, results)
    for name, value in REFERENCE_FIRST_CYCLE.items():
        assert getattr(performance, name) == pytest.approx(value, rel=1e-3)


def test_splitting_a_component_changes_nothing():
    """
    Three components where "N2" is split into two identical halves must give
    the same CO2 breakthrough as the binary model: a check of the
    n-component equations.
    """
    binary = h.model(adiabatic=True)
    iso = binary.adsorbent.isotherm
    split_iso = DualSiteLangmuir(qsb=(3.09, 5.84, 5.84), qsd=(2.54, 0, 0), b0=(8.65e-7, 2.5e-6, 2.5e-6),
                                 d0=(2.63e-8, 0, 0), dUb=(-36641.21, -1.58e4, -1.58e4), dUd=(-35690.66, 0, 0))
    assert np.allclose(iso.qsb, split_iso.qsb[:2])
    adsorbent = Adsorbent("13X split", split_iso, (36641.21, 1.58e4, 1.58e4), density=1130.0,
                          particle_radius=1e-3)
    gas = GasMixture(("CO2", "N2a", "N2b"), molar_masses=(44.01e-3, 28.014e-3, 28.014e-3))
    ternary = ColumnModel(h.column(adiabatic=True), adsorbent, gas, reference_loading=5.84)
    step = Step("FEED", 600.0, feed_end=FeedInlet(1.0, (0.15, 0.425, 0.425)),
                product_end=PressureOutlet(1.0))
    r3 = ternary.simulate(step, ternary.initial_state(1.0, (0.0, 0.5, 0.5)), n_output=61)
    r2 = binary.simulate(h.breakthrough_step(600.0), binary.initial_state(1.0, {"N2": 1.0}), n_output=61)
    y3 = ternary.end_history(r3)["mole_fractions"][0]
    y2 = binary.end_history(r2)["mole_fractions"][0]
    assert np.max(np.abs(y3 - y2)) < 1e-4


def test_ergun_reduces_to_darcy_at_low_velocity():
    darcy = h.model()
    ergun = ColumnModel(Column(length=1.0, inner_radius=0.1445, pressure_drop="ergun"), h.zeolite_13x(), h.flue_gas())
    g = 1e-6
    v_darcy = darcy.velocity_from_gradient(g, 0.0)
    v_ergun = ergun.velocity_from_gradient(g, ergun._inertial(np.array([0.15]), 1.0, 1.0))
    assert v_ergun == pytest.approx(v_darcy, rel=1e-4)
    assert ergun.velocity_from_gradient(50.0, ergun._inertial(np.array([0.15]), 1.0, 1.0)) < darcy.velocity_from_gradient(50.0, 0.0)


def test_composition_dependent_gamma():
    from swingpy import LinearCO2AdiabaticConstant
    gas = GasMixture(("CO2", "N2"), adiabatic_constant=LinearCO2AdiabaticConstant())
    assert gas.gamma([0.0, 1.0]) == pytest.approx(1.395)
    assert gas.gamma([1.0, 0.0]) == pytest.approx(1.29)


def test_vacuum_pump_imposes_its_volumetric_flow():
    from swingpy import VacuumPump
    from swingpy.model import Cell
    model = h.model()
    pump = VacuumPump(volumetric_flow=500.0)
    face = pump.face(model, "feed", 0.0, Cell(0.5, np.array([0.5]), 1.0))
    expected = 500.0 / 3600.0 / (model.column.bed_voidage * model.column.cross_section) / model.v0
    assert face.velocity == pytest.approx(-expected)
    assert face.pressure < 0.5


def test_profile_criterion_needs_settled_profiles():
    """The profile test must not pass while the column still changes, and profile_change is recorded."""
    model, cycle = h.model(), h.four_step_vsa()
    run = cycle.run(model, model.initial_state(0.1, {"N2": 1.0}), max_cycles=3, verbose=False,
                    criterion="profiles", profile_tolerance=1e-3)
    assert not run.converged
    assert all(np.isfinite(p.profile_change) and p.profile_change > 1e-3 for p in run.history)
    assert len(run.recent) == 3
    with pytest.raises(ValueError):
        cycle.run(model, model.initial_state(0.1, {"N2": 1.0}), max_cycles=1, verbose=False, criterion="nope")


def test_inputs_are_checked_with_clear_messages():
    from swingpy import Column, Toth, data
    with pytest.raises(ValueError, match="bed_voidage"):
        Column(length=1.0, inner_radius=0.1, bed_voidage=1.2)
    with pytest.raises(ValueError, match="one value per component"):
        Toth(qs=(1.0, 2.0), b0=(1e-6,), dU=(-3e4, -1e4), t=(0.5, 0.5))
    with pytest.raises(ValueError, match="heterogeneity"):
        Toth(qs=(1.0,), b0=(1e-6,), dU=(-3e4,), t=(1.5,))
    gas = data.gas(["CO2", "N2", "O2"])
    with pytest.raises(ValueError, match="2 components but the gas has 3"):
        ColumnModel(h.column(), h.adsorbent("zeolite-13x"), gas)


def test_data_bank_reproduces_the_library_parameters_and_reorders_components():
    from swingpy import data
    bank, reference = h.adsorbent("zeolite-13x"), h.zeolite_13x()
    c = np.array([[5.0, 1.0], [30.0, 10.0]])
    assert np.allclose(bank.isotherm.loadings(c, 310.0), reference.isotherm.loadings(c, 310.0), rtol=1e-12)
    swapped = h.adsorbent("zeolite-13x", components=["N2", "CO2"])
    assert np.allclose(swapped.isotherm.loadings(c[::-1], 310.0), bank.isotherm.loadings(c, 310.0)[::-1])
    gas = data.gas(["CO2", "N2"], {"CO2": 0.15, "N2": 0.85})
    assert gas.molecular_diffusivity == pytest.approx(1.6e-5, rel=0.05)
    assert gas.viscosity == pytest.approx(1.72e-5, rel=0.05)


def test_user_adsorbents_can_be_added_from_a_file(tmp_path):
    import json
    entry = {"components": ["CO2", "N2"], "isotherm": {"type": "Toth", "qs": [4.0, 2.0], "b0": [1e-6, 1e-7],
                                                       "dU": [-30000.0, -15000.0], "t": [0.6, 0.9]},
             "heat_of_adsorption": [30000.0, 15000.0], "density": 1100.0, "particle_radius": 1e-3}
    path = tmp_path / "Mine.json"
    path.write_text(json.dumps({"my-sorbent": entry}))
    assert h.load_adsorbents(path) == ["my-sorbent"]
    assert h.adsorbent("my-sorbent").density == 1100.0
