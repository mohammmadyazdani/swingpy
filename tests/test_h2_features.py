"""Features added for hydrogen PSA: the multisite Langmuir isotherm, pure-gas equalization replay,
pressurization with a recorded product."""

import numpy as np
import pytest

from swingpy import (Adsorbent, BufferInlet, Closed, Column, ColumnModel, Cycle, FlowInlet, GasMixture,
                      MultisiteLangmuir, PressureOutlet, PressurizationInlet, Step)
import swingpy.isotherms as isotherms
from swingpy.adsorbent import ConstantLDF
from swingpy.gas import GAS_CONSTANT as R

"""Activated carbon of Ribeiro et al. (2008), Table 4: CO2, CH4, H2."""
AC = dict(qmax=(7.8550, 6.7329, 23.565), a=(3.0, 3.5, 1.0), K_inf=(2.125e-11, 7.904e-11, 7.233e-11),
          dH=(-29084.0, -22701.0, -12843.0))


@pytest.mark.parametrize("compiled", [True, False])
def test_multisite_langmuir_solves_its_equation(compiled, monkeypatch):
    if not compiled:
        monkeypatch.setattr(isotherms, "_multisite_theta", None)
    iso = MultisiteLangmuir(**AC)
    T = np.array([303.0, 330.0, 280.0])
    p = np.array([[1.16e5, 0.5e5, 3e5], [0.25e5, 0.1e5, 0.5e5], [5.1e5, 1e5, 2e5]])
    """p: pa."""
    c = p / (R * T)
    q = iso.loadings(c, T)
    theta = np.sum(q / np.array(AC["qmax"])[:, None], axis=0)
    K = np.array(AC["K_inf"])[:, None] * np.exp(-np.array(AC["dH"])[:, None] / (R * T))
    a = np.array(AC["a"])[:, None]
    expected = np.array(AC["qmax"])[:, None] * a * K * p * (1 - theta) ** a
    assert q == pytest.approx(expected, rel=1e-10)
    assert np.all((theta > 0) & (theta < 1))


def test_multisite_langmuir_limits():
    """One site per molecule: the Langmuir isotherm."""
    iso = MultisiteLangmuir(qmax=(2.0,), a=(1.0,), K_inf=(1e-5,), dH=(0.0,))
    p = np.array([[1e4, 1e5, 1e6]])
    q = iso.loadings(p / (R * 300.0), np.full(3, 300.0))
    assert q[0] == pytest.approx(2.0 * 1e-5 * p[0] / (1 + 1e-5 * p[0]), rel=1e-10)
    """Henry's law at low pressure: q = qmax a K p."""
    iso = MultisiteLangmuir(**AC)
    q = iso.loadings(np.array([[1e-6], [0.0], [0.0]]), np.array([303.0]))
    K = AC["K_inf"][0] * np.exp(-AC["dH"][0] / (R * 303.0))
    assert q[0, 0] == pytest.approx(7.855 * 3.0 * K * 1e-6 * R * 303.0, rel=1e-6)
    """A_i qmax_i must be the same."""
    with pytest.raises(ValueError):
        MultisiteLangmuir(qmax=(1.0, 2.0), a=(1.0, 1.0), K_inf=(1e-5, 1e-5), dH=(0.0, 0.0))


def small_model():
    gas = GasMixture(("CO2", "H2"), molar_masses=(0.044, 0.002), heat_capacity=29.0, viscosity=1.2e-5,
                     molecular_diffusivity=1e-5)
    sorbent = Adsorbent(name="AC", isotherm=MultisiteLangmuir(qmax=(7.855, 23.565), a=(3.0, 1.0),
                                                            K_inf=(2.125e-11, 7.233e-11), dH=(-29084.0, -12843.0)),
                        heat_of_adsorption=(29084.0, 12843.0), density=842.0, particle_radius=1.17e-3,
                        particle_voidage=0.566, heat_capacity=709.0, adsorbed_phase_heat_capacity=21.0,
                        kinetics=ConstantLDF((0.19, 1.3)))
    column = Column(length=0.2, inner_radius=0.02, outer_radius=0.022, bed_voidage=0.38, pressure_drop="ergun",
                    axial_thermal_conductivity=1.0, inner_heat_transfer=50.0, outer_heat_transfer=0.0,
                    wall_density=8000.0, wall_heat_capacity=500.0, wall_thermal_conductivity=0.0,
                    ambient_temperature=303.0, include_pore_gas=True)
    return ColumnModel(column, sorbent, gas, n_cells=10, scheme="weno", reference_pressure=7.0,
                       reference_temperature=303.0, reference_velocity=0.05)


def test_equalization_with_pure_gas_and_product_pressurization():
    m = small_model()
    feed = {"CO2": 0.3, "H2": 0.7}
    flow = 0.002
    cycle = Cycle([
        Step("FEED", 20.0, feed_end=FlowInlet(flow, feed, temperature=303.0),
             product_end=PressureOutlet(7.0, stream="product", record_as="product")),
        Step("EQDN", 10.0, product_end=PressureOutlet(4.0, stream="pe", record_as="pe"), feed_end=Closed()),
        Step("BLOW", 10.0, feed_end=PressureOutlet(1.0, stream="waste"), product_end=Closed()),
        Step("EQUP", 10.0, product_end=BufferInlet("pe", stream="pe_in", composition={"H2": 1.0}), feed_end=Closed()),
        Step("PRES", 10.0, product_end=PressurizationInlet(7.0, "product", temperature=303.0, supply_pressure=None,
                                                           default_composition={"H2": 1.0}, stream="pres"),
             feed_end=Closed()),
    ], key_component="H2", product_stream="product")
    state = m.initial_state(7.0, {"H2": 1.0}, 303.0)
    for _ in range(2):
        results = cycle.run_once(m, state)
        state = results[-1].final_state
    perf = cycle.evaluate(m, results)
    s = perf.streams
    """The replay carries the recorded molar flow, but only H2."""
    assert s["pe_in"][0] == pytest.approx(0.0, abs=1e-12)
    assert -s["pe_in"].sum() == pytest.approx(s["pe"].sum(), rel=0.02)
    """Pressurization with the mean product gas of the feed step."""
    product = np.asarray(s["product"]) / np.sum(s["product"])
    pres = np.asarray(s["pres"]) / np.sum(s["pres"])
    assert pres == pytest.approx(product, abs=1e-9)
    with pytest.raises(ValueError):
        PressurizationInlet(7.0, "product")


def test_hydrogen_features_survive_a_specification_round_trip():
    import yaml
    from swingpy.spec import load_problem, to_spec
    m = small_model()
    cycle = Cycle([
        Step("FEED", 20.0, feed_end=FlowInlet(0.002, {"CO2": 0.3, "H2": 0.7}, temperature=303.0),
             product_end=PressureOutlet(7.0, stream="product", record_as="product")),
        Step("EQDN", 10.0, product_end=PressureOutlet(4.0, stream="pe", record_as="pe"), feed_end=Closed()),
        Step("EQUP", 10.0, product_end=BufferInlet("pe", stream="pe_in", composition={"H2": 1.0}), feed_end=Closed()),
        Step("PRES", 10.0, product_end=PressurizationInlet(7.0, "product", temperature=303.0, supply_pressure=None,
                                                           default_composition={"H2": 1.0}, stream="pres"),
             feed_end=Closed()),
    ], key_component="H2", product_stream="product")
    spec = to_spec(m, cycle, dict(pressure=7.0, temperature=303.0, composition={"H2": 1.0}))
    problem = load_problem(yaml.safe_load(yaml.safe_dump(spec, sort_keys=False)))
    assert isinstance(problem.model.adsorbent.isotherm, MultisiteLangmuir)
    p_step = next(s for s in problem.cycle.steps if s.name == "EQUP").product_end
    assert p_step.composition == {"CO2": 0.0, "H2": 1.0}
    pres = next(s for s in problem.cycle.steps if s.name == "PRES").product_end
    assert pres.buffer == "product" and pres.supply_pressure is None
    state = problem.initial_state()
    x = state * (1.0 + 0.05 * np.random.default_rng(1).random(state.size))
    for a, b in zip(problem.cycle.steps, cycle.steps):
        """K_inf goes through 1/bar in the specification (x 1e5, then x 1e-5): equal to rounding."""
        assert np.allclose(problem.model.rhs(0.7, x, a), m.rhs(0.7, x, b), rtol=1e-12, atol=1e-15), a.name


def test_a_hard_extrapolated_start_is_discarded(monkeypatch):
    from swingpy.model import StepTooStiff
    m = small_model()
    step = Step("BLOW", 5.0, product_end=PressureOutlet(1.0), feed_end=Closed())
    with pytest.raises(StepTooStiff):
        m.simulate(step, m.initial_state(7.0, {"H2": 1.0}, 303.0), max_evaluations=1, fallback=False)
    cycle = Cycle([Step("FEED", 20.0, feed_end=FlowInlet(0.002, {"CO2": 0.3, "H2": 0.7}, temperature=303.0),
                        product_end=PressureOutlet(7.0, stream="product")),
                   Step("BLOW", 10.0, feed_end=PressureOutlet(1.0, stream="waste"), product_end=Closed()),
                   Step("PRES", 10.0, feed_end=FlowInlet(0.004, {"CO2": 0.3, "H2": 0.7}, temperature=303.0,
                                                         max_pressure=7.0), product_end=Closed())],
                  key_component="H2", product_stream="product")
    original = Cycle.run_once
    calls = {"guarded": 0}

    def run_once(self, model, state, *args, fallback=True, **kwargs):
        """Every extrapolated start fails."""
        if not fallback:
            calls["guarded"] += 1
            raise StepTooStiff("test")
        return original(self, model, state, *args, **kwargs)
    monkeypatch.setattr(Cycle, "run_once", run_once)
    run = cycle.run(m, m.initial_state(7.0, {"H2": 1.0}, 303.0), max_cycles=8, verbose=False, keep_cycles=1,
                    criterion="mass_balance", tolerance=1e-12, accelerate="graves_morris")
    assert calls["guarded"] >= 1 and len(run.history) == 8
