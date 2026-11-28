"""Valves, dead volumes, extended Langmuir and pressure-scaled kinetics."""

import numpy as np
import pytest

import setups
from swingpy import (Closed, Column, ColumnModel, DualSiteLangmuir, ExtendedLangmuir, FeedInlet, PressureOutlet,
                      PressureScaledLDF, RecordedPressure, Step, Valve)
import setups as h


def test_extended_langmuir_reduces_to_competitive_langmuir_and_allows_henry_components():
    qs, b = np.array([3.0, 2.0]), np.array([1e-3, 2e-4])
    ext = ExtendedLangmuir(henry=qs * b, affinity=b, dU=(-30000.0, -15000.0))
    lang = DualSiteLangmuir(qsb=qs, qsd=(0, 0), b0=b * np.exp(-np.array([30000.0, 15000.0]) / (8.314 * 298.15)),
                            d0=(0, 0), dUb=(-30000.0, -15000.0), dUd=(0, 0))
    c = np.array([[5.0, 50.0], [20.0, 1.0]])
    assert np.allclose(ext.loadings(c, 320.0), lang.loadings(c, 320.0), rtol=1e-12)
    henry = ExtendedLangmuir(henry=(0.02, 0.4), affinity=(0.0, 0.16), dU=(-7000.0, -21000.0), pressure_based=True)
    q_low = henry.loadings(np.array([[10.0], [0.0]]), 298.15)[0, 0]
    q_high = henry.loadings(np.array([[20.0], [0.0]]), 298.15)[0, 0]
    """Linear in its own concentration."""
    assert q_high == pytest.approx(2.0 * q_low)


def test_pressure_scaled_ldf():
    kinetics = PressureScaledLDF((3.0, 1.5), reference_pressure=1.0)
    rates = kinetics.rate_constants(np.ones((2, 3)), None, None, pressure=np.array([1.0, 2.0, 10.0]))
    assert np.allclose(rates, [[3.0, 1.5, 0.3], [1.5, 0.75, 0.15]])


def test_valve_obeys_its_flow_law_and_conserves_mass():
    model = h.model(n_cells=20)
    cv = 2.0
    """cv: mol/(s bar)."""
    step = Step("VENT", 30.0, feed_end=Closed(), product_end=Valve(cv, 0.5, stream="waste"))
    result = model.simulate(step, model.initial_state(1.0, {"CO2": 0.15, "N2": 0.85}), n_output=31)
    history = model.end_history(result, "product")
    record = model.outflow_record(result, "product")
    expected = cv * np.maximum(history["pressure"] - 0.5, 0.0)
    assert np.allclose(record.molar_flow[1:], expected[1:], rtol=1e-6, atol=1e-9)
    stored = model.component_inventory(result.final_state) - model.component_inventory(result.initial_state)
    assert np.allclose(stored, result.moles_in["product"], rtol=1e-4, atol=1e-6)
    assert history["pressure"][-1] < 1.0


def test_valve_letting_gas_in_and_recorded_pressure():
    model = h.model(n_cells=20)
    inlet = Valve(1.0, RecordedPressure("upstream", lambda t: 1.0 + 0.01 * t), direction="in",
                  composition={"N2": 1.0}, stream="feed")
    result = model.simulate(Step("FILL", 20.0, feed_end=inlet), model.initial_state(0.5, {"N2": 1.0}))
    p = model.unpack(result.final_state)["P"]
    assert 0.5 < p.min() and p.max() <= 1.2 + 1e-9


def test_dead_volume_slows_the_ends_and_keeps_the_mass_balance():
    plain = ColumnModel(h.column(), setups.adsorbent("zeolite-13x"), h.flue_gas(), n_cells=20)
    column = h.column()
    """M3, about a fifth of the bed gas volume."""
    column.dead_volume_feed = column.dead_volume_product = 0.01
    dead = ColumnModel(column, setups.adsorbent("zeolite-13x"), h.flue_gas(), n_cells=20)
    step = Step("FEED", 800.0, feed_end=FeedInlet(1.0, {"CO2": 0.15, "N2": 0.85}), product_end=PressureOutlet(1.0))
    results = [m.simulate(step, m.initial_state(1.0, {"N2": 1.0})) for m in (plain, dead)]
    for m, r in zip((plain, dead), results):
        stored = m.component_inventory(r.final_state) - m.component_inventory(r.initial_state)
        assert np.allclose(stored, r.moles_in["feed"] + r.moles_in["product"], rtol=1e-4, atol=1e-6)

    def breakthrough(m, r, level):
        """Time at which the CO2 mole fraction at the outlet first reaches ``level``, interpolated linearly."""
        y, t = m.end_history(r, "product")["mole_fractions"][0], r.time
        i = int(np.argmax(y >= level))
        return t[i - 1] + (level - y[i - 1]) * (t[i] - t[i - 1]) / (y[i] - y[i - 1])

    """
    The front reaches the outlet after about 500 s. The gas held in the two
    dead volumes delays it by somewhat less than a second at every level of
    the breakthrough curve, far more than the error of the integration.
    """
    for level in (0.0075, 0.03, 0.075):
        plain_time, dead_time = (breakthrough(m, r, level) for m, r in zip((plain, dead), results))
        assert 450.0 < plain_time < 600.0
        assert 0.2 < dead_time - plain_time < 5.0


def test_layered_bed_with_one_material_matches_the_plain_bed_and_layers_conserve_mass():
    from swingpy import LayeredBed
    sorbent = setups.adsorbent("zeolite-13x")
    step = Step("FEED", 300.0, feed_end=FeedInlet(1.0, {"CO2": 0.15, "N2": 0.85}), product_end=PressureOutlet(1.0))
    plain = ColumnModel(h.column(), sorbent, h.flue_gas(), n_cells=20, backend="numpy")
    split = ColumnModel(h.column(), LayeredBed([(sorbent, 0.4), (sorbent, 0.6)]), h.flue_gas(), n_cells=20)
    a = plain.simulate(step, plain.initial_state(1.0, {"N2": 1.0}))
    b = split.simulate(step, split.initial_state(1.0, {"N2": 1.0}))
    assert np.allclose(a.final_state, b.final_state, rtol=1e-6, atol=1e-8)

    carbon = setups.adsorbent("utsa-16")
    bed = LayeredBed([(carbon, 0.3), (sorbent, 0.7)])
    model = ColumnModel(h.column(), bed, h.flue_gas(), n_cells=20)
    result = model.simulate(step, model.initial_state(1.0, {"N2": 1.0}))
    stored = model.component_inventory(result.final_state) - model.component_inventory(result.initial_state)
    assert np.allclose(stored, result.moles_in["feed"] + result.moles_in["product"], rtol=1e-4, atol=1e-6)
    x = model.unpack(result.final_state)["x"][0]
    cells = bed.layer_cells(20)
    assert len(cells[0]) == 6 and len(cells[1]) == 14
    with pytest.raises(ValueError, match="add up to 1"):
        LayeredBed([(sorbent, 0.5), (carbon, 0.6)])
    assert np.all(np.isfinite(x))


def test_iast_equals_extended_langmuir_for_equal_capacities_and_runs_in_a_column():
    from swingpy import IAST, Toth
    qs, b = 3.0, np.array([2e-3, 3e-4])
    iast = IAST([ExtendedLangmuir(henry=(qs * b[i],), affinity=(b[i],), dU=(-30000.0,)) for i in range(2)])
    langmuir = ExtendedLangmuir(henry=qs * b, affinity=b, dU=(-30000.0, -30000.0))
    c = np.array([[0.0, 1.0, 40.0, 300.0], [50.0, 10.0, 40.0, 0.0]])
    assert np.allclose(iast.loadings(c, 310.0), langmuir.loadings(c, 310.0), rtol=1e-10, atol=1e-14)
    toth = IAST([Toth(qs=(5.09,), b0=(4.305e-4,), dU=(-29380.0,), t=(0.429,), pressure_based=True),
                 Toth(qs=(3.08,), b0=(0.882e-4,), dU=(-17190.0,), t=(0.869,), pressure_based=True)])
    sorbent = setups.adsorbent("zeolite-13x", isotherm=toth)
    model = ColumnModel(h.column(), sorbent, h.flue_gas(), n_cells=15)
    step = Step("FEED", 100.0, feed_end=FeedInlet(1.0, {"CO2": 0.15, "N2": 0.85}), product_end=PressureOutlet(1.0))
    result = model.simulate(step, model.initial_state(1.0, {"N2": 1.0}))
    stored = model.component_inventory(result.final_state) - model.component_inventory(result.initial_state)
    assert np.allclose(stored, result.moles_in["feed"] + result.moles_in["product"], rtol=1e-4, atol=1e-6)


def test_an_insulated_column_exchanges_no_heat_with_its_surroundings():
    insulated = Column(length=0.5, inner_radius=0.025, outer_heat_transfer=0.0)
    plain = ColumnModel(insulated, setups.adsorbent("zeolite-13x"), h.flue_gas(), n_cells=10)
    result = plain.simulate(Step("REST", 10.0), plain.initial_state(1.0, {"N2": 1.0}))
    assert plain.wall_heat(result) == 0.0


def test_layers_with_their_own_voidage_and_particles_conserve_mass_and_both_backends_agree():
    pytest.importorskip("numba")
    from dataclasses import replace

    from swingpy import IAST, LayeredBed, Toth
    carbon = replace(setups.adsorbent("utsa-16"), particle_radius=1.5e-3, kinetics=PressureScaledLDF((0.3, 0.5)))
    zeolite = setups.adsorbent("zeolite-13x")
    bed = LayeredBed([(carbon, 0.3, 0.42), (zeolite, 0.7)])
    column = Column(length=1.0, inner_radius=0.1445, bed_voidage=0.37, pressure_drop="ergun")
    step = Step("FEED", 300.0, feed_end=FeedInlet(1.0, {"CO2": 0.15, "N2": 0.85}), product_end=PressureOutlet(1.0))
    results = {}
    for backend in ("numpy", "numba"):
        model = ColumnModel(column, bed, h.flue_gas(), n_cells=20, backend=backend)
        assert model.backend == backend
        result = model.simulate(step, model.initial_state(1.0, {"N2": 1.0}))
        stored = model.component_inventory(result.final_state) - model.component_inventory(result.initial_state)
        assert np.allclose(stored, result.moles_in["feed"] + result.moles_in["product"], rtol=1e-4, atol=1e-6)
        results[backend] = (model, result)
    (m, r) = results["numpy"]
    state = r.states[:, len(r.time) // 2]
    reference = m.rhs(0.3, state, step)
    assert np.max(np.abs(results["numba"][0].rhs(0.3, state, step) - reference)) < 1e-9 * np.max(np.abs(reference))
    assert np.allclose(m._eps_cell[:6], 0.42) and np.allclose(m._eps_cell[6:], 0.37)

    iast = IAST([Toth(qs=(5.09,), b0=(4.305e-4,), dU=(-29380.0,), t=(0.429,), pressure_based=True),
                 Toth(qs=(3.08,), b0=(0.882e-4,), dU=(-17190.0,), t=(0.869,), pressure_based=True)])
    sorbent = replace(zeolite, isotherm=iast)
    fast = ColumnModel(h.column(), sorbent, h.flue_gas(), n_cells=15, backend="numba")
    slow = ColumnModel(h.column(), sorbent, h.flue_gas(), n_cells=15, backend="numpy")
    state = fast.initial_state(1.0, {"CO2": 0.1, "N2": 0.9})
    a, b = fast.rhs(0.0, state, step), slow.rhs(0.0, state, step)
    assert np.max(np.abs(a - b)) < 1e-9 * np.max(np.abs(b))


def test_pore_gas_holdup_conserves_mass_and_slows_a_weakly_adsorbed_front():
    pytest.importorskip("numba")
    sorbent = setups.adsorbent("zeolite-13x")
    step = Step("FEED", 200.0, feed_end=FeedInlet(1.0, {"CO2": 0.15, "N2": 0.85}), product_end=PressureOutlet(1.0))
    outlets = {}
    for pores in (False, True):
        column = Column(length=1.0, inner_radius=0.1445, include_pore_gas=pores)
        fast = ColumnModel(column, sorbent, h.flue_gas(), n_cells=15, backend="numba")
        slow = ColumnModel(column, sorbent, h.flue_gas(), n_cells=15, backend="numpy")
        result = fast.simulate(step, fast.initial_state(1.0, {"N2": 1.0}))
        stored = fast.component_inventory(result.final_state) - fast.component_inventory(result.initial_state)
        assert np.allclose(stored, result.moles_in["feed"] + result.moles_in["product"], rtol=1e-4, atol=1e-6)
        state = result.states[:, len(result.time) // 2]
        a, b = fast.rhs(0.1, state, step), slow.rhs(0.1, state, step)
        assert np.max(np.abs(a - b)) < 1e-9 * np.max(np.abs(b))
        outlets[pores] = fast.end_history(result, "product")["mole_fractions"][0]
    """More gas holdup delays the CO2 that reaches the outlet early."""
    assert outlets[True][len(outlets[True]) // 3] <= outlets[False][len(outlets[False]) // 3] + 1e-12


def test_steffensen_methods_are_exact_for_a_slow_linear_mode():
    """
    X -> a x has the fixed point 0. The Steffensen variants of Subraveti, Karimi and Gazzani (2026), Table 1, reduce to
    Aitken's process on one mode and land on it; Graves-Morris and vector epsilon2 as printed are exact only in the
    limit of a slow mode (a -> 1), the case they are meant for.
    """
    from swingpy.acceleration import STEFFENSEN_METHODS
    x = np.array([0.3, -1.2, 0.7])
    for a in (0.9, 0.99):
        D, F = x, a * x
        FF = a * F
        u, v = F - D, FF - F
        for name, method in STEFFENSEN_METHODS.items():
            new = method(D, F, FF, u, v, u * 0 + v - u)
            if name in ("graves_morris", "epsilon2"):
                assert np.linalg.norm(new) < (12.0 * (1.0 - a) if name == "graves_morris" else 120.0) * np.linalg.norm(x)
            else:
                assert np.linalg.norm(new) < 1e-9 * np.linalg.norm(x), name


def test_acceleration_reaches_cyclic_steady_state_in_fewer_cycles():
    """
    The basic four-step VSA cycle of Haghpanah et al. (2013) on a coarse grid: Anderson mixing and the Graves-Morris
    method of Subraveti, Karimi and Gazzani (2026) both reach cyclic steady state within 100 cycles and agree with
    each other, while plain successive substitution has not reached it after as many cycles as Anderson needed.
    """
    from setups import cycles
    point = cycles.EXAMPLE_POINTS["basic four step"]
    model = cycles.model(feed_velocity=point["feed_velocity"], n_cells=8)
    start = cycles.initial_state(model)

    def run(**options):
        return point["cycle"](**point["kwargs"]).run(model, start, verbose=False, keep_cycles=0, **options)

    anderson = run(max_cycles=100, accelerate=True)
    graves_morris = run(max_cycles=100, accelerate="graves_morris", criterion="residual", consecutive=1)
    assert anderson.converged and graves_morris.converged
    assert np.isfinite(graves_morris.history[-1].residual)
    assert graves_morris.performance.purity == pytest.approx(anderson.performance.purity, abs=0.5)
    assert graves_morris.performance.recovery == pytest.approx(anderson.performance.recovery, abs=0.5)
    plain = run(max_cycles=len(anderson.history))
    assert not plain.converged
    with pytest.raises(ValueError):
        run(accelerate="no_such_method", max_cycles=1)


def test_sips_linear_exponent_form():
    """
    exponent_form="linear": n(T) = n_ref + alpha (T/Tref - 1), capacity n_inf(T) = n_ref exp(-chi (T/Tref - 1)),
    b = b0 exp(-dU/RT), q = n_inf (b p)^n / (1 + (b p)^n).
    """
    from swingpy.gas import GAS_CONSTANT as R
    from swingpy.isotherms import Sips
    iso = Sips(qs=(6.0,), b0=(1.0e-4,), dU=(-28000.0,), n=(0.5,), alpha=(0.6,), chi=(0.5,),
               reference_temperature=298.15, pressure_based=True, exponent_form="linear")
    for T, p in ((298.15, 0.15), (373.15, 1.0), (413.15, 0.5)):
        c = (0.5 + 0.6 * (T / 298.15 - 1.0))
        n_inf = 6.0 * np.exp(-0.5 * (T / 298.15 - 1.0))
        bp = 1.0e-4 * np.exp(28000.0 / (R * T)) * p
        expected = n_inf * bp ** c / (1.0 + bp ** c)
        conc = np.array([[p * 1e5 / (R * T)]])
        assert iso.loadings(conc, T)[0, 0] == pytest.approx(expected, rel=1e-10)
    with pytest.raises(ValueError):
        Sips(qs=(1.0,), b0=(1.0,), dU=(0.0,), n=(1.0,), exponent_form="other")


def test_mixing_inlet_counts_fresh_feed_and_recycle_apart():
    from swingpy import Cycle, MixingInlet
    model = ColumnModel(Column(length=0.5, inner_radius=0.025, axial_thermal_conductivity=0.0),
                        setups.adsorbent("zeolite-13x"), h.flue_gas(), n_cells=8, reference_velocity=0.1)
    feed = {"CO2": 0.15, "N2": 0.85}
    ads = Step("FEED", 100.0, feed_end=MixingInlet(0.01, feed, buffer="desorbed", temperature=298.15),
               product_end=PressureOutlet(1.0, stream="raffinate"))
    des = Step("DESO", 100.0, feed_end=PressureOutlet(0.2, stream="recycle", record_as="desorbed"))
    cycle = Cycle([ads, des], key_component="CO2", product_stream="raffinate")
    state = model.initial_state(1.0, {"N2": 1.0})
    """No record yet: fresh feed only."""
    first = cycle.run_once(model, state)
    second = cycle.run_once(model, first[-1].final_state)
    p = cycle.evaluate(model, second)
    fresh = 0.01 * 100.0 * np.array([0.15, 0.85])
    assert -p.streams["feed"] == pytest.approx(fresh, rel=1e-6)
    recorded = second[0].moles_in["feed"] - fresh
    """All of the desorbed gas returns."""
    assert recorded == pytest.approx(first[1].moles_in["feed"] * -1.0, rel=1e-3)


def test_steffensen_accelerator_discards_a_runaway_extrapolation():
    from swingpy.acceleration import SteffensenAccelerator
    model = ColumnModel(Column(length=0.5, inner_radius=0.025), setups.adsorbent("zeolite-13x"), h.flue_gas(),
                        n_cells=6, reference_velocity=0.1)
    base = model.initial_state(1.0, {"CO2": 0.15, "N2": 0.85})
    size = model.state_size - 2 * (model.n_components + 2)
    rng = np.random.default_rng(1)
    u = 1e-3 * rng.standard_normal(size)
    v = u + 1e-3 * rng.standard_normal(size)
    """V.w close to zero: lambda of ZL blows up."""
    v = v - (v @ (v - u)) / ((v - u) @ (v - u)) * (v - u) * 0.999
    D = base.copy()
    F = D.copy(); F[:size] += u
    FF = F.copy(); FF[:size] += v
    acc = SteffensenAccelerator(model, method="zienkiewicz_lohner", start=2, every=2)
    acc.next_state(D, F)
    out = acc.next_state(F, FF)
    assert acc.rejected == 1 and acc.extrapolations == 0
    """The plain cycle is kept."""
    assert np.array_equal(out, FF)
