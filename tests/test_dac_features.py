"""Features added for direct air capture: HumidToth, instantaneous pressurization, the LSODA fallback,
feed inlets from a recorded stream."""

import numpy as np
import pytest

from swingpy import (Adsorbent, Closed, Column, ColumnModel, Cycle, FeedInlet, GasMixture, HumidToth,
                      PressureOutlet, Step, instant_pressurization)
from swingpy.adsorbent import ConstantLDF
from swingpy.gas import GAS_CONSTANT as R

APDES = dict(components=3, co2=0, water=1, qs0=2.2, b0=0.373, dH0=60000.0, t0=0.4247, alpha=-0.4921,
             reference_temperature=296.0, Cm=36.48, CG=0.1489, K=0.5751, antoine=(4.6543, 1435.264, -64.848),
             psi=0.00958, beta=3.448)


def test_humid_toth_follows_its_equations():
    iso = HumidToth(**APDES)
    T = 293.15
    psat = 1e5 * 10 ** (4.6543 - 1435.264 / (T - 64.848))
    p_co2, rh = 40.0, 0.5
    c = np.array([[p_co2 / (R * T)], [rh * psat / (R * T)], [1e5 / (R * T)]])
    q = iso.loadings(c, T)[:, 0]
    x = rh
    q_w = 36.48 * 0.1489 * 0.5751 * x / ((1 - 0.5751 * x) * (1 + (0.1489 - 1) * 0.5751 * x))
    qs = 2.2 / (1 - 0.00958 * q_w)
    b = 0.373 * np.exp(60000.0 / (R * 296.0) * (296.0 / T - 1.0)) * (1 + 3.448 * q_w)
    t = 0.4247 - 0.4921 * (1 - 296.0 / T)
    expected = qs * b * p_co2 / (1 + (b * p_co2) ** t) ** (1 / t)
    assert q[1] == pytest.approx(q_w, rel=1e-12)
    assert q[0] == pytest.approx(expected, rel=1e-12)
    assert q[2] == 0.0
    dry = iso.loadings(np.array([[p_co2 / (R * T)], [0.0], [1e5 / (R * T)]]), T)[0, 0]
    """Water enhances CO2 uptake on the amine (cooperative)"""
    assert q[0] > dry
    with pytest.raises(ValueError):
        HumidToth(**dict(APDES, water=0))


def small_model():
    gas = GasMixture(("CO2", "H2O", "N2"), molar_masses=(0.044, 0.018, 0.028), heat_capacity=30.1,
                     viscosity=1.72e-5, molecular_diffusivity=2.05e-5, adiabatic_constant=1.4)
    sorbent = Adsorbent(name="APDES", isotherm=HumidToth(**APDES), heat_of_adsorption=(60000.0, 43800.0, 0.0),
                        density=61.0, particle_radius=0.00375, particle_voidage=0.9616, heat_capacity=2070.0,
                        adsorbed_phase_heat_capacity=30.1, kinetics=ConstantLDF((0.0002, 0.2, 1.0)))
    column = Column(length=0.01, inner_radius=0.04, outer_radius=0.041, bed_voidage=0.092,
                    axial_thermal_conductivity=0.0, inner_heat_transfer=3.0, outer_heat_transfer=1e4,
                    wall_density=7800.0, wall_heat_capacity=500.0, wall_thermal_conductivity=0.0,
                    ambient_temperature=293.15)
    return ColumnModel(column, sorbent, gas, n_cells=10, scheme="van_leer", reference_pressure=1.0,
                       reference_temperature=293.15, reference_velocity=0.1)


AIR = {"CO2": 0.0004, "H2O": 0.0115, "N2": 0.9881}


def test_instant_pressurization_replaces_the_gas_and_freezes_the_solid():
    m = small_model()
    evacuated = m.simulate(Step("EVAC", 600.0, product_end=PressureOutlet(0.07), feed_end=Closed(),
                                ambient_temperature=393.15), m.initial_state(1.0, AIR, 293.15)).final_state
    r = m.simulate(instant_pressurization("PRES", AIR, 1.0), evacuated)
    before, after = m.unpack(evacuated), m.unpack(r.final_state)
    assert np.allclose(after["x"], before["x"]) and np.allclose(after["T"], before["T"])
    assert np.allclose(after["P"] * m.P0 / 1e5, 1.0)
    assert np.allclose(m.full_composition(after["y"]), np.array([[0.0004], [0.0115], [0.9881]]))
    assert r.moles_in["feed"] == pytest.approx(m.component_inventory(r.final_state) - m.component_inventory(evacuated))
    assert r.time[-1] == 0.0 and r.work["feed"] == 0.0
    """Counted as feed in a cycle."""
    cycle = Cycle([Step("FEED", 100.0, feed_end=FeedInlet(0.1, AIR, temperature=293.15),
                        product_end=PressureOutlet(1.0, stream="waste")),
                   Step("DESO", 600.0, product_end=PressureOutlet(0.07, stream="product"), feed_end=Closed(),
                        ambient_temperature=393.15),
                   instant_pressurization("PRES", AIR, 1.0)], key_component="CO2", product_stream="product")
    results = cycle.run_once(m, m.initial_state(1.0, AIR, 293.15))
    perf = cycle.evaluate(m, results)
    assert -perf.streams["feed"] == pytest.approx(results[0].moles_in["feed"] + results[2].moles_in["feed"])


def test_lsoda_fallback_agrees_with_bdf():
    m = small_model()
    state = m.initial_state(1.0, AIR, 293.15)
    step = Step("BLOW", 1.0, product_end=PressureOutlet(0.07), feed_end=Closed())
    bdf = m.simulate(step, state, max_evaluations=None)
    lsoda = m.simulate(step, state, max_evaluations=1)
    n = m.state_size - 2 * (m.n_components + 2)
    assert np.abs(bdf.final_state[:n] - lsoda.final_state[:n]).max() < 1e-5
    assert lsoda.moles_in["product"] == pytest.approx(bdf.moles_in["product"], rel=1e-3)


def test_feed_inlet_from_a_recorded_stream():
    m = small_model()
    with pytest.raises(ValueError):
        FeedInlet(0.1, "tank")
    inlet = FeedInlet(0.1, "tank", temperature=293.15, default_composition={"N2": 1.0})
    assert inlet.composition == {"N2": 1.0}

    class Record:
        def total_moles(self):
            return np.array([1.0, 0.0, 3.0])
    inlet.attach(Record(), 10.0)
    assert inlet.composition == pytest.approx([0.25, 0.0, 0.75])
    assert m.to_mole_fractions(inlet.composition) == pytest.approx([0.25, 0.0])
