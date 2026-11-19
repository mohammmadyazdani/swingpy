"""Water vapour: saturation pressure, humidity, water isotherms and humid gas in a column."""

import numpy as np
import pytest

import setups
from swingpy import AranovichDonohue, Column, ColumnModel, FeedInlet, PressureOutlet, Sips, Step, data, humidity


def test_saturation_pressure_relative_humidity_and_dew_point():
    assert humidity.saturation_pressure(298.15) == pytest.approx(0.03169, rel=2e-3)
    assert humidity.saturation_pressure(373.15) == pytest.approx(1.0142, rel=3e-3)
    y = humidity.mole_fraction(0.5, 313.15, 1.0)
    assert humidity.relative_humidity(y, 313.15, 1.0) == pytest.approx(0.5)
    assert humidity.dew_point(humidity.mole_fraction(1.0, 313.15, 1.0), 1.0) == pytest.approx(313.15, abs=1e-6)
    humid = humidity.humid_composition({"CO2": 0.15, "N2": 0.85}, 0.5, 313.15, 1.0)
    assert sum(humid.values()) == pytest.approx(1.0) and humid["H2O"] == pytest.approx(y)


def test_water_isotherm_of_son_et_al_reproduces_their_data():
    water = AranovichDonohue(Sips(qs=(18.87,), b0=(1.353e-8,), dU=(-8150.0 * 8.314,), n=(0.288,), pressure_based=True),
                             component=0, d=0.02772)
    for celsius, p_kpa, measured in ((25, 0.128, 12.8), (25, 1.28, 15.5), (70, 1.67, 11.8), (100, 25.33, 12.4)):
        T = celsius + 273.15
        q = water.loadings(np.array([[p_kpa * 1e3 / (8.314 * T)]]), T)[0, 0]
        assert q == pytest.approx(measured, rel=0.1)


def test_humid_zeolite_entry_and_water_displacing_co2():
    sorbent = setups.adsorbent("zeolite-13x-humid", components=["CO2", "H2O", "N2"])
    T = 313.15
    c = np.array([[0.15], [0.0], [0.85]]) * 1e5 / (8.314 * T)
    dry = sorbent.isotherm.loadings(c, T)[0, 0]
    c[1] = 0.03 * 1e5 / (8.314 * T)
    wet = sorbent.isotherm.loadings(c, T)
    """Water takes the sites."""
    assert wet[0, 0] < 0.25 * dry and wet[1, 0] > 10.0
    feed = humidity.humid_composition({"CO2": 0.15, "N2": 0.85}, 0.5, T, 1.0)
    model = ColumnModel(Column(length=1.0, inner_radius=0.1445, ambient_temperature=T), sorbent,
                        data.gas(["CO2", "H2O", "N2"], feed, temperature=T), n_cells=10, reference_temperature=T)
    step = Step("FEED", 600.0, feed_end=FeedInlet(1.0, feed, T), product_end=PressureOutlet(1.0))
    result = model.simulate(step, model.initial_state(1.0, {"N2": 1.0}, T))
    stored = model.component_inventory(result.final_state) - model.component_inventory(result.initial_state)
    assert np.allclose(stored, result.moles_in["feed"] + result.moles_in["product"], rtol=1e-4, atol=1e-6)
