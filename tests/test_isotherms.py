"""Tests of the Sips and Toth isotherms."""

import numpy as np
import pytest

from swingpy import Adsorbent, Column, ColumnModel, DualSiteLangmuir, FeedInlet, PressureOutlet, Sips, Step, Toth
import setups as h

QS, B0, DU = (3.0, 2.0), (8.0e-7, 2.5e-6), (-36000.0, -16000.0)
LANGMUIR = DualSiteLangmuir(qsb=QS, qsd=(0.0, 0.0), b0=B0, d0=(0.0, 0.0), dUb=DU, dUd=(0.0, 0.0))
C = np.array([[0.5, 5.0, 20.0], [30.0, 10.0, 1.0]])


def test_sips_with_unit_exponent_is_langmuir():
    sips = Sips(qs=QS, b0=B0, dU=DU, n=(1.0, 1.0))
    assert np.allclose(sips.loadings(C, 310.0), LANGMUIR.loadings(C, 310.0), rtol=1e-12)


def test_toth_with_unit_exponent_is_langmuir():
    toth = Toth(qs=QS, b0=B0, dU=DU, t=(1.0, 1.0))
    assert np.allclose(toth.loadings(C, 310.0), LANGMUIR.loadings(C, 310.0), rtol=1e-12)


def test_single_component_toth_matches_its_formula():
    toth = Toth(qs=(4.0, 1.0), b0=(1e-6, 1e-6), dU=(-30000.0, -10000.0), t=(0.5, 1.0))
    c, T = 8.0, 300.0
    b = 1e-6 * np.exp(30000.0 / (8.314 * T))
    expected = 4.0 * b * c / (1.0 + (b * c) ** 0.5) ** 2.0
    assert toth.loadings(np.array([[c], [0.0]]), T)[0, 0] == pytest.approx(expected, rel=1e-12)


def test_sips_stays_finite_at_zero_concentration():
    sips = Sips(qs=QS, b0=B0, dU=DU, n=(0.6, 0.9))
    c = np.array([[0.0, 1e-12, 1.0], [0.0, 1.0, 1e-12]])
    assert np.all(np.isfinite(sips.slopes(c, 300.0)))
    assert np.allclose(sips.loadings(c, 300.0)[:, 0], 0.0)


def test_capacity_falls_with_temperature_when_chi_is_positive():
    toth = Toth(qs=QS, b0=B0, dU=(0.0, 0.0), t=(0.8, 0.8), chi=(0.5, 0.5))
    assert toth.loadings(C, 350.0)[0, 2] < toth.loadings(C, 300.0)[0, 2]


@pytest.mark.parametrize("isotherm", [Sips(qs=(5.6, 5.8), b0=(8.0e-7, 2.5e-6), dU=(-36600.0, -15800.0), n=(0.7, 1.0)),
                                      Toth(qs=(5.6, 5.8), b0=(8.0e-7, 2.5e-6), dU=(-36600.0, -15800.0), t=(0.6, 1.0))])
def test_breakthrough_with_new_isotherms_conserves_mass_and_backends_agree(isotherm):
    adsorbent = Adsorbent("test", isotherm, heat_of_adsorption=(36600.0, 15800.0), density=1130.0, particle_radius=1e-3)
    column = Column(length=1.0, inner_radius=0.1445)
    step = Step("FEED", 300.0, feed_end=FeedInlet(1.0, {"CO2": 0.15, "N2": 0.85}), product_end=PressureOutlet(1.0))
    model = ColumnModel(column, adsorbent, h.flue_gas())
    result = model.simulate(step, model.initial_state(1.0, {"N2": 1.0}))
    stored = model.component_inventory(result.final_state) - model.component_inventory(result.initial_state)
    entered = result.moles_in["feed"] + result.moles_in["product"]
    assert np.allclose(stored, entered, rtol=1e-4, atol=1e-6)
    reference = ColumnModel(column, adsorbent, h.flue_gas(), backend="numpy")
    state = result.states[:, 20]
    assert np.max(np.abs(model.rhs(0.1, state, step) - reference.rhs(0.1, state, step))) < 1e-9 * np.max(
        np.abs(reference.rhs(0.1, state, step)))


def test_non_competitive_toth_ignores_the_other_component():
    toth = Toth(qs=(5.09, 3.08), b0=(4.305e-4, 0.882e-4), dU=(-29380.0, -17190.0), t=(0.429, 0.869),
                competitive=False, pressure_based=True)
    alone = toth.loadings(np.array([[8.0], [0.0]]), 323.15)[0, 0]
    mixed = toth.loadings(np.array([[8.0], [40.0]]), 323.15)[0, 0]
    assert mixed == pytest.approx(alone, rel=1e-12)


def test_pressure_based_toth_uses_partial_pressure():
    toth = Toth(qs=(5.09, 3.08), b0=(4.305e-4, 0.882e-4), dU=(-29380.0, -17190.0), t=(0.429, 0.869),
                competitive=False, pressure_based=True)
    T, p_bar = 323.15, 0.195
    K = 4.305e-4 * np.exp(29380.0 / (8.314 * T))
    expected = 5.09 * K * p_bar / (1.0 + (K * p_bar) ** 0.429) ** (1.0 / 0.429)
    c = p_bar * 1.0e5 / (8.314 * T)
    assert toth.loadings(np.array([[c], [0.0]]), T)[0, 0] == pytest.approx(expected, rel=1e-10)
