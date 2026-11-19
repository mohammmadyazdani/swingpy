"""Virial isotherm (Lopes et al.) and series LDF kinetics."""

import numpy as np
import pytest
from scipy.optimize import brentq

import swingpy.isotherms as isotherms
from swingpy import (Adsorbent, Closed, Column, ColumnModel, Cycle, FlowInlet, GasMixture, PressureOutlet,
                      SeriesLDF, Step, Virial)
from swingpy.gas import GAS_CONSTANT as R

"""Lopes et al., Fuel 93 (2012) 510, Table 4 and Chem. Eng. Sci. 66 (2011) 303, Table 3: AC5-KS, order CO2, H2, N2."""
LOPES = dict(K_inf=(8.06e-10, 2.12e-8, 3.06e-9), dH=(-28.02e3, -7.26e3, -18.27e3),
             A0=(0.081e5, -1.917e5, 11.989e5), A1=(45.31e6, 0.1994e6, -278.79e6),
             B0=(-0.365e11, -4.015e11, -13.758e11), B1=(92.247e11, 7315.4e11, 3988.6e11))
AREA = 971e3


def pure_pressure(i, q, T):
    """The pure component virial equation p(q) in Pa."""
    K = LOPES["K_inf"][i] * np.exp(-LOPES["dH"][i] / (R * T))
    A = LOPES["A0"][i] + LOPES["A1"][i] / T
    B = LOPES["B0"][i] + LOPES["B1"][i] / T
    with np.errstate(over="ignore"):
        return q / K * np.exp(2 * A * q / AREA + 1.5 * B * q * q / AREA ** 2)


@pytest.mark.parametrize("compiled", [True, False])
def test_virial_pure_components_follow_the_published_equation(compiled, monkeypatch):
    if not compiled:
        monkeypatch.setattr(isotherms, "_virial_solve", None)
    iso = Virial(specific_area=AREA, **LOPES)
    for T in (298.0, 303.0, 323.0):
        for i in range(3):
            for p in (1e3, 1e5, 5e5, 1e6):
                c = np.zeros((3, 1))
                c[i, 0] = p / (R * T)
                q = iso.loadings(c, T)[:, 0]
                """The physical root: the first crossing of p(q) (negative B makes p(q) fold back at large q)"""
                grid = np.geomspace(1e-12, 30.0, 4000)
                above = np.nonzero(pure_pressure(i, grid, T) > p)[0][0]
                expected = brentq(lambda x: pure_pressure(i, x, T) - p, grid[above - 1], grid[above], xtol=1e-15)
                assert q[i] == pytest.approx(expected, rel=1e-9)
                assert np.all(np.delete(q, i) == 0.0)


def test_virial_matches_figure_2_of_lopes_2012():
    """
    Fig. 2: AC5-KS at 303 K, read from the figure: CO2 about 2.55 mol/kg at 1 bar and 5.8 at 5 bar, H2 about 0.18 mol/kg
    at 5 bar and 0.24 at 7 bar.
    """
    iso = Virial(specific_area=AREA, **LOPES)
    T = 303.0
    def q(i, bar):
        c = np.zeros((3, 1))
        c[i, 0] = bar * 1e5 / (R * T)
        return iso.loadings(c, T)[i, 0]
    assert q(0, 1.0) == pytest.approx(2.55, abs=0.1)
    assert q(0, 5.0) == pytest.approx(5.8, abs=0.15)
    assert q(1, 5.0) == pytest.approx(0.18, abs=0.01)
    assert q(1, 7.0) == pytest.approx(0.24, abs=0.01)


def test_virial_mixture_equations_and_slopes():
    iso = Virial(specific_area=AREA, **LOPES)
    T = np.array([298.0, 303.0, 320.0, 300.0])
    p = np.array([[0.51e5, 1e5, 0.02e5, 0.0], [2.49e5, 4e5, 0.3e5, 3e5], [0.1e5, 0.0, 0.01e5, 0.0]])
    c = p / (R * T)
    q = iso.loadings(c, T)
    A = np.array(LOPES["A0"])[:, None] + np.array(LOPES["A1"])[:, None] / T
    B = np.array(LOPES["B0"])[:, None] + np.array(LOPES["B1"])[:, None] / T
    K = np.array(LOPES["K_inf"])[:, None] * np.exp(-np.array(LOPES["dH"])[:, None] / (R * T))
    for j in range(len(T)):
        for i in range(3):
            if p[i, j] == 0.0:
                assert q[i, j] == 0.0
                continue
            Aij = (A[i, j] + A[:, j]) / 2
            Bijk = (B[i, j] + B[:, j][:, None] + B[:, j][None, :]) / 3
            E = 2 / AREA * Aij @ q[:, j] + 1.5 / AREA ** 2 * q[:, j] @ Bijk @ q[:, j]
            assert p[i, j] == pytest.approx(q[i, j] / K[i, j] * np.exp(E), rel=1e-9)
    H = iso.slopes(c, T)
    assert np.allclose(H * c, q, rtol=1e-12, atol=0.0)
    """Finite where c = 0 too (the Henry limit given the other loadings)"""
    assert np.all(H > 0.0)
    """The slope of an absent component is the limit q_i/c_i as c_i -> 0."""
    small = c.copy()
    small[2, 1] = 1e-9
    assert iso.slopes(small, T)[2, 1] == pytest.approx(H[2, 1], rel=1e-6)


def test_series_ldf_adds_the_resistances():
    kin = SeriesLDF(micropore_rates=(35.64, 46.35), activation_energies=(17.6e3, 8.51e3),
                    macropore_diffusivities=(10.7e-6, 10.7e-6), film_coefficient=4.37e-2, micropore_factor=15.0,
                    macropore_factor=15.0, area_factor=3.0)
    sorbent = Adsorbent(name="AC5-KS", isotherm=Virial(specific_area=AREA, **{k: v[:2] for k, v in LOPES.items()}),
                        heat_of_adsorption=(28.02e3, 7.26e3), density=786.0, particle_radius=1.45e-3,
                        particle_voidage=0.627, heat_capacity=709.0, kinetics=kin)
    slopes = np.array([[0.08, 0.1], [0.002, 0.002]])
    T = np.array([298.0, 320.0])
    k = kin.rate_constants(slopes, sorbent, None, temperature=T)
    for i in range(2):
        for j in range(2):
            K = 786.0 * slopes[i, j]
            micro = 15.0 * kin.micropore_rates[i] * np.exp(-kin.activation_energies[i] / (R * T[j]))
            tau = 1 / micro + K * 1.45e-3 ** 2 / (15 * 0.627 * 10.7e-6) + K * 1.45e-3 / (3 * 4.37e-2)
            assert k[i, j] == pytest.approx(1 / tau, rel=1e-12)
    """Without micropore and film resistance it is the macropore LDF of MacroporeLDF."""
    only_macro = SeriesLDF(micropore_rates=(np.inf, np.inf), activation_energies=(0.0, 0.0),
                           macropore_diffusivities=(1e-5, 1e-5), film_coefficient=np.inf, micropore_factor=15.0,
                           macropore_factor=15.0, area_factor=3.0, max_rate=1e9)
    k = only_macro.rate_constants(slopes, sorbent, None, temperature=T)
    assert k == pytest.approx(15 * 0.627 * 1e-5 / 1.45e-3 ** 2 / (786.0 * slopes), rel=1e-12)
    with pytest.raises(ValueError):
        kin.rate_constants(slopes, sorbent, None)


def small_model():
    gas = GasMixture(("CO2", "H2"), molar_masses=(0.044, 0.002), heat_capacity=30.0, viscosity=1e-5,
                     molecular_diffusivity=1e-5)
    kin = SeriesLDF(micropore_rates=(35.64, 46.35), activation_energies=(17.6e3, 8.51e3),
                    macropore_diffusivities=(10.7e-6, 10.7e-6), film_coefficient=4.37e-2, micropore_factor=15.0,
                    macropore_factor=15.0, area_factor=3.0)
    sorbent = Adsorbent(name="AC5-KS", isotherm=Virial(specific_area=AREA, **{k: v[:2] for k, v in LOPES.items()}),
                        heat_of_adsorption=(28.02e3, 7.26e3), density=786.0, particle_radius=1.45e-3,
                        particle_voidage=0.627, heat_capacity=709.0, adsorbed_phase_heat_capacity=22.0, kinetics=kin)
    column = Column(length=0.313, inner_radius=0.0105, outer_radius=0.0125, bed_voidage=0.396, pressure_drop="ergun",
                    axial_thermal_conductivity=1.125, inner_heat_transfer=80.0, outer_heat_transfer=22.0,
                    wall_density=8238.0, wall_heat_capacity=500.0, wall_thermal_conductivity=0.0,
                    ambient_temperature=298.0, include_pore_gas=True)
    return ColumnModel(column, sorbent, gas, n_cells=12, scheme="weno", reference_pressure=3.0,
                       reference_temperature=298.0, reference_velocity=0.12, axial_dispersion=7.03e-5)


def test_virial_breakthrough_conserves_mass():
    m = small_model()
    step = Step("FEED", 60.0, feed_end=FlowInlet(2.0e-3, {"CO2": 0.17, "H2": 0.83}, temperature=298.0),
                product_end=PressureOutlet(3.0, stream="product"))
    r = m.simulate(step, m.initial_state(3.0, {"H2": 1.0}, 298.0), n_output=61)
    held = m.component_inventory(r.final_state) - m.component_inventory(r.initial_state)
    entered = np.asarray(r.moles_in["feed"]) + np.asarray(r.moles_in["product"])
    assert np.allclose(held, entered, rtol=1e-5, atol=1e-9)
    """CO2 is taken up."""
    assert held[0] > 0.0


def test_virial_and_series_ldf_survive_a_specification_round_trip():
    import yaml
    from swingpy.spec import load_problem, to_spec
    m = small_model()
    cycle = Cycle([Step("FEED", 10.0, feed_end=FlowInlet(2.0e-3, {"CO2": 0.17, "H2": 0.83}, temperature=298.0),
                        product_end=PressureOutlet(3.0, stream="product")),
                   Step("BLOW", 5.0, feed_end=PressureOutlet(0.3, stream="blow"), product_end=Closed())],
                  key_component="H2", product_stream="product")
    spec = to_spec(m, cycle, dict(pressure=3.0, temperature=298.0, composition={"H2": 1.0}))
    problem = load_problem(yaml.safe_load(yaml.safe_dump(spec, sort_keys=False)))
    assert isinstance(problem.model.adsorbent.isotherm, Virial)
    assert isinstance(problem.model.adsorbent.kinetics, SeriesLDF)
    state = problem.initial_state()
    x = state * (1.0 + 0.05 * np.random.default_rng(2).random(state.size))
    for a, b in zip(problem.cycle.steps, cycle.steps):
        assert np.allclose(problem.model.rhs(0.7, x, a), m.rhs(0.7, x, b), rtol=1e-10, atol=1e-13), a.name
