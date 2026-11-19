"""
Particle diffusion against exact solutions.

    Single particle: Crank, The Mathematics of Diffusion, 2nd ed. (1975), Eq. 6.20.
    Column: exact Laplace transform of the linear, isothermal column with axial
    dispersion and diffusion in spheres (Rosen's problem; Ruthven 1984, Ch. 8),
    inverted by the fixed Talbot method (Abate and Valko 2004).
"""

import numpy as np
import pytest
from analytical import crank_sphere_uptake, linear_column_outlet, sphere_transfer, talbot_inverse
from scipy.integrate import solve_ivp

import setups
from swingpy import (Adsorbent, Column, ColumnModel, ConstantLDF, ExtendedLangmuir, FeedInlet, GasMixture,
                      ParticleDiffusion, PressureOutlet, Step)
import setups as h


def test_inverse_laplace_reproduces_known_solutions():
    t = np.array([0.002, 0.01, 0.05, 0.2, 1.0])
    assert np.max(np.abs(talbot_inverse(lambda s: sphere_transfer(s, 1.0) / s, t) - crank_sphere_uptake(t, 1.0))) < 1e-9
    assert talbot_inverse(lambda s: 1.0 / s ** 2, [3.0])[0] == pytest.approx(3.0, rel=1e-9)


def test_single_particle_uptake_matches_crank():
    particle = ParticleDiffusion((1.0,), shells=12)
    t = np.array([0.002, 0.005, 0.01, 0.03, 0.1, 0.3])

    def rhs(_, q):
        d_shells, _ = particle.derivatives(q.reshape(1, 12, 1), np.ones((1, 1)))
        return d_shells.ravel()
    solution = solve_ivp(rhs, (0.0, t[-1]), np.zeros(12), method="BDF", t_eval=t, rtol=1e-10, atol=1e-12)
    uptake = particle.average(solution.y.reshape(1, 12, -1))[0]
    assert np.max(np.abs(uptake - crank_sphere_uptake(t, 1.0))) < 0.006


def _linear_column(kinetics, n_cells=100):
    gas = GasMixture(("CO2", "N2"), heat_capacity=29.0, viscosity=1.8e-5, molecular_diffusivity=1.6e-5)
    sorbent = Adsorbent("linear", ExtendedLangmuir(henry=(0.01, 0.0), affinity=(0.0, 0.0), dU=(0.0, 0.0)),
                        heat_of_adsorption=(0.0, 0.0), density=1000.0, particle_radius=5e-3, kinetics=kinetics)
    column = Column(length=1.0, inner_radius=0.1, bed_voidage=0.4, inner_heat_transfer=0.0, outer_heat_transfer=0.0)
    return ColumnModel(column, sorbent, gas, n_cells=n_cells, reference_velocity=0.1, axial_dispersion=1.0e-3)


def test_column_with_particle_diffusion_matches_the_exact_breakthrough():
    """Slow diffusion, where the particle model matters."""
    rate = 0.005
    t = np.linspace(20.0, 500.0, 49)
    exact = linear_column_outlet(t, velocity=0.1, dispersion=1.0e-3, length=1.0, capacity=0.6 / 0.4 * 1000.0 * 0.01,
                                 rate=rate)
    errors = {}
    for name, kinetics in (("diffusion", ParticleDiffusion((rate, rate))), ("ldf", ConstantLDF((15 * rate, 15 * rate)))):
        model = _linear_column(kinetics)
        step = Step("FEED", 500.0, feed_end=FeedInlet(0.1, {"CO2": 1e-3, "N2": 1 - 1e-3}), product_end=PressureOutlet(1.0))
        result = model.simulate(step, model.initial_state(1.0, {"N2": 1.0}), n_output=501)
        outlet = model.end_history(result, "product")["mole_fractions"][0] / 1e-3
        errors[name] = np.max(np.abs(np.interp(t, result.time, outlet) - exact))
    assert errors["diffusion"] < 0.004
    """The linear driving force is only an approximation here."""
    assert errors["ldf"] > 3 * errors["diffusion"]


def test_particle_diffusion_conserves_mass_and_both_backends_agree():
    pytest.importorskip("numba")
    from dataclasses import replace
    sorbent = replace(setups.adsorbent("zeolite-13x"), kinetics=ParticleDiffusion((0.02, 0.5), shells=8))
    step = Step("FEED", 300.0, feed_end=FeedInlet(1.0, {"CO2": 0.15, "N2": 0.85}), product_end=PressureOutlet(1.0))
    fast = ColumnModel(h.column(), sorbent, h.flue_gas(), n_cells=15, backend="numba")
    slow = ColumnModel(h.column(), sorbent, h.flue_gas(), n_cells=15, backend="numpy")
    result = fast.simulate(step, fast.initial_state(1.0, {"N2": 1.0}))
    stored = fast.component_inventory(result.final_state) - fast.component_inventory(result.initial_state)
    assert np.allclose(stored, result.moles_in["feed"] + result.moles_in["product"], rtol=1e-4, atol=1e-6)
    state = result.states[:, len(result.time) // 2]
    a, b = fast.rhs(0.3, state, step), slow.rhs(0.3, state, step)
    assert np.max(np.abs(a - b)) < 1e-9 * np.max(np.abs(b))
    shells = fast.unpack(result.final_state)["shells"]
    """
    The mean of the shells and the stored loading are integrated separately; they agree to well within the solver's
    absolute tolerance (1e-7; measured: 1.5e-9).
    """
    assert np.allclose(sorbent.kinetics.average(shells), fast.unpack(result.final_state)["x"], rtol=1e-6, atol=1e-8)
