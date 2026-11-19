"""
Peng-Robinson equation of state.

Reference values from CoolProp 8.0.0 (Bell et al., Ind. Eng. Chem. Res. 53
(2014) 2498): its Peng-Robinson backend with k_ij = 0 (an independent
implementation of the same equation) and its reference equations of state
with the GERG-2008 mixture model (the accurate behaviour of the gas).
"""

import numpy as np
import pytest

from swingpy.eos import PengRobinson


def test_matches_an_independent_peng_robinson_implementation():
    pr = PengRobinson(["CO2", "H2"], constants=[(304.1282, 73.773, 0.22394), (33.145, 12.964, -0.219)])
    assert pr.compressibility(298.15, 35.0, [0.5, 0.5]) == pytest.approx(0.9449857317, rel=1e-8)
    assert pr.fugacity_coefficients(298.15, 35.0, [0.5, 0.5]) == pytest.approx([0.8533748666, 1.0461492442], rel=1e-8)


@pytest.mark.parametrize("components, y, T, p, z_reference, tolerance", [
    (["CO2", "N2"], [0.15, 0.85], 298.15, 1.0, 0.999313, 0.001),
    (["CO2", "H2"], [0.50, 0.50], 298.15, 35.0, 0.961899, 0.02),
    (["CO2", "H2"], [0.75, 0.25], 298.15, 25.0, 0.927412, 0.015),
    (["CH4", "H2"], [0.30, 0.70], 303.15, 25.0, 1.007239, 0.012),
])
def test_accuracy_against_reference_equations_of_state(components, y, T, p, z_reference, tolerance):
    z = PengRobinson(components).compressibility(T, p, y)
    assert z == pytest.approx(z_reference, rel=tolerance)


def test_ideal_gas_limit_and_fugacity_consistency():
    pr = PengRobinson(["CO2", "N2"])
    assert pr.compressibility(300.0, 1e-6, [0.5, 0.5]) == pytest.approx(1.0, abs=1e-8)
    assert np.allclose(pr.fugacity_coefficients(300.0, 1e-6, [0.5, 0.5]), 1.0, atol=1e-8)
    """Thermodynamic consistency: d(sum_i y_i ln phi_i)/dp at constant T and y equals (Z - 1)/p."""
    y, T, p = np.array([0.3, 0.7]), 300.0, 30.0
    phi = pr.fugacity_coefficients(T, p, y)
    h = 1e-6
    lnphi_mix = lambda pp: float(y @ np.log(pr.fugacity_coefficients(T, pp, y)))
    dlnphi_dp = (lnphi_mix(p + h) - lnphi_mix(p - h)) / (2 * h)
    assert dlnphi_dp == pytest.approx((pr.compressibility(T, p, y) - 1.0) / p, rel=1e-5)
    assert np.all(phi > 0)


"""Vectorized use, more references, consistency."""

from swingpy.eos import IdealGas  # noqa: E402
from swingpy.gas import GAS_CONSTANT as R  # noqa: E402

"""CoolProp 8.0.0 PR backend constants: Tc [K], pc [bar], w."""
COOLPROP = {"CO2": (304.1282, 73.773, 0.22394), "N2": (126.192, 33.958, 0.0372),
            "H2": (33.145, 12.964, -0.219), "CH4": (190.564, 45.992, 0.01142)}


@pytest.mark.parametrize("fluids, y, T, p, Z, phi", [
    (("CO2",), (1.0,), 313.15, 40.0, 0.791844, (0.821144,)),
    (("CO2",), (1.0,), 400.0, 100.0, 0.810969, (0.818625,)),
    (("N2",), (1.0,), 400.0, 100.0, 1.025241, (1.019755,)),
    (("H2",), (1.0,), 330.0, 60.0, 1.019176, (1.018265,)),
    (("CH4",), (1.0,), 313.15, 40.0, 0.931108, (0.930301,)),
    (("H2", "CH4", "CO2", "N2"), (0.7, 0.1, 0.15, 0.05), 303.15, 25.0, 0.993415,
     (1.011711, 0.963207, 0.927914, 0.991315)),
])
def test_more_coolprop_references(fluids, y, T, p, Z, phi):
    pr = PengRobinson(fluids, constants=[COOLPROP[f] for f in fluids])
    assert pr.compressibility(T, p, y) == pytest.approx(Z, rel=2e-6)
    assert pr.fugacity_coefficients(T, p, y) == pytest.approx(phi, rel=2e-6)


def test_vectorized_over_points():
    pr = PengRobinson(["CO2", "N2"], constants=[COOLPROP["CO2"], COOLPROP["N2"]])
    y = np.array([[0.15, 0.5, 0.9], [0.85, 0.5, 0.1]])
    T = np.array([298.15, 320.0, 350.0])
    Z = pr.compressibility(T, 20.0, y)
    phi = pr.fugacity_coefficients(T, 20.0, y)
    for k in range(3):
        assert Z[k] == pytest.approx(pr.compressibility(T[k], 20.0, y[:, k]), rel=1e-12)
        assert phi[:, k] == pytest.approx(pr.fugacity_coefficients(T[k], 20.0, y[:, k]), rel=1e-12)
    assert pr.molar_density(300.0, 20.0, [0.5, 0.5]) == pytest.approx(
        20.0e5 / (pr.compressibility(300.0, 20.0, [0.5, 0.5]) * R * 300.0))


def test_partial_molar_and_enthalpy_consistency():
    pr = PengRobinson(["CO2", "H2", "CH4"], interaction={("CO2", "H2"): -0.16, ("CO2", "CH4"): 0.09},
                      constants=[COOLPROP["CO2"], COOLPROP["H2"], COOLPROP["CH4"]])
    T, p, y = 310.0, 30.0, np.array([0.3, 0.6, 0.1])
    ln_phi = np.log(pr.fugacity_coefficients(T, p, y))

    """N G^R / RT."""
    def n_g_residual(n):
        return float(np.sum(n * np.log(pr.fugacity_coefficients(T, p, n / n.sum()))))
    """Ln phi_i = d(n G^R / RT) / d n_i."""
    for i in range(3):
        dn = np.zeros(3)
        dn[i] = 1e-6
        assert (n_g_residual(y + dn) - n_g_residual(y - dn)) / 2e-6 == pytest.approx(ln_phi[i], rel=1e-6, abs=1e-9)

    def g(t):
        return float(y @ np.log(pr.fugacity_coefficients(t, p, y)))
    """H^R = -R T^2 d(G^R/RT)/dT."""
    expected = -R * T ** 2 * (g(T + 1e-3) - g(T - 1e-3)) / 2e-3
    assert pr.residual_enthalpy(T, p, y) == pytest.approx(expected, rel=1e-6)


def test_ideal_gas_and_input_checks():
    ideal = IdealGas()
    assert ideal.compressibility(300.0, 30.0, [0.5, 0.5]) == 1.0
    assert np.all(ideal.fugacity_coefficients(300.0, 30.0, [0.5, 0.5]) == 1.0)
    assert ideal.molar_density(300.0, 1.0, [1.0]) == pytest.approx(1e5 / (R * 300.0))
    with pytest.raises(ValueError):
        PengRobinson(["CO2", "N2"], constants=[COOLPROP["CO2"]])
    with pytest.raises(ValueError):
        PengRobinson(["CO2", "N2"], constants=[COOLPROP["CO2"], COOLPROP["N2"]]).compressibility(300.0, 1.0, [1.0])
