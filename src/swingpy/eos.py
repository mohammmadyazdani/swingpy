"""
Equations of state of the gas phase.

    from swingpy.eos import PengRobinson

    eos = PengRobinson(["CO2", "H2"])                     # constants of the gas data bank
    eos = PengRobinson(["CO2", "H2"], constants=[(304.13, 73.77, 0.2239), (33.15, 12.96, -0.219)])
    Z = eos.compressibility(T, P, y)        # T in K, P in bar; y one composition or (n_components, n_points)
    phi = eos.fugacity_coefficients(T, P, y)

``IdealGas`` gives Z = 1 and fugacity coefficients of one. ``PengRobinson``
is the cubic equation of Peng and Robinson, Ind. Eng. Chem. Fundam. 15 (1976)
59, https://doi.org/10.1021/i160057a011, with the van der Waals one-fluid
mixing rules

    a = sum_i sum_j y_i y_j (1 - k_ij) sqrt(a_i a_j),   b = sum_i y_i b_i,
    a_i = Omega_a R^2 Tc_i^2 / pc_i [1 + kappa_i (1 - sqrt(T / Tc_i))]^2,   b_i = Omega_b R Tc_i / pc_i,
    kappa_i = 0.37464 + 1.54226 w_i - 0.26992 w_i^2   (``kappa="1978"``: the form of
        Robinson and Peng (1978) for w > 0.491)
    Z^3 - (1 - B) Z^2 + (A - 3 B^2 - 2 B) Z - (A B - B^2 - B^3) = 0,
        A = a p / (R T)^2, B = b p / (R T); the gas takes the largest root

Omega_a = 0.457235529 and Omega_b = 0.0777960739 are the exact values that
follow from the conditions at the critical point; the article rounds them to
0.45724 and 0.07780 (which changes Z by about 3e-6 at 35 bar).

Critical constants and acentric factors come from the gas data
(``data.gas_properties``) or are given (``constants``). Binary interaction
parameters are zero unless given; they are mixture specific and should come
from a fit to data.

The column model uses the ideal gas law; this equation of state checks how
far the gas is from ideal (``analysis.compressibility_range``, the
``gas.compressibility_check`` section of a specification).
"""

from __future__ import annotations

import numpy as np

from .gas import GAS_CONSTANT

_SQRT2 = np.sqrt(2.0)
OMEGA_A, OMEGA_B = 0.457235529, 0.0777960739


def _as_points(T, P, y):
    y = np.asarray(y, dtype=float)
    if y.ndim == 1:
        y = y[:, None]
    n_points = y.shape[1]
    T = np.broadcast_to(np.asarray(T, dtype=float), (n_points,)) if np.ndim(T) else np.full(n_points, float(T))
    P = np.broadcast_to(np.asarray(P, dtype=float), (n_points,)) if np.ndim(P) else np.full(n_points, float(P))
    return T, P, y


class IdealGas:
    """The ideal gas: Z = 1, fugacity coefficients 1, no residual enthalpy."""

    def compressibility(self, temperature, pressure, mole_fractions):
        T, P, y = _as_points(temperature, pressure, mole_fractions)
        return 1.0 if np.ndim(mole_fractions) == 1 else np.ones_like(T)

    def fugacity_coefficients(self, temperature, pressure, mole_fractions):
        T, P, y = _as_points(temperature, pressure, mole_fractions)
        return np.ones(y.shape[0]) if np.ndim(mole_fractions) == 1 else np.ones_like(y)

    def residual_enthalpy(self, temperature, pressure, mole_fractions):
        T, P, y = _as_points(temperature, pressure, mole_fractions)
        return 0.0 if np.ndim(mole_fractions) == 1 else np.zeros_like(T)

    def molar_density(self, temperature, pressure, mole_fractions):
        """Moles per m3 (p in bar)."""
        T, P, y = _as_points(temperature, pressure, mole_fractions)
        rho = P * 1.0e5 / (GAS_CONSTANT * T)
        return float(rho[0]) if np.ndim(mole_fractions) == 1 else rho


def _cubic_largest_root(c2, c1, c0):
    """Largest real root of z^3 + c2 z^2 + c1 z + c0 = 0 (arrays), by Cardano and one Newton polish."""
    p = c1 - c2 ** 2 / 3.0
    q = 2.0 * c2 ** 3 / 27.0 - c2 * c1 / 3.0 + c0
    disc = (q / 2.0) ** 2 + (p / 3.0) ** 3
    z = np.empty_like(c2)
    one = disc > 0.0
    if np.any(one):
        s = np.sqrt(disc[one])
        z[one] = np.cbrt(-q[one] / 2.0 + s) + np.cbrt(-q[one] / 2.0 - s)
    three = ~one
    if np.any(three):
        m = 2.0 * np.sqrt(-p[three] / 3.0)
        arg = np.clip(3.0 * q[three] / (p[three] * m), -1.0, 1.0)
        """The largest of the three."""
        z[three] = m * np.cos(np.arccos(arg) / 3.0)
    z -= c2 / 3.0
    for _ in range(2):
        f = ((z + c2) * z + c1) * z + c0
        d = (3.0 * z + 2.0 * c2) * z + c1
        z = z - np.where(d != 0.0, f / d, 0.0)
    return z


class PengRobinson:
    """
    Peng-Robinson equation of state for a gas mixture.

    Args:
        components: Gas names, in the order of the compositions.
        interaction: Binary interaction parameters, {(name_i, name_j): k_ij}.
        constants: Optional list of (Tc in K, pc in bar, acentric factor),
            one per component, instead of the data bank values.
        kappa: "1976" (default) or "1978", the form of kappa(w).
        name: Label (for example the source of the constants).
    """

    def __init__(self, components, interaction=None, constants=None, kappa="1976", name=""):
        self.components = [str(c) for c in components]
        if constants is None:
            from . import data
            props = [data.gas_properties(c) for c in self.components]
            constants = [(p["critical_temperature"], p["critical_pressure"], p["acentric_factor"]) for p in props]
        constants = [tuple(float(v) for v in c) for c in constants]
        n = len(self.components)
        if len(constants) != n or any(len(c) != 3 for c in constants):
            raise ValueError("PengRobinson: give one (Tc, pc, acentric factor) per component.")
        self.Tc, pc, self.omega = (np.array(x, dtype=float) for x in zip(*constants))
        if np.any(self.Tc <= 0.0) or np.any(pc <= 0.0):
            raise ValueError("PengRobinson: critical temperatures and pressures must be positive.")
        if kappa not in ("1976", "1978"):
            raise ValueError("PengRobinson: kappa must be '1976' or '1978'.")
        self.kappa_form, self.name = kappa, name
        self.pc = pc * 1.0e5
        self.kappa = 0.37464 + 1.54226 * self.omega - 0.26992 * self.omega ** 2
        if kappa == "1978":
            w = self.omega
            self.kappa = np.where(w > 0.491, 0.379642 + 1.48503 * w - 0.164423 * w ** 2 + 0.016666 * w ** 3,
                                  self.kappa)
        self.b_i = OMEGA_B * GAS_CONSTANT * self.Tc / self.pc
        self._ac = OMEGA_A * (GAS_CONSTANT * self.Tc) ** 2 / self.pc
        self.interaction = dict(interaction or {})
        self.k = np.zeros((n, n))
        for (a, b), value in self.interaction.items():
            i, j = self.components.index(a), self.components.index(b)
            self.k[i, j] = self.k[j, i] = float(value)

    @property
    def n_components(self):
        return len(self.components)

    @property
    def constants(self):
        """(Tc in K, pc in bar, acentric factor) of each component."""
        return [(float(t), float(p) / 1.0e5, float(w)) for t, p, w in zip(self.Tc, self.pc, self.omega)]

    def _parameters(self, T):
        """a_i(T) and d a_i / dT, shape (n, n_points)."""
        sqrt_tr = np.sqrt(T[None, :] / self.Tc[:, None])
        m = 1.0 + self.kappa[:, None] * (1.0 - sqrt_tr)
        a = self._ac[:, None] * m ** 2
        da = -self._ac[:, None] * m * self.kappa[:, None] * sqrt_tr / T[None, :]
        return a, da

    def _mixture(self, T, P, y):
        if y.shape[0] != self.n_components:
            raise ValueError(f"PengRobinson has {self.n_components} components, the composition {y.shape[0]}.")
        a_i, da_i = self._parameters(T)
        sq = np.sqrt(a_i)
        one_minus_k = 1.0 - self.k
        ya = np.einsum("ij,ip,jp->ip", one_minus_k, sq, y * sq)
        """ya: sum_j y_j a_ij."""
        a = np.sum(y * ya, axis=0)
        da = 2.0 * np.einsum("ij,ip,jp->p", one_minus_k, y * da_i / (2.0 * sq), y * sq)
        b = np.sum(y * self.b_i[:, None], axis=0)
        RT = GAS_CONSTANT * T
        A = a * P * 1.0e5 / RT ** 2
        B = b * P * 1.0e5 / RT
        Z = _cubic_largest_root(B - 1.0, A - 3.0 * B ** 2 - 2.0 * B, B ** 3 + B ** 2 - A * B)
        return a, da, b, A, B, Z, ya

    def compressibility(self, temperature, pressure, mole_fractions):
        """Compressibility factor Z = p / (c R T): a float for one composition, else one per point."""
        T, P, y = _as_points(temperature, pressure, mole_fractions)
        Z = self._mixture(T, P, y)[5]
        return float(Z[0]) if np.ndim(mole_fractions) == 1 else Z

    def molar_density(self, temperature, pressure, mole_fractions):
        """Moles per m3 (p in bar)."""
        T, P, y = _as_points(temperature, pressure, mole_fractions)
        rho = P * 1.0e5 / (self._mixture(T, P, y)[5] * GAS_CONSTANT * T)
        return float(rho[0]) if np.ndim(mole_fractions) == 1 else rho

    def fugacity_coefficients(self, temperature, pressure, mole_fractions):
        """phi_i = f_i / (y_i p): shape (n_components,) for one composition, else (n_components, n_points)."""
        T, P, y = _as_points(temperature, pressure, mole_fractions)
        a, _, b, A, B, Z, ya = self._mixture(T, P, y)
        bi = self.b_i[:, None] / b[None, :]
        log_term = np.log((Z + (1.0 + _SQRT2) * B) / (Z + (1.0 - _SQRT2) * B))
        ln_phi = bi * (Z - 1.0) - np.log(Z - B) - A / (2.0 * _SQRT2 * B) * (2.0 * ya / a - bi) * log_term
        phi = np.exp(ln_phi)
        return phi[:, 0] if np.ndim(mole_fractions) == 1 else phi

    def residual_enthalpy(self, temperature, pressure, mole_fractions):
        """H - H(ideal gas) at the same T and p, J/mol."""
        T, P, y = _as_points(temperature, pressure, mole_fractions)
        a, da, b, A, B, Z, _ = self._mixture(T, P, y)
        log_term = np.log((Z + (1.0 + _SQRT2) * B) / (Z + (1.0 - _SQRT2) * B))
        h = GAS_CONSTANT * T * (Z - 1.0) + (T * da - a) / (2.0 * _SQRT2 * b) * log_term
        return float(h[0]) if np.ndim(mole_fractions) == 1 else h
