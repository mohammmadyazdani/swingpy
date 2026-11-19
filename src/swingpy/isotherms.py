"""
Multicomponent adsorption isotherms.

Every isotherm in this module returns its result through the "Henry slope"
H_i = q_i* / c_i, in m3/kg. The loading follows as q_i* = H_i c_i, and the
linear driving force model needs c_i / q_i* = 1 / H_i, which can then be
computed without dividing by a loading that may be zero.

Available: ``DualSiteLangmuir`` (computed inside the compiled kernel when
numba is installed), ``Sips``, ``Toth``, ``ExtendedLangmuir``,
``MultisiteLangmuir``, ``Virial`` and ``IAST``. To add a new isotherm, write a
class with a ``slopes(c, T)`` method that returns H with the same shape as
``c``, a ``loadings(c, T)`` method and a ``total_saturation`` property.
"""

from __future__ import annotations

from dataclasses import dataclass

from .checks import positive, same_length
from typing import Sequence

import numpy as np

from .gas import GAS_CONSTANT


@dataclass
class DualSiteLangmuir:
    """
    Competitive dual-site Langmuir isotherm with Arrhenius temperature dependence.

    The equilibrium loading of component i is

        q_i* = qsb_i b_i c_i / (1 + sum_k b_k c_k) + qsd_i d_i c_i / (1 + sum_k d_k c_k)

    with b_i = b0_i exp(-dUb_i / RT) and d_i = d0_i exp(-dUd_i / RT).
    Concentrations are in mol/m3 and loadings in mol/kg. Each parameter is a
    sequence with one value per gas component, in the order of
    ``GasMixture.components``. A single-site Langmuir isotherm is obtained
    with ``qsd`` and ``d0`` equal to zero.

    Attributes:
        qsb: Saturation capacity of site 1, mol/kg.
        qsd: Saturation capacity of site 2, mol/kg.
        b0: Pre-exponential factor of site 1, m3/mol.
        d0: Pre-exponential factor of site 2, m3/mol.
        dUb: Internal energy of adsorption on site 1, J/mol (negative).
        dUd: Internal energy of adsorption on site 2, J/mol (negative).
    """

    qsb: Sequence[float]
    qsd: Sequence[float]
    b0: Sequence[float]
    d0: Sequence[float]
    dUb: Sequence[float]
    dUd: Sequence[float]

    def __post_init__(self):
        for name in ("qsb", "qsd", "b0", "d0", "dUb", "dUd"):
            setattr(self, name, np.asarray(getattr(self, name), dtype=float))

        same_length("DualSiteLangmuir", qsb=self.qsb, qsd=self.qsd, b0=self.b0, d0=self.d0, dUb=self.dUb, dUd=self.dUd)
        positive("DualSiteLangmuir", "saturation capacity (mol/kg)", self.qsb, allow_zero=True)
        positive("DualSiteLangmuir", "b0", self.b0, allow_zero=True)

    def slopes(self, c, temperature):
        """
        Henry slopes q_i*/c_i in m3/kg.

        Args:
            c: Concentrations, array of shape (n_components, ...) in mol/m3.
            temperature: Temperature in K, broadcastable to c[0].

        Returns:
            Array with the shape of ``c``.
        """
        RT = GAS_CONSTANT * np.asarray(temperature)
        b = self.b0[:, None] * np.exp(-self.dUb[:, None] / RT)
        d = self.d0[:, None] * np.exp(-self.dUd[:, None] / RT)
        c = np.asarray(c).reshape(b.shape[0], -1)
        site1 = 1.0 + np.sum(b * c, axis=0)
        site2 = 1.0 + np.sum(d * c, axis=0)
        return self.qsb[:, None] * b / site1 + self.qsd[:, None] * d / site2

    def loadings(self, c, temperature):
        """Equilibrium loadings q_i* in mol/kg, with the shape of ``c``."""
        c = np.asarray(c, dtype=float).reshape(len(self.qsb), -1)
        return self.slopes(c, temperature) * c

    @property
    def total_saturation(self):
        """qsb + qsd for each component, mol/kg."""
        return self.qsb + self.qsd


def _temperature_terms(T, reference_temperature, exponent0, exponent_alpha, saturation0, saturation_chi,
                       exponent_form="inverse"):
    """
    Exponent n(T) = n0 + alpha (1 - T_ref / T) ("inverse") or n0 + alpha (T / T_ref - 1) ("linear"), and
    capacity q_s(T) = q_s0 exp(chi (1 - T / T_ref)).
    """
    T = np.asarray(T, dtype=float)
    shape = (1.0 - reference_temperature / T) if exponent_form == "inverse" else (T / reference_temperature - 1.0)
    exponent = exponent0[:, None] + exponent_alpha[:, None] * shape
    saturation = saturation0[:, None] * np.exp(saturation_chi[:, None] * (1.0 - T / reference_temperature))
    return exponent, saturation


@dataclass
class Sips:
    """
    Extended Sips (Langmuir-Freundlich) isotherm for mixtures.

        q_i* = qs_i (b_i c_i)^n_i / (1 + sum_k (b_k c_k)^n_k)

    with b_i = b0_i exp(-dU_i / RT), n_i(T) = n0_i + alpha_i (1 - T_ref / T) and
    qs_i(T) = qs0_i exp(chi_i (1 - T / T_ref)). With ``exponent_form="linear"``
    the exponent is n_i(T) = n0_i + alpha_i (T / T_ref - 1), the form of Do
    (1998) used by Hefti et al. (2015) for CO2 and N2 on zeolite 13X; their
    capacity exp(chi (T / T_ref - 1)) is this one with the sign of chi
    reversed. With n = 1 it is the
    competitive Langmuir isotherm. Concentrations in mol/m3, loadings in
    mol/kg, one value per gas component for every parameter.

    For n < 1 the slope q*/c grows without bound as c goes to zero (the
    Freundlich limit). The slope is therefore evaluated at no less than
    ``minimum_concentration``, so the linear driving force rate stays finite,
    while the loading itself still goes to zero with c.

    Attributes:
        qs: Saturation capacity at T_ref, mol/kg.
        b0: Pre-exponential factor, m3/mol.
        dU: Internal energy of adsorption, J/mol (negative).
        n: Heterogeneity exponent at T_ref, -.
        alpha: Temperature coefficient of n, -.
        chi: Temperature coefficient of the capacity, -.
        reference_temperature: T_ref, K.
        exponent_form: "inverse" (default) or "linear", the temperature
            dependence of n above.
        minimum_concentration: Floor of c in the slope, mol/m3.
        pressure_based: True when b0 is a constant in 1/bar multiplying partial
            pressures (b c = K p), as for the Toth isotherm.
        saturation_function, exponent_function: Optional functions of the
            temperature (K, any shape) returning q_s,i and n_i with an extra
            leading axis of one value per component, for temperature
            dependences other than the default forms (they replace qs, chi
            and n, alpha respectively).
    """

    qs: Sequence[float]
    b0: Sequence[float]
    dU: Sequence[float]
    n: Sequence[float]
    alpha: Sequence[float] = None
    chi: Sequence[float] = None
    reference_temperature: float = 298.15
    minimum_concentration: float = 1.0e-6
    pressure_based: bool = False
    saturation_function: object = None
    exponent_function: object = None
    exponent_form: str = "inverse"

    def __post_init__(self):
        if self.exponent_form not in ("inverse", "linear"):
            raise ValueError("Sips: exponent_form must be 'inverse' or 'linear'.")
        for name in ("qs", "b0", "dU", "n"):
            setattr(self, name, np.asarray(getattr(self, name), dtype=float))
        self.alpha = np.zeros_like(self.qs) if self.alpha is None else np.asarray(self.alpha, dtype=float)
        self.chi = np.zeros_like(self.qs) if self.chi is None else np.asarray(self.chi, dtype=float)

        same_length("Sips", qs=self.qs, b0=self.b0, dU=self.dU, n=self.n, alpha=self.alpha, chi=self.chi)
        positive("Sips", "saturation capacity (mol/kg)", self.qs, allow_zero=True)
        positive("Sips", "b0", self.b0, allow_zero=True)
        positive("Sips", "exponent n", self.n)

    def slopes(self, c, temperature):
        """Henry slopes q_i*/c_i in m3/kg, with the shape of ``c``."""
        T = np.asarray(temperature, dtype=float)
        c = np.asarray(c, dtype=float).reshape(len(self.qs), -1)
        b = self.b0[:, None] * np.exp(-self.dU[:, None] / (GAS_CONSTANT * T))
        if self.pressure_based:
            b = b * GAS_CONSTANT * T / 1.0e5
        n, qs = self._terms(T)
        bc = b * np.maximum(c, 0.0)
        denominator = 1.0 + np.sum(bc ** n, axis=0)
        c_floor = np.maximum(c, self.minimum_concentration)
        return qs * (b * c_floor) ** n / c_floor / denominator

    def _terms(self, T):
        """Exponents n_i(T) and capacities q_s,i(T), each with a leading axis of components."""
        n, qs = _temperature_terms(T, self.reference_temperature, self.n, self.alpha, self.qs, self.chi,
                                   self.exponent_form)
        if self.exponent_function is not None:
            n = np.broadcast_to(np.asarray(self.exponent_function(T), dtype=float).reshape(len(self.qs), -1), n.shape)
        if self.saturation_function is not None:
            qs = np.broadcast_to(np.asarray(self.saturation_function(T), dtype=float).reshape(len(self.qs), -1),
                                 qs.shape)
        return n, qs

    def loadings(self, c, temperature):
        """Equilibrium loadings q_i* in mol/kg, with the shape of ``c`` (exact above the concentration floor)."""
        c = np.asarray(c, dtype=float).reshape(len(self.qs), -1)
        return self.slopes(c, temperature) * np.maximum(c, 0.0)

    @property
    def total_saturation(self):
        """Saturation capacity at T_ref, mol/kg."""
        return self.qs


@dataclass
class Toth:
    """
    Toth isotherm for mixtures, competitive (extended) or not.

    Competitive (default):
        q_i* = qs_i b_i c_i / (1 + sum_k (b_k c_k)^t_i)^(1 / t_i)
    Non-competitive (``competitive=False``), each component on its own sites:
        q_i* = qs_i b_i c_i / (1 + (b_i c_i)^t_i)^(1 / t_i)

    with b_i = b0_i exp(-dU_i / RT), t_i(T) = t0_i + alpha_i (1 - T_ref / T)
    and qs_i(T) = qs0_i exp(chi_i (1 - T / T_ref)). With t = 1 the competitive
    form is the competitive Langmuir isotherm. Unlike the Sips isotherm it has
    a finite Henry constant (qs b), so it behaves well at low concentration.

    The Toth isotherm is often written with partial pressures,
    q = qs K p / (1 + (K p)^t)^(1/t), with K = K0 exp(-dH / RT) in 1/bar. Use
    ``pressure_based=True`` and give K0 as ``b0`` (1/bar) and dH as ``dU``:
    the model then uses b = K RT / 1e5, so that b c = K p exactly.

    Attributes:
        qs: Saturation capacity at T_ref, mol/kg.
        b0: Pre-exponential factor, m3/mol (or 1/bar with ``pressure_based``).
        dU: Energy of adsorption in the exponent, J/mol (negative).
        t: Heterogeneity parameter at T_ref (0 < t <= 1), -.
        alpha: Temperature coefficient of t, -.
        chi: Temperature coefficient of the capacity, -.
        reference_temperature: T_ref, K.
        competitive: False when every component adsorbs on its own sites.
        pressure_based: True when b0 is a pressure-based constant in 1/bar.
    """

    qs: Sequence[float]
    b0: Sequence[float]
    dU: Sequence[float]
    t: Sequence[float]
    alpha: Sequence[float] = None
    chi: Sequence[float] = None
    reference_temperature: float = 298.15
    competitive: bool = True
    pressure_based: bool = False

    def __post_init__(self):
        for name in ("qs", "b0", "dU", "t"):
            setattr(self, name, np.asarray(getattr(self, name), dtype=float))
        self.alpha = np.zeros_like(self.qs) if self.alpha is None else np.asarray(self.alpha, dtype=float)
        self.chi = np.zeros_like(self.qs) if self.chi is None else np.asarray(self.chi, dtype=float)

        same_length("Toth", qs=self.qs, b0=self.b0, dU=self.dU, t=self.t, alpha=self.alpha, chi=self.chi)
        positive("Toth", "saturation capacity (mol/kg)", self.qs, allow_zero=True)
        positive("Toth", "b0", self.b0, allow_zero=True)
        if np.any(self.t <= 0.0) or np.any(self.t > 1.0):
            raise ValueError(f"Toth: the heterogeneity parameter t must lie in (0, 1], got {self.t}.")
    def _b(self, T):
        b = self.b0[:, None] * np.exp(-self.dU[:, None] / (GAS_CONSTANT * T))
        return b * GAS_CONSTANT * T / 1.0e5 if self.pressure_based else b

    def slopes(self, c, temperature):
        """Henry slopes q_i*/c_i in m3/kg, with the shape of ``c``."""
        T = np.asarray(temperature, dtype=float)
        c = np.asarray(c, dtype=float).reshape(len(self.qs), -1)
        b = self._b(T)
        t, qs = _temperature_terms(T, self.reference_temperature, self.t, self.alpha, self.qs, self.chi)
        bc = b * np.maximum(c, 0.0)
        slopes = np.empty_like(bc)
        for i in range(len(self.qs)):
            terms = np.sum(bc ** t[i], axis=0) if self.competitive else bc[i] ** t[i]
            slopes[i] = qs[i] * b[i] / (1.0 + terms) ** (1.0 / t[i])
        return slopes

    def loadings(self, c, temperature):
        """Equilibrium loadings q_i* in mol/kg, with the shape of ``c``."""
        c = np.asarray(c, dtype=float).reshape(len(self.qs), -1)
        return self.slopes(c, temperature) * np.maximum(c, 0.0)

    @property
    def total_saturation(self):
        """Saturation capacity at T_ref, mol/kg."""
        return self.qs


@dataclass
class HumidToth:
    """
    CO2 and water on an amine sorbent: a Toth isotherm for CO2 whose
    capacity and affinity grow with the water loading, and a GAB isotherm for
    water (Stampi-Bombelli et al., Adsorption 26 (2020) 1183,
    https://doi.org/10.1007/s10450-020-00249-w; used for direct air capture by
    Balasubramaniam et al., Chem. Eng. J. 485 (2024) 149568). Other
    components do not adsorb.

        q_H2O* = Cm CG K x / [(1 - K x)(1 + (CG - 1) K x)],   x = p_H2O / p_sat(T)
        q_CO2* = qs_w b_w p / (1 + (b_w p)^t)^(1/t),
        qs_w = qs / (1 - psi q_H2O*),   b_w = b (1 + beta q_H2O*),
        qs = qs0 exp(chi (1 - T / T0)),   b = b0 exp(dH0 / (R T0) (T0 / T - 1)),
        t = t0 + alpha (1 - T0 / T),

    with p the partial pressure in Pa and log10 p_sat[bar] = A - B / (T + C)
    (Antoine). The water loading does not depend on CO2.

    Attributes:
        components: Number of components of the gas.
        co2, water: Positions of CO2 and water among them.
        qs0: CO2 capacity at T0, mol/kg.
        b0: CO2 affinity at T0, 1/Pa.
        dH0: Heat of adsorption of CO2 at zero coverage, J/mol (positive).
        t0, alpha, chi: Toth heterogeneity, its temperature coefficient, and
            the temperature coefficient of the capacity.
        reference_temperature: T0, K.
        Cm, CG, K: GAB constants (Cm in mol/kg).
        antoine: (A, B, C) of the saturation pressure of water, bar and K.
        psi, beta: Water coefficients of the capacity and the affinity, kg/mol.
    """

    components: int
    co2: int
    water: int
    qs0: float
    b0: float
    dH0: float
    t0: float
    Cm: float
    CG: float
    K: float
    antoine: tuple
    alpha: float = 0.0
    chi: float = 0.0
    reference_temperature: float = 298.15
    psi: float = 0.0
    beta: float = 0.0

    def __post_init__(self):
        if not (0 <= self.co2 < self.components and 0 <= self.water < self.components) or self.co2 == self.water:
            raise ValueError("HumidToth: co2 and water must be two different positions among the components.")
        if self.qs0 < 0.0 or self.b0 < 0.0 or self.Cm < 0.0 or not 0.0 <= self.K < 1.0:
            raise ValueError("HumidToth: qs0, b0 and Cm must be non-negative and K in [0, 1).")
        if len(self.antoine) != 3:
            raise ValueError("HumidToth: antoine is (A, B, C).")

    def saturation_pressure(self, temperature):
        """Saturation pressure of water, Pa."""
        A, B, C = self.antoine
        return 1.0e5 * 10.0 ** (A - B / (np.asarray(temperature, dtype=float) + C))

    def _water(self, c_w, T):
        """GAB loading (mol/kg) and its slope q/c (m3/kg)."""
        psat = self.saturation_pressure(T)
        x = np.clip(c_w * GAS_CONSTANT * T / psat, 0.0, 0.999 / max(self.K, 1e-12))
        denominator = (1.0 - self.K * x) * (1.0 + (self.CG - 1.0) * self.K * x)
        slope = self.Cm * self.CG * self.K * GAS_CONSTANT * T / psat / denominator
        return slope * c_w, slope

    def slopes(self, c, temperature):
        """Henry slopes q_i*/c_i in m3/kg, with the shape of ``c``."""
        T = np.broadcast_to(np.asarray(temperature, dtype=float), np.shape(np.asarray(c))[1:] or (1,))
        c = np.maximum(np.asarray(c, dtype=float).reshape(self.components, -1), 0.0)
        out = np.zeros_like(c)
        q_w, out[self.water] = self._water(c[self.water], T)
        T0 = self.reference_temperature
        qs = self.qs0 * np.exp(self.chi * (1.0 - T / T0)) / (1.0 - self.psi * q_w)
        b = self.b0 * np.exp(self.dH0 / (GAS_CONSTANT * T0) * (T0 / T - 1.0)) * (1.0 + self.beta * q_w)
        t = self.t0 + self.alpha * (1.0 - T0 / T)
        bp = b * c[self.co2] * GAS_CONSTANT * T
        out[self.co2] = qs * b * GAS_CONSTANT * T / (1.0 + bp ** t) ** (1.0 / t)
        return out

    def loadings(self, c, temperature):
        """Equilibrium loadings q_i* in mol/kg, with the shape of ``c``."""
        c = np.asarray(c, dtype=float).reshape(self.components, -1)
        return self.slopes(c, temperature) * np.maximum(c, 0.0)

    @property
    def total_saturation(self):
        """Capacities used to scale the loadings, mol/kg."""
        q = np.zeros(self.components)
        q[self.co2], q[self.water] = self.qs0, self.Cm
        return q


@dataclass
class ExtendedLangmuir:
    """
    Extended Langmuir isotherm with separate capacity and affinity constants,

        q_i* = A_i e_i c_i / (1 + sum_j B_j e_j c_j),   e_i = exp(-dU_i / R (1/T - 1/T_ref)),

    so that a component can adsorb in proportion to its concentration without
    taking up sites (B_i = 0, a "Henry" component such as H2 on many
    adsorbents). With A_i = q_s,i B_i it is the competitive Langmuir isotherm.

    Attributes:
        henry: A_i at T_ref, m3/kg (or mol/(kg bar) with ``pressure_based``).
        affinity: B_i at T_ref, m3/mol (or 1/bar with ``pressure_based``).
        dU: Energy of adsorption in the exponent, J/mol (negative for adsorption).
        reference_temperature: T_ref, K.
        pressure_based: True when the constants multiply partial pressures in bar.
    """

    henry: Sequence[float]
    affinity: Sequence[float]
    dU: Sequence[float]
    reference_temperature: float = 298.15
    pressure_based: bool = False

    def __post_init__(self):
        for name in ("henry", "affinity", "dU"):
            setattr(self, name, np.asarray(getattr(self, name), dtype=float))
        same_length("ExtendedLangmuir", henry=self.henry, affinity=self.affinity, dU=self.dU)
        positive("ExtendedLangmuir", "henry", self.henry, allow_zero=True)
        positive("ExtendedLangmuir", "affinity", self.affinity, allow_zero=True)

    def _factors(self, T):
        T = np.asarray(T, dtype=float)
        e = np.exp(-self.dU[:, None] / GAS_CONSTANT * (1.0 / T - 1.0 / self.reference_temperature))
        scale = GAS_CONSTANT * T / 1.0e5 if self.pressure_based else 1.0
        return self.henry[:, None] * e * scale, self.affinity[:, None] * e * scale

    def slopes(self, c, temperature):
        """Henry slopes q_i*/c_i in m3/kg, with the shape of ``c``."""
        c = np.asarray(c, dtype=float).reshape(len(self.henry), -1)
        A, B = self._factors(temperature)
        return A / (1.0 + np.sum(B * np.maximum(c, 0.0), axis=0))

    def loadings(self, c, temperature):
        """Equilibrium loadings q_i* in mol/kg, with the shape of ``c``."""
        c = np.asarray(c, dtype=float).reshape(len(self.henry), -1)
        return self.slopes(c, temperature) * np.maximum(c, 0.0)

    @property
    def total_saturation(self):
        """Capacity scale for the loading reference, mol/kg (A/B, or 1 for Henry components)."""
        with np.errstate(divide="ignore", invalid="ignore"):
            capacity = np.where(self.affinity > 0.0, self.henry / np.where(self.affinity > 0, self.affinity, 1.0), 0.0)
        return np.where(capacity > 0.0, capacity, 1.0)


@dataclass
class MultisiteLangmuir:
    """
    Multisite Langmuir isotherm of Nitta, Shigetomi, Kuro-Oka and Katayama
    (J. Chem. Eng. Japan 17 (1984) 39), for mixtures,

        q_i* / qmax_i = a_i K_i p_i (1 - sum_k q_k* / qmax_k)^a_i,
        K_i = K_inf,i exp(-dH_i / RT),

    where a molecule of component i occupies a_i sites. The model is
    thermodynamically consistent when a_i qmax_i is the same for every
    component (the number of sites); this is checked. The loadings are
    implicit: the fraction of occupied sites theta = sum_k q_k*/qmax_k is the
    root of f(theta) = theta - sum_k a_k K_k p_k (1 - theta)^a_k, which is
    increasing and concave on [0, 1) when every a_k >= 1, so Newton's method
    started at theta = 0 converges monotonically (a bisection safeguard
    covers a_k < 1). Used by the Rodrigues group (LSRE, Porto) for H2
    purification on activated carbon and zeolites, for example Lopes et al.
    (Adsorption 15 (2009) 125) and Ribeiro et al. (Chem. Eng. Sci. 63 (2008)
    5258).

    Attributes:
        qmax: Saturation capacity of each component, mol/kg.
        a: Number of sites a molecule occupies, -.
        K_inf: Pre-exponential factor, 1/Pa.
        dH: Enthalpy of adsorption in the exponent, J/mol (negative).
        tolerance: Absolute tolerance on theta.
    """

    qmax: Sequence[float]
    a: Sequence[float]
    K_inf: Sequence[float]
    dH: Sequence[float]
    tolerance: float = 1.0e-13

    def __post_init__(self):
        for name in ("qmax", "a", "K_inf", "dH"):
            setattr(self, name, np.asarray(getattr(self, name), dtype=float))
        same_length("MultisiteLangmuir", qmax=self.qmax, a=self.a, K_inf=self.K_inf, dH=self.dH)
        positive("MultisiteLangmuir", "qmax (mol/kg)", self.qmax)
        positive("MultisiteLangmuir", "a", self.a)
        positive("MultisiteLangmuir", "K_inf", self.K_inf, allow_zero=True)
        sites = self.a * self.qmax
        if np.ptp(sites) > 1e-3 * sites.max():
            raise ValueError(f"MultisiteLangmuir: a_i qmax_i must be the same for every component "
                             f"(thermodynamic consistency), got {sites}.")

    def coverage(self, c, temperature):
        """Fraction of occupied sites theta and the products a_i K_i p_i, for concentrations c (mol/m3)."""
        T = np.asarray(temperature, dtype=float)
        c = np.maximum(np.asarray(c, dtype=float).reshape(len(self.qmax), -1), 0.0)
        akp = self.a[:, None] * self.K_inf[:, None] * np.exp(-self.dH[:, None] / (GAS_CONSTANT * T)) \
            * c * GAS_CONSTANT * T
        if _multisite_theta is not None:
            return _multisite_theta(np.ascontiguousarray(akp), self.a, self.tolerance), akp
        a = self.a[:, None]
        theta = np.zeros(c.shape[1])
        lo, hi = np.zeros_like(theta), np.ones_like(theta)
        for _ in range(200):
            free = 1.0 - theta
            f = theta - np.sum(akp * free ** a, axis=0)
            lo, hi = np.where(f < 0.0, theta, lo), np.where(f >= 0.0, theta, hi)
            df = 1.0 + np.sum(a * akp * free ** (a - 1.0), axis=0)
            step = theta - f / df
            bad = (step <= lo) | (step >= hi)
            new = np.where(bad, 0.5 * (lo + hi), step)
            if np.max(np.abs(new - theta)) < self.tolerance:
                theta = new
                break
            theta = new
        return theta, akp

    def slopes(self, c, temperature):
        """Henry slopes q_i*/c_i in m3/kg, with the shape of ``c``."""
        T = np.asarray(temperature, dtype=float)
        c = np.asarray(c, dtype=float).reshape(len(self.qmax), -1)
        theta, _ = self.coverage(c, T)
        K = self.K_inf[:, None] * np.exp(-self.dH[:, None] / (GAS_CONSTANT * T))
        return self.qmax[:, None] * self.a[:, None] * K * GAS_CONSTANT * T * (1.0 - theta) ** self.a[:, None]

    def loadings(self, c, temperature):
        """Equilibrium loadings q_i* in mol/kg, with the shape of ``c``."""
        c = np.asarray(c, dtype=float).reshape(len(self.qmax), -1)
        return self.slopes(c, temperature) * np.maximum(c, 0.0)

    @property
    def total_saturation(self):
        """Saturation capacity of each component, mol/kg."""
        return self.qmax


"""Compiled root finder for MultisiteLangmuir (the numpy loop above is the fallback)"""
try:
    from numba import njit

    @njit(cache=True)
    def _multisite_theta(akp, a, tolerance):
        n, m = akp.shape
        theta = np.zeros(m)
        for j in range(m):
            t, lo, hi = 0.0, 0.0, 1.0
            for _ in range(200):
                free = 1.0 - t
                f, df = t, 1.0
                for i in range(n):
                    term = akp[i, j] * free ** a[i]
                    f -= term
                    if free > 0.0:
                        df += a[i] * term / free
                if f < 0.0:
                    lo = t
                else:
                    hi = t
                new = t - f / df
                if new <= lo or new >= hi:
                    new = 0.5 * (lo + hi)
                if abs(new - t) < tolerance:
                    t = new
                    break
                t = new
            theta[j] = t
        return theta
except ImportError:  # pragma: no cover
    _multisite_theta = None

@dataclass
class Virial:
    """
    Multicomponent virial isotherm in the form used by the Rodrigues group
    (LSRE, Porto) for activated carbon, Lopes et al. (Ind. Eng. Chem. Res. 48
    (2009) 3978; Chem. Eng. Sci. 66 (2011) 303, Table 4; Fuel 93 (2012) 510,
    Table 3), after Taqvi and LeVan (Ind. Eng. Chem. Res. 36 (1997) 2197):

        p_i = q_i / K_i exp( (2/S) sum_j A_ij q_j + (3/(2 S^2)) sum_j sum_k B_ijk q_j q_k ),
        K_i = K_inf,i exp(-dH_i / RT),   A_i = A0_i + A1_i / T,   B_i = B0_i + B1_i / T,
        A_ij = (A_i + A_j) / 2,   B_ijk = (B_i + B_j + B_k) / 3,

    with S the specific area of the adsorbent. For one component it reduces
    to the pure component virial equation that the coefficients are fitted
    to. The loadings are implicit; they are found by Newton's method on
    ln q_i (components with zero partial pressure have zero loading), and the
    Henry slope follows exactly as q_i*/c_i = K_i R T exp(-E_i), E_i being the
    exponent above, so it stays finite when c_i = 0.

    Attributes:
        K_inf: Adsorption constant at infinite temperature, mol/(kg Pa)
            (papers give mol/(kg bar): multiply by 1e-5).
        dH: Enthalpy of adsorption in the exponent, J/mol (negative).
        A0, A1: Virial coefficients A = A0 + A1/T, m2/mol and m2 K/mol.
        B0, B1: Virial coefficients B = B0 + B1/T, m4/mol2 and m4 K/mol2.
        specific_area: S, m2/kg (papers usually give m2/g: multiply by 1e3).
        tolerance: Absolute tolerance on ln q.
    """

    K_inf: Sequence[float]
    dH: Sequence[float]
    A0: Sequence[float]
    A1: Sequence[float]
    B0: Sequence[float]
    B1: Sequence[float]
    specific_area: float
    tolerance: float = 1.0e-12

    def __post_init__(self):
        for name in ("K_inf", "dH", "A0", "A1", "B0", "B1"):
            setattr(self, name, np.asarray(getattr(self, name), dtype=float))
        same_length("Virial", K_inf=self.K_inf, dH=self.dH, A0=self.A0, A1=self.A1, B0=self.B0, B1=self.B1)
        positive("Virial", "K_inf (mol/(kg Pa))", self.K_inf, allow_zero=True)
        positive("Virial", "specific_area (m2/kg)", self.specific_area)

    def _terms(self, T):
        T = np.asarray(T, dtype=float).reshape(-1)
        K = self.K_inf[:, None] * np.exp(-self.dH[:, None] / (GAS_CONSTANT * T))
        A = self.A0[:, None] + self.A1[:, None] / T
        B = self.B0[:, None] + self.B1[:, None] / T
        return K, A, B

    def solve(self, c, temperature):
        """Loadings q_i* (mol/kg) and exponents E_i (-), for concentrations c (mol/m3), shapes (n, N)."""
        c = np.maximum(np.asarray(c, dtype=float).reshape(len(self.K_inf), -1), 0.0)
        T = np.broadcast_to(np.asarray(temperature, dtype=float).reshape(-1), (c.shape[1],))
        K, A, B = self._terms(T)
        Kp = np.ascontiguousarray(K * c * GAS_CONSTANT * T)
        A, B = np.ascontiguousarray(np.broadcast_to(A, c.shape)), np.ascontiguousarray(np.broadcast_to(B, c.shape))
        S = float(self.specific_area)
        if _virial_solve is not None:
            return _virial_solve(Kp, A, B, S, self.tolerance)
        return _virial_solve_numpy(Kp, A, B, S, self.tolerance)

    def slopes(self, c, temperature):
        """Henry slopes q_i*/c_i in m3/kg, with the shape of ``c``."""
        c = np.asarray(c, dtype=float).reshape(len(self.K_inf), -1)
        T = np.broadcast_to(np.asarray(temperature, dtype=float).reshape(-1), (c.shape[1],))
        _, E = self.solve(c, T)
        K, _, _ = self._terms(T)
        return K * GAS_CONSTANT * T * np.exp(-E)

    def loadings(self, c, temperature):
        """Equilibrium loadings q_i* in mol/kg, with the shape of ``c``."""
        return self.solve(c, temperature)[0]

    @property
    def total_saturation(self):
        """Loading scale of each component, mol/kg: the pure component loading at 10 bar and 298.15 K.

        The virial isotherm has no saturation capacity; this value only sets
        the scale of the dimensionless loadings in ``ColumnModel``.
        """
        T = 298.15
        c = 10.0e5 / (GAS_CONSTANT * T)
        out = np.ones(len(self.K_inf))
        for i in range(len(out)):
            ci = np.zeros((len(out), 1))
            ci[i, 0] = c
            q = float(self.loadings(ci, T)[i, 0])
            out[i] = q if q > 0.0 else 1.0
        return out


def _virial_exponents(q, A, B, S):
    """E_i = (2/S) sum_j A_ij q_j + 3/(2 S^2) sum_jk B_ijk q_j q_k for one point, and dE_i/dq_j."""
    n = q.shape[0]
    sq = q.sum()
    Aq = (A * q).sum()
    Bq = (B * q).sum()
    E = np.empty(n)
    J = np.empty((n, n))
    for i in range(n):
        """Sum_j A_ij q_j = (A_i sq + Aq) / 2;  sum_jk B_ijk q_j q_k = (B_i sq^2 + 2 Bq sq) / 3."""
        E[i] = (A[i] * sq + Aq) / S + 0.5 / (S * S) * (B[i] * sq * sq + 2.0 * Bq * sq)
        for j in range(n):
            J[i, j] = (A[i] + A[j]) / S + 1.0 / (S * S) * (B[i] * sq + Bq + B[j] * sq)
    return E, J


def _virial_point(kp, A, B, S, tolerance):
    """Newton's method on u_i = ln q_i: u_i - ln(K_i p_i) + E_i(q) = 0, components with p_i = 0 held at q_i = 0."""
    n = kp.shape[0]
    active = kp > 0.0
    """
    Start below the physical root: Henry loadings shrunk Langmuir-fashion with a 1 mol/kg scale. Starting at the Henry
    loadings can land beyond the fold of p(q) that negative B coefficients produce, where Newton's method finds a
    second, unphysical root at a much larger loading.
    """
    total = 0.0
    for i in range(n):
        if active[i]:
            total += kp[i]
    q = np.zeros(n)
    for i in range(n):
        if active[i]:
            q[i] = kp[i] / (1.0 + total)
    E, J = _virial_exponents(q, A, B, S)
    for _ in range(100):
        """Residual in ln q, Jacobian d/du_j = delta_ij + J_ij q_j."""
        m = 0
        for i in range(n):
            if active[i]:
                m += 1
        idx = np.empty(m, dtype=np.int64)
        k = 0
        for i in range(n):
            if active[i]:
                idx[k] = i
                k += 1
        if m == 0:
            break
        r = np.empty(m)
        M = np.empty((m, m))
        for a in range(m):
            i = idx[a]
            r[a] = np.log(q[i]) - np.log(kp[i]) + E[i]
            for b in range(m):
                j = idx[b]
                M[a, b] = J[i, j] * q[j] + (1.0 if a == b else 0.0)
        du = np.linalg.solve(M, -r)
        step = np.abs(du).max()
        """Damp large steps; ln q changes by at most 0.5 per iteration."""
        if step > 0.5:
            du = du * (0.5 / step)
        for a in range(m):
            q[idx[a]] = q[idx[a]] * np.exp(du[a])
        E, J = _virial_exponents(q, A, B, S)
        if step < tolerance:
            break
    return q, E


def _virial_solve_numpy(Kp, A, B, S, tolerance):
    n, N = Kp.shape
    q, E = np.zeros((n, N)), np.zeros((n, N))
    for j in range(N):
        q[:, j], E[:, j] = _virial_point(Kp[:, j], A[:, j], B[:, j], S, tolerance)
    return q, E


"""Compiled versions of the virial root finder."""
try:
    from numba import njit as _njit

    _virial_exponents = _njit(cache=True)(_virial_exponents)
    _virial_point = _njit(cache=True)(_virial_point)

    @_njit(cache=True)
    def _virial_solve(Kp, A, B, S, tolerance):
        n, N = Kp.shape
        q, E = np.zeros((n, N)), np.zeros((n, N))
        for j in range(N):
            qj, Ej = _virial_point(Kp[:, j].copy(), A[:, j].copy(), B[:, j].copy(), S, tolerance)
            q[:, j] = qj
            E[:, j] = Ej
        return q, E
except ImportError:  # pragma: no cover
    _virial_solve = None


_GL_POINTS, _GL_WEIGHTS = np.polynomial.legendre.leggauss(32)
_GL_S = 0.5 * (_GL_POINTS + 1.0)
_GL_W = 0.5 * _GL_WEIGHTS


def _pure(isotherm, c0, T):
    """Loading q (mol/kg) and reduced spreading pressure pi = int_0^c0 q/c dc (mol/kg) of a pure component."""
    c0 = np.maximum(np.asarray(c0, dtype=float), 0.0)
    q = isotherm.loadings(c0[None, :], T)[0]
    if isinstance(isotherm, DualSiteLangmuir):
        b = isotherm.b0[0] * np.exp(-isotherm.dUb[0] / (GAS_CONSTANT * T))
        d = isotherm.d0[0] * np.exp(-isotherm.dUd[0] / (GAS_CONSTANT * T))
        return q, isotherm.qsb[0] * np.log1p(b * c0) + isotherm.qsd[0] * np.log1p(d * c0)
    if isinstance(isotherm, ExtendedLangmuir):
        A, B = (f[0] for f in isotherm._factors(T))
        with np.errstate(divide="ignore", invalid="ignore"):
            pi = np.where(B > 0.0, A / np.where(B > 0.0, B, 1.0) * np.log1p(B * c0), A * c0)
        return q, pi
    if isinstance(isotherm, Sips):
        b = isotherm.b0[0] * np.exp(-isotherm.dU[0] / (GAS_CONSTANT * T))
        if isotherm.pressure_based:
            b = b * GAS_CONSTANT * T / 1.0e5
        n, qs = (a[0] for a in isotherm._terms(T))
        return q, qs / n * np.log1p((b * c0) ** n)
    """
    Any other isotherm: pi = int_0^1 q(c0 s) / s ds by Gauss-Legendre quadrature (the integrand is finite at s = 0).
    """
    s = _GL_S[:, None] * c0[None, :]
    Ts = np.broadcast_to(np.asarray(T, dtype=float), c0.shape)
    qs_ = isotherm.loadings(s.reshape(1, -1), np.broadcast_to(Ts, s.shape).reshape(-1))[0].reshape(s.shape)
    with np.errstate(divide="ignore", invalid="ignore"):
        integrand = np.where(s > 0.0, qs_ * c0[None, :] / np.where(s > 0.0, s, 1.0), 0.0)
    return q, np.sum(_GL_W[:, None] * integrand, axis=0)


class IAST:
    """
    Ideal adsorbed solution theory: mixture equilibrium from pure-component isotherms.

    For a gas of concentrations c_i, IAST finds the concentrations c_i0 of
    the pure components that have the same spreading pressure and whose
    adsorbed mole fractions x_i = c_i / c_i0 add up to one; then
    1 / q_total = sum_i x_i / q_i0(c_i0) and q_i = x_i q_total. It needs only
    the pure-component isotherms and is the usual prediction of mixture
    adsorption when mixture data are lacking.

    The equations are solved in every cell by a damped Newton method on
    ln c_i0. Spreading pressures use closed forms for single-component
    DualSiteLangmuir, ExtendedLangmuir and Sips isotherms and Gauss-Legendre
    quadrature for any other (for example Toth).

    Args:
        pure: One single-component isotherm per gas component, in the order of
            the gas (for example ``Toth(qs=(5.1,), b0=(4.3e-4,), ...)``).
        tolerance: Convergence tolerance of the Newton iterations.
        max_iterations: Iteration limit.
    """

    def __init__(self, pure, tolerance=1.0e-10, max_iterations=50):
        self.pure = list(pure)
        for iso in self.pure:
            if len(np.atleast_1d(iso.total_saturation)) != 1:
                raise ValueError("IAST needs single-component isotherms, one per gas component.")
        self.tolerance, self.max_iterations = tolerance, max_iterations
        """Last solution, to start the next one from (consecutive calls change little)"""
        self._last = None

    @property
    def total_saturation(self):
        return np.array([float(np.atleast_1d(iso.total_saturation)[0]) for iso in self.pure])

    def _solve(self, c, temperature):
        n = len(self.pure)
        c = np.maximum(np.asarray(c, dtype=float).reshape(n, -1), 0.0)
        M = c.shape[1]
        T = np.broadcast_to(np.asarray(temperature, dtype=float), (M,)).astype(float)
        total = c.sum(axis=0)
        active = total > 1.0e-12
        """Initial guess: c_i0 = c_total."""
        u = np.log(np.where(active, total, 1.0))[None, :].repeat(n, axis=0)
        if self._last is not None and self._last.shape == u.shape and np.all(np.isfinite(self._last)):
            u = self._last.copy()
            """u: warm start."""
        q0 = np.zeros((n, M))
        for _ in range(self.max_iterations):
            c0 = np.exp(u)
            pis = np.empty((n, M))
            for i, iso in enumerate(self.pure):
                q0[i], pis[i] = _pure(iso, c0[i], T)
            residual = np.empty((n, M))
            residual[0] = np.sum(c / c0, axis=0) - 1.0
            residual[1:] = pis[1:] - pis[0]
            jacobian = np.zeros((M, n, n))
            jacobian[:, 0, :] = (-c / c0).T
            for k in range(1, n):
                jacobian[:, k, k] = q0[k]
                jacobian[:, k, 0] = -q0[0]
            jacobian[:, range(n), range(n)] += 1.0e-300
            step = np.linalg.solve(jacobian, -residual.T[..., None])[..., 0].T
            u += np.clip(step, -2.0, 2.0)
            if np.max(np.abs(step[:, active])) < self.tolerance if active.any() else True:
                break
        self._last = u.copy()
        c0 = np.exp(u)
        for i, iso in enumerate(self.pure):
            q0[i], _ = _pure(iso, c0[i], T)
        x = c / c0
        q_total = 1.0 / np.sum(np.where(x > 0.0, x / np.maximum(q0, 1e-300), 0.0), axis=0)
        slopes = np.empty((n, M))
        henry = np.array([iso.slopes(np.full((1, M), 1.0e-9), T)[0] for iso in self.pure])
        slopes[:, active] = (q_total[active] / c0[:, active])
        slopes[:, ~active] = henry[:, ~active]
        return slopes

    def slopes(self, c, temperature):
        """Henry slopes q_i*/c_i in m3/kg, with the shape of ``c``."""
        return self._solve(c, temperature)

    def loadings(self, c, temperature):
        """Equilibrium loadings q_i* in mol/kg, with the shape of ``c``."""
        c = np.asarray(c, dtype=float).reshape(len(self.pure), -1)
        return self._solve(c, temperature) * np.maximum(c, 0.0)



class AranovichDonohue:
    """
    Multilayer correction of Aranovich and Donohue for a condensable component,

        q_k* = f_k(c, T) / (1 - p_k / p_sat(T))^d,

    applied to one component of any isotherm f. It describes the steep rise
    of the loading of a vapour such as water towards saturation (multilayer
    adsorption and capillary condensation in the meso- and macropores of a
    pellet); far from saturation it changes nothing. p_k / p_sat is limited
    to 0.98 so that the loading stays finite.

    Args:
        isotherm: The isotherm f (any SwingPy isotherm).
        component: Index of the condensable component.
        d: Exponent of the correction.
        saturation_pressure: Function of temperature (K) giving p_sat in bar;
            water by default.
    """

    def __init__(self, isotherm, component, d, saturation_pressure=None):
        from .humidity import saturation_pressure as water
        self.isotherm, self.component, self.d = isotherm, int(component), float(d)
        self.saturation_pressure = saturation_pressure or water

    @property
    def total_saturation(self):
        return self.isotherm.total_saturation

    def _factor(self, c, T):
        T = np.broadcast_to(np.asarray(T, dtype=float), c.shape[1:])
        partial = np.maximum(c[self.component], 0.0) * GAS_CONSTANT * T / 1.0e5
        ratio = np.minimum(partial / self.saturation_pressure(T), 0.98)
        return (1.0 - ratio) ** (-self.d)

    def slopes(self, c, temperature):
        c = np.asarray(c, dtype=float).reshape(len(np.atleast_1d(self.total_saturation)), -1)
        slopes = np.array(self.isotherm.slopes(c, temperature), dtype=float)
        slopes[self.component] *= self._factor(c, temperature)
        return slopes

    def loadings(self, c, temperature):
        c = np.asarray(c, dtype=float).reshape(len(np.atleast_1d(self.total_saturation)), -1)
        return self.slopes(c, temperature) * np.maximum(c, 0.0)
