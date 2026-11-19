"""Exact solutions used to verify the numerics of SwingPy (not a test module itself)."""

import numpy as np


def crank_sphere_uptake(t, rate):
    """
    Fractional uptake of a sphere at constant surface concentration,
    Crank, The Mathematics of Diffusion, 2nd ed. (1975), Eq. 6.20:
    F = 1 - 6/pi^2 sum_n exp(-n^2 pi^2 D t / r^2) / n^2, with rate = D / r^2.
    """
    n = np.arange(1, 2000)[:, None]
    t = np.atleast_1d(np.asarray(t, dtype=float))
    return 1.0 - 6.0 / np.pi ** 2 * np.sum(np.exp(-rate * n ** 2 * np.pi ** 2 * t[None, :]) / n ** 2, axis=0)


def sphere_transfer(s, rate):
    """Laplace transform of the particle average over the surface value, 3 (lambda coth lambda - 1) / lambda^2."""
    lam = np.sqrt(s / rate)
    return 3.0 * (lam / np.tanh(lam) - 1.0) / lam ** 2


def talbot_inverse(transform, t, terms=32):
    """
    Numerical inverse Laplace transform by the fixed Talbot method,
    Abate and Valko, Int. J. Numer. Meth. Eng. 60 (2004) 979-993.
    """
    r = 2.0 * terms / 5.0
    theta = np.arange(1, terms) * np.pi / terms
    cot = 1.0 / np.tan(theta)
    delta = np.concatenate(([r], r * theta * (cot + 1j)))
    gamma = np.concatenate(([0.5 * np.exp(r)], (1.0 + 1j * theta * (1.0 + cot ** 2) - 1j * cot) * np.exp(delta[1:])))
    out = [r / (terms * time) * np.sum(np.real(gamma * transform(delta / time))) for time in np.atleast_1d(t)]
    return np.array(out)


def linear_column_outlet(t, velocity, dispersion, length, capacity, rate):
    """
    Outlet concentration over the feed concentration after a step at the inlet,
    for an isothermal column at constant velocity with axial dispersion,
    a linear isotherm and diffusion in spherical particles (the problem of
    Rosen; Ruthven, Principles of Adsorption and Adsorption Processes, 1984,
    Ch. 8), from its exact Laplace transform with a Danckwerts inlet and a
    zero gradient outlet.

    Args:
        velocity: Interstitial velocity, m/s.
        dispersion: Axial dispersion coefficient, m2/s.
        length: m.
        capacity: (1 - eps) / eps x particle density x Henry constant, dimensionless.
        rate: D / r^2 of the particles, 1/s (or None for local equilibrium).
    """
    def outlet(s):
        uptake = sphere_transfer(s, rate) if rate is not None else 1.0
        root = np.sqrt(velocity ** 2 + 4.0 * dispersion * s * (1.0 + capacity * uptake))
        m1, m2 = (velocity + root) / (2.0 * dispersion), (velocity - root) / (2.0 * dispersion)
        ratio = (m2 / m1) * np.exp((m2 - m1) * length)
        B = velocity / s / ((velocity - dispersion * m2) - (velocity - dispersion * m1) * ratio)
        return B * np.exp(m2 * length) * (1.0 - m2 / m1)
    return talbot_inverse(outlet, t)
