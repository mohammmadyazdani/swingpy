"""
Face reconstruction schemes of the finite volume method.

A finite volume needs the value of each transported quantity at the faces
between cells. The schemes below compute it from the cell averages on the
upwind side: first order upwind, two flux-limited TVD schemes and WENO.
"""

from __future__ import annotations

import numpy as np

SCHEMES = ("uds", "van_leer", "superbee", "weno")
"""Names accepted by ``reconstruct``."""


def reconstruct(cells, left_ghost, scheme="weno", delta=1.0e-10):
    """
    Values at the interior faces for flow in the +Z direction.

    Args:
        cells: Cell averages, array of shape (..., N). Leading dimensions
            (for example one row per component) are reconstructed independently.
        left_ghost: Value of a fictitious cell before the first one, from the
            half-cell approximation f_0 = 2 f_1/2 - f_1, shape (...,).
        scheme: "uds", "van_leer", "superbee" or "weno".
        delta: Small number that keeps the smoothness ratios finite.

    Returns:
        Array of shape (..., N - 1) with the values at faces 1+1/2 ... N-1/2.
    """
    extended = np.concatenate((np.asarray(left_ghost)[..., None], cells), axis=-1)
    f_minus, f_center, f_plus = extended[..., :-2], extended[..., 1:-1], extended[..., 2:]

    if scheme == "uds":
        return f_center.copy()
    if scheme in ("van_leer", "superbee"):
        r = (f_center - f_minus + delta) / (f_plus - f_center + delta)
        if scheme == "van_leer":
            limiter = (r + np.abs(r)) / (1.0 + np.abs(r))
        else:
            limiter = np.maximum(0.0, np.maximum(np.minimum(2.0 * r, 1.0), np.minimum(r, 2.0)))
        return f_center + 0.5 * limiter * (f_plus - f_center)
    if scheme == "weno":
        alpha_0 = (2.0 / 3.0) / (f_plus - f_center + delta) ** 4
        alpha_1 = (1.0 / 3.0) / (f_center - f_minus + delta) ** 4
        return (alpha_0 * 0.5 * (f_center + f_plus) + alpha_1 * (1.5 * f_center - 0.5 * f_minus)) / (alpha_0 + alpha_1)
    raise ValueError(f"Unknown scheme '{scheme}'. Choose one of {SCHEMES}.")


def upwind_faces(cells, feed_face, product_face, direction, scheme):
    """
    Interior face values reconstructed from the upwind side.

    For flow towards Z = 0 the column is read backwards, reconstructed with
    the same function and turned around again ("ends interchanged").
    """
    if direction > 0:
        return reconstruct(cells, 2.0 * feed_face - cells[..., 0], scheme)
    mirrored = reconstruct(cells[..., ::-1], 2.0 * product_face - cells[..., -1], scheme)
    return mirrored[..., ::-1]
