"""
Compiled core of the right hand side (optional, requires numba).

``ColumnModel.rhs`` spends almost all its time in many small numpy
operations on arrays of 30 to 100 cells, where the cost is the Python
overhead of each call rather than the arithmetic. This module writes the
same equations as explicit loops and compiles them with numba, which makes
the right hand side several times faster. The boundary conditions are still
evaluated in Python, because they are user-extensible objects and only
handle a few numbers.

The kernel must stay equation for equation identical to
``ColumnModel._rhs_numpy``; ``tests/test_backends.py`` checks this.

Layout of the ``params`` vector (all floats):

    0 dz            1 darcy_group      2 ergun_group     3 peclet
    4 psi           5 T0               6 P0              7 R
    8 qs0           9 time_scale      10 solid heat capacity per volume (rho_s Cps)
   11 adsorbed heat capacity per volume and unit loading (qs0_vol Cpa)
   12 bed voidage  13 conduction group Kz / (v0 eps L)
   14 gas heat capacity per volume (Cpg c0)          15 qs0_vol
   16 wall group 2 h_in / r_in L / v0               17 pi_1   18 pi_2   19 pi_3
   20 ambient temperature (dimensionless)            21 conduction number at the ends
   22 macropore LDF constant 15 eps_p D_p / r_p^2 / rho_s (1/s per kg/m3)
   23 largest LDF rate constant of the macropore model, 1/s   24 scheme (0 uds, 1 van Leer, 2 Superbee, 3 WENO)
   25 isotherm mode (0 dual-site Langmuir in the kernel, 1 slopes given)
   26 kinetics mode (0 macropore LDF in the kernel, 1 rates given, 2 uptake rates dX given)

Properties that vary along the column (a layered bed) come in two arrays:

    cells (7, N): gas holdup voidage, psi, rho_s Cps, qs0_vol Cpa, qs0_vol, eps_ref / eps, bed voidage
    faces (3, N + 1): permeability (darcy) group, Ergun group, 1 / Peclet

and the scalar entries 1 to 4, 10 to 12 and 15 of ``params`` are not used.
"""

import numpy as np
from numba import njit

SCHEME_IDS = {"uds": 0, "van_leer": 1, "superbee": 2, "weno": 3}


@njit(cache=True)
def _reconstruct(cells, ghost, scheme, out):
    """Interior face values for flow in +Z; writes len(cells) - 1 values into ``out``."""
    n = cells.shape[0]
    delta = 1.0e-10
    for j in range(n - 1):
        fc = cells[j]
        fp = cells[j + 1]
        fm = ghost if j == 0 else cells[j - 1]
        if scheme == 0:
            out[j] = fc
        elif scheme == 3:
            a0 = (2.0 / 3.0) / (fp - fc + delta) ** 4
            a1 = (1.0 / 3.0) / (fc - fm + delta) ** 4
            out[j] = (a0 * 0.5 * (fc + fp) + a1 * (1.5 * fc - 0.5 * fm)) / (a0 + a1)
        else:
            r = (fc - fm + delta) / (fp - fc + delta)
            if scheme == 1:
                limiter = (r + abs(r)) / (1.0 + abs(r))
            else:
                limiter = max(0.0, max(min(2.0 * r, 1.0), min(r, 2.0)))
            out[j] = fc + 0.5 * limiter * (fp - fc)


@njit(cache=True)
def _upwind(cells, first_face, last_face, direction, scheme, out):
    """Interior faces from the upwind side; the column is read backwards for direction -1."""
    n = cells.shape[0]
    if direction > 0:
        _reconstruct(cells, 2.0 * first_face - cells[0], scheme, out)
    else:
        mirrored = cells[::-1].copy()
        temp = np.empty(n - 1)
        _reconstruct(mirrored, 2.0 * last_face - cells[n - 1], scheme, temp)
        for j in range(n - 1):
            out[j] = temp[n - 2 - j]


@njit(cache=True)
def rhs_kernel(Y, P, X, T, Tw, feed, feed_y, product, product_y, params, isotherm, masses, heat,
               rates_given, slopes_given, cells, faces):
    """
    Derivatives of the profiles and the fluxes through the two end faces.

    Args:
        Y (nc, N), P (N), X (n, N), T (N), Tw (N): dimensionless profiles.
        feed, product: (velocity, pressure, temperature) at the end faces.
        feed_y, product_y: independent mole fractions at the end faces.
        params: see the module docstring.
        isotherm: (6, n) dual-site Langmuir parameters qsb, qsd, b0, d0, dUb, dUd.
        masses: molar masses (n), for the Ergun term.
        heat: heats of adsorption -dH (n, N).
        rates_given: LDF rates (n, N) in 1/s, used when kinetics mode is 1.
        slopes_given: (n, N) Henry slopes, used when isotherm mode is 1.

    Returns:
        dY, dP, dX, dT, dTw, end component fluxes F (nc, 2), total fluxes
        G (2) and enthalpy fluxes (2), all in the +Z direction.
    """
    nc, N = Y.shape
    n = nc + 1
    dz = params[0]
    T0, P0, R, qs0, ts = params[5], params[6], params[7], params[8], params[9]
    cond, cpg_c0 = params[13], params[14]
    wall4, pi1, pi2, pi3 = params[16], params[17], params[18], params[19]
    Ta, cnum, ldf = params[20], params[21], params[22]
    scheme = int(params[24])
    iso_mode, kin_mode = int(params[25]), int(params[26])

    """Face velocity, pressure, composition and temperature."""
    v = np.empty(N + 1)
    Pf = np.empty(N + 1)
    Tf = np.empty(N + 1)
    yf = np.empty((nc, N + 1))
    v[0], Pf[0], Tf[0] = feed[0], feed[1], feed[2]
    v[N], Pf[N], Tf[N] = product[0], product[1], product[2]
    for j in range(N - 1):
        pm = 0.5 * (P[j] + P[j + 1])
        Pf[j + 1] = pm
        viscous = 1.0 / faces[0, j + 1]
        ergun = faces[1, j + 1]
        inertial = 0.0
        if ergun > 0.0:
            ysum = 0.0
            mass = 0.0
            for i in range(nc):
                ym = min(max(0.5 * (Y[i, j] + Y[i, j + 1]), 0.0), 1.0)
                mass += masses[i] * ym
                ysum += ym
            mass += masses[n - 1] * min(max(1.0 - ysum, 0.0), 1.0)
            inertial = ergun * mass * pm / (0.5 * (T[j] + T[j + 1]))
        g = -(P[j + 1] - P[j]) / dz
        v[j + 1] = 2.0 * g / (viscous + np.sqrt(viscous * viscous + 4.0 * inertial * abs(g)))
    """
    Each interior face is reconstructed from the side the gas comes from, following the sign of the local velocity (the
    flow may reverse inside the column even when the step has a main direction).
    """
    forward = np.empty(N - 1)
    backward = np.empty(N - 1)
    for i in range(nc):
        yf[i, 0] = feed_y[i]
        yf[i, N] = product_y[i]
        if N > 1:
            _upwind(Y[i], feed_y[i], product_y[i], 1, scheme, forward)
            _upwind(Y[i], feed_y[i], product_y[i], -1, scheme, backward)
            for j in range(N - 1):
                yf[i, j + 1] = forward[j] if v[j + 1] >= 0.0 else backward[j]
    if N > 1:
        _upwind(T, feed[2], product[2], 1, scheme, forward)
        _upwind(T, feed[2], product[2], -1, scheme, backward)
        for j in range(N - 1):
            Tf[j + 1] = forward[j] if v[j + 1] >= 0.0 else backward[j]

    """Gradients at the faces (half cells at the ends)."""
    dyz = np.empty((nc, N + 1))
    for i in range(nc):
        dyz[i, 0] = (Y[i, 0] - yf[i, 0]) / (0.5 * dz)
        dyz[i, N] = (yf[i, N] - Y[i, N - 1]) / (0.5 * dz)
        for j in range(N - 1):
            dyz[i, j + 1] = (Y[i, j + 1] - Y[i, j]) / dz
    dTz = np.empty(N + 1)
    dTz[0] = (T[0] - Tf[0]) / (0.5 * dz)
    dTz[N] = (Tf[N] - T[N - 1]) / (0.5 * dz)
    for j in range(N - 1):
        dTz[j + 1] = (T[j + 1] - T[j]) / dz

    dX = np.empty((n, N))
    """dX: uptake."""
    c = np.empty(n)
    yfull = np.empty(n)
    for j in range(N):
        Tk = T0 * min(max(T[j], 0.5), 2.0)
        ctot = P0 * min(max(P[j], 1.0e-4), 1.0e2) / (R * Tk)
        ysum = 0.0
        for i in range(nc):
            yi = min(max(Y[i, j], 0.0), 1.0)
            yfull[i] = yi
            ysum += yi
        yfull[n - 1] = min(max(1.0 - ysum, 0.0), 1.0)
        for i in range(n):
            c[i] = yfull[i] * ctot
        if iso_mode == 0:
            s1 = 1.0
            s2 = 1.0
            for i in range(n):
                s1 += isotherm[2, i] * np.exp(-isotherm[4, i] / (R * Tk)) * c[i]
                s2 += isotherm[3, i] * np.exp(-isotherm[5, i] / (R * Tk)) * c[i]
        for i in range(n):
            if iso_mode == 0:
                b = isotherm[2, i] * np.exp(-isotherm[4, i] / (R * Tk))
                d = isotherm[3, i] * np.exp(-isotherm[5, i] / (R * Tk))
                slope = isotherm[0, i] * b / s1 + isotherm[1, i] * d / s2
            else:
                slope = slopes_given[i, j]
            if kin_mode == 2:
                """Uptake given (particle diffusion)"""
                dX[i, j] = rates_given[i, j]
            else:
                k = min(ldf / slope, params[23]) if kin_mode == 0 else rates_given[i, j]
                dX[i, j] = k * ts * (slope * c[i] / qs0 - X[i, j])

    """Energy groups, total mass and energy balances."""
    G = np.empty(N + 1)
    vP = np.empty(N + 1)
    for j in range(N + 1):
        G[j] = Pf[j] * v[j] / Tf[j]
        vP[j] = v[j] * Pf[j]
    dP = np.empty(N)
    dT = np.empty(N)
    dTw = np.empty(N)
    dY = np.empty((nc, N))
    F = np.empty((nc, N + 1))
    for i in range(nc):
        for j in range(N + 1):
            F[i, j] = yf[i, j] * G[j] - faces[2, j] * (Pf[j] / Tf[j]) * dyz[i, j]
    for j in range(N):
        eps, psi, rho_cps, qcpa, qs0v, scale = cells[0, j], cells[1, j], cells[2, j], cells[3, j], cells[4, j], cells[5, j]
        eps_bed = cells[6, j]
        xsum = 0.0
        dxsum = 0.0
        released = 0.0
        for i in range(n):
            xsum += X[i, j]
            dxsum += dX[i, j]
            released += heat[i, j] * dX[i, j]
        hc = rho_cps + qcpa * xsum
        solid = (1.0 - eps_bed) / eps * hc
        o1 = cond * scale / solid
        o2 = cpg_c0 / solid
        o3 = qcpa / hc
        o4 = wall4 / ((1.0 - eps_bed) * hc)
        heat_term = (qs0v / T0) * released / hc
        dPp = -T[j] * scale * (G[j + 1] - G[j]) / dz - psi * T[j] * dxsum
        dTp = (o1 * (dTz[j + 1] - dTz[j]) / dz - o2 * scale * (vP[j + 1] - vP[j]) / dz
               - o3 * T[j] * dxsum + heat_term - o4 * (T[j] - Tw[j]))
        pot = P[j] / T[j]
        dT[j] = (dTp - o2 * dPp) / (1.0 + o2 * pot)
        dP[j] = dPp + pot * dT[j]
        for i in range(nc):
            dY[i, j] = (-(T[j] / P[j]) * scale * (F[i, j + 1] - F[i, j]) / dz - psi * (T[j] / P[j]) * dX[i, j]
                        - (Y[i, j] / P[j]) * dP[j] + (Y[i, j] / T[j]) * dT[j])
        left = 2.0 * Ta - Tw[0] if j == 0 else Tw[j - 1]
        right = 2.0 * Ta - Tw[N - 1] if j == N - 1 else Tw[j + 1]
        dTw[j] = pi1 * (right - 2.0 * Tw[j] + left) / (dz * dz) + pi2 * (T[j] - Tw[j]) - pi3 * (Tw[j] - Ta)

    F_end = np.empty((nc, 2))
    for i in range(nc):
        F_end[i, 0] = F[i, 0]
        F_end[i, 1] = F[i, N]
    G_end = np.array([G[0], G[N]])
    H_end = np.array([vP[0] - cnum * dTz[0], vP[N] - cnum * dTz[N]])
    return dY, dP, dX, dT, dTw, F_end, G_end, H_end
