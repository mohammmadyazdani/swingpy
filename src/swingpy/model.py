"""
Finite volume model of a nonisothermal, nonisobaric adsorption column.

The column is described by a one-dimensional, axially dispersed plug flow
model of an ideal gas mixture with any number of components, linear driving
force uptake, a bed energy balance with heat of adsorption and heat loss to
the wall, a wall energy balance, and Darcy or Ergun pressure drop. The
equations are written in dimensionless, conservative form and solved on N
equal cells.

Scaling. Pressure is divided by a reference pressure P0, temperature by T0,
velocity by v0, length by the column length L, time by L / v0 and loadings
by a reference loading q_s0. With L = 1 m and v0 = 1 m/s the dimensionless
time equals time in seconds.

State vector. For n components the state holds, in this order:

    y          n - 1 independent mole fractions, N values each
    P          pressure, N values
    x          n loadings, N values each
    T          bed temperature, N values
    T_wall     wall temperature, N values
    bookkeeping  running integrals at both ends: n component fluxes,
                 the enthalpy flux and the machine work, per end

The bookkeeping integrals make purity, recovery, energy and balance errors
as accurate as the time integration itself.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
from scipy.integrate import solve_ivp
from scipy.sparse import lil_matrix

from .gas import GAS_CONSTANT
from .schemes import SCHEMES, upwind_faces

ENDS = ("feed", "product")


class Cell(NamedTuple):
    """Dimensionless state of the cell next to a column end, handed to boundary conditions."""

    pressure: float
    mole_fractions: np.ndarray
    temperature: float


@dataclass
class StepResult:
    """
    Outcome of one integrated step.

    Attributes:
        step: The step that was simulated.
        time: Output times from the start of the step, s.
        states: State vectors at those times, shape (state size, len(time)).
        moles_in: For each end ("feed", "product"), the moles of each
            component that entered the column (negative when they left).
        enthalpy_in: For each end, the gas enthalpy that entered, J.
        work: For each end, the compressor or vacuum pump work, J.
    """

    step: object
    time: np.ndarray
    states: np.ndarray
    moles_in: dict
    enthalpy_in: dict
    work: dict

    @property
    def final_state(self) -> np.ndarray:
        return self.states[:, -1]

    @property
    def initial_state(self) -> np.ndarray:
        return self.states[:, 0]


class StepTooStiff(RuntimeError):
    """A step needed more right hand side evaluations than allowed, with the LSODA fallback switched off."""


class _TooManyEvaluations(Exception):
    """Raised inside the solver when a step takes too many evaluations with BDF."""


class ColumnModel:
    """
    Discretized column ready to simulate cycle steps.

    Args:
        column: Geometry and thermal properties (``Column``).
        adsorbent: Adsorbent, isotherm and kinetics (``Adsorbent``).
        gas: Gas components and properties (``GasMixture``).
        n_cells: Number of finite volumes.
        scheme: Face reconstruction, one of "uds", "van_leer", "superbee", "weno".
        reference_pressure: P0 in bar, usually the highest pressure of the cycle.
        reference_temperature: T0 in K, usually the feed temperature.
        reference_velocity: v0 in m/s, usually the feed velocity. It also
            sets the axial dispersion coefficient, D_L = 0.7 D_m + 0.5 v0 d_p.
        reference_loading: q_s0 in mol/kg. Only scales the loadings; by
            default the largest total saturation capacity of the isotherm.
        backend: "auto" (numba when installed, numpy otherwise), "numba" or
            "numpy". Both solve the same equations; numba is several times
            faster and worth installing for cyclic steady state runs.
        axial_dispersion: Axial dispersion coefficient D_L in m2/s, to use
            the value of a particular correlation; by default
            D_L = 0.7 D_m + 0.5 v0 d_p.
    """

    def __init__(self, column, adsorbent, gas, n_cells=30, scheme="weno",
                 reference_pressure=1.0, reference_temperature=298.15, reference_velocity=1.0,
                 reference_loading=None, backend="auto", axial_dispersion=None):
        from .adsorbent import LayeredBed
        if isinstance(adsorbent, LayeredBed):
            adsorbent = adsorbent.cell_view(int(n_cells))
        n_iso = len(np.atleast_1d(getattr(adsorbent.isotherm, "total_saturation", np.zeros(gas.n_components))))
        if n_iso != gas.n_components:
            raise ValueError(f"The isotherm of {adsorbent.name} has parameters for {n_iso} components but the gas "
                             f"has {gas.n_components} ({', '.join(gas.components)}); give one value per gas "
                             f"component, in the same order.")
        if int(n_cells) < 2:
            raise ValueError(f"n_cells must be at least 2, got {n_cells}.")
        for label, value in (("reference_pressure (bar)", reference_pressure),
                             ("reference_temperature (K)", reference_temperature),
                             ("reference_velocity (m/s)", reference_velocity)):
            if not value > 0.0:
                raise ValueError(f"{label} must be positive, got {value!r}.")
        if scheme not in SCHEMES:
            raise ValueError(f"Unknown scheme '{scheme}'. Choose one of {SCHEMES}.")
        self.column, self.adsorbent, self.gas = column, adsorbent, gas
        self.n_cells, self.scheme = n_cells, scheme
        """The options as given, to save the problem as a specification (swingpy.spec.to_spec)."""
        self.options = dict(backend=backend, axial_dispersion=axial_dispersion, reference_loading=reference_loading)
        self.n_components = gas.n_components
        self.n_independent = gas.n_components - 1
        self.cell_width = 1.0 / n_cells

        eps = column.bed_voidage
        self.P0 = reference_pressure * 1.0e5
        self.T0 = reference_temperature
        self.v0 = reference_velocity
        self.time_scale = column.length / reference_velocity
        self.c0 = self.P0 / (GAS_CONSTANT * self.T0)
        if reference_loading is None:
            reference_loading = float(np.max(adsorbent.isotherm.total_saturation))
        self.qs0 = reference_loading
        self.qs0_volumetric = reference_loading * adsorbent.density

        """
        Bed voidage and particle size of every cell and face. They are uniform except in a layered bed. The velocity v
        is the superficial velocity divided by the column voidage eps (eps_ref), so the end faces convert flows as in a
        uniform bed; a cell of voidage eps_j accumulates gas in proportion to eps_j, so its flux divergences are scaled
        by eps_ref/eps_j.
        """
        N = self.n_cells
        eps_cell = np.asarray(getattr(adsorbent, "bed_voidage_cells", None) if
                              getattr(adsorbent, "bed_voidage_cells", None) is not None else np.full(N, eps), dtype=float)
        eps_cell = np.where(np.isnan(eps_cell), eps, eps_cell)
        radius_cell = np.asarray(getattr(adsorbent, "particle_radius_cells", None) if
                                 getattr(adsorbent, "particle_radius_cells", None) is not None
                                 else np.full(N, adsorbent.particle_radius), dtype=float)
        eps_face = np.concatenate(([eps_cell[0]], 0.5 * (eps_cell[1:] + eps_cell[:-1]), [eps_cell[-1]]))
        radius_face = np.concatenate(([radius_cell[0]], 0.5 * (radius_cell[1:] + radius_cell[:-1]), [radius_cell[-1]]))
        """Gas holdup: the bed voidage, plus the particle pores when asked for."""
        eps_bed = eps_cell
        if column.include_pore_gas:
            particle_voidage = np.asarray(getattr(adsorbent, "particle_voidage_cells", None)
                                          if getattr(adsorbent, "particle_voidage_cells", None) is not None
                                          else np.full(N, adsorbent.particle_voidage), dtype=float)
            eps_cell = eps_bed + (1.0 - eps_bed) * particle_voidage
        self._eps_bed_cell = eps_bed
        self._eps_cell, self._scale = eps_cell, eps / eps_cell
        cell_gas_volume = eps_cell * column.cross_section * column.length / N
        self._gas_weight = eps_cell / eps
        self._gas_weight[0] += column.dead_volume_feed / (eps * column.cross_section * column.length / N)
        self._gas_weight[-1] += column.dead_volume_product / (eps * column.cross_section * column.length / N)
        self._dead_volume_factor = ((1.0 + column.dead_volume_feed / cell_gas_volume[0],
                                     1.0 + column.dead_volume_product / cell_gas_volume[-1])
                                    if column.dead_volume_feed > 0.0 or column.dead_volume_product > 0.0 else None)

        """Axial dispersion and the mass and heat Peclet numbers (per face)."""
        dispersion_face = (np.full(N + 1, float(axial_dispersion)) if axial_dispersion else
                           0.7 * gas.molecular_diffusivity + 0.5 * reference_velocity * 2.0 * radius_face)
        self._inv_peclet_face = eps_face / eps * dispersion_face / (reference_velocity * column.length)
        dispersion = axial_dispersion or (0.7 * gas.molecular_diffusivity
                                          + 0.5 * reference_velocity * 2.0 * adsorbent.particle_radius)
        self.peclet = reference_velocity * column.length / dispersion
        """Without axial conduction the heat Peclet number is infinite: an inlet face takes the inlet temperature."""
        self.heat_peclet = (eps * reference_velocity * column.length * self.c0 * gas.heat_capacity
                            / column.axial_thermal_conductivity if column.axial_thermal_conductivity > 0.0 else np.inf)

        """Coupling of the solid uptake to the gas balances."""
        self.psi = GAS_CONSTANT * self.T0 * self.qs0_volumetric / self.P0 * (1.0 - eps_bed) / eps_cell
        self._psi_inventory = GAS_CONSTANT * self.T0 * self.qs0_volumetric / self.P0 * (1.0 - eps_bed) / eps

        """
        Pressure drop: v = darcy_group * (-dP/dZ) for Darcy's law; the Ergun equation adds an inertial term proportional
        to the gas density.
        """
        self.darcy_group = ((4.0 / 150.0) * (eps / (1.0 - eps)) ** 2 * adsorbent.particle_radius ** 2
                            * self.P0 / (gas.viscosity * reference_velocity * column.length))
        self.ergun_group = 0.0
        if column.pressure_drop == "ergun":
            self.ergun_group = ((1.75 / 2.0) / adsorbent.particle_radius * (1.0 - eps) / eps
                                * column.length * reference_velocity ** 2 / (GAS_CONSTANT * self.T0))
        """
        Per face, for v = superficial velocity / eps_ref: the permeability scales as eps_f^3 / (1 - eps_f)^2 and the
        inertial term as (1 - eps_f) / eps_f^3, with the particle size of the face.
        """
        self._darcy_face = ((4.0 / 150.0) * eps_face ** 3 / ((1.0 - eps_face) ** 2 * eps) * radius_face ** 2
                            * self.P0 / (gas.viscosity * reference_velocity * column.length))
        self._ergun_face = (((1.75 / 2.0) / radius_face * (1.0 - eps_face) * eps ** 2 / eps_face ** 3
                             * column.length * reference_velocity ** 2 / (GAS_CONSTANT * self.T0))
                            if column.pressure_drop == "ergun" else np.zeros(N + 1))

        wall_capacity = column.wall_density * column.wall_heat_capacity
        """wall_capacity: wall energy balance."""
        wall_ring = column.outer_radius ** 2 - column.inner_radius ** 2
        self.pi_1 = column.wall_thermal_conductivity / (wall_capacity * reference_velocity * column.length)
        self.pi_2 = 2.0 * column.inner_radius * column.inner_heat_transfer / wall_ring * self.time_scale / wall_capacity
        self.pi_3 = 2.0 * column.outer_radius * column.outer_heat_transfer / wall_ring * self.time_scale / wall_capacity

        """Ratio of axial conduction to gas enthalpy convection at the ends."""
        self.conduction_number = (column.axial_thermal_conductivity * GAS_CONSTANT * self.T0
                                  / (eps * gas.heat_capacity * reference_velocity * self.P0 * column.length))
        self._sparsity = self._build_sparsity()
        self.backend = self._choose_backend(backend)

    def _choose_backend(self, backend):
        """Pick the right hand side implementation and prepare the compiled one if used."""
        if backend not in ("auto", "numba", "numpy"):
            raise ValueError("backend must be 'auto', 'numba' or 'numpy'.")
        if backend == "numpy":
            return "numpy"
        try:
            from . import kernels
        except ImportError:
            if backend == "numba":
                raise ImportError("backend='numba' needs numba: pip install numba")
            return "numpy"
        from .adsorbent import ConstantLDF, MacroporeLDF
        from .isotherms import DualSiteLangmuir
        kinetics, isotherm = self.adsorbent.kinetics, self.adsorbent.isotherm
        a, c, n, N = self.adsorbent, self.column, self.n_components, self.n_cells
        layered = hasattr(a, "layered_bed")
        self._kernel = kernels.rhs_kernel
        """
        Dual-site Langmuir uptake with macropore or constant rates runs entirely in the kernel; any other isotherm or
        kinetics (and layered beds) gives its slopes and rates from Python, and the kernel does the rest.
        """
        self._kinetics_in_kernel = isinstance(kinetics, MacroporeLDF) and not layered
        self._isotherm_in_kernel = isinstance(isotherm, DualSiteLangmuir) and (
            self._kinetics_in_kernel or isinstance(kinetics, ConstantLDF))
        self._kernel_isotherm = (np.vstack([isotherm.qsb, isotherm.qsd, isotherm.b0, isotherm.d0,
                                            isotherm.dUb, isotherm.dUd]) if self._isotherm_in_kernel
                                 else np.zeros((6, n)))
        self._constant_rates = (np.repeat(np.asarray(kinetics.rates, dtype=float)[:, None], N, axis=1)
                                if isinstance(kinetics, ConstantLDF) else None)
        ldf = (15.0 * a.particle_voidage * self.gas.molecular_diffusivity / a.tortuosity / a.particle_radius ** 2
               / float(np.mean(a.density)))
        self._kernel_params = np.array([
            self.cell_width, 0.0, 0.0, 0.0, 0.0,
            self.T0, self.P0, GAS_CONSTANT, self.qs0, self.time_scale,
            0.0, 0.0, 0.0,
            c.axial_thermal_conductivity / (self.v0 * c.bed_voidage * c.length), self.gas.heat_capacity * self.c0,
            0.0, 2.0 * c.inner_heat_transfer / c.inner_radius * self.time_scale,
            self.pi_1, self.pi_2, self.pi_3, 1.0, self.conduction_number, ldf,
            float(getattr(kinetics, "max_rate", 1.0e3)), float(kernels.SCHEME_IDS[self.scheme]), 0.0 if self._isotherm_in_kernel else 1.0,
            2.0 if self.n_shells else (0.0 if self._kinetics_in_kernel else 1.0),
        ])
        ones = np.ones(N)
        self._kernel_cells = np.ascontiguousarray(np.vstack([
            self._eps_cell, self.psi * ones, a.density * a.heat_capacity * ones,
            self.qs0_volumetric * a.adsorbed_phase_heat_capacity * ones, self.qs0_volumetric * ones, self._scale,
            self._eps_bed_cell]))
        self._kernel_faces = np.ascontiguousarray(np.vstack([self._darcy_face, self._ergun_face,
                                                             self._inv_peclet_face]))
        self._kernel_heat = np.ascontiguousarray(
            np.asarray(a.heat_of_adsorption, dtype=float).reshape(n, -1) * np.ones((n, N)))
        self._no_slopes = np.zeros((n, N))
        return "numba"

    """Unit conversions used by the boundary conditions."""

    def to_pressure(self, bar):
        return bar * 1.0e5 / self.P0

    def to_velocity(self, metres_per_second):
        return metres_per_second / self.v0

    def to_temperature(self, kelvin):
        return kelvin / self.T0

    def to_mole_fractions(self, composition):
        return self.gas.composition(composition)[: self.n_independent]

    def full_composition(self, independent):
        """All n mole fractions from the n - 1 independent ones."""
        independent = np.asarray(independent)
        return np.concatenate((independent, [1.0 - np.sum(independent, axis=0)]), axis=0) \
            if independent.ndim == 1 else np.vstack((independent, 1.0 - np.sum(independent, axis=0)))

    """State vector layout."""

    @property
    def n_shells(self) -> int:
        """Shells per particle when particle diffusion is resolved, zero otherwise."""
        from .adsorbent import ParticleDiffusion
        kinetics = self.adsorbent.kinetics
        return kinetics.shells if isinstance(kinetics, ParticleDiffusion) else 0

    @property
    def state_size(self) -> int:
        n, N = self.n_components, self.n_cells
        return (n - 1) * N + N + n * N + 2 * N + n * self.n_shells * N + 2 * (n + 2)

    def unpack(self, state):
        """Split a state vector into its named parts (views, not copies)."""
        n, nc, N = self.n_components, self.n_independent, self.n_cells
        i = 0
        parts = {}
        for name, rows in (("y", nc), ("P", 1), ("x", n), ("T", 1), ("T_wall", 1)):
            block = state[i:i + rows * N]
            parts[name] = block.reshape(rows, N) if name in ("y", "x") else block
            i += rows * N
        if self.n_shells:
            size = n * self.n_shells * N
            parts["shells"] = state[i:i + size].reshape(n, self.n_shells, N)
            i += size
        book = state[i:]
        parts["bookkeeping"] = {"feed": book[: n + 2], "product": book[n + 2:]}
        return parts

    def initial_state(self, pressure, composition, temperature=None):
        """
        Column filled with a uniform gas and the solid in equilibrium with it.

        Args:
            pressure: bar.
            composition: Mole fractions (dict or sequence).
            temperature: K, the reference temperature by default.
        """
        temperature = self.T0 if temperature is None else temperature
        y = self.gas.composition(composition)
        c = y[:, None] * pressure * 1.0e5 / (GAS_CONSTANT * temperature)
        N = self.n_cells
        """Per cell, so that a layered bed starts at equilibrium in every layer."""
        loadings = self.adsorbent.isotherm.loadings(np.repeat(np.asarray(c, dtype=float).reshape(-1, 1), N, axis=1),
                                                    temperature)
        state = np.concatenate([
            np.repeat(y[:-1], N), np.full(N, self.to_pressure(pressure)),
            (loadings / self.qs0).ravel(), np.full(N, self.to_temperature(temperature)),
            np.full(N, self.to_temperature(self.column.ambient_temperature)),
            np.repeat((loadings / self.qs0)[:, None, :], self.n_shells, axis=1).ravel(),
            np.zeros(2 * (self.n_components + 2)),
        ])
        return state

    """Pressure drop and boundary helpers."""

    def _end_index(self, end):
        return 0 if end == "feed" else -1

    def _inertial(self, y, P, T, group=None):
        """Coefficient of v|v| in the dimensionless pressure gradient (zero for Darcy's law)."""
        group = self.ergun_group if group is None else group
        if np.all(np.asarray(group) == 0.0):
            return 0.0
        """
        Mole fractions are clipped to [0, 1] so that the gas density stays positive for the unphysical trial states of a
        diverging Newton iteration; the solver then rejects them instead of failing.
        """
        y_full = np.clip(self.full_composition(np.clip(y, 0.0, 1.0)), 0.0, 1.0)
        molar_mass = np.tensordot(self.gas.molar_masses, y_full, axes=1)
        return group * molar_mass * P / T

    def velocity_from_gradient(self, gradient, inertial, darcy=None):
        """
        Velocity from g = -dP/dZ. Darcy: v = darcy_group g. Ergun:
        g = v / darcy_group + inertial v|v|, solved in a form that stays
        accurate when the inertial term vanishes.
        """
        viscous = 1.0 / (self.darcy_group if darcy is None else darcy)
        return 2.0 * gradient / (viscous + np.sqrt(viscous ** 2 + 4.0 * inertial * np.abs(gradient)))

    def _half_cell_drop(self, velocity, inertial, end=None):
        darcy = self.darcy_group if end is None else self._darcy_face[self._end_index(end)]
        return (velocity / darcy + inertial * velocity * abs(velocity)) * self.cell_width / 2.0

    def end_inertial(self, cell: Cell, end):
        """Inertial coefficient at one end face (zero for Darcy's law)."""
        return self._inertial(cell.mole_fractions, cell.pressure, cell.temperature,
                              group=self._ergun_face[self._end_index(end)])

    def pressure_for_velocity(self, velocity, cell: Cell, end):
        """Face pressure that drives a given velocity through half a cell."""
        drop = self._half_cell_drop(velocity, self.end_inertial(cell, end), end)
        face = cell.pressure + drop if end == "feed" else cell.pressure - drop
        return max(face, 1.0e-6)

    def velocity_for_pressure(self, face_pressure, cell: Cell, end):
        """Velocity through half a cell for a given face pressure."""
        difference = face_pressure - cell.pressure if end == "feed" else cell.pressure - face_pressure
        inertial = self.end_inertial(cell, end)
        return float(self.velocity_from_gradient(difference / (self.cell_width / 2.0), inertial,
                                                 self._darcy_face[self._end_index(end)]))

    def danckwerts(self, cell: Cell, y_in, t_in, inflow_speed, end=None):
        """
        Face composition and temperature of an inlet (Danckwerts conditions):
        f = (f_cell + f_in a) / (1 + a) with a = v Pe dZ / 2. With no inflow
        this reduces to the zero gradient condition.
        """
        peclet = self.peclet if end is None else 1.0 / self._inv_peclet_face[self._end_index(end)]
        a = inflow_speed * peclet * self.cell_width / 2.0
        y = (cell.mole_fractions + y_in * a) / (1.0 + a)
        if np.isinf(self.heat_peclet):
            t = t_in if inflow_speed > 0.0 else cell.temperature
        else:
            a_heat = inflow_speed * self.heat_peclet * self.cell_width / 2.0
            t = (cell.temperature + t_in * a_heat) / (1.0 + a_heat)
        return y, t

    def molar_inflow_face(self, molar_flow, y_full, temperature_k, cell: Cell, end):
        """
        Face state of an inlet with an imposed molar flow (mol/s), used by
        recorded and tank streams. The velocity is v = G T / P with G the
        dimensionless molar flux, and the face pressure must drive that
        velocity through half a cell. For Darcy's law the two conditions give
        a quadratic in the face pressure, solved exactly; the Ergun term is
        added by a few fixed-point iterations.
        """
        c = self.column
        flux = molar_flow / (c.bed_voidage * c.cross_section * self.v0 * self.c0)
        t_in = self.to_temperature(temperature_k)
        k = flux * t_in * self.cell_width / (2.0 * self._darcy_face[self._end_index(end)])
        pressure = 0.5 * (cell.pressure + np.sqrt(cell.pressure ** 2 + 4.0 * k))
        inertial = self.end_inertial(cell, end)
        if inertial:
            for _ in range(5):
                speed = flux * t_in / pressure
                pressure = cell.pressure + self._half_cell_drop(speed, inertial, end)
        speed = flux * t_in / pressure
        y, t = self.danckwerts(cell, np.asarray(y_full)[: self.n_independent], t_in, speed, end)
        from .boundaries import FaceState
        return FaceState(speed if end == "feed" else -speed, pressure, y, t)

    """The right hand side."""

    def rhs(self, tau, state, step):
        """Time derivative of the state vector during ``step``."""
        out = self._rhs_numba(tau, state, step) if self.backend == "numba" else self._rhs_numpy(tau, state, step)
        if self._dead_volume_factor is not None:
            self._apply_dead_volume(state, out)
        return out

    def _apply_dead_volume(self, state, out):
        """
        Extra gas holdup of the end cells: the gas content (P/T) of a cell with
        dead volume changes (1 + d) times more slowly, where d is the dead
        volume over the gas volume of the cell; the bed temperature equation
        is unchanged and the mole fractions slow by the same factor.
        """
        N, nc = self.n_cells, self.n_independent
        p, d = self.unpack(state), self.unpack(out)
        for cell, factor in ((0, self._dead_volume_factor[0]), (N - 1, self._dead_volume_factor[1])):
            if factor == 1.0:
                continue
            P, T, dT = p["P"][cell], p["T"][cell], d["T"][cell]
            dP_old = out[nc * N + cell]
            """T d(P/T)/dt with the extra holdup."""
            gas_rate = (dP_old - P * dT / T) / factor
            out[nc * N + cell] = gas_rate + P * dT / T
            for i in range(nc):
                out[i * N + cell] /= factor

    def _ambient(self, step):
        return self.to_temperature(self.column.ambient_temperature)

    def _rhs_numba(self, tau, state, step):
        """The right hand side through the compiled kernel (same equations as ``_rhs_numpy``)."""
        p = self.unpack(state)
        Y, P, X, T, T_wall = p["y"], p["P"], p["x"], p["T"], p["T_wall"]
        time_s = tau * self.time_scale
        feed = step.feed_end.face(self, "feed", time_s, Cell(P[0], Y[:, 0], T[0]))
        product = step.product_end.face(self, "product", time_s, Cell(P[-1], Y[:, -1], T[-1]))
        params = self._kernel_params
        params[20] = self._ambient(step)
        slopes, rates = self._no_slopes, self._no_slopes
        if not self._isotherm_in_kernel:
            T_k = self.T0 * np.clip(T, 0.5, 2.0)
            c_total = self.P0 * np.clip(P, 1.0e-4, 1.0e2) / (GAS_CONSTANT * T_k)
            c = np.clip(self.full_composition(np.clip(Y, 0.0, 1.0)), 0.0, 1.0) * c_total
            slopes = np.ascontiguousarray(self.adsorbent.isotherm.slopes(c, T_k))
        shell_rates = np.zeros(0)
        if self.n_shells:
            p = self.unpack(state)
            shell_rates, d_average = self.adsorbent.kinetics.derivatives(
                p["shells"], slopes * c / self.qs0, self.time_scale)
            rates = np.ascontiguousarray(d_average)
        elif not self._kinetics_in_kernel:
            if self._constant_rates is not None:
                rates = self._constant_rates
            else:
                T_k = self.T0 * np.clip(T, 0.5, 2.0)
                rates = np.ascontiguousarray(self.adsorbent.kinetics.rate_constants(
                    slopes, self.adsorbent, self.gas, pressure=self.P0 * np.clip(P, 1.0e-4, 1.0e2) / 1.0e5,
                    temperature=T_k) * np.ones_like(slopes))
        try:
            dY, dP, dX, dT, dT_wall, F_end, G_end, H_end = self._kernel(
                np.ascontiguousarray(Y), P, np.ascontiguousarray(X), T, T_wall,
                np.array([feed.velocity, feed.pressure, feed.temperature]),
                np.asarray(feed.mole_fractions, dtype=float),
                np.array([product.velocity, product.pressure, product.temperature]),
                np.asarray(product.mole_fractions, dtype=float), params, self._kernel_isotherm,
                self.gas.molar_masses, self._kernel_heat, rates, slopes, self._kernel_cells, self._kernel_faces)
        except ZeroDivisionError:
            """
            Only happens for wildly unphysical trial states of the implicit solver (a diverging Newton iteration). Numba
            raises where numpy returns inf or nan, so such a state is evaluated with numpy: the solver then rejects it
            and shrinks its step, exactly as with the numpy backend.
            """
            with np.errstate(all="ignore"):
                return self._rhs_numpy(tau, state, step)
        book = []
        for k, sign, face, boundary in ((0, 1.0, feed, step.feed_end), (1, -1.0, product, step.product_end)):
            component_flux = np.concatenate((F_end[:, k], [G_end[k] - F_end[:, k].sum()]))
            work = self._work_integrand(boundary, sign * face.velocity, face.pressure, face.mole_fractions)
            book.append(np.concatenate((sign * component_flux, [sign * H_end[k], work])))
        return np.concatenate((dY.ravel(), dP, dX.ravel(), dT, dT_wall, shell_rates.ravel(), book[0], book[1]))

    def _rhs_numpy(self, tau, state, step):
        """The right hand side written with numpy array operations (reference implementation)."""
        p = self.unpack(state)
        Y, P, X, T, T_wall = p["y"], p["P"], p["x"], p["T"], p["T_wall"]
        N, dz, nc = self.n_cells, self.cell_width, self.n_independent
        time_s = tau * self.time_scale

        feed = step.feed_end.face(self, "feed", time_s, Cell(P[0], Y[:, 0], T[0]))
        product = step.product_end.face(self, "product", time_s, Cell(P[-1], Y[:, -1], T[-1]))

        """Face values: boundaries at the ends, reconstruction inside."""
        v = np.empty(N + 1)
        v[0], v[-1] = feed.velocity, product.velocity
        y_mid, P_mid, T_mid = 0.5 * (Y[:, 1:] + Y[:, :-1]), 0.5 * (P[1:] + P[:-1]), 0.5 * (T[1:] + T[:-1])
        v[1:-1] = self.velocity_from_gradient(-(P[1:] - P[:-1]) / dz,
                                              self._inertial(y_mid, P_mid, T_mid, group=self._ergun_face[1:-1]),
                                              self._darcy_face[1:-1])
        P_face = np.concatenate(([feed.pressure], P_mid, [product.pressure]))
        y_face = np.empty((nc, N + 1))
        y_face[:, 0], y_face[:, -1] = feed.mole_fractions, product.mole_fractions
        T_face = np.empty(N + 1)
        T_face[0], T_face[-1] = feed.temperature, product.temperature
        if N > 1:
            """
            Each interior face takes its value from the side the gas actually comes from. A step has a main flow
            direction, but the flow can reverse locally (for example around a pressure maximum created by desorption
            during evacuation); reconstructing from the downstream side there would be unstable.
            """
            forward = v[1:-1] >= 0.0
            y_face[:, 1:-1] = np.where(forward, upwind_faces(Y, y_face[:, 0], y_face[:, -1], 1, self.scheme),
                                       upwind_faces(Y, y_face[:, 0], y_face[:, -1], -1, self.scheme))
            T_face[1:-1] = np.where(forward, upwind_faces(T, T_face[0], T_face[-1], 1, self.scheme),
                                    upwind_faces(T, T_face[0], T_face[-1], -1, self.scheme))

        def gradient(cells, first, last):
            g = np.empty(cells.shape[:-1] + (N + 1,))
            g[..., 1:-1] = (cells[..., 1:] - cells[..., :-1]) / dz
            g[..., 0] = (cells[..., 0] - first) / (dz / 2.0)
            g[..., -1] = (last - cells[..., -1]) / (dz / 2.0)
            return g

        dy_dZ = gradient(Y, y_face[:, 0], y_face[:, -1])
        dT_dZ = gradient(T, T_face[0], T_face[-1])

        """Uptake (LDF). Clipping only guards trial states of the implicit solver."""
        T_k = self.T0 * np.clip(T, 0.5, 2.0)
        c_total = self.P0 * np.clip(P, 1.0e-4, 1.0e2) / (GAS_CONSTANT * T_k)
        y_full = np.clip(self.full_composition(np.clip(Y, 0.0, 1.0)), 0.0, 1.0)
        c = y_full * c_total
        slopes = self.adsorbent.isotherm.slopes(c, T_k)
        shell_rates = np.zeros(0)
        if self.n_shells:
            shell_rates, dX = self.adsorbent.kinetics.derivatives(self.unpack(state)["shells"], slopes * c / self.qs0,
                                                                  self.time_scale)
        else:
            rates = self.adsorbent.kinetics.rate_constants(slopes, self.adsorbent, self.gas,
                                                           pressure=self.P0 * np.clip(P, 1.0e-4, 1.0e2) / 1.0e5,
                                                           temperature=T_k)
            dX = rates * self.time_scale * (slopes * c / self.qs0 - X)
        dX_total = dX.sum(axis=0)

        """Energy groups (they depend on the loading through the heat capacity)."""
        eps, eps_bed, a, scale = self._eps_cell, self._eps_bed_cell, self.adsorbent, self._scale
        heat_capacity = a.density * a.heat_capacity + self.qs0_volumetric * a.adsorbed_phase_heat_capacity * X.sum(axis=0)
        solid_ratio = (1.0 - eps_bed) / eps * heat_capacity
        omega_1 = self.column.axial_thermal_conductivity / (self.v0 * eps * self.column.length) / solid_ratio
        omega_2 = self.gas.heat_capacity * self.c0 / solid_ratio
        omega_3 = self.qs0_volumetric * a.adsorbed_phase_heat_capacity / heat_capacity
        omega_4 = (2.0 * self.column.inner_heat_transfer / self.column.inner_radius * self.time_scale
                   / ((1.0 - eps_bed) * heat_capacity))
        heat_released = (self.qs0_volumetric / self.T0) * np.sum(np.asarray(a.heat_of_adsorption, dtype=float).reshape(self.n_components, -1) * dX, axis=0) / heat_capacity

        """Total mass and energy balances, solved together for dP and dT."""
        G = P_face * v / T_face
        dP_partial = -T * scale * (G[1:] - G[:-1]) / dz - self.psi * T * dX_total
        vP = v * P_face
        dT_partial = (omega_1 * (dT_dZ[1:] - dT_dZ[:-1]) / dz - omega_2 * scale * (vP[1:] - vP[:-1]) / dz
                      - omega_3 * T * dX_total + heat_released - omega_4 * (T - T_wall))
        P_over_T = P / T
        dT = (dT_partial - omega_2 * dP_partial) / (1.0 + omega_2 * P_over_T)
        dP = dP_partial + P_over_T * dT

        """Component balances in conservative flux form."""
        F = y_face * G - self._inv_peclet_face * (P_face / T_face) * dy_dZ
        dY = (-(T / P) * scale * (F[:, 1:] - F[:, :-1]) / dz - self.psi * (T / P) * dX[:nc]
              - (Y / P) * dP + (Y / T) * dT)

        """Wall, with both flanges at ambient temperature."""
        T_a = self._ambient(step)
        wall_left = np.concatenate(([2.0 * T_a - T_wall[0]], T_wall[:-1]))
        wall_right = np.concatenate((T_wall[1:], [2.0 * T_a - T_wall[-1]]))
        dT_wall = (self.pi_1 * (wall_right - 2.0 * T_wall + wall_left) / dz ** 2
                   + self.pi_2 * (T - T_wall) - self.pi_3 * (T_wall - T_a))

        """Bookkeeping: what enters the column through each end."""
        enthalpy = vP - self.conduction_number * dT_dZ
        book = []
        for end, index, sign, boundary in (("feed", 0, 1.0, step.feed_end), ("product", -1, -1.0, step.product_end)):
            component_flux = np.concatenate((F[:, index], [G[index] - F[:, index].sum()]))
            work = self._work_integrand(boundary, sign * v[index], P_face[index], y_face[:, index])
            book.append(np.concatenate((sign * component_flux, [sign * enthalpy[index], work])))

        return np.concatenate((dY.ravel(), dP, dX.ravel(), dT, dT_wall, shell_rates.ravel(), book[0], book[1]))

    def _work_integrand(self, boundary, velocity_into, face_pressure, y_face):
        """
        Isentropic work of the machine at one end, per unit dimensionless time:
        gamma/(gamma-1) |v| P [ratio^((gamma-1)/gamma) - 1] / efficiency, with the
        ratio from the supply pressure up to the face (compressor) or from the
        face up to the discharge pressure (vacuum pump).
        """
        pressure_bar = max(face_pressure * self.P0 / 1.0e5, 1.0e-12)
        if velocity_into > 0.0 and boundary.supply_pressure is not None:
            ratio = pressure_bar / boundary.supply_pressure
        elif velocity_into < 0.0 and boundary.discharge_pressure is not None:
            ratio = boundary.discharge_pressure / pressure_bar
        else:
            return 0.0
        if ratio <= 1.0:
            return 0.0
        gamma = self.gas.gamma(self.full_composition(y_face))
        exponent = (gamma - 1.0) / gamma
        work = abs(velocity_into) * face_pressure * (ratio ** exponent - 1.0)
        return work * gamma / (gamma - 1.0) / boundary.machine_efficiency(pressure_bar)

    """The methods below integrate the equations of one step in time."""

    def _build_sparsity(self):
        """
        Jacobian pattern given to the stiff solver, as tight as the equations allow.

        The finite difference Jacobian costs one right hand side evaluation
        per group of columns that share no row, so every unnecessary entry
        slows the solver down. The couplings are:

            y, P, T rows   y, P, T of cells j-2 ... j+2 (face reconstruction),
                           x and T_wall of cell j only
            x rows         y, P, T and x of cell j only (local uptake)
            T_wall rows    T of cell j and T_wall of cells j-1 ... j+1

        The bookkeeping integrals (moles, enthalpy and work through the two
        ends) never feed back into the profiles, but their rows must be in
        the pattern: they depend on the cells next to each end, and without
        those entries Newton's method updates them one iteration late, which
        the solver can read as divergence. In a short column with a large
        permeability (a fast pressure response) this made a blowdown take
        millions of right hand side evaluations instead of about a thousand.
        Only the three cells next to each end are coupled, so the extra cost
        is a few column groups.
        """
        N, n, nc = self.n_cells, self.n_components, self.n_independent
        offsets = {"y": 0, "P": nc * N, "x": (nc + 1) * N, "T": (nc + 1 + n) * N, "T_wall": (nc + 2 + n) * N,
                   "shells": (nc + 3 + n) * N}
        rows_of = {"y": nc, "P": 1, "x": n, "T": 1, "T_wall": 1, "shells": n * self.n_shells}
        pattern = lil_matrix((self.state_size, self.state_size), dtype=int)

        def couple(row_block, column_block, j, cells):
            for r in range(rows_of[row_block]):
                for c in range(rows_of[column_block]):
                    for k in cells:
                        if 0 <= k < N:
                            pattern[offsets[row_block] + r * N + j, offsets[column_block] + c * N + k] = 1

        for j in range(N):
            band, local = range(j - 2, j + 3), [j]
            for row_block in ("y", "P", "T"):
                for column_block in ("y", "P", "T"):
                    couple(row_block, column_block, j, band)
                couple(row_block, "x", j, local)
                couple(row_block, "T_wall", j, local)
            for column_block in ("y", "P", "T", "x"):
                couple("x", column_block, j, local)
            couple("T_wall", "T", j, local)
            couple("T_wall", "T_wall", j, range(j - 1, j + 2))
            if self.n_shells:
                """The uptake is the surface flux of the shells, which see the gas of their cell."""
                for row_block in ("y", "P", "T", "x"):
                    couple(row_block, "shells", j, local)
                for column_block in ("y", "P", "T", "shells"):
                    couple("shells", column_block, j, local)
        n_profiles = self.state_size - 2 * (n + 2)
        end_cells = [k for k in (0, 1, 2, N - 3, N - 2, N - 1) if 0 <= k < N]
        for row in range(n_profiles, self.state_size):
            for column_block in ("y", "P", "T"):
                for c in range(rows_of[column_block]):
                    for k in end_cells:
                        pattern[row, offsets[column_block] + c * N + k] = 1
        return pattern.tocsr()

    def simulate(self, step, state, n_output=50, rtol=1.0e-5, atol=1.0e-7, max_evaluations=40000,
                 fallback=True) -> StepResult:
        """
        Integrate one step from its start to ``step.duration``, or until its
        ``stop`` condition is met.

        The bookkeeping integrals are reset at the start, so the returned
        flows and work belong to this step only. A step whose stop condition
        already holds is skipped and returns a zero-length result.

        The stiff BDF solver is used first. The upwind side of each face
        follows the local velocity, so the right hand side is not smooth
        where the velocity changes sign; in a step whose gas is nearly at
        rest over part of the column (a blowdown or evacuation of a short,
        very permeable bed) Newton's method then keeps failing and BDF can
        take hundreds of thousands of evaluations. After ``max_evaluations``
        right hand side evaluations the step is restarted with LSODA, which
        switches between non-stiff and stiff methods and handles this far
        better (a 2 s blowdown of a 1 cm bed: 81 000 evaluations with BDF,
        11 500 with LSODA). ``max_evaluations=None``
        keeps BDF however long it takes. With ``fallback=False`` the step
        raises ``StepTooStiff`` instead of switching to LSODA (used to discard
        an extrapolated cycle start that turns out to be hard to integrate).
        """
        y0 = np.array(state, dtype=float)
        y0[self.state_size - 2 * (self.n_components + 2):] = 0.0
        tau_end = step.duration / self.time_scale
        events = None
        if step.stop is not None:
            def event(tau, s):
                return step.stop.value(self, s)
            event.terminal, event.direction = True, 1
            if event(0.0, y0) >= 0.0:
                zero = {end: np.zeros(self.n_components) for end in ENDS}
                return StepResult(step, np.array([0.0]), y0[:, None], zero,
                                  {end: 0.0 for end in ENDS}, {end: 0.0 for end in ENDS})
            events = [event]
        """
        The bookkeeping columns of the Jacobian are exactly zero; SciPy's finite differences overflow harmlessly on
        them, so those warnings are silenced.
        """
        with np.errstate(over="ignore", invalid="ignore"):
            """
            A step with a stop condition may end long before its maximum duration, so its output points are taken
            afterwards from the solver's dense output, evenly over the actual duration.
            """
            options = dict(t_eval=None if events else np.linspace(0.0, tau_end, n_output),
                           dense_output=bool(events), rtol=rtol, atol=atol, events=events)
            count = [0]

            def counted(tau, s):
                count[0] += 1
                if max_evaluations is not None and count[0] > max_evaluations:
                    raise _TooManyEvaluations
                return self.rhs(tau, s, step)
            try:
                solution = solve_ivp(counted, (0.0, tau_end), y0, method="BDF", jac_sparsity=self._sparsity, **options)
            except _TooManyEvaluations:
                if not fallback:
                    raise StepTooStiff(f"Step '{step.name}' needed more than {max_evaluations} evaluations.")
                solution = solve_ivp(lambda tau, s: self.rhs(tau, s, step), (0.0, tau_end), y0, method="LSODA",
                                     **options)
        if not solution.success:
            raise RuntimeError(f"Step '{step.name}' failed: {solution.message}")
        times, states = solution.t, solution.y
        if events:
            end = solution.t_events[0][0] if solution.status == 1 and len(solution.t_events[0]) else solution.t[-1]
            times = np.linspace(0.0, end, n_output)
            states = solution.sol(times)
            if solution.status == 1 and len(solution.t_events[0]):
                states[:, -1] = solution.y_events[0][0]
        book = self.unpack(states[:, -1])["bookkeeping"]
        n = self.n_components
        return StepResult(
            step=step, time=times * self.time_scale, states=states,
            moles_in={end: self.moles(book[end][:n]) for end in ENDS},
            enthalpy_in={end: self.enthalpy_joule(book[end][n]) for end in ENDS},
            work={end: self.work_joule(book[end][n + 1]) for end in ENDS},
        )

    """Conversions of the bookkeeping integrals."""

    def moles(self, integral):
        """Dimensionless flux integral to moles: eps A L P0 / (R T0)."""
        c = self.column
        return c.bed_voidage * c.cross_section * c.length * self.c0 * np.asarray(integral)

    def work_joule(self, integral):
        """Dimensionless work integral to joules: eps A v0 P0 (L / v0)."""
        c = self.column
        return float(c.bed_voidage * c.cross_section * self.v0 * self.P0 * self.time_scale * integral)

    def enthalpy_joule(self, integral):
        """Dimensionless enthalpy integral to joules: eps A (Cpg / R) P0 L."""
        c = self.column
        return float(c.bed_voidage * c.cross_section * self.gas.heat_capacity / GAS_CONSTANT
                     * self.P0 * c.length * integral)

    """The methods below measure what the column holds and how well the balances close."""

    def component_inventory(self, state):
        """Moles of each component held in the column (gas, dead volumes and solid)."""
        p = self.unpack(state)
        gas = self.full_composition(p["y"]) * p["P"] / p["T"] * self._gas_weight
        solid = self._psi_inventory * p["x"]
        return self.moles(np.sum(gas + solid, axis=1) * self.cell_width)

    def enthalpy_inventory(self, state):
        """Enthalpy held in the column (solid, adsorbed phase and gas, from 0 K), J."""
        p, c, a = self.unpack(state), self.column, self.adsorbent
        T_k = p["T"] * self.T0
        solid = (1.0 - self._eps_bed_cell) * (a.density * a.heat_capacity
                                          + self.qs0_volumetric * a.adsorbed_phase_heat_capacity * p["x"].sum(axis=0)) * T_k
        gas = c.bed_voidage * self.gas.heat_capacity / GAS_CONSTANT * p["P"] * self.P0 * self._gas_weight
        return float(np.sum(solid + gas) * c.cross_section * c.length * self.cell_width)

    def wall_heat(self, result: StepResult, positive_only=False) -> float:
        """
        Heat supplied to the column from its surroundings through the wall
        during a step, J (negative when the column loses heat), from the
        outer heat transfer coefficient and the wall temperatures over time.

        With ``positive_only=True`` only the heat flowing into the column
        counts, the integral of max(0, Q_in) over time: heat removed by the
        cooling fluid is not credited back.
        """
        c = self.column
        if c.outer_heat_transfer <= 0.0 or len(result.time) < 2:
            return 0.0
        ambient = c.ambient_temperature
        wall = np.array([self.unpack(s)["T_wall"] for s in result.states.T]) * self.T0
        area_per_cell = 2.0 * np.pi * c.outer_radius * c.length / self.n_cells
        rate = c.outer_heat_transfer * area_per_cell * np.sum(ambient - wall, axis=1)
        if positive_only:
            rate = np.maximum(rate, 0.0)
        integrate = getattr(np, "trapezoid", None) or np.trapz
        return float(integrate(rate, result.time))

    def heat_of_adsorption_released(self, initial_state, final_state):
        """Heat released by adsorption between two states, J."""
        change = self.unpack(final_state)["x"] - self.unpack(initial_state)["x"]
        c = self.column
        volume = c.cross_section * c.length * self.cell_width
        heat = np.asarray(self.adsorbent.heat_of_adsorption, dtype=float).reshape(self.n_components, -1)
        return float(np.sum((1.0 - self._eps_bed_cell) * heat * change * self.qs0_volumetric) * volume)

    def end_history(self, result: StepResult, end="product"):
        """
        Velocity (m/s), pressure (bar), mole fractions and temperature (K) at
        one end face over the output times of a step, for plots.
        """
        boundary = result.step.feed_end if end == "feed" else result.step.product_end
        index = 0 if end == "feed" else -1
        rows = []
        for t, state in zip(result.time, result.states.T):
            p = self.unpack(state)
            face = boundary.face(self, end, t, Cell(p["P"][index], p["y"][:, index], p["T"][index]))
            rows.append((face.velocity * self.v0, face.pressure * self.P0 / 1.0e5,
                         self.full_composition(face.mole_fractions), face.temperature * self.T0))
        velocity, pressure, y, temperature = zip(*rows)
        return {"velocity": np.array(velocity), "pressure": np.array(pressure),
                "mole_fractions": np.array(y).T, "temperature": np.array(temperature)}

    def outflow_record(self, result: StepResult, end):
        """
        Record of the gas leaving one end during a step (molar flow in mol/s,
        composition, temperature), to be replayed by a later step.
        """
        from .streams import StreamRecord
        history = self.end_history(result, end)
        c = self.column
        sign = -1.0 if end == "feed" else 1.0
        flux = (sign * history["velocity"] / self.v0) * (history["pressure"] * 1.0e5 / self.P0) \
            / (history["temperature"] / self.T0)
        flow = np.maximum(flux, 0.0) * c.bed_voidage * c.cross_section * self.v0 * self.c0
        n = self.n_components
        cumulative = np.array([-self.moles(self.unpack(state)["bookkeeping"][end][:n])
                               for state in result.states.T]).T
        return StreamRecord(result.time.copy(), flow, history["mole_fractions"], history["temperature"],
                            history["pressure"].copy(), cumulative)

    def profile_change(self, previous_state, state):
        """
        Largest change of any profile (y, P, x, T, T_wall) between two states,
        in dimensionless units, where every profile is of order one: 1e-3
        means about 0.3 K at T0 = 300 K, or 0.1 % of the reference pressure.
        Used as a cyclic steady state test on the column itself.
        """
        n_profiles = self.state_size - 2 * (self.n_components + 2)
        return float(np.max(np.abs(np.asarray(state)[:n_profiles] - np.asarray(previous_state)[:n_profiles])))
