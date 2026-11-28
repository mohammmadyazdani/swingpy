"""
Adsorbent particles and their mass transfer kinetics.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .checks import fraction, positive, same_length
from typing import Sequence

import numpy as np


@dataclass
class MacroporeLDF:
    """
    Linear driving force rate controlled by molecular diffusion in macropores.

    The rate constant of component i is

        k_i = (c_i / q_i*) 15 eps_p D_p / r_p^2,   with D_p = D_m / tortuosity,

    where q_i* is expressed per cubic metre of adsorbent so that c_i / q_i*
    is dimensionless (q_i* in mol/kg times the particle density), the
    Glueckauf approximation for diffusion in a spherical particle.

    A component that the solid barely holds (q_i* / c_i close to zero, for
    example nitrogen displaced by water) gets an enormous rate constant,
    which only means that it is at equilibrium; such rates are capped at
    ``max_rate`` (1/s), far above any rate that matters, so that the
    equations do not become needlessly stiff.
    """

    max_rate: float = 1.0e3

    def rate_constants(self, slopes, adsorbent, gas, pressure=None, temperature=None):
        """LDF rate constants in 1/s, with the shape of ``slopes`` (m3/kg)."""
        diffusivity = gas.molecular_diffusivity / adsorbent.tortuosity
        prefactor = 15.0 * adsorbent.particle_voidage * diffusivity / adsorbent.particle_radius ** 2
        return np.minimum(prefactor / (adsorbent.density * slopes), self.max_rate)


@dataclass
class ConstantLDF:
    """
    Linear driving force with fixed rate constants.

    Attributes:
        rates: One rate constant per gas component, 1/s.
    """

    rates: Sequence[float]

    def rate_constants(self, slopes, adsorbent, gas, pressure=None, temperature=None):
        return np.broadcast_to(np.asarray(self.rates, dtype=float)[:, None], slopes.shape)


@dataclass
class PressureScaledLDF:
    """
    Linear driving force with rate constants inversely proportional to pressure,

        k_i = rates_i * reference_pressure / P,

    the behaviour of uptake controlled by molecular diffusion in the
    macropores (the molecular diffusivity scales as 1/P). Uptake is then much
    faster at low pressure (blowdown, purge) than at high pressure (feed).

    Attributes:
        rates: Rate constants at the reference pressure, 1/s, one per component.
        reference_pressure: bar.
    """

    rates: Sequence[float]
    reference_pressure: float = 1.0

    def rate_constants(self, slopes, adsorbent, gas, pressure=None, temperature=None):
        rates = np.asarray(self.rates, dtype=float)[:, None]
        if pressure is None:
            return np.broadcast_to(rates, slopes.shape)
        return rates * self.reference_pressure / np.maximum(np.asarray(pressure, dtype=float), 1.0e-3)


@dataclass
class SeriesLDF:
    """
    Linear driving force for a bidisperse particle, with the film, macropore
    and micropore resistances added in series (Glueckauf; Ruthven, Principles
    of Adsorption and Adsorption Processes, 1984, Eq. 8.53 and 8.54):

        1 / k_i = K_i / (a_p k_f) + K_i R_p^2 / (Omega_m eps_p D_p,i) + r_c^2 / (Omega_c D_c,i),
        K_i = rho_p q_i* / c_i,   D_c,i / r_c^2 = (D_c0 / r_c^2)_i exp(-E_a,i / RT),

    where K_i is the dimensionless slope of the isotherm per volume of particle
    (the secant q*/c, as in ``MacroporeLDF``), a_p the external area per volume
    of particle (3 / R_p for spheres, 2 / R_p for cylinders) and Omega the LDF
    factors (15 for spheres, 8 for cylinders). This is a single LDF that
    stands in for a bi-LDF model, where the macropore gas and the micropore
    loading are separate states; it is exact for a linear isotherm at long times and an
    approximation otherwise, so say so when you use it.

    Attributes:
        micropore_rates: (D_c0 / r_c^2)_i, 1/s, one per component (``inf`` for
            no micropore resistance).
        activation_energies: E_a,i, J/mol (zero for a constant D_c/r_c^2).
        macropore_diffusivities: D_p,i, m2/s (``inf`` for none).
        film_coefficient: k_f, m/s (``inf`` for no film resistance).
        micropore_factor: Omega_c, -.
        macropore_factor: Omega_m, -.
        area_factor: a_p R_p, - (3 for spheres, 2 for cylinders).
        max_rate: Cap on the rate constants, 1/s (see ``MacroporeLDF``).
    """

    micropore_rates: Sequence[float]
    activation_energies: Sequence[float]
    macropore_diffusivities: Sequence[float]
    film_coefficient: float
    micropore_factor: float
    macropore_factor: float
    area_factor: float
    max_rate: float = 1.0e3

    def __post_init__(self):
        for name in ("micropore_rates", "activation_energies", "macropore_diffusivities"):
            setattr(self, name, np.asarray(getattr(self, name), dtype=float))
        same_length("SeriesLDF", micropore_rates=self.micropore_rates,
                    activation_energies=self.activation_energies,
                    macropore_diffusivities=self.macropore_diffusivities)
        for name, value in (("micropore_rates (1/s)", self.micropore_rates),
                            ("macropore_diffusivities (m2/s)", self.macropore_diffusivities),
                            ("film_coefficient (m/s)", self.film_coefficient)):
            values = np.atleast_1d(np.asarray(value, dtype=float))
            """Inf: no resistance of that kind."""
            if np.any(np.isnan(values)) or np.any(values <= 0.0):
                raise ValueError(f"SeriesLDF: {name} must be positive (inf for no resistance), got {value!r}.")
        positive("SeriesLDF", "activation_energies (J/mol)", self.activation_energies, allow_zero=True)
        positive("SeriesLDF", "micropore_factor", self.micropore_factor)
        positive("SeriesLDF", "macropore_factor", self.macropore_factor)
        positive("SeriesLDF", "area_factor", self.area_factor)

    def rate_constants(self, slopes, adsorbent, gas, pressure=None, temperature=None):
        """LDF rate constants in 1/s, with the shape of ``slopes`` (m3/kg)."""
        from .gas import GAS_CONSTANT
        if temperature is None:
            raise ValueError("SeriesLDF needs the temperature (activation energies of micropore diffusion).")
        K = adsorbent.density * np.asarray(slopes, dtype=float)
        T = np.asarray(temperature, dtype=float)
        Rp = adsorbent.particle_radius
        micro = self.micropore_factor * self.micropore_rates[:, None] \
            * np.exp(-self.activation_energies[:, None] / (GAS_CONSTANT * T))
        with np.errstate(divide="ignore"):
            time_constant = 1.0 / micro \
                + K * Rp ** 2 / (self.macropore_factor * adsorbent.particle_voidage * self.macropore_diffusivities[:, None]) \
                + K * Rp / (self.area_factor * self.film_coefficient)
            return np.minimum(1.0 / time_constant, self.max_rate)


@dataclass
class Adsorbent:
    """
    An adsorbent material in particle form.

    Attributes:
        name: Display name, for reports and plots.
        isotherm: Equilibrium model, for example ``DualSiteLangmuir``.
        heat_of_adsorption: One positive value per gas component, J/mol
            (-dH). Isotherm papers often give only internal energies of
            adsorption; state clearly which value you use.
        density: Particle density, kg/m3.
        particle_radius: m.
        particle_voidage: Macropore porosity of the particle, -.
        tortuosity: Macropore tortuosity, -.
        heat_capacity: Specific heat capacity of the solid, J/(kg K).
        adsorbed_phase_heat_capacity: Molar heat capacity of the adsorbed
            phase, J/(mol K).
        kinetics: Mass transfer model, ``MacroporeLDF()`` by default.
        notes: Free text about the parameters (origin, assumptions).
    """

    name: str
    isotherm: object
    heat_of_adsorption: Sequence[float]
    density: float
    particle_radius: float
    particle_voidage: float = 0.35
    tortuosity: float = 3.0
    heat_capacity: float = 1070.0
    adsorbed_phase_heat_capacity: float = 30.7
    kinetics: object = field(default_factory=MacroporeLDF)
    notes: str = ""

    def __post_init__(self):
        positive(f"Adsorbent {self.name}", "density (kg/m3)", self.density)
        positive(f"Adsorbent {self.name}", "particle_radius (m)", self.particle_radius)
        fraction(f"Adsorbent {self.name}", "particle_voidage", self.particle_voidage)
        positive(f"Adsorbent {self.name}", "tortuosity", self.tortuosity)
        positive(f"Adsorbent {self.name}", "heat_capacity (J/(kg K))", self.heat_capacity)
        positive(f"Adsorbent {self.name}", "heat_of_adsorption (J/mol)", self.heat_of_adsorption, allow_zero=True)
        saturation = getattr(self.isotherm, "total_saturation", None)
        if saturation is not None:
            same_length(f"Adsorbent {self.name}", isotherm=np.atleast_1d(saturation),
                        heat_of_adsorption=self.heat_of_adsorption)
        self.heat_of_adsorption = np.asarray(self.heat_of_adsorption, dtype=float)


class LayeredBed:
    """
    A bed of several adsorbent layers along the column, for example a guard
    layer of alumina or activated carbon at the feed end and a zeolite above
    it, as in most industrial PSA units.

    Pass it to ``ColumnModel`` in place of an ``Adsorbent``. Every layer keeps
    its own isotherm, kinetics, density, heat capacity, heats of adsorption,
    particle size and, optionally, bed voidage, applied to the cells it fills;
    the pressure drop and axial dispersion of every face follow from the
    particle size and voidage on both sides of it.

    Args:
        layers: (Adsorbent, length fraction) or (Adsorbent, length fraction,
            bed voidage) tuples from the feed end to the product end; the
            fractions must add up to 1. A layer without a voidage takes that
            of the column.
        name: Display name.
    """

    def __init__(self, layers, name=None):
        layers = [tuple(layer) for layer in layers]
        self.voidages = [float(layer[2]) if len(layer) > 2 and layer[2] is not None else None for layer in layers]
        for v in self.voidages:
            if v is not None and not 0.0 < v < 1.0:
                raise ValueError(f"LayeredBed: a layer voidage must lie between 0 and 1, got {v}.")
        layers = [(layer[0], float(layer[1])) for layer in layers]
        if not layers:
            raise ValueError("LayeredBed needs at least one layer.")
        fractions = np.array([f for _, f in layers])
        if np.any(fractions <= 0.0) or abs(fractions.sum() - 1.0) > 1e-6:
            raise ValueError(f"LayeredBed: layer fractions must be positive and add up to 1, got {fractions}.")
        counts = {len(np.atleast_1d(a.heat_of_adsorption)) for a, _ in layers}
        if len(counts) != 1:
            raise ValueError("LayeredBed: every layer needs parameters for the same gas components.")
        self.layers = layers
        self.name = name or " / ".join(a.name for a, _ in layers)

    def layer_cells(self, n_cells):
        """Cell indices of every layer (each cell belongs to the layer that holds its centre)."""
        centres = (np.arange(n_cells) + 0.5) / n_cells
        bounds = np.concatenate(([0.0], np.cumsum([f for _, f in self.layers])))
        bounds[-1] = 1.0 + 1e-12
        return [np.where((centres >= lo) & (centres < hi))[0] for lo, hi in zip(bounds[:-1], bounds[1:])]

    def cell_view(self, n_cells):
        """An Adsorbent-like object whose properties are arrays over the cells."""
        return _CellAdsorbent(self, n_cells)


class _LayeredIsotherm:
    def __init__(self, bed, cells, n_cells):
        self.bed, self.cells, self.n_cells = bed, cells, n_cells

    def _per_cell(self, method, c, temperature):
        c = np.asarray(c, dtype=float)
        if c.ndim == 1:
            c = c[:, None]
        if c.shape[1] != self.n_cells:
            c = np.repeat(c[:, :1], self.n_cells, axis=1)
        T = np.broadcast_to(np.asarray(temperature, dtype=float), (self.n_cells,))
        out = np.empty_like(c)
        for (adsorbent, _), idx in zip(self.bed.layers, self.cells):
            if len(idx):
                out[:, idx] = getattr(adsorbent.isotherm, method)(c[:, idx], T[idx])
        return out

    def slopes(self, c, temperature):
        return self._per_cell("slopes", c, temperature)

    def loadings(self, c, temperature):
        return self._per_cell("loadings", c, temperature)

    @property
    def total_saturation(self):
        return np.max([np.atleast_1d(a.isotherm.total_saturation) for a, _ in self.bed.layers], axis=0)


class _LayeredKinetics:
    def __init__(self, bed, cells):
        self.bed, self.cells = bed, cells

    def rate_constants(self, slopes, adsorbent, gas, pressure=None, temperature=None):
        rates = np.empty_like(slopes)
        for (layer, _), idx in zip(self.bed.layers, self.cells):
            if len(idx):
                rates[:, idx] = layer.kinetics.rate_constants(
                    slopes[:, idx], layer, gas,
                    pressure=None if pressure is None else np.asarray(pressure)[idx],
                    temperature=None if temperature is None else np.asarray(temperature)[idx])
        return rates


class _CellAdsorbent:
    """Per-cell properties of a layered bed, with the attributes the column model reads from an Adsorbent."""

    def __init__(self, bed, n_cells):
        cells = bed.layer_cells(n_cells)
        fractions = np.array([f for _, f in bed.layers])

        def per_cell(attribute):
            out = np.empty(n_cells)
            for (a, _), idx in zip(bed.layers, cells):
                out[idx] = getattr(a, attribute)
            return out

        self.name, self.layered_bed, self.layer_indices = bed.name, bed, cells
        self.particle_radius_cells = per_cell("particle_radius")
        self.particle_voidage_cells = per_cell("particle_voidage")
        if any(v is not None for v in bed.voidages):
            self.bed_voidage_cells = np.full(n_cells, np.nan)
            for v, idx in zip(bed.voidages, cells):
                self.bed_voidage_cells[idx] = np.nan if v is None else v
        else:
            self.bed_voidage_cells = None
        self.density = per_cell("density")
        self.heat_capacity = per_cell("heat_capacity")
        self.adsorbed_phase_heat_capacity = per_cell("adsorbed_phase_heat_capacity")
        n = len(np.atleast_1d(bed.layers[0][0].heat_of_adsorption))
        self.heat_of_adsorption = np.empty((n, n_cells))
        for (a, _), idx in zip(bed.layers, cells):
            self.heat_of_adsorption[:, idx] = np.asarray(a.heat_of_adsorption, dtype=float)[:, None]
        mean = lambda attribute: float(np.dot(fractions, [getattr(a, attribute) for a, _ in bed.layers]))  # noqa: E731
        self.particle_radius = mean("particle_radius")
        self.particle_voidage = mean("particle_voidage")
        self.tortuosity = mean("tortuosity")
        self.isotherm = _LayeredIsotherm(bed, cells, n_cells)
        self.kinetics = _LayeredKinetics(bed, cells)
        self.notes = "layered bed"


@dataclass
class ParticleDiffusion:
    """
    Fickian diffusion inside spherical particles (or crystals), resolved in
    concentric shells, instead of a linear driving force:

        dq/dt = D / r^2 d/dr (r^2 dq/dr),   q(r_p) = q*(c, T) at the surface.

    This is the model of uptake controlled by diffusion in the micropores of
    a crystal or a carbon molecular sieve, where it matters for kinetic
    separations; for a single particle at constant surface loading it
    reproduces the exact solution of Crank (The Mathematics of Diffusion,
    1975, Eq. 6.20), to which a linear driving force with k = 15 D / r_p^2 is
    the Glueckauf approximation.

    The particle is divided into ``shells`` shells, thinner towards the
    surface, and solved by finite volumes; the uptake of a cell is the flux through the
    particle surface, so mass is conserved exactly.

    Attributes:
        rates: Diffusional time constants D_i / r_p^2, 1/s, one per component.
        shells: Number of shells (12 by default).
    """

    rates: Sequence[float]
    shells: int = 12

    def __post_init__(self):
        self.rates = np.asarray(self.rates, dtype=float)
        if self.shells < 2:
            raise ValueError("ParticleDiffusion needs at least 2 shells.")
        if np.any(self.rates < 0.0):
            raise ValueError("ParticleDiffusion rates (D/r^2, 1/s) must not be negative.")
        """
        Shells thin towards the surface, r = 1 - (1 - s)^2, where the steep profiles of the first moments of uptake
        form: 12 such shells match the exact uptake of a sphere to 0.5 %, against 4 % for 12 equal shells.
        """
        edges = 1.0 - (1.0 - np.linspace(0.0, 1.0, self.shells + 1)) ** 2
        """Times 4 pi / 3."""
        self._volume = edges[1:] ** 3 - edges[:-1] ** 3
        self._inner_area, self._outer_area = edges[:-1] ** 2, edges[1:] ** 2
        """self._inner_area, self._outer_area: times 4 pi."""
        centres = 0.5 * (edges[1:] + edges[:-1])
        self._gap = np.diff(centres)
        self._surface_gap = 1.0 - centres[-1]

    def derivatives(self, shells, surface, time_scale=1.0):
        """
        Time derivatives of the shell loadings and of the particle average.

        Args:
            shells: Loadings of the shells, shape (n, shells, N).
            surface: Equilibrium loading at the particle surface, shape (n, N).
            time_scale: Seconds per unit of model time.

        Returns:
            (d shells / dt, d average / dt), shapes (n, shells, N) and (n, N).
        """
        D = (self.rates * time_scale)[:, None, None]
        """D dq/dr at the outer face of each shell."""
        outward = np.empty_like(shells)
        outward[:, :-1, :] = D * (shells[:, 1:, :] - shells[:, :-1, :]) / self._gap[None, :, None]
        outward[:, -1, :] = D[:, 0, :] * (surface - shells[:, -1, :]) / self._surface_gap
        inward = np.zeros_like(shells)
        inward[:, 1:, :] = outward[:, :-1, :]
        d_shells = 3.0 * (self._outer_area[None, :, None] * outward - self._inner_area[None, :, None] * inward) \
            / self._volume[None, :, None]
        """Surface flux, area 4 pi over volume 4 pi / 3."""
        d_average = 3.0 * outward[:, -1, :]
        return d_shells, d_average

    def average(self, shells):
        """Volume average of the shell loadings, shape (n, N)."""
        return np.tensordot(self._volume, shells, axes=([0], [1])) / self._volume.sum()

    def rate_constants(self, slopes, adsorbent, gas, pressure=None, temperature=None):
        raise TypeError("ParticleDiffusion has no rate constants; the column model resolves the particles.")
