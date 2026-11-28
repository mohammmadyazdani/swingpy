"""
Geometry and thermal properties of a packed adsorption column.
"""

from __future__ import annotations

from dataclasses import dataclass

from .checks import fraction, positive

import numpy as np


@dataclass
class Column:
    """
    A cylindrical packed bed with a steel wall.

    Attributes:
        length: Bed length, m.
        inner_radius: m.
        outer_radius: m. Only used by the wall energy balance.
        bed_voidage: Interparticle voidage, -.
        pressure_drop: "darcy" (viscous term only) or "ergun" (viscous and inertial terms).
        axial_thermal_conductivity: Effective axial conductivity of the bed, W/(m K).
        inner_heat_transfer: Bed to wall coefficient, W/(m2 K). Zero for an adiabatic column.
        outer_heat_transfer: Wall to ambient coefficient, W/(m2 K).
        wall_density: kg/m3.
        wall_heat_capacity: J/(kg K).
        wall_thermal_conductivity: W/(m K).
        include_pore_gas: Count the gas in the pores of the particles as gas
            holdup: the gas balance then uses the total voidage
            eps_t = eps_b + (1 - eps_b) eps_p (eps_p is the particle voidage
            of the adsorbent), while the flow through the bed and the solid
            fraction use the bed voidage eps_b. Off by default, as in models
            that lump the pore gas into the solid; it matters most for weakly
            adsorbed gases such as nitrogen, whose front it slows down.
        dead_volume_feed, dead_volume_product: Well-mixed gas volume at each
            end of the bed (pipes, distributors, valve bodies), m3. It holds
            gas but no adsorbent, and slows the pressure and composition
            changes at that end; it is often what makes small laboratory
            columns behave differently from the model of the bed alone.
        ambient_temperature: Temperature outside the wall, K.
    """

    length: float
    inner_radius: float
    outer_radius: float = None
    bed_voidage: float = 0.37
    pressure_drop: str = "darcy"
    axial_thermal_conductivity: float = 0.09
    inner_heat_transfer: float = 0.0
    outer_heat_transfer: float = 0.0
    wall_density: float = 7800.0
    wall_heat_capacity: float = 502.0
    wall_thermal_conductivity: float = 16.0
    ambient_temperature: float = 298.15
    dead_volume_feed: float = 0.0
    dead_volume_product: float = 0.0
    include_pore_gas: bool = False

    def __post_init__(self):
        positive("Column", "length (m)", self.length)
        positive("Column", "inner_radius (m)", self.inner_radius)
        fraction("Column", "bed_voidage", self.bed_voidage)
        for name in ("axial_thermal_conductivity", "inner_heat_transfer", "outer_heat_transfer"):
            positive("Column", name, getattr(self, name), allow_zero=True)
        positive("Column", "ambient_temperature (K)", self.ambient_temperature)
        positive("Column", "dead_volume_feed (m3)", self.dead_volume_feed, allow_zero=True)
        positive("Column", "dead_volume_product (m3)", self.dead_volume_product, allow_zero=True)
        if self.outer_radius is None:
            self.outer_radius = 1.1 * self.inner_radius
        if self.outer_radius < self.inner_radius:
            raise ValueError(f"Column: outer_radius ({self.outer_radius}) is smaller than inner_radius ({self.inner_radius}).")
        if self.pressure_drop not in ("darcy", "ergun"):
            raise ValueError("pressure_drop must be 'darcy' or 'ergun'.")

    @property
    def cross_section(self) -> float:
        """Inner cross-sectional area, m2."""
        return np.pi * self.inner_radius ** 2

    @property
    def adsorbent_volume(self) -> float:
        """Volume of the solid particles, (1 - eps) A L, m3."""
        return (1.0 - self.bed_voidage) * self.cross_section * self.length
