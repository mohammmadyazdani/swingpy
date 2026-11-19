"""
Properties of the gas mixture that flows through the column.

The gas is an ideal mixture of any number of components. Its transport
properties are taken as constant, which is the usual assumption of the
detailed adsorption models this package follows. The adiabatic constant,
which only enters the energy of compressors and vacuum pumps, may depend on
composition.
"""

from __future__ import annotations

from dataclasses import dataclass

from .checks import positive
from typing import Callable, Sequence, Union

import numpy as np

GAS_CONSTANT = 8.314
"""Universal gas constant in J/(mol K)."""

MOLAR_MASSES = {
    "CO2": 44.01e-3, "N2": 28.014e-3, "O2": 31.999e-3, "Ar": 39.948e-3,
    "H2O": 18.015e-3, "H2": 2.016e-3, "CH4": 16.043e-3, "CO": 28.010e-3,
}
"""Molar masses in kg/mol of common gases, used when none are given."""


@dataclass
class GasMixture:
    """
    Components and physical properties of the gas phase.

    Attributes:
        components: Names of the components, in the order used everywhere
            else (isotherm parameters, feed compositions, results). The
            last component is obtained by closure, so it is best to list
            the dominant carrier gas (for example N2) last.
        heat_capacity: Molar heat capacity of the gas, J/(mol K).
        viscosity: Dynamic viscosity, Pa s.
        molecular_diffusivity: Molecular diffusivity used for macropore
            diffusion and axial dispersion, m2/s.
        adiabatic_constant: Ratio of heat capacities used for compression
            work. Either a number or a function of the mole fractions
            (a sequence ordered like ``components``).
        molar_masses: Molar masses in kg/mol. Taken from ``MOLAR_MASSES``
            when not given. Only needed by the Ergun equation.
    """

    components: Sequence[str]
    heat_capacity: float = 30.7
    viscosity: float = 1.72e-5
    molecular_diffusivity: float = 1.6e-5
    adiabatic_constant: Union[float, Callable] = 1.4
    molar_masses: Sequence[float] = None

    def __post_init__(self):
        if len(set(self.components)) != len(tuple(self.components)) or len(tuple(self.components)) < 1:
            raise ValueError(f"GasMixture: components must be distinct and at least one, got {self.components!r}.")
        positive("GasMixture", "heat_capacity (J/(mol K))", self.heat_capacity)
        positive("GasMixture", "viscosity (Pa s)", self.viscosity)
        positive("GasMixture", "molecular_diffusivity (m2/s)", self.molecular_diffusivity)
        self.components = tuple(self.components)
        if len(self.components) < 2:
            raise ValueError("A gas mixture needs at least two components.")
        if self.molar_masses is None:
            missing = [c for c in self.components if c not in MOLAR_MASSES]
            if missing:
                raise ValueError(f"Give molar_masses explicitly; unknown components: {missing}.")
            self.molar_masses = tuple(MOLAR_MASSES[c] for c in self.components)
        self.molar_masses = np.asarray(self.molar_masses, dtype=float)

    @property
    def n_components(self) -> int:
        return len(self.components)

    def index(self, component: str) -> int:
        """Position of a component in ``components``."""
        return self.components.index(component)

    def composition(self, mole_fractions) -> np.ndarray:
        """
        Turn a composition into an array ordered like ``components``.

        Args:
            mole_fractions: Either a sequence ordered like ``components``
                or a dict such as ``{"CO2": 0.15, "N2": 0.85}``. Missing
                components of a dict are zero.

        Returns:
            Mole fractions as a numpy array that sums to one.
        """
        if isinstance(mole_fractions, dict):
            unknown = set(mole_fractions) - set(self.components)
            if unknown:
                raise ValueError(f"Unknown components in composition: {sorted(unknown)}.")
            values = np.array([mole_fractions.get(c, 0.0) for c in self.components], dtype=float)
        else:
            values = np.asarray(mole_fractions, dtype=float)
        if values.shape != (self.n_components,) or abs(values.sum() - 1.0) > 1e-6:
            raise ValueError("A composition must give one mole fraction per component, summing to 1.")
        return values

    def gamma(self, mole_fractions) -> float:
        """Adiabatic constant for a given composition."""
        if callable(self.adiabatic_constant):
            return float(self.adiabatic_constant(mole_fractions))
        return float(self.adiabatic_constant)


@dataclass
class LinearCO2AdiabaticConstant:
    """
    Adiabatic constant of CO2/N2 mixtures as a linear function of the CO2
    mole fraction, gamma = 1.395 - 0.105 y_CO2, a linear fit of NIST
    reference data between pure N2 and pure CO2.

    Pass an instance as ``GasMixture.adiabatic_constant``. It matters for the
    vacuum pumps that handle CO2-rich gas: compared with a constant
    gamma = 1.4, it lowers the evacuation work by about 10 % at 0.02 to
    0.03 bar.

    Attributes:
        co2_index: Position of CO2 in ``GasMixture.components``.
    """

    co2_index: int = 0

    def __call__(self, mole_fractions) -> float:
        return 1.395 - 0.105 * float(mole_fractions[self.co2_index])
