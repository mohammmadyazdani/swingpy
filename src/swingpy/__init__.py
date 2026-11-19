"""
SwingPy: simulation of pressure, vacuum and temperature swing adsorption columns.

A column model is built from three objects (``Column``, ``Adsorbent`` and
``GasMixture``), and a cycle from ``Step`` objects whose two ends are
boundary conditions such as ``FeedInlet``, ``PressureOutlet`` or ``Closed``.
See the README and the examples folder for a walk-through.
"""

from .adsorbent import Adsorbent, ConstantLDF, LayeredBed, MacroporeLDF, ParticleDiffusion, PressureScaledLDF, SeriesLDF
from .boundaries import (Boundary, BufferInlet, MixingInlet, Closed, Constant, ExponentialRamp, FeedInlet, LinearRamp, PressureOutlet,
                         PressurizationInlet, TankInlet, VacuumPump, falling_vacuum_efficiency,
                         FlowInlet, RecordedPressure, Valve, standard_litres_per_minute)
from .column import Column
from .cycle import Cycle, CyclePerformance, CycleRun
from .gas import GasMixture, LinearCO2AdiabaticConstant
from .isotherms import IAST, AranovichDonohue, DualSiteLangmuir, ExtendedLangmuir, HumidToth, MultisiteLangmuir, Sips, Toth, Virial
from .model import ColumnModel, StepResult, StepTooStiff
from .steps import PressureReached, Step, instant_pressurization
from .streams import StreamRecord

__version__ = "1.0.0"

from . import data, eos, humidity  # noqa: E402

__all__ = ["data", "eos", "humidity",
    "Adsorbent", "ConstantLDF", "LayeredBed", "ParticleDiffusion", "MacroporeLDF", "PressureScaledLDF", "Boundary", "Closed", "Constant", "ExponentialRamp", "LinearRamp",
    "FeedInlet", "PressureOutlet", "PressurizationInlet", "falling_vacuum_efficiency", "Column",
    "Cycle", "CyclePerformance", "CycleRun", "GasMixture", "LinearCO2AdiabaticConstant", "AranovichDonohue", "DualSiteLangmuir", "ExtendedLangmuir", "HumidToth", "MultisiteLangmuir", "Virial", "SeriesLDF", "IAST", "Sips", "Toth", "ColumnModel", "StepTooStiff",
    "StepResult", "Step", "PressureReached", "instant_pressurization", "BufferInlet", "MixingInlet", "TankInlet", "VacuumPump", "FlowInlet", "RecordedPressure", "Valve", "standard_litres_per_minute", "StreamRecord",
]
