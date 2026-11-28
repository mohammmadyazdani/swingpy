"""
Cycle configurations for flue gas (15 % CO2) on zeolite 13X in a 1 m column,
after Haghpanah et al., AIChE J. 59 (2013) 4735, https://doi.org/10.1002/aic.14192:
the basic four-step cycle, four steps with light product pressurization
(LPPR), five steps with light product pressurization and light reflux
(LREF), and five steps with pressure equalization (EQDN and EQUP) and
pressurization with feed (PRES).

``EXAMPLE_POINTS`` holds ready operating points of these cycles and
``example_case(name)`` returns model, cycle and initial state for one.
Every column starts full of N2 at 1 bar and 25 C; the prescribed pressure
histories relax at 0.5 1/s.
"""

from __future__ import annotations

from dataclasses import replace

from swingpy.cycles import HAGHPANAH_2013 as CYCLES
from swingpy.cycles import (basic_four_step, five_step_lpp_lr, five_step_pe_fp, five_step_pe_lpp, four_step_lpp,
                            six_step_lpp_lr_hr, six_step_pe_lpp_hr)
from swingpy.gas import GasMixture
from swingpy.model import ColumnModel
from . import vsa as iecr

"""The cycle configurations of the article are presets of ``swingpy.cycles``, re-exported here."""
__all__ = ["CYCLES", "EXAMPLE_POINTS", "FEED", "HIGH_PRESSURE", "RATE", "basic_four_step", "example_case",
           "five_step_lpp_lr", "five_step_pe_fp", "five_step_pe_lpp", "four_step_lpp", "initial_state", "model",
           "six_step_lpp_lr_hr", "six_step_pe_lpp_hr"]

FEED = {"CO2": 0.15, "N2": 0.85}
HIGH_PRESSURE = 1.0
RATE = 0.5


def model(feed_velocity=1.0, n_cells=30, scheme="weno", backend="auto", adiabatic_constant=1.4) -> ColumnModel:
    """
    Column of Table 2, scaled with v0 equal to the feed velocity.

    ``adiabatic_constant`` defaults to 1.4, as in the article; pass
    ``LinearCO2AdiabaticConstant()`` for the value of the actual gas
    mixture (see ``setups.flue_gas``).
    """
    gas = GasMixture(("CO2", "N2"), heat_capacity=30.7, viscosity=1.72e-5, molecular_diffusivity=1.30e-5,
                     adiabatic_constant=adiabatic_constant)
    adsorbent = replace(iecr.zeolite_13x(), particle_radius=7.5e-4,
                        notes="Haghpanah et al. (2013) AIChE J., Tables 1 and 2")
    return ColumnModel(iecr.column(), adsorbent, gas, n_cells=n_cells, scheme=scheme,
                       reference_pressure=HIGH_PRESSURE, reference_temperature=298.15,
                       reference_velocity=feed_velocity, backend=backend)


def initial_state(column_model):
    """Pure N2 at 1 bar and 25 C."""
    return column_model.initial_state(1.0, {"N2": 1.0})


"""Ready operating points of these cycles (from the article above)."""


EXAMPLE_POINTS = {
    "basic four step": dict(
        cycle=basic_four_step, feed_velocity=0.27,
        kwargs=dict(t_ads=74.6, t_bd=60.1, t_evac=85.4, intermediate=0.17, low=0.03, feed_velocity=0.27)),
    "four step with light product pressurization, high recovery": dict(
        cycle=four_step_lpp, feed_velocity=0.23,
        kwargs=dict(t_ads=64.64, t_bd=172.80, t_evac=141.30, intermediate=0.07, low=0.03, feed_velocity=0.23)),
    "four step with light product pressurization, high purity": dict(
        cycle=four_step_lpp, feed_velocity=0.33,
        kwargs=dict(t_ads=53.15, t_bd=141.76, t_evac=131.21, intermediate=0.05, low=0.03, feed_velocity=0.33)),
    "four step with light product pressurization, purity and recovery above 90 %": dict(
        cycle=four_step_lpp, feed_velocity=0.46,
        kwargs=dict(t_ads=99.86, t_bd=30.53, t_evac=116.82, intermediate=0.16, low=0.03, feed_velocity=0.46)),
    "five step with pressure equalization": dict(
        cycle=five_step_pe_fp, feed_velocity=0.56,
        kwargs=dict(t_ads=83.59, t_pe=60.0, t_evac=130.0, low=0.03, feed_velocity=0.56)),
}


def example_case(name, adiabatic_constant=1.4, pressure_limited_lpp=True):
    """Model, cycle and initial state of one of ``EXAMPLE_POINTS``."""
    point = EXAMPLE_POINTS[name]
    column_model = model(feed_velocity=point["feed_velocity"], adiabatic_constant=adiabatic_constant)
    kwargs = dict(point["kwargs"])
    if point["cycle"] in (four_step_lpp, five_step_lpp_lr):
        kwargs["pressure_limited_lpp"] = pressure_limited_lpp
    return column_model, point["cycle"](**kwargs), initial_state(column_model)
