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

from swingpy.boundaries import BufferInlet, ExponentialRamp, FeedInlet, PressureOutlet, PressurizationInlet
from swingpy.cycle import Cycle
from swingpy.gas import GasMixture
from swingpy.model import ColumnModel
from swingpy.steps import PressureReached, Step
from . import vsa as iecr

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


def _adsorption(t_ads, feed_velocity):
    return Step("FEED", t_ads, feed_end=FeedInlet(feed_velocity, FEED),
                product_end=PressureOutlet(HIGH_PRESSURE, stream="raffinate", record_as="raffinate"))


def _blowdown(t_bd, intermediate):
    return Step("BLOW", t_bd, product_end=PressureOutlet(ExponentialRamp(HIGH_PRESSURE, intermediate, RATE),
                                                       stream="blowdown"))


def _evacuation(t_evac, start, low, record=None):
    return Step("EVAC", t_evac, feed_end=PressureOutlet(ExponentialRamp(start, low, RATE), stream="product",
                                                        record_as=record))


def _light_product_pressurization(max_time, offset=0.0, pressure_limited=True):
    """
    Light product pressurization (LPPR) with the raffinate of the feed step, which can enter at most
    at P_H, the pressure it is produced at. The step ends when the closed
    feed end reaches 0.99 P_H, at most after ``max_time``; a feed
    pressurization step (PRES) then completes the pressurization.
    With ``pressure_limited=False`` the recorded flow is imposed whatever the
    pressure and the step ends at P_H, the unibed treatment of the article,
    which over-pressurizes the product end.
    """
    return [
        Step("LPPR", max_time, product_end=BufferInlet("raffinate", offset=offset, stream="raffinate",
                                                      max_pressure=HIGH_PRESSURE if pressure_limited else None),
             stop=PressureReached((0.99 if pressure_limited else 1.0) * HIGH_PRESSURE, location="feed",
                                  rising=True)),
        Step("PRES", 20.0, feed_end=PressurizationInlet(HIGH_PRESSURE, FEED),
             stop=PressureReached(0.999 * HIGH_PRESSURE, location="product", rising=True)),
    ]


def basic_four_step(t_ads, t_bd, t_evac, intermediate, low, feed_velocity, t_press=20.0) -> Cycle:
    """
    Basic four-step cycle of Figure 2, run in the order FEED, BLOW, EVAC, PRES.

    The article starts every cycle from a column at 1 bar. Feed
    pressurization starts from P_L, so beginning with it would expose the
    1 bar column to a 0.03 bar face for no physical reason. Starting with
    adsorption gives the same cyclic steady state without that first shock.
    """
    cycle = iecr.four_step_vsa(t_press=t_press, t_ads=t_ads, t_bd=t_bd, t_evac=t_evac, high_pressure=HIGH_PRESSURE,
                               intermediate_pressure=intermediate, low_pressure=low, feed_velocity=feed_velocity,
                               rate=RATE)
    cycle.steps = cycle.steps[1:] + cycle.steps[:1]
    return cycle


def four_step_lpp(t_ads, t_bd, t_evac, intermediate, low, feed_velocity, pressure_limited_lpp=True) -> Cycle:
    """Four-step cycle with light product pressurization, Figure 6a: FEED, BLOW, EVAC, LPPR, PRES."""
    steps = [_adsorption(t_ads, feed_velocity), _blowdown(t_bd, intermediate),
             _evacuation(t_evac, intermediate, low)] + _light_product_pressurization(t_ads, 0.0, pressure_limited_lpp)
    return Cycle(steps, key_component="CO2")


def five_step_lpp_lr(t_ads, t_bd, t_evac, t_lr, intermediate, low, feed_velocity, pressure_limited_lpp=True) -> Cycle:
    """
    Five-step cycle with light product pressurization and light reflux, Figure 6b: FEED, BLOW, EVAC, LREF, LPPR. The
    first t_LR seconds of the raffinate purge the column at P_L, the rest
    pressurizes it.
    """
    light_reflux = Step("LREF", t_lr, feed_end=PressureOutlet(low, stream="product"),
                        product_end=BufferInlet("raffinate", stream="raffinate"))
    steps = [_adsorption(t_ads, feed_velocity), _blowdown(t_bd, intermediate),
             _evacuation(t_evac, intermediate, low), light_reflux] + \
        _light_product_pressurization(t_ads - t_lr, t_lr, pressure_limited_lpp)
    return Cycle(steps, key_component="CO2")


def five_step_pe_fp(t_ads, t_pe, t_evac, low, feed_velocity, t_press=20.0, kappa=0.041) -> Cycle:
    """
    Five-step cycle with one pressure equalization and pressurization with
    feed, Figure 6d: FEED, EQDN, EVAC, EQUP, PRES.

    The two columns of the equalization are coupled through a recorded
    stream: the donor blows down from P_H through its product end to the
    intermediate pressure of eq. 9 of the article,
    P_I = (P_H + P_L) / 2 - kappa with kappa = 0.041 bar, and the receiver
    takes in the recorded gas through its product end. The receiver then
    ends at P_REC, which the article reports to lie 0.01 to 0.1 bar below
    P_I, as it should for gas flowing from the donor to the receiver.
    Feed pressurization then completes the pressurization to P_H.
    """
    intermediate = 0.5 * (HIGH_PRESSURE + low) - kappa
    steps = [
        _adsorption(t_ads, feed_velocity),
        Step("EQDN", t_pe, product_end=PressureOutlet(ExponentialRamp(HIGH_PRESSURE, intermediate, RATE),
                                                         discharge_pressure=None, stream="equalization",
                                                         record_as="equalization")),
        _evacuation(t_evac, intermediate, low),
        Step("EQUP", t_pe, product_end=BufferInlet("equalization", stream="equalization")),
        Step("PRES", t_press, feed_end=PressurizationInlet(ExponentialRamp(intermediate, HIGH_PRESSURE, RATE), FEED)),
    ]
    return Cycle(steps, key_component="CO2")


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
