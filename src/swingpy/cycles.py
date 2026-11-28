"""
A family of single-column VSA cycles built from optional steps.

``vsa_cycle`` assembles one cycle from the steps of the cycle synthesis of
Haghpanah et al. (2013), AIChE J. 59, 4735, https://doi.org/10.1002/aic.14192
(their Figures 2 and 6):

    FEED   adsorption at P_H; the raffinate is recorded
    BLOW   co-current blowdown to P_I            (depressurization="blowdown")
    EQDN   co-current pressure equalization       (depressurization="equalization")
    HREF   heavy reflux at P_I, optional: a fraction of the extract enters
           the feed end at a constant molar flow while the product end is
           held at P_I; the refluxed gas is subtracted from the product
    EVAC   counter-current evacuation to P_L; recorded as the extract
    LREF   light reflux at P_L, optional: raffinate purges the column from
           the product end, the gas leaving the feed end is product
    EQUP   receiving side of the equalization (depressurization="equalization")
    LPPR   light product pressurization with the raffinate, then PRES with
           feed to P_H                            (pressurization="light product")
    PRES   pressurization with feed               (pressurization="feed")

The seven cycles of the article are presets of this function in this
module (``HAGHPANAH_2013``).

Pressure changes are either prescribed, as in the article (an exponential
approach at ``rate`` over a given step time), or driven by vacuum pumps of
given interstitial velocity, ended when the pressure is reached (give
``blowdown_velocity`` or ``evacuation_velocity``). The equalization is
always the single column treatment: the donor follows a prescribed
pressure history and its outflow is recorded, the receiver takes that
recorded flow in through its product end.

The function builds the cycle only; the column model (pressure drop law,
mass transfer, heat transfer) is chosen independently when the cycle is run.
"""

from __future__ import annotations

from .boundaries import (BufferInlet, ExponentialRamp, FeedInlet, PressureOutlet, PressurizationInlet,
                         TankInlet, VacuumPump)
from .cycle import Cycle
from .steps import PressureReached, Step

__all__ = ["vsa_cycle", "HAGHPANAH_2013", "basic_four_step", "four_step_lpp", "five_step_lpp_lr",
           "six_step_lpp_lr_hr", "five_step_pe_fp", "five_step_pe_lpp", "six_step_pe_lpp_hr"]


def vsa_cycle(feed, *, t_ads, feed_velocity, high, low, intermediate,
              depressurization="blowdown", t_blow=None, blowdown_velocity=None, t_eq=None,
              t_hr=0.0, reflux_ratio=0.0,
              t_evac=None, evacuation_velocity=None,
              t_lr=0.0,
              pressurization="light product", t_press=20.0, pressure_limited_lpp=True,
              final_pressurization_time=20.0,
              rate=0.5, supply_pressure=1.0, discharge_pressure=1.0, efficiency=0.72,
              pump_efficiency=None, max_step_time=5000.0, record_raffinate=True,
              key_component="CO2") -> Cycle:
    """
    One single-column VSA cycle; see the module docstring for the steps.

    Args:
        feed: Feed composition, e.g. {"CO2": 0.15, "N2": 0.85}.
        t_ads: Adsorption time, s.
        feed_velocity: Interstitial feed velocity, m/s.
        high: Adsorption pressure P_H, bar.
        low: Evacuation pressure P_L, bar.
        intermediate: Pressure at the end of blowdown or equalization, P_I, bar.
        depressurization: "blowdown" (BLOW through a vacuum pump) or
            "equalization" (EQDN, and EQUP after evacuation and light reflux).
        t_blow: Blowdown time, s, for a prescribed blowdown.
        blowdown_velocity: Interstitial velocity of the blowdown pump, m/s,
            instead of ``t_blow``; the step ends at P_I.
        t_eq: Time of each equalization step, s.
        t_hr: Heavy reflux time, s (0 leaves the step out).
        reflux_ratio: Fraction of the extract fed back in the heavy reflux.
        t_evac: Evacuation time, s, for a prescribed evacuation.
        evacuation_velocity: Interstitial velocity of the evacuation pump,
            m/s, instead of ``t_evac``; the step ends at P_L.
        t_lr: Light reflux time, s (0 leaves the step out). The light reflux
            uses the first t_lr seconds of the recorded raffinate and the
            light product pressurization the rest.
        pressurization: "light product" (LPPR then PRES) or "feed" (PRES with
            a prescribed rise from the pressure reached before it to P_H over
            ``t_press``).
        t_press: Feed pressurization time, s.
        final_pressurization_time: Longest time of the feed pressurization
            (PRES) that follows LPPR, s; the step ends earlier when the
            product end reaches 0.999 P_H. If it is too short for the column,
            the feed step starts below P_H.
        pressure_limited_lpp: Light product enters at most at P_H, and LPPR
            ends at 0.99 P_H (True), or the recorded flow is imposed whatever
            the pressure and LPPR ends at P_H (False, the unibed treatment of
            Haghpanah et al.).
        rate: Rate constant of the prescribed pressure histories, 1/s.
        supply_pressure: Pressure of the feed supply, bar.
        discharge_pressure: Delivery pressure of vacuum pumps and outlets, bar.
        efficiency: Efficiency of compressors and of prescribed outlets.
        pump_efficiency: Efficiency of the vacuum pumps given by velocity
            (a number or a function of suction pressure); ``efficiency`` if None.
        max_step_time: Upper bound of steps ended by a pressure, s.
        record_raffinate: Record the raffinate of FEED (needed for LPPR and LREF).
        key_component: Component whose purity and recovery are reported.
    """
    if depressurization not in ("blowdown", "equalization"):
        raise ValueError('depressurization must be "blowdown" or "equalization".')
    if pressurization not in ("light product", "feed"):
        raise ValueError('pressurization must be "light product" or "feed".')
    if not low < intermediate < high:
        raise ValueError("Pressures must satisfy P_L < P_I < P_H.")
    if (t_lr > 0.0 or pressurization == "light product") and not record_raffinate:
        raise ValueError("Light reflux and light product pressurization need record_raffinate=True.")
    pump_efficiency = efficiency if pump_efficiency is None else pump_efficiency

    steps = [Step("FEED", t_ads,
                  feed_end=FeedInlet(feed_velocity, feed, supply_pressure=supply_pressure, efficiency=efficiency),
                  product_end=PressureOutlet(high, discharge_pressure=discharge_pressure, efficiency=efficiency,
                                             stream="raffinate", record_as="raffinate" if record_raffinate else None))]

    if depressurization == "blowdown":
        if blowdown_velocity is not None:
            steps.append(Step("BLOW", max_step_time,
                              product_end=VacuumPump(stream="blowdown", velocity=blowdown_velocity,
                                                     discharge_pressure=discharge_pressure, efficiency=pump_efficiency),
                              stop=PressureReached(intermediate, location="product", rising=False)))
        elif t_blow is not None:
            steps.append(Step("BLOW", t_blow,
                              product_end=PressureOutlet(ExponentialRamp(high, intermediate, rate),
                                                         discharge_pressure=discharge_pressure, efficiency=efficiency,
                                                         stream="blowdown")))
        else:
            raise ValueError("Give t_blow or blowdown_velocity.")
    else:
        if t_eq is None:
            raise ValueError("Give t_eq for the equalization.")
        steps.append(Step("EQDN", t_eq,
                          product_end=PressureOutlet(ExponentialRamp(high, intermediate, rate), discharge_pressure=None,
                                                     efficiency=efficiency, stream="equalization",
                                                     record_as="equalization")))

    if t_hr > 0.0:
        steps.append(Step("HREF", t_hr,
                          feed_end=TankInlet("extract", reflux_ratio, supply_pressure=supply_pressure,
                                             efficiency=efficiency, stream="product"),
                          product_end=PressureOutlet(intermediate, discharge_pressure=discharge_pressure,
                                                     efficiency=efficiency, stream="heavy reflux effluent")))

    extract = "extract" if t_hr > 0.0 else None
    if evacuation_velocity is not None:
        steps.append(Step("EVAC", max_step_time,
                          feed_end=VacuumPump(stream="product", velocity=evacuation_velocity, record_as=extract,
                                              discharge_pressure=discharge_pressure, efficiency=pump_efficiency),
                          stop=PressureReached(low, location="feed", rising=False)))
    elif t_evac is not None:
        steps.append(Step("EVAC", t_evac,
                          feed_end=PressureOutlet(ExponentialRamp(intermediate, low, rate),
                                                  discharge_pressure=discharge_pressure, efficiency=efficiency,
                                                  stream="product", record_as=extract)))
    else:
        raise ValueError("Give t_evac or evacuation_velocity.")

    if t_lr > 0.0:
        steps.append(Step("LREF", t_lr,
                          feed_end=PressureOutlet(low, discharge_pressure=discharge_pressure, efficiency=efficiency,
                                                  stream="product"),
                          product_end=BufferInlet("raffinate", stream="raffinate")))

    if depressurization == "equalization":
        steps.append(Step("EQUP", t_eq, product_end=BufferInlet("equalization", stream="equalization")))

    if pressurization == "light product":
        steps += [
            Step("LPPR", t_ads - t_lr,
                 product_end=BufferInlet("raffinate", offset=t_lr, stream="raffinate",
                                         max_pressure=high if pressure_limited_lpp else None),
                 stop=PressureReached((0.99 if pressure_limited_lpp else 1.0) * high, location="feed", rising=True)),
            Step("PRES", final_pressurization_time,
                 feed_end=PressurizationInlet(high, feed, supply_pressure=supply_pressure, efficiency=efficiency),
                 stop=PressureReached(0.999 * high, location="product", rising=True)),
        ]
    else:
        start = intermediate if depressurization == "equalization" else low
        steps.append(Step("PRES", t_press,
                          feed_end=PressurizationInlet(ExponentialRamp(start, high, rate), feed,
                                                       supply_pressure=supply_pressure, efficiency=efficiency)))
    return Cycle(steps, key_component=key_component)


# The seven configurations of Haghpanah et al. (2013), AIChE J. 59, 4735, Figures 2 and 6, for 15 % CO2 at
# P_H = 1 bar with the prescribed pressure histories of the article (rate 0.5 1/s, as in the companion
# Ind. Eng. Chem. Res. article). Every step time and pressure is an argument;
# ``examples/setups/cycles.py`` pairs them with the column model and operating points of the article.

HAGHPANAH_FEED = {"CO2": 0.15, "N2": 0.85}
HAGHPANAH_HIGH_PRESSURE = 1.0
HAGHPANAH_RATE = 0.5


def _cycle(**kwargs) -> Cycle:
    """``vsa_cycle`` with the feed, pressures and pressure histories of Haghpanah et al. (2013)."""
    return vsa_cycle(HAGHPANAH_FEED, high=HAGHPANAH_HIGH_PRESSURE, rate=HAGHPANAH_RATE, **kwargs)


def _equalization_pressure(low, kappa):
    """Pressure at the end of the equalization, eq. 9 of the article: P_I = (P_H + P_L) / 2 - kappa."""
    return 0.5 * (HAGHPANAH_HIGH_PRESSURE + low) - kappa


def basic_four_step(t_ads, t_bd, t_evac, intermediate, low, feed_velocity, t_press=20.0) -> Cycle:
    """
    Basic four-step cycle of Figure 2, run in the order FEED, BLOW, EVAC, PRES.

    The article starts every cycle from a column at 1 bar. Feed
    pressurization starts from P_L, so beginning with it would expose the
    1 bar column to a 0.03 bar face for no physical reason. Starting with
    adsorption gives the same cyclic steady state without that first shock.
    """
    return _cycle(t_ads=t_ads, feed_velocity=feed_velocity, low=low, intermediate=intermediate, t_blow=t_bd,
                  t_evac=t_evac, pressurization="feed", t_press=t_press, record_raffinate=False)


def four_step_lpp(t_ads, t_bd, t_evac, intermediate, low, feed_velocity, pressure_limited_lpp=True) -> Cycle:
    """
    Four-step cycle with light product pressurization, Figure 6a: FEED, BLOW, EVAC, LPPR, PRES.

    Light product pressurization uses the raffinate of the feed step, which
    can enter at most at P_H, the pressure it is produced at. The step ends
    when the closed feed end reaches 0.99 P_H, at most after t_ads; a feed
    pressurization step (PRES) then completes the pressurization. With
    ``pressure_limited_lpp=False`` the recorded flow is imposed whatever the
    pressure and the step ends at P_H, the unibed treatment of the article,
    which over-pressurizes the product end.
    """
    return _cycle(t_ads=t_ads, feed_velocity=feed_velocity, low=low, intermediate=intermediate, t_blow=t_bd,
                  t_evac=t_evac, pressure_limited_lpp=pressure_limited_lpp)


def five_step_lpp_lr(t_ads, t_bd, t_evac, t_lr, intermediate, low, feed_velocity, pressure_limited_lpp=True) -> Cycle:
    """
    Five-step cycle with light product pressurization and light reflux, Figure 6b: FEED, BLOW, EVAC, LREF, LPPR.
    The first t_LR seconds of the raffinate purge the column at P_L, the rest
    pressurizes it.
    """
    return _cycle(t_ads=t_ads, feed_velocity=feed_velocity, low=low, intermediate=intermediate, t_blow=t_bd,
                  t_evac=t_evac, t_lr=t_lr, pressure_limited_lpp=pressure_limited_lpp)


def six_step_lpp_lr_hr(t_ads, t_bd, t_hr, t_evac, t_lr, intermediate, low, reflux_ratio, feed_velocity,
                       pressure_limited_lpp=True) -> Cycle:
    """
    Six-step cycle with light product pressurization, light and heavy reflux, Figure 6c: FEED, BLOW, HREF, EVAC, LREF, LPPR.
    A fraction ``reflux_ratio`` of the evacuation stream is fed back at the
    feed end during HREF, at a constant flow (eq. 8 of the article), while a
    vacuum pump holds the product end at P_I. The refluxed gas is subtracted
    from the product.
    """
    return _cycle(t_ads=t_ads, feed_velocity=feed_velocity, low=low, intermediate=intermediate, t_blow=t_bd,
                  t_hr=t_hr, reflux_ratio=reflux_ratio, t_evac=t_evac, t_lr=t_lr,
                  pressure_limited_lpp=pressure_limited_lpp)


def five_step_pe_fp(t_ads, t_pe, t_evac, low, feed_velocity, t_press=20.0, kappa=0.041) -> Cycle:
    """
    5-step cycle with one pressure equalization and feed pressurization,
    Figure 6d: FEED, EQDN, EVAC, EQUP, PRES.

    The two columns of the equalization are coupled through a recorded
    stream: the donor blows down from P_H through its product end to the
    intermediate pressure of eq. 9 of the article,
    P_I = (P_H + P_L) / 2 - kappa with kappa = 0.041 bar, and the receiver
    takes in the recorded gas through its product end. The receiver then
    ends at P_REC, which the article reports to lie 0.01 to 0.1 bar below
    P_I, as it should for gas flowing from the donor to the receiver.
    Feed pressurization then completes the pressurization to P_H.
    """
    return _cycle(t_ads=t_ads, feed_velocity=feed_velocity, low=low, intermediate=_equalization_pressure(low, kappa),
                  depressurization="equalization", t_eq=t_pe, t_evac=t_evac, pressurization="feed", t_press=t_press)


def five_step_pe_lpp(t_ads, t_pe, t_evac, low, feed_velocity, kappa=0.041, pressure_limited_lpp=True) -> Cycle:
    """
    5-step cycle with one pressure equalization and light product
    pressurization, Figure 6e: FEED, EQDN, EVAC, EQUP, LPPR (then PRES to P_H).

    The equalization is that of ``five_step_pe_fp``; light product
    pressurization replaces feed pressurization from P_REC to P_H.
    """
    return _cycle(t_ads=t_ads, feed_velocity=feed_velocity, low=low, intermediate=_equalization_pressure(low, kappa),
                  depressurization="equalization", t_eq=t_pe, t_evac=t_evac,
                  pressure_limited_lpp=pressure_limited_lpp)


def six_step_pe_lpp_hr(t_ads, t_pe, t_hr, t_evac, low, reflux_ratio, feed_velocity, kappa=0.041,
                       pressure_limited_lpp=True) -> Cycle:
    """
    6-step cycle with pressure equalization, light product pressurization and
    heavy reflux, Figure 6f: FEED, EQDN, HREF, EVAC, EQUP, LPPR (then PRES).

    The heavy reflux follows the equalization at its end pressure P_I, as in
    ``six_step_lpp_lr_hr``: a fraction ``reflux_ratio`` of the extract enters
    the feed end while a vacuum pump holds the product end at P_I.
    """
    return _cycle(t_ads=t_ads, feed_velocity=feed_velocity, low=low, intermediate=_equalization_pressure(low, kappa),
                  depressurization="equalization", t_eq=t_pe, t_hr=t_hr, reflux_ratio=reflux_ratio, t_evac=t_evac,
                  pressure_limited_lpp=pressure_limited_lpp)


HAGHPANAH_2013 = {
    "basic four step (Figure 2)": basic_four_step,
    "four step with LPP (Figure 6a)": four_step_lpp,
    "five step with LPP and LR (Figure 6b)": five_step_lpp_lr,
    "six step with LPP, LR and HR (Figure 6c)": six_step_lpp_lr_hr,
    "five step with PE and FP (Figure 6d)": five_step_pe_fp,
    "five step with PE and LPP (Figure 6e)": five_step_pe_lpp,
    "six step with PE, LPP and HR (Figure 6f)": six_step_pe_lpp_hr,
}
"""The seven cycle configurations of Haghpanah et al. (2013), by name."""
