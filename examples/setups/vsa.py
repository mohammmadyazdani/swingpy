"""
Ready-made setups: a column, gas, adsorbent and cycle to start from.

Two families of standard cases, with the parameters of published studies:

- post-combustion flue gas (15 % CO2) on zeolite 13X in a 1 m column with
  a four-step vacuum swing cycle, after Haghpanah et al., Ind. Eng. Chem.
  Res. 52 (2013) 4249, https://doi.org/10.1021/ie302658y: ``flue_gas``,
  ``zeolite_13x``, ``column``, ``model``, ``breakthrough_step``,
  ``four_step_vsa``;
- steam-methane-reformer flue gas (20 % CO2) on zeolite 13X, UTSA-16 or
  IISERP MOF2 with a four-step cycle with light product pressurization,
  after Subraveti et al., Sep. Purif. Technol. 256 (2021) 117832,
  https://doi.org/10.1016/j.seppur.2020.117832: the ``smr_`` functions.

Each function returns ordinary SwingPy objects to change as needed.
"""

from __future__ import annotations

from dataclasses import replace

from swingpy.adsorbent import Adsorbent
from swingpy.boundaries import (BufferInlet, ExponentialRamp, FeedInlet, PressureOutlet, PressurizationInlet,
                         falling_vacuum_efficiency,
                         VacuumPump)
from swingpy.column import Column
from swingpy.cycle import Cycle
from swingpy.gas import GasMixture, LinearCO2AdiabaticConstant
from swingpy.isotherms import DualSiteLangmuir
from swingpy.model import ColumnModel
from swingpy.steps import PressureReached, Step

FEED = {"CO2": 0.15, "N2": 0.85}


def flue_gas(adiabatic_constant=1.4) -> GasMixture:
    """
    Dry flue gas, CO2 and N2, with the properties of Table 3.

    The adiabatic constant of the pumped gas is 1.4 by default, as in the
    article; ``LinearCO2AdiabaticConstant()`` uses the value of the actual
    CO2/N2 mixture instead, which lowers the vacuum work of CO2-rich
    streams by about 10 %.
    """
    return GasMixture(components=("CO2", "N2"), heat_capacity=30.7, viscosity=1.72e-5,
                      molecular_diffusivity=1.6e-5, adiabatic_constant=adiabatic_constant)


def zeolite_13x() -> Adsorbent:
    """Zeochem zeolite 13X: dual-site Langmuir parameters of Table 2, particle data of Table 3."""
    isotherm = DualSiteLangmuir(
        qsb=(3.09, 5.84), qsd=(2.54, 0.0),
        b0=(8.65e-7, 2.50e-6), d0=(2.63e-8, 0.0),
        dUb=(-36641.21, -1.58e4), dUd=(-35690.66, 0.0),
    )
    return Adsorbent(
        name="Zeolite 13X", isotherm=isotherm, heat_of_adsorption=(36641.21, 1.58e4),
        density=1130.0, particle_radius=1.0e-3, particle_voidage=0.35, tortuosity=3.0,
        heat_capacity=1070.0, adsorbed_phase_heat_capacity=30.7,
        notes="Haghpanah et al. (2013), Table 2; heat of adsorption taken as -dU of site 1",
    )


def column(adiabatic=False) -> Column:
    """The 1 m column of Table 3. ``adiabatic=True`` for the breakthrough case of Section 5."""
    base = Column(length=1.0, inner_radius=0.1445, outer_radius=0.1620, bed_voidage=0.37,
                  axial_thermal_conductivity=0.09, inner_heat_transfer=8.6, outer_heat_transfer=2.5,
                  wall_density=7800.0, wall_heat_capacity=502.0, wall_thermal_conductivity=16.0,
                  ambient_temperature=298.15)
    return replace(base, inner_heat_transfer=0.0, outer_heat_transfer=0.0) if adiabatic else base


def model(adiabatic=False, n_cells=30, scheme="weno", feed_velocity=1.0, high_pressure=1.0,
          adiabatic_constant=1.4) -> ColumnModel:
    """Column model scaled with P0 = P_H, T0 = T_feed and v0 = v_feed, as in the article."""
    return ColumnModel(column(adiabatic), zeolite_13x(), flue_gas(adiabatic_constant), n_cells=n_cells, scheme=scheme,
                       reference_pressure=high_pressure, reference_temperature=298.15,
                       reference_velocity=feed_velocity)


def breakthrough_step(duration=8000.0) -> Step:
    """Dynamic column breakthrough of Section 5: feed at 1 m/s, outlet held at 1 bar."""
    return Step("FEED", duration,
                feed_end=FeedInlet(velocity=1.0, composition=FEED, temperature=298.15),
                product_end=PressureOutlet(pressure=1.0, stream="raffinate"))


def four_step_vsa(t_press=15.0, t_ads=15.0, t_bd=30.0, t_evac=40.0, high_pressure=1.0,
                  intermediate_pressure=0.2, low_pressure=0.1, feed_velocity=1.0, rate=0.5,
                  flue_gas_pressure=1.0, efficiency=0.72) -> Cycle:
    """
    The four-step VSA cycle of Figure 7 (defaults: Table 4).

    Pressures in bar, times in s, velocity in m/s. Every step uses the same
    machine efficiency, and vacuum pumps discharge at 1 bar.
    """
    steps = [
        Step("PRES", t_press,
             feed_end=PressurizationInlet(ExponentialRamp(low_pressure, high_pressure, rate), FEED,
                                          supply_pressure=flue_gas_pressure, efficiency=efficiency)),
        Step("FEED", t_ads,
             feed_end=FeedInlet(feed_velocity, FEED, supply_pressure=flue_gas_pressure, efficiency=efficiency),
             product_end=PressureOutlet(high_pressure, efficiency=efficiency, stream="raffinate")),
        Step("BLOW", t_bd,
             product_end=PressureOutlet(ExponentialRamp(high_pressure, intermediate_pressure, rate),
                                        efficiency=efficiency, stream="blowdown")),
        Step("EVAC", t_evac,
             feed_end=PressureOutlet(ExponentialRamp(intermediate_pressure, low_pressure, rate),
                                     efficiency=efficiency, stream="product")),
    ]
    return Cycle(steps, key_component="CO2")


"""SMR flue gas cases."""

SMR_FEED = {"CO2": 0.20, "N2": 0.80}
SMR_HIGH_PRESSURE = 1.02
SMR_DISCHARGE_PRESSURE = 1.01325
SMR_COMPRESSOR_EFFICIENCY = 0.80

"""Dual-site Langmuir parameters qsb, qsd, b0, d0, dUb, dUd for (CO2, N2): SI Table S2."""
_SMR_ISOTHERMS = {
    "Zeolite 13X": ((3.09, 3.09), (2.54, 2.54), (8.65e-7, 2.69e-6), (2.63e-8, 2.69e-6),
                    (-36641.0, -15710.0), (-35690.0, -15710.0)),
    "UTSA-16": ((4.08, 1.33), (1.29, 1.77), (2.52e-7, 9.17e-5), (1.75e-9, 9.42e-9),
                (-32800.0, -7500.0), (-35040.0, -27760.0)),
    "IISERP MOF2": ((3.29, 3.29), (1.89, 1.89), (9.39e-8, 2.55e-7), (5.23e-7, 2.55e-7),
                    (-31135.0, -11890.0), (-31135.0, -11890.0)),
}
_SMR_DENSITIES = {"Zeolite 13X": 1130.0, "UTSA-16": 1171.0, "IISERP MOF2": 937.7}


def smr_adsorbent(name) -> Adsorbent:
    """One of "Zeolite 13X", "UTSA-16" or "IISERP MOF2", as 1.5 mm particles (SI Tables S2 and S4)."""
    qsb, qsd, b0, d0, dUb, dUd = _SMR_ISOTHERMS[name]
    return Adsorbent(name=name, isotherm=DualSiteLangmuir(qsb, qsd, b0, d0, dUb, dUd),
                     heat_of_adsorption=(-dUb[0], -dUb[1]), density=_SMR_DENSITIES[name], particle_radius=0.75e-3,
                     particle_voidage=0.35, tortuosity=3.0, heat_capacity=1070.0, adsorbed_phase_heat_capacity=30.7,
                     notes="Subraveti et al. (2021), SI Tables S2 and S4; -dH taken as -dU of site 1")


def smr_flue_gas() -> GasMixture:
    """Dry flue gas as a CO2/N2 binary (O2 and Ar counted as N2), SI Table S4."""
    return GasMixture(("CO2", "N2"), heat_capacity=30.7, viscosity=1.72e-5, molecular_diffusivity=1.6e-5,
                      adiabatic_constant=LinearCO2AdiabaticConstant(0))


def smr_column(length, length_to_diameter=3.0) -> Column:
    """Adiabatic column with Ergun pressure drop; the diameter follows from L/D."""
    return Column(length=length, inner_radius=length / length_to_diameter / 2.0, bed_voidage=0.37,
                  pressure_drop="ergun", axial_thermal_conductivity=0.09, inner_heat_transfer=0.0,
                  outer_heat_transfer=0.0)


def smr_model(adsorbent_name, length, feed_velocity, length_to_diameter=3.0, n_cells=30, backend="auto") -> ColumnModel:
    return ColumnModel(smr_column(length, length_to_diameter), smr_adsorbent(adsorbent_name), smr_flue_gas(),
                       n_cells=n_cells,
                       reference_pressure=SMR_HIGH_PRESSURE, reference_temperature=298.15,
                       reference_velocity=feed_velocity, backend=backend)


def smr_four_step_lpp_cycle(t_ads, blowdown_flow, evacuation_flow, intermediate, low, feed_velocity,
                    max_step_time=5000.0, pressure_limited_lpp=True, blowdown_velocity=None,
                    evacuation_velocity=None) -> Cycle:
    """
    FEED, BLOW, EVAC and LPPR of Figure 4 of the article.

    Args:
        t_ads: Adsorption time, s.
        blowdown_flow, evacuation_flow: Vacuum pump capacities, m3/h (or None
            when given as velocities).
        blowdown_velocity, evacuation_velocity: Interstitial velocities the
            pumps draw, m/s, instead of their capacities.
        intermediate, low: P_I and P_L, bar.
        feed_velocity: Interstitial feed velocity, m/s.
        max_step_time: Safety limit for the pressure-terminated steps, s.
        pressure_limited_lpp: True (default) lets the raffinate enter the LPP
            column at most at P_H, the pressure it is produced at. False
            imposes the recorded molar flow whatever the pressure, which is
            the boundary condition of the article (SI Table S3); it ends the
            LPP step sooner but pushes the product end above P_H, which the
            next adsorption step then releases as a burst.
    """
    lpp_limit = SMR_HIGH_PRESSURE if pressure_limited_lpp else None
    lpp_stop = 0.99 * SMR_HIGH_PRESSURE if pressure_limited_lpp else SMR_HIGH_PRESSURE
    pump = dict(discharge_pressure=SMR_DISCHARGE_PRESSURE, efficiency=falling_vacuum_efficiency)
    steps = [
        Step("FEED", t_ads,
             feed_end=FeedInlet(feed_velocity, SMR_FEED, supply_pressure=SMR_HIGH_PRESSURE, efficiency=SMR_COMPRESSOR_EFFICIENCY),
             product_end=PressureOutlet(SMR_HIGH_PRESSURE, discharge_pressure=None, stream="raffinate",
                                        record_as="raffinate")),
        Step("BLOW", max_step_time, product_end=VacuumPump(blowdown_flow, stream="blowdown",
                                                          velocity=blowdown_velocity, **pump),
             stop=PressureReached(intermediate, location="product", rising=False)),
        Step("EVAC", max_step_time, feed_end=VacuumPump(evacuation_flow, stream="product",
                                                        velocity=evacuation_velocity, **pump),
             stop=PressureReached(low, location="feed", rising=False)),
        Step("LPPR", t_ads, product_end=BufferInlet("raffinate", stream="raffinate", max_pressure=lpp_limit),
             stop=PressureReached(lpp_stop, location="feed", rising=True)),
        Step("PRES", 60.0, feed_end=PressurizationInlet(SMR_HIGH_PRESSURE, SMR_FEED, supply_pressure=SMR_HIGH_PRESSURE,
                                                             efficiency=SMR_COMPRESSOR_EFFICIENCY),
             stop=PressureReached(0.999 * SMR_HIGH_PRESSURE, location="product", rising=True)),
    ]
    return Cycle(steps, key_component="CO2")


SMR_OPERATING_POINTS = {
    "Zeolite 13X": dict(t_ads=212.0, blowdown_flow=7784.0, evacuation_flow=12682.0, intermediate=0.11, low=0.022,
                        feed_velocity=1.14, length=8.0),
    "UTSA-16": dict(t_ads=159.0, blowdown_flow=6086.0, evacuation_flow=8631.0, intermediate=0.11, low=0.026,
                    feed_velocity=1.20, length=5.4),
    "IISERP MOF2": dict(t_ads=118.0, blowdown_flow=5434.0, evacuation_flow=16341.0, intermediate=0.31, low=0.036,
                        feed_velocity=1.20, length=8.8),
}
"""Optimal operating conditions of the three adsorbents (Subraveti et al. 2021, Table 5)."""


def smr_example_case(adsorbent_name, pressure_limited_lpp=True):
    """
    Model, cycle and initial state at one of ``SMR_OPERATING_POINTS``.

    The column starts full of feed gas at the high pressure, because the
    cycle begins with adsorption; the cyclic steady state does not depend on
    the initial state.
    """
    point = SMR_OPERATING_POINTS[adsorbent_name]
    column_model = smr_model(adsorbent_name, point["length"], point["feed_velocity"])
    cycle = smr_four_step_lpp_cycle(point["t_ads"], point["blowdown_flow"], point["evacuation_flow"],
                                    point["intermediate"], point["low"], point["feed_velocity"],
                                    pressure_limited_lpp=pressure_limited_lpp)
    return column_model, cycle, column_model.initial_state(SMR_HIGH_PRESSURE, SMR_FEED)
