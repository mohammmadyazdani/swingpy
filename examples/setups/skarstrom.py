"""
Two-bed Skarstrom cycle separating CO2 from N2 (15 % CO2, 1.3 bar) on
zeolite 13X in a laboratory column, after Dantas et al., Chem. Eng. J. 172
(2011) 698, https://doi.org/10.1016/j.cej.2011.06.037 (the particle diameter,
not reported there, is taken as 2 mm). ``RUNS`` holds three operating
points; ``example_case(run)`` returns model, cycle and initial state.
"""

from __future__ import annotations
from swingpy.adsorbent import Adsorbent, ConstantLDF
from swingpy.boundaries import Closed, FlowInlet, PressureOutlet, standard_litres_per_minute
from swingpy.column import Column
from swingpy.cycle import Cycle
from swingpy.gas import GAS_CONSTANT, GasMixture
from swingpy.isotherms import Toth
from swingpy.model import ColumnModel
from swingpy.steps import Step

FEED = {"CO2": 0.15, "N2": 0.85}
NITROGEN = {"N2": 1.0}
FEED_PRESSURE = 1.3
"""FEED_PRESSURE: bar."""
PURGE_PRESSURE = 0.1
"""PURGE_PRESSURE: bar."""
FEED_FLOW = standard_litres_per_minute(3.0)
PRESSURIZATION_FLOW = standard_litres_per_minute(3.0)
PURGE_FLOW = standard_litres_per_minute(0.5)
PARTICLE_DIAMETER = 2.0e-3
PARTICLE_POROSITY, TORTUOSITY = 0.37, 2.2
DIFFUSIVITY, VISCOSITY, HEAT_CAPACITY, CONDUCTIVITY = 1.4e-5, 1.9e-5, 29.5, 0.028

RUNS = {
    1: dict(temperature=323.15, t_feed=100.0, t_purge=70.0, k_co2=0.090),
    2: dict(temperature=323.15, t_feed=120.0, t_purge=50.0, k_co2=0.090),
    3: dict(temperature=373.15, t_feed=120.0, t_purge=50.0, k_co2=0.292),
}
"""Table 3 of the article."""



def isotherm() -> Toth:
    """Toth parameters of Table 5 (K0 in 1/bar, heats of adsorption in J/mol)."""
    return Toth(qs=(5.09, 3.08), b0=(4.305e-4, 0.882e-4), dU=(-29380.0, -17190.0), t=(0.429, 0.869),
                competitive=False, pressure_based=True)


def nitrogen_ldf(temperature):
    """LDF coefficient of N2 from Eq. 20 of the article at the feed conditions, 1/s."""
    p_n2 = 0.85 * FEED_PRESSURE * 1.0e5
    c0 = p_n2 / (GAS_CONSTANT * temperature)
    q0 = isotherm().loadings([[0.0], [c0]], temperature)[1, 0] * 1940.0
    return 15.0 * PARTICLE_POROSITY * DIFFUSIVITY * c0 / (TORTUOSITY * (PARTICLE_DIAMETER / 2.0) ** 2 * q0)


def feed_velocity(temperature):
    """Interstitial feed velocity at the inlet conditions, m/s."""
    column = Column(length=0.83, inner_radius=0.0105, bed_voidage=0.62)
    volume_flow = FEED_FLOW * GAS_CONSTANT * temperature / (FEED_PRESSURE * 1.0e5)
    return volume_flow / (column.cross_section * column.bed_voidage)


def model(run, n_cells=30, backend="auto") -> ColumnModel:
    """Column, adsorbent and gas of Tables 1 to 3 for one run, scaled on the feed conditions."""
    conditions = RUNS[run]
    T = conditions["temperature"]
    v = feed_velocity(T)
    density = FEED_PRESSURE * 1.0e5 * (0.15 * 44.01e-3 + 0.85 * 28.014e-3) / (GAS_CONSTANT * T)
    reynolds = density * v * 0.62 * PARTICLE_DIAMETER / VISCOSITY
    schmidt = VISCOSITY / (density * DIFFUSIVITY)
    prandtl = HEAT_CAPACITY / ((0.15 * 44.01e-3 + 0.85 * 28.014e-3)) * VISCOSITY / CONDUCTIVITY
    dispersion = DIFFUSIVITY * (20.0 + 0.5 * schmidt * reynolds) / 0.62
    axial_conduction = CONDUCTIVITY * (10.0 + 0.5 * prandtl * reynolds)
    gas = GasMixture(("CO2", "N2"), heat_capacity=HEAT_CAPACITY, viscosity=VISCOSITY,
                     molecular_diffusivity=DIFFUSIVITY)
    adsorbent = Adsorbent(
        name="Zeolite 13X (Dantas et al. 2011)", isotherm=isotherm(), heat_of_adsorption=(29380.0, 17190.0),
        density=1940.0, particle_radius=PARTICLE_DIAMETER / 2.0, particle_voidage=PARTICLE_POROSITY,
        tortuosity=TORTUOSITY, heat_capacity=920.0, adsorbed_phase_heat_capacity=0.0,
        kinetics=ConstantLDF((conditions["k_co2"], nitrogen_ldf(T))),
        notes="Dantas et al. (2011), Tables 1, 3 and 5; particle diameter assumed 2 mm")
    column = Column(length=0.83, inner_radius=0.0105, outer_radius=0.0105 + 0.0041, bed_voidage=0.62,
                    pressure_drop="ergun", axial_thermal_conductivity=axial_conduction, inner_heat_transfer=18.0,
                    outer_heat_transfer=8.0, wall_density=8238.0, wall_heat_capacity=500.0,
                    wall_thermal_conductivity=13.4, ambient_temperature=T)
    return ColumnModel(column, adsorbent, gas, n_cells=n_cells, reference_pressure=FEED_PRESSURE,
                       reference_temperature=T, reference_velocity=v, backend=backend, axial_dispersion=dispersion)


def skarstrom_cycle(run) -> Cycle:
    """Pressurization (20 s), feed, counter-current blowdown (70 s) and counter-current N2 purge."""
    conditions = RUNS[run]
    T = conditions["temperature"]
    steps = [
        Step("PRES", 20.0,
             feed_end=FlowInlet(PRESSURIZATION_FLOW, NITROGEN, T, max_pressure=FEED_PRESSURE, stream="nitrogen"),
             product_end=Closed()),
        Step("FEED", conditions["t_feed"], feed_end=FlowInlet(FEED_FLOW, FEED, T, stream="feed"),
             product_end=PressureOutlet(FEED_PRESSURE, discharge_pressure=None, stream="raffinate")),
        Step("BLOW", 70.0, feed_end=PressureOutlet(PURGE_PRESSURE, discharge_pressure=None, stream="product"),
             product_end=Closed()),
        Step("PURG", conditions["t_purge"],
             feed_end=PressureOutlet(PURGE_PRESSURE, discharge_pressure=None, stream="product"),
             product_end=FlowInlet(PURGE_FLOW, NITROGEN, T, stream="nitrogen")),
    ]
    return Cycle(steps, key_component="CO2")


def example_case(run):
    """Model, cycle and initial state (N2 at the purge pressure) of one run."""
    column_model = model(run)
    return column_model, skarstrom_cycle(run), column_model.initial_state(PURGE_PRESSURE, NITROGEN,
                                                                           RUNS[run]["temperature"])
