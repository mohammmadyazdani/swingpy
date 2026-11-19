"""
Two-bed hydrogen PSA with pressure equalization, driven by valves.

Hydrogen is recovered at 15 bar from a 75 % H2 / 25 % N2 feed on a molecular
sieve. Every step is set by valves (molar flow = cv x pressure difference)
and fixed flows, as in an industrial unit, rather than by prescribed
pressures. The second bed is represented by recorded streams: the gas a
bed sends to the other one (equalization, purge, repressurization) is
recorded and fed back when this bed takes the place of the other one, and
the donor vents into the pressure history the receiving step recorded in
the previous cycle; at cyclic steady state the two beds are consistent.

    Step  Name                                   Time    Ends
    1     Adsorption                             60 s    feed 0.03 mol/s; product valve to 14.997 bar
    2     Adsorption, supply to repressurization 180 s   same, and 0.008 mol/s of product to the other bed
    3     Depressurization, equalization         60 s    product end valve to the other bed
    4     Depressurization, supply of purge      180 s   product end valve to the other bed
    5     Blowdown                               60 s    feed end valve to 1.013 bar
    6     Purge                                  180 s   gas of step 4 in; feed end valve to 1.013 bar
    7     Repressurization, equalization         60 s    gas of step 3 in
    8     Repressurization with product          180 s   product of step 2 in

Valve coefficients are in mol/(s bar). Isotherm: extended Langmuir on partial pressures, hydrogen as a Henry
component (it adsorbs a little but takes up no sites). Mass transfer:
linear driving force with k = 2.95 / P(bar) 1/s. Bed: 2 m x 5.5 cm,
voidage 0.40, particles of 1.05 mm radius. These data are those of the
H2/N2 PSA example case distributed with a commercial adsorption simulator.

    python "examples/06 Hydrogen PSA.py" --max-cycles 60
"""

import argparse

from swingpy import (Adsorbent, BufferInlet, Column, ColumnModel, Cycle, ExtendedLangmuir, FlowInlet,
                      PressureScaledLDF, RecordedPressure, Step, Valve, data)
from swingpy.plotting import plot_all
from swingpy.reporting import print_heading, print_table

FEED = {"H2": 0.75, "N2": 0.25}
FEED_FLOW = 0.03
"""FEED_FLOW: mol/s."""
HIGH, PRODUCT_PRESSURE, WASTE = 15.0, 14.997, 1.013
"""HIGH, PRODUCT_PRESSURE, WASTE: bar."""


def model(n_cells=30):
    adsorbent = Adsorbent(
        "molecular sieve",
        ExtendedLangmuir(henry=(0.015625, 0.44643), affinity=(0.0, 0.160), dU=(-7140.0, -21000.0),
                         reference_temperature=293.0, pressure_based=True),
        heat_of_adsorption=(7140.0, 21000.0), density=720.0 / (1.0 - 0.40), particle_radius=1.05e-3,
        particle_voidage=0.47, heat_capacity=840.0, adsorbed_phase_heat_capacity=29.0,
        kinetics=PressureScaledLDF((2.95, 2.95), reference_pressure=1.0))
    gas = data.gas(["H2", "N2"], FEED, temperature=298.15, pressure=HIGH)
    column = Column(length=2.0, inner_radius=0.0275, bed_voidage=0.40, pressure_drop="ergun")
    v0 = FEED_FLOW * 8.314 * 298.15 / (HIGH * 1e5) / (column.cross_section * column.bed_voidage)
    return ColumnModel(column, adsorbent, gas, n_cells=n_cells, reference_pressure=HIGH, reference_velocity=v0)


def cycle():
    product = dict(stream="product")
    receive_equalization = BufferInlet("equalization gas", stream="equalization")
    receive_equalization.record_as = "equalization pressure"
    receive_purge = BufferInlet("purge gas", stream="purge")
    receive_purge.record_as = "purge pressure"
    steps = [
        Step("FEED", 60.0, feed_end=FlowInlet(FEED_FLOW, FEED, 298.15),
             product_end=Valve(58.0, PRODUCT_PRESSURE, **product)),
        Step("FEED", 180.0, feed_end=FlowInlet(FEED_FLOW, FEED, 298.15),
             product_end=Valve(56.0, PRODUCT_PRESSURE, extra_outflow=0.008, record_as="product", **product)),
        Step("EQDN", 60.0,
             product_end=Valve(0.0057, RecordedPressure("equalization pressure", lambda t: 1.0 + 6.5 * t / 60.0),
                               stream="equalization", record_as="equalization gas")),
        Step("SUPP", 180.0,
             product_end=Valve(0.001, RecordedPressure("purge pressure", WASTE), stream="purge",
                               record_as="purge gas")),
        Step("BLOW", 60.0, feed_end=Valve(0.035, WASTE, stream="waste")),
        Step("PURG", 180.0, feed_end=Valve(0.6, WASTE, stream="waste"), product_end=receive_purge),
        Step("EQUP", 60.0, product_end=receive_equalization),
        Step("PRES", 180.0,
             product_end=FlowInlet(0.008, "product", 298.15, default_composition={"H2": 1.0},
                                   max_pressure=PRODUCT_PRESSURE, stream="product")),
    ]
    return Cycle(steps, key_component="H2")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-cycles", type=int, default=150)
    parser.add_argument("--cells", type=int, default=30)
    arguments = parser.parse_args()
    m, c = model(arguments.cells), cycle()
    run = c.run(m, m.initial_state(HIGH, {"H2": 1.0}), max_cycles=arguments.max_cycles, report_every=10)
    p = run.performance
    print_heading(f"Hydrogen PSA after {len(run.history)} cycles ({'steady state' if run.converged else 'not converged'})")
    print_table([("H2 purity", (p.purity, 3), "%"), ("H2 recovery", (p.recovery, 2), "%"),
                 ("Mass balance error", (p.mass_balance_error, 3), "%")], show_reference=False)
    plot_all(m, run.recent, run.history, "Hydrogen PSA")
