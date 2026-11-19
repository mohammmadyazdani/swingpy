"""
Temperature swing adsorption of CO2 with a heating and cooling jacket.

A small column of zeolite 13X takes up CO2 from a 15 % CO2 flue gas at
25 C, is heated through its wall by a jacket at 150 C while the desorbed
CO2 leaves at the feed end at 1 bar, and is cooled back by the jacket at
25 C under nitrogen. The heat that crosses the wall is the main energy
demand of such a cycle: it is reported per tonne of CO2 recovered, with
the purity and recovery of the CO2 product.

    python "examples/07 Temperature swing.py" --max-cycles 30
"""

import argparse

import setups
from swingpy import Closed, Column, ColumnModel, Cycle, FeedInlet, PressureOutlet, PressurizationInlet, Step, data
from swingpy.plotting import plot_all
from swingpy.reporting import print_heading, print_table

FLUE = {"CO2": 0.15, "N2": 0.85}


def model(n_cells=20):
    column = Column(length=0.5, inner_radius=0.025, outer_radius=0.028, bed_voidage=0.37, inner_heat_transfer=60.0,
                    outer_heat_transfer=150.0, wall_density=7800.0, wall_heat_capacity=500.0,
                    ambient_temperature=298.15)
    return ColumnModel(column, setups.adsorbent("zeolite-13x"), data.gas(["CO2", "N2"], FLUE), n_cells=n_cells,
                       reference_velocity=0.1)


def cycle(t_ads=600.0, t_heat=1800.0, t_cool=1800.0, hot=423.15, cold=298.15):
    return Cycle([
        Step("FEED", t_ads, feed_end=FeedInlet(0.1, FLUE), product_end=PressureOutlet(1.0, stream="raffinate"),
             ambient_temperature=cold),
        Step("HEAT", t_heat, feed_end=PressureOutlet(1.0, stream="product"), product_end=Closed(),
             ambient_temperature=hot),
        Step("COOL", t_cool, feed_end=Closed(),
             product_end=PressurizationInlet(1.0, {"N2": 1.0}, stream="nitrogen"), ambient_temperature=cold),
    ], key_component="CO2")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-cycles", type=int, default=60)
    arguments = parser.parse_args()
    m, c = model(), cycle()
    run = c.run(m, m.initial_state(1.0, {"N2": 1.0}), max_cycles=arguments.max_cycles, report_every=5)
    p = run.performance
    print_heading(f"TSA after {len(run.history)} cycles ({'steady state' if run.converged else 'not converged'})")
    print_table([("CO2 purity", (p.purity, 2), "%"), ("CO2 recovery", (p.recovery, 2), "%"),
                 ("Heat through the wall", (p.thermal_energy, 2), "GJ/t CO2"),
                 ("Heat supplied in the heating step", (p.heat_by_step["HEAT"] / 1e3, 1), "kJ per cycle"),
                 ("Heat removed in the cooling step", (-p.heat_by_step["COOL"] / 1e3, 1), "kJ per cycle")],
                show_reference=False)
    plot_all(m, run.recent, run.history, "Temperature swing", positions=(0.0, 0.5, 1.0))
