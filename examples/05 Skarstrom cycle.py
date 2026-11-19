"""
Two-bed Skarstrom cycle in a laboratory column.

Pressurization, feed at 1.3 bar, blowdown and purge with nitrogen, for
15 % CO2 in N2 on zeolite 13X, at one of three operating points
(``setups.skarstrom.RUNS``). The CO2 product is the gas leaving
the feed end during blowdown and purge.

    python "examples/05 Skarstrom cycle.py" --run 2
"""

import argparse

from swingpy.plotting import plot_profiles
from setups import skarstrom
from swingpy.reporting import print_heading, print_table

parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
parser.add_argument("--run", type=int, default=1, choices=sorted(skarstrom.RUNS))
parser.add_argument("--max-cycles", type=int, default=200)
arguments = parser.parse_args()

model, cycle, state = skarstrom.example_case(arguments.run)
run = cycle.run(model, state, max_cycles=arguments.max_cycles, report_every=10)
p = run.performance

print_heading(f"Skarstrom cycle, run {arguments.run} "
              f"({len(run.history)} cycles{', cyclic steady state' if run.converged else ''})")
print_table([
    ("CO2 purity", (p.purity, 1), "%"),
    ("CO2 recovery", (p.recovery, 1), "%"),
], show_reference=False)
plot_profiles(model, run.results, f"Skarstrom cycle run {arguments.run} profiles.png")
