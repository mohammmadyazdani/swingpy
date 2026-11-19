"""
Cycle configurations for post-combustion CO2 capture on zeolite 13X.

Runs one of the ready operating points of ``setups.cycles`` to
cyclic steady state: the basic four-step cycle, four steps with light
product pressurization (LPPR), or five steps with pressure equalization.
In the cycles with light product pressurization the gas leaving the feed step is recorded and
replayed to pressurize the column from its light end in the next cycle.

    python "examples/04 Cycle configurations.py" --list
    python "examples/04 Cycle configurations.py" --case "four step with light product pressurization, high purity"
"""

import argparse

from swingpy.plotting import plot_convergence, plot_profiles
from setups import cycles
from swingpy.reporting import print_heading, print_table

parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
parser.add_argument("--case", default="four step with light product pressurization, high purity",
                    choices=list(cycles.EXAMPLE_POINTS))
parser.add_argument("--list", action="store_true", help="list the ready operating points")
parser.add_argument("--max-cycles", type=int, default=800)
arguments = parser.parse_args()

if arguments.list:
    for name, point in cycles.EXAMPLE_POINTS.items():
        print(f"{name}: {point['kwargs']}")
    raise SystemExit

model, cycle, state = cycles.example_case(arguments.case)
run = cycle.run(model, state, max_cycles=arguments.max_cycles, report_every=50)
p = run.performance

print_heading(f"{arguments.case} at {'cyclic steady state' if run.converged else 'the last cycle'} "
              f"({len(run.history)} cycles)")
print_table([
    ("CO2 purity", (p.purity, 2), "%"),
    ("CO2 recovery", (p.recovery, 2), "%"),
    ("Specific energy", (p.specific_energy, 1), "kWh/t CO2"),
], show_reference=False)

tag = arguments.case.replace(",", "").replace("%", "percent").capitalize()
plot_profiles(model, run.results, f"{tag} profiles.png")
plot_convergence(run.history, f"{tag} convergence.png", tolerance=0.5, profile_tolerance=1e-3)
