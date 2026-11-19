"""
Four-step vacuum swing adsorption cycle for post-combustion CO2 capture.

Pressurization with feed, adsorption at 1 bar, co-current blowdown to an
intermediate pressure and counter-current evacuation, on zeolite 13X with
flue gas of 15 % CO2. The cycle is run to cyclic steady state; the script
prints purity, recovery, energy and productivity, the vacuum work of each
step, and plots the column profiles and the approach to steady state.

    python "examples/02 Four step VSA.py"
    python "examples/02 Four step VSA.py" --low-pressure 0.03 --max-cycles 400
"""

import argparse

import setups as presets
from swingpy.plotting import plot_convergence, plot_profiles
from swingpy.reporting import print_heading, print_table

parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
parser.add_argument("--low-pressure", type=float, default=0.1, help="evacuation pressure, bar")
parser.add_argument("--max-cycles", type=int, default=600)
arguments = parser.parse_args()

model = presets.model()
cycle = presets.four_step_vsa(low_pressure=arguments.low_pressure)
run = cycle.run(model, model.initial_state(arguments.low_pressure, {"N2": 1.0}), max_cycles=arguments.max_cycles,
                report_every=50)
p = run.performance

print_heading(f"Four-step VSA at {'cyclic steady state' if run.converged else 'the last cycle'} "
              f"({len(run.history)} cycles)")
print_table([
    ("CO2 purity", (p.purity, 2), "%"),
    ("CO2 recovery", (p.recovery, 2), "%"),
    ("Specific energy", (p.specific_energy, 1), "kWh/t CO2"),
    ("Productivity", (p.productivity, 3), "mol/m3/s"),
], show_reference=False)
print_heading("Work of each step")
print_table([(name, f"{work / 1000:.2f}", "kJ/cycle") for name, work in p.energy_by_step.items()],
            show_reference=False)

plot_profiles(model, run.results, "Four step VSA profiles.png")
plot_convergence(run.history, "Four step VSA convergence.png", tolerance=0.5, profile_tolerance=1e-3)
