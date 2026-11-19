"""
Dynamic column breakthrough.

A 1 m column of zeolite 13X, initially full of N2 at 1 bar, is fed with
15 % CO2 in N2 at 1 m/s. The script reports when the CO2 and heat fronts
reach the exit, checks the mass and energy balances, and plots the exit
velocity, temperature and composition.

    python "examples/01 Breakthrough.py"
"""

import time

import matplotlib.pyplot as plt
import numpy as np

import setups as h
from swingpy.plotting import apply_style, label_panels, save_figure, single_style
from swingpy.reporting import print_heading, print_table


def crossing_time(t, values, level, start=0, falling=False):
    for k in range(start + 1, len(values)):
        a, b = values[k - 1], values[k]
        if (a > level >= b) if falling else (a < level <= b):
            return t[k - 1] + (level - a) * (t[k] - t[k - 1]) / (b - a)
    return float("nan")


"""An adiabatic column, 30 cells, WENO reconstruction, and a feed step."""
model = h.model(adiabatic=True)
step = h.breakthrough_step(duration=8000.0)
start = time.time()
result = model.simulate(step, model.initial_state(pressure=1.0, composition={"N2": 1.0}), n_output=801)
wall_time = time.time() - start

"""Exit histories: velocity in m/s (v0 = 1 m/s), temperature in K, mole fractions."""
exit_ = model.end_history(result, end="product")
t, y, T, v = result.time, exit_["mole_fractions"][0], exit_["temperature"] / 298.15, exit_["velocity"]
co2_front = crossing_time(t, y, 0.075)
peak = int(np.argmax(T))
heat_front = crossing_time(t, T, 0.5 * (T[peak] + T[-1]), start=peak, falling=True)
plateau = int(np.argmin(np.abs(t - 0.5 * (co2_front + heat_front))))

"""Balances over the whole run."""
co2 = model.gas.index("CO2")
stored = model.component_inventory(result.final_state)[co2] - model.component_inventory(result.initial_state)[co2]
entered = sum(result.moles_in[end][co2] for end in ("feed", "product"))
heat_stored = model.enthalpy_inventory(result.final_state) - model.enthalpy_inventory(result.initial_state)
heat_in = sum(result.enthalpy_in.values()) + model.heat_of_adsorption_released(result.initial_state, result.final_state)

print_heading(f"Dynamic column breakthrough ({model.n_cells} cells, {model.scheme}, {wall_time:.1f} s)")
print_table([
    ("CO2 front at the exit", (co2_front, 1), "s"),
    ("Heat front at the exit", (heat_front, 0), "s"),
    ("Exit temperature on the plateau", (T[plateau], 3), "T/T0"),
    ("Exit CO2 mole fraction on the plateau", (y[plateau], 3), "-"),
    ("CO2 mass balance error", 100 * abs(entered - stored) / stored, "%"),
    ("Energy balance error", 100 * abs(heat_in - heat_stored) / abs(heat_stored), "%"),
], show_reference=False)

apply_style()
figure, axes = plt.subplots(3, 1, figsize=(6.0, 6.8), sharex=True)
for axis, values, label in zip(axes, (v, T, y), (r"$v/v_0$ [-]", r"$T/T_0$ [-]", r"$y_{\mathrm{CO_2}}$ [-]")):
    axis.plot(t, values, **single_style())
    axis.set_ylabel(label)
    axis.set_xlim(t[0], t[-1])
axes[-1].set_xlabel("Time [s]")
label_panels(axes)
figure.align_ylabels(axes)
figure.tight_layout()
save_figure(figure, "Breakthrough histories.png")
