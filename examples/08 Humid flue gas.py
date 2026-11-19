"""
Humid flue gas on zeolite 13X: water pushes CO2 off the adsorbent.

Zeolite 13X takes up water far more strongly than CO2, so even a few per
cent of water vapour in the feed matters. This example feeds a flue gas
with 15 % CO2 (dry basis) at 50 % relative humidity and 40 C to a column of
zeolite 13X and follows the water and CO2 fronts.

Equilibrium is the competitive dual-site Langmuir isotherm of the
"zeolite-13x-humid" entry of the data bank: the CO2 and N2 parameters of
the other zeolite 13X entry and, for water, a fit to the isotherms measured
by Son, Richardson and Cmarik (J. Chem. Eng. Data 64 (2019) 1063) below
60 % relative humidity. Water takes up sites that CO2 would otherwise use,
so it pushes CO2 off the adsorbent where it arrives.

    python "examples/08 Humid flue gas.py"
"""

import matplotlib
import numpy as np

import setups
from swingpy import Column, ColumnModel, FeedInlet, PressureOutlet, Step, data, humidity
from swingpy.plotting import apply_style, label_panels, save_figure, series_style

matplotlib.use("Agg")
TEMPERATURE, PRESSURE = 313.15, 1.0
FEED = humidity.humid_composition({"CO2": 0.15, "N2": 0.85}, 0.5, TEMPERATURE, PRESSURE)


def model(n_cells=20):
    """
    The last component is the one the model does not solve for (its mole fraction is one minus the others): it should be
    the abundant carrier gas, never a trace component with a sharp front such as water.
    """
    components = ["CO2", "H2O", "N2"]
    gas = data.gas(components, FEED, temperature=TEMPERATURE, pressure=PRESSURE)
    column = Column(length=1.0, inner_radius=0.1445, bed_voidage=0.37, inner_heat_transfer=0.0,
                    outer_heat_transfer=0.0, ambient_temperature=TEMPERATURE)
    return ColumnModel(column, setups.adsorbent("zeolite-13x-humid", components=components), gas, n_cells=n_cells,
                       reference_temperature=TEMPERATURE)


if __name__ == "__main__":
    m = model()
    step = Step("FEED", 4000.0, feed_end=FeedInlet(1.0, FEED, TEMPERATURE), product_end=PressureOutlet(PRESSURE))
    result = m.simulate(step, m.initial_state(PRESSURE, {"N2": 1.0}, TEMPERATURE), n_output=61)
    print(f"Feed: {FEED['H2O'] * 100:.2f} % water, {FEED['CO2'] * 100:.2f} % CO2 (50 % relative humidity at 40 C)")
    z = (np.arange(m.n_cells) + 0.5) * m.cell_width
    apply_style()
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(1, 2, figsize=(11.0, 4.0))
    for k, t in enumerate((500.0, 1500.0, 4000.0)):
        state = result.states[:, np.argmin(np.abs(result.time - t))]
        q = m.unpack(state)["x"] * m.qs0
        style = dict(series_style(k, len(z)), label=f"{t:.0f} s")
        axes[0].plot(z, q[0], **style)
        axes[1].plot(z, q[1], **style)
    axes[0].set_ylabel("CO$_2$ loading [mol/kg]")
    axes[1].set_ylabel("H$_2$O loading [mol/kg]")
    for axis in axes:
        axis.set_xlabel("Z [-]")
        axis.set_xlim(0, 1)
        axis.legend()
    label_panels(axes)
    figure.tight_layout()
    save_figure(figure, "Humid flue gas fronts.png")
    q_end = m.unpack(result.final_state)["x"] * m.qs0
    print(f"After {step.duration:.0f} s: water loading at the inlet {q_end[1, 0]:.1f} mol/kg, CO2 there "
          f"{q_end[0, 0]:.2f} mol/kg against {q_end[0, -1]:.2f} mol/kg near the outlet")
