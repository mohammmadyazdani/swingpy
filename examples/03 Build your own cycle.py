"""
Tutorial: build a model and a cycle from scratch.

This example shows the four objects a simulation needs and how a cycle is
assembled from boundary conditions, without using a ready-made case.

1. GasMixture: which components flow, and their properties.
2. Adsorbent: isotherm, heat of adsorption, particle properties.
3. Column: geometry and heat transfer.
4. Step: a duration and one boundary condition at each end.

The cycle built here is a vacuum swing cycle with a heated evacuation: the
wall is held at 340 K while the CO2 product is pumped out, which shows how a
step can change the column's surroundings (a first step towards temperature
swing operation).

    python "examples/03 Build your own cycle.py"
"""

from swingpy import (Adsorbent, Closed, Column, ColumnModel, Cycle, DualSiteLangmuir, ExponentialRamp,
                      FeedInlet, GasMixture, PressureOutlet, PressurizationInlet, Step)
from swingpy.reporting import print_heading, print_table

"""1. The gas. The last component is obtained by closure, so list the carrier gas last."""
gas = GasMixture(components=("CO2", "N2"), heat_capacity=30.7, viscosity=1.72e-5,
                 molecular_diffusivity=1.6e-5)

"""
2. The adsorbent. One isotherm parameter per component, in the order of the gas. DualSiteLangmuir can be swapped for
Sips or Toth, for example Toth(qs=(5.6, 5.8), b0=(8.0e-7, 2.5e-6), dU=(-36600.0, -15800.0), t=(0.6, 1.0)).
"""
"""The heat of adsorption is -dH in J/mol, one value per component."""
adsorbent = Adsorbent(
    name="My zeolite",
    isotherm=DualSiteLangmuir(qsb=(3.09, 5.84), qsd=(2.54, 0.0), b0=(8.65e-7, 2.50e-6), d0=(2.63e-8, 0.0),
                              dUb=(-36641.21, -1.58e4), dUd=(-35690.66, 0.0)),
    heat_of_adsorption=(36641.21, 1.58e4),
    density=1130.0, particle_radius=1.0e-3,
)

"""3. The column. Heat transfer through the wall matters here, because the evacuation step heats it."""
column = Column(length=1.0, inner_radius=0.1445, outer_radius=0.1620, inner_heat_transfer=8.6,
                outer_heat_transfer=2.5)
model = ColumnModel(column, adsorbent, gas, n_cells=30, scheme="weno")

"""
4. The steps. Each end takes a boundary condition; the stream labels tell the cycle which flows are feed and product.
"""
feed = {"CO2": 0.15, "N2": 0.85}
steps = [
    Step("PRES", 15.0, feed_end=PressurizationInlet(ExponentialRamp(0.1, 1.0, rate=0.5), feed)),
    Step("FEED", 20.0, feed_end=FeedInlet(velocity=0.5, composition=feed),
         product_end=PressureOutlet(1.0, stream="raffinate")),
    Step("BLOW", 30.0, product_end=PressureOutlet(ExponentialRamp(1.0, 0.2), stream="blowdown")),
    Step("HEVA", 40.0, feed_end=PressureOutlet(ExponentialRamp(0.2, 0.1), stream="product"),
         product_end=Closed(), ambient_temperature=340.0),
]
cycle = Cycle(steps, key_component="CO2")

"""A few cycles only, to keep the tutorial fast. Use cycle.run(...) to reach cyclic steady state."""
state = model.initial_state(pressure=0.1, composition={"N2": 1.0})
for number in range(1, 4):
    results = cycle.run_once(model, state)
    state = results[-1].final_state
    performance = cycle.evaluate(model, results)

print_heading("Heated four-step cycle after 3 cycles (far from steady state)")
print_table([
    ("CO2 purity", (performance.purity, 2), "%"),
    ("CO2 recovery", (performance.recovery, 2), "%"),
    ("Specific energy", (performance.specific_energy, 1), "kWh/t CO2"),
    ("Wall temperature at the end", (model.unpack(state)["T_wall"].mean() * model.T0, 1), "K"),
], show_reference=False)
