# Quick start

## A breakthrough

```python
from swingpy import (Adsorbent, Column, ColumnModel, DualSiteLangmuir, FeedInlet, MacroporeLDF, PressureOutlet,
                      Step, data)

gas = data.gas(["CO2", "N2"], {"CO2": 0.15, "N2": 0.85})
sorbent = Adsorbent(
    name="zeolite 13X",
    isotherm=DualSiteLangmuir(qsb=(3.09, 5.84), qsd=(2.54, 0.0), b0=(8.65e-7, 2.5e-6), d0=(2.63e-8, 0.0),
                              dUb=(-36641.0, -15800.0), dUd=(-35691.0, 0.0)),
    heat_of_adsorption=(36641.0, 15800.0), density=1130.0, particle_radius=1e-3, particle_voidage=0.35,
    tortuosity=3.0, heat_capacity=1070.0, adsorbed_phase_heat_capacity=30.7, kinetics=MacroporeLDF())
column = Column(length=1.0, inner_radius=0.1445, bed_voidage=0.37)
model = ColumnModel(column, sorbent, gas, n_cells=30)

feed = Step("FEED", 3000.0,
            feed_end=FeedInlet(1.0, {"CO2": 0.15, "N2": 0.85}),
            product_end=PressureOutlet(1.0))
result = model.simulate(feed, model.initial_state(1.0, {"N2": 1.0}), n_output=301)
exit_gas = model.end_history(result, "product")     # velocity, pressure, temperature, composition
```

## A four-step VSA cycle

```python
from swingpy import Cycle, ExponentialRamp, PressurizationInlet
from swingpy.plotting import plot_all

high, intermediate, low = 1.0, 0.1, 0.03
cycle = Cycle([
    Step("PRES", 15.0,
         feed_end=PressurizationInlet(ExponentialRamp(low, high, 0.5), {"CO2": 0.15, "N2": 0.85})),
    Step("FEED", 80.0, feed_end=FeedInlet(1.0, {"CO2": 0.15, "N2": 0.85}),
         product_end=PressureOutlet(high, stream="raffinate")),
    Step("BLOW", 30.0, product_end=PressureOutlet(ExponentialRamp(high, intermediate, 0.5),
                                                      stream="waste")),
    Step("EVAC", 40.0, feed_end=PressureOutlet(ExponentialRamp(intermediate, low, 0.5),
                                                     stream="product")),
], key_component="CO2")

run = cycle.run(model, model.initial_state(low, {"N2": 1.0}))
print(run.performance.purity, run.performance.recovery, run.performance.specific_energy)
plot_all(model, run.recent, run.history, "Four step VSA")      # every figure of the run
```

`run.performance` holds the purity, recovery, specific energy and productivity of the last cycle, `run.history` those of every cycle, and `run.recent` the full results of the last ten cycles.

## Examples

Each example runs from the repository root, for example `python "examples/02 Four step VSA.py"`, and writes its figures to the current folder.

| Example | What it shows |
|---|---|
| `01 Breakthrough.py` | a breakthrough of CO<sub>2</sub> through a column of zeolite 13X, with its mass and energy balances |
| `02 Four step VSA.py` | the four-step vacuum swing cycle for post-combustion capture, run to cyclic steady state |
| `03 Build your own cycle.py` | a tutorial: gas, adsorbent, column and steps built from scratch, run as a four-step VSA cycle |
| `04 Cycle configurations.py` | ready-made cycles with light product pressurization and pressure equalization |
