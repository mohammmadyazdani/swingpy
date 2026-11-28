# Gas properties and adsorbents

## Pure-component gas properties

`swingpy.data` holds pure-component properties of common gases (CO<sub>2</sub>, N<sub>2</sub>, O<sub>2</sub>, Ar, CH<sub>4</sub>, CO, He): molar mass, ideal-gas heat capacity and viscosity at 298.15 K, and Fuller diffusion volume. `data.gas_sources()` names the source of each kind of property. Your own components are added with `data.register_gas`.

```python
from swingpy import data

data.list_gases()
data.gas_properties("CO2")
```

## Adsorbents

SwingPy does not ship adsorbents: the isotherm, heats of adsorption and particle properties of a material come from you, in Python or in a specification file (see [Problem specifications](specification.md)).

```python
from swingpy import Adsorbent, DualSiteLangmuir, MacroporeLDF

sorbent = Adsorbent(
    name="my zeolite",
    isotherm=DualSiteLangmuir(qsb=(3.09, 5.84), qsd=(2.54, 0.0), b0=(8.65e-7, 2.5e-6), d0=(2.63e-8, 0.0),
                              dUb=(-36641.0, -15800.0), dUd=(-35691.0, 0.0)),
    heat_of_adsorption=(36641.0, 15800.0), density=1130.0, particle_radius=1e-3, particle_voidage=0.35,
    tortuosity=3.0, heat_capacity=1070.0, adsorbed_phase_heat_capacity=30.7, kinetics=MacroporeLDF())
```

The examples use adsorbents reported in the literature, each with its sources: `examples/setups/adsorbents.json`, read by `examples/setups/adsorbents.py` (`import setups; setups.adsorbent("zeolite-13x")` from the examples folder).

## Gas mixtures

```python
gas = data.gas(["CO2", "N2"], {"CO2": 0.15, "N2": 0.85}, temperature=298.15, pressure=1.0)
```

The heat capacity is the mole-fraction average, the viscosity follows Wilke's mixing rule and the molecular diffusivity the method of Fuller, Schettler and Giddings for the first two components. They are constants of the model: evaluate them at the feed conditions. `data.gas_properties("CO2")` returns the stored properties of a pure gas.

## Your own gases

```python
data.register_gas("C2H6", molar_mass=30.07e-3, heat_capacity=52.5, viscosity=0.93e-5, diffusion_volume=44.88)
```
