# Problem specifications

A whole problem can be written as a YAML or JSON file (or a Python dict):
components, gas, adsorbent, bed, model options, streams, initial state and
cycle. `load_problem` checks it, reports every error at once with its path,
and builds the model and the cycle.

```python
from swingpy.spec import load_problem

problem = load_problem("examples/specs/VSA zeolite 13X flue gas.yaml")
run = problem.run(max_cycles=300)
print(run.performance.purity, run.performance.recovery)
```

## Principles

- **Every physical quantity comes from the specification.** Nothing about the gas, the adsorbent, the bed, the machines or the feed has a hidden default; a missing value is an error.
- **Units are explicit.** A value is `"<number> <unit>"` (`"1.2 m"`, `"25 C"`, `"36.6 kJ/mol"`) or a plain number in the base unit of its kind (`swingpy.spec.BASE_UNITS`). A unit of the wrong kind is an error.
- **Only numerical options have defaults:** `model.cells` (30), `model.scheme` (`weno`), `model.backend` (`auto`), `model.axial_dispersion` (`correlation`), `bed.include_pore_gas` (false), `bed.dead_volume` (none), `kinetics.max_rate` (1000 1/s) and `kinetics.shells` (12).

## Layout

| Section | Contents |
|---|---|
| `components` | Names, two or more. Any names: the data bank is not consulted. |
| `gas` | `equation_of_state` (`ideal`), `molar_masses` per component, `heat_capacity`, `viscosity`, `molecular_diffusivity`, `adiabatic_constant`. |
| `adsorbent` | `particle_density`, `particle_radius`, `particle_voidage`, `tortuosity`, `heat_capacity`, `adsorbed_phase_heat_capacity`, `heat_of_adsorption` per component, `isotherm`, `kinetics`. |
| `bed` | `length`, `inner_radius`, `outer_radius`, `voidage`, `pressure_drop` (`darcy`, `ergun`), `axial_thermal_conductivity`, `wall` (density, heat capacity, thermal conductivity), `heat_transfer` (inner, outer, ambient temperature), optional `dead_volume` and `include_pore_gas`. |
| `model` | `reference` (pressure, temperature, velocity used to scale the equations), numerical options. |
| `streams` | Named gas streams: `composition` per component (adding up to 1) and `temperature`. |
| `initial` | `pressure`, `temperature`, `composition` of the column at the start. |
| `cycle` | `key_component`, optional `machines` (efficiency, supply and discharge pressures every machine inherits unless it gives its own), `steps`. |

### Isotherms

`model` is one of `langmuir`, `dual_site_langmuir`, `sips`, `toth`, or `python` (with `object: "module:function"` for an isotherm object of your own), followed by one block of parameters per component. For `sips` and `toth` the unit of `b0` sets the basis: m3/mol for concentrations, 1/bar (or 1/Pa, 1/kPa) for partial pressures; it must be the same for every component. The temperature dependence of the Sips exponent is n0 + alpha (1 − T_ref/T) by default; `exponent_form: linear` (at the level of `model`) selects n0 + alpha (T/T_ref − 1), the form of Do used by Hefti et al. (2015). For `virial`, give `specific_area` (m2/kg or m2/g) at the level of `model`, and per component `K_inf` (mol/(kg bar), as the papers give it), `dH` (negative), `A0` (m2/mol), `A1` (m2 K/mol), `B0` (m4/mol2) and `B1` (m4 K/mol2).

### Kinetics

`macropore_ldf` (rate from the particle properties), `constant_ldf` and `pressure_scaled_ldf` (with `rates` per component), `particle_diffusion` (with `rates` D/r<sup>2</sup> per component and `shells`), or `series_ldf` (with `micropore_rates` D_c0/r_c<sup>2</sup>, `activation_energies` and `macropore_diffusivities` per component, and `film_coefficient`, `micropore_factor`, `macropore_factor`, `area_factor`; see `SeriesLDF`).

### Steps

Each step has a `name`, and either a `duration` or `until` (`pressure`, `at`: feed, product or mean, `direction`: rising or falling) with `max_duration`. Each end (`feed_end`, `product_end`, closed if omitted) is one of:

| `type` | Parameters |
|---|---|
| `closed` | |
| `feed` | `stream`, `velocity`, `supply_pressure`, `efficiency` |
| `pressure_inlet` | `stream`, `pressure` (a value or `{ramp: exponential, from, to, rate}`), `supply_pressure`, `efficiency`, optional `record` (the mean gas of a recorded stream; `stream` is used until it exists) |
| `outlet` | `pressure` (value or ramp), `label`, `discharge_pressure`, `efficiency`, optional `record_as`, `check_valve` |
| `vacuum_pump` | `volumetric_flow` or `velocity`, `label`, `discharge_pressure`, `efficiency` (a number or `falling_vacuum`), optional `record_as` |
| `replay` | `record` (a stream recorded with `record_as`), `label`, optional `offset`, `max_pressure`, `stream` (replay the recorded flow with this gas instead) |
| `tank` | `record`, `fraction`, `temperature`, `label`, `supply_pressure`, `efficiency` |
| `mixing_inlet` | `stream` (the fresh gas), `molar_flow`, `record` (the recycled stream), optional `fraction`, `label` (fresh, `feed` by default), `recycle_label` (`recycle` by default), `supply_pressure`, `efficiency` |

`label` names the stream for the performance indicators (`product`, `raffinate`, ...); `cycle.product_stream` (default `product`) is the one purity and recovery refer to.

## Saving a problem built in Python

`to_spec(model, cycle, initial)` returns the specification of a problem built in Python, and `save_problem(path, model, cycle, initial, header=...)` writes it as YAML; `load_problem` builds an identical model and cycle from it. What a specification cannot express yet (layered beds, valves, functions of your own) is named in the error. The files in `examples/specs` were written this way from the cases they describe, and each is checked to give the same model, step by step.

```python
from swingpy.spec import save_problem
save_problem("My case.yaml", model, cycle, dict(pressure=0.1, temperature=298.15, composition={"N2": 1.0}))
```

Further boundary types: `flow_inlet` (a fixed `molar_flow`, in mol/s, kmol/h or SLPM, from a `stream` or from a `record` with a `stream` used until the record exists; optional `max_pressure`). `discharge_pressure: none` or `supply_pressure: none` marks a stream without a machine (no work counted). For Toth isotherms, `competitive: false` treats the components independently; `gas.adiabatic_constant: linear_co2` uses gamma = 1.395 - 0.105 y_CO2.

## Errors

```
the specification has 3 error(s):
  adsorbent.heat_capacity: missing
  bed.length: unit 'bar' is not a length unit (that is a pressure unit); use one of m, cm, mm, um, in, ft
  cycle.steps[1].feed_end.stream: no stream 'syngas' in the streams section
```
