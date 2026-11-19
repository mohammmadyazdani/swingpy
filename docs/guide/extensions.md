# Valves, layers, dead volume, IAST and TSA

## Valves

`Valve(cv, pressure)` connects a column end to a reservoir with a linear valve, molar flow = cv × (upstream − downstream pressure), cv in mol/(s bar). The face pressure and the flow are solved together with the pressure drop over half a cell.

```python
Valve(0.035, 1.013, stream="waste")                                     # blowdown to the waste line
Valve(1.0, 3.0, direction="in", composition={"N2": 1.0}, stream="feed")   # filling from a 3 bar header
Valve(56.0, 14.997, extra_outflow=0.008, stream="product")               # product valve and a fixed side draw
```

The reservoir pressure can be a number, a function of time, or a `RecordedPressure`: the face pressure that another step recorded (`record_as`) in the previous cycle. This is how a bed equalizes with, or supplies purge to, another bed in a single-column simulation: the donor vents through its valve into the pressure history of the receiving step, the receiving step replays the donor's gas (`BufferInlet`), and at cyclic steady state the two are consistent.

A gas supplied from a line (a recorded product, a header) can be limited to the pressure of that line: `FlowInlet(..., max_pressure=15.0)`, `BufferInlet(..., max_pressure=...)`.

`examples/06 Hydrogen PSA.py` is a complete two-bed hydrogen PSA with pressure equalization driven by valves.

## Dead volume

`Column(dead_volume_feed=..., dead_volume_product=...)` adds well-mixed gas volumes (m<sup>3</sup>) at the ends of the bed: pipes, distributors, valve bodies. They hold gas but no adsorbent and slow the pressure and composition changes at that end; the mass and energy inventories include them.

## Layered beds

```python
from swingpy import LayeredBed
bed = LayeredBed([(carbon, 0.3), (zeolite, 0.7)])        # two Adsorbent objects of your own
model = ColumnModel(column, bed, gas)
```

Layers are listed from the feed end, as (adsorbent, length fraction) or (adsorbent, length fraction, bed voidage). Every layer keeps its own isotherm, kinetics, density, heat capacity, heats of adsorption, particle size and voidage. The balances use the superficial flux through every face and the voidage of every cell, and the permeability, inertial term and dispersion of every face follow from the particle size and voidage on both sides of it, so gas and heat are conserved exactly across layer boundaries.

Layered beds, IAST, Sips, Toth and any kinetics model run with the compiled (numba) backend: the isotherm slopes and rate constants are evaluated in Python and passed to the compiled transport and energy equations.

## Mixture equilibrium by IAST

`IAST([pure_1, pure_2, ...])` predicts mixture adsorption from single-component isotherms (any of `DualSiteLangmuir`, `ExtendedLangmuir`, `Sips`, `Toth` with one component), by solving the ideal adsorbed solution theory in every cell. For equal saturation capacities it reduces to the extended Langmuir isotherm, which SwingPy reproduces to machine precision.

## Extended Langmuir and pressure-dependent kinetics

`ExtendedLangmuir(henry, affinity, dU)` has separate capacity and affinity constants, so a component can adsorb in proportion to its concentration without taking up sites (affinity zero), as hydrogen on many adsorbents. `PressureScaledLDF(rates, reference_pressure)` gives rate constants inversely proportional to pressure, the behaviour of macropore molecular diffusion.

## Multisite Langmuir

`MultisiteLangmuir(qmax, a, K_inf, dH)` is the isotherm of Nitta et al. (1984) for mixtures, q_i/qmax_i = a_i K_i p_i (1 − Σ_k q_k/qmax_k)^a_i with K_i = K_inf,i exp(−dH_i/RT) (K_inf in 1/Pa, dH negative). A molecule of component i covers a_i sites, and a_i qmax_i must be the same for all components (checked). The loadings are implicit; the fraction of covered sites is found by a safeguarded Newton iteration, compiled with numba when available. It is the isotherm of the Rodrigues group for hydrogen purification on activated carbon and zeolites (Ribeiro et al., Chem. Eng. Sci. 63 (2008) 5258). In a specification: `model: multisite_langmuir` with `qmax`, `a`, `K_inf` (a unit of inverse pressure) and `dH` per component.

## Virial isotherm and series LDF

`Virial(K_inf, dH, A0, A1, B0, B1, specific_area)` is the multicomponent virial isotherm used by the Rodrigues group for activated carbon (Lopes et al., Chem. Eng. Sci. 66 (2011) 303; Fuel 93 (2012) 510, after Taqvi and LeVan 1997):

p_i = q_i/K_i · exp((2/S) Σ_j A_ij q_j + (3/(2S<sup>2</sup>)) Σ_j Σ_k B_ijk q_j q_k), with K_i = K_inf,i exp(−dH_i/RT), A = A0 + A1/T, B = B0 + B1/T, A_ij = (A_i + A_j)/2 and B_ijk = (B_i + B_j + B_k)/3.

S is the specific area of the adsorbent (m2/kg) and K_inf is in mol/(kg Pa) (the papers give mol/(kg bar): multiply by 10<sup>−5</sup>). The loadings are implicit and are found by Newton's method on ln q, compiled with numba when available. A negative B makes p(q) fold back at large loadings, so the equation has a second, unphysical root; the iteration starts below the physical one. Pure component loadings match the published equation and Fig. 2 of Lopes et al. (2012) (`tests/test_virial.py`). Outside the pressure range of the fit the fold can come within reach, so check the loadings you get at your highest pressure.

`SeriesLDF(micropore_rates, activation_energies, macropore_diffusivities, film_coefficient, micropore_factor, macropore_factor, area_factor)` is one LDF rate with the film, macropore and micropore resistances added in series (Glueckauf; Ruthven 1984): 1/k_i = K_i/(a_p k_f) + K_i R_p<sup>2</sup>/(Ω_m ε_p D_p,i) + r_c<sup>2</sup>/(Ω_c D_c,i), with K_i = ρ_p q_i*/c_i and D_c/r_c<sup>2</sup> = (D_c0/r_c<sup>2</sup>) exp(−E_a/RT). It stands in for the bi-LDF model of the same papers, which keeps the macropore gas and the micropore loading as separate states; that model is not in SwingPy. The series form is exact for a linear isotherm at long times and an approximation otherwise.

## Temperature swing

A step can set the temperature of the surroundings, `Step(..., ambient_temperature=423.15)`, which acts as a heating or cooling jacket through the wall heat transfer coefficients of the column. `ColumnModel.wall_heat(result)` gives the heat that crossed the wall in a step; `CyclePerformance.heat_by_step` lists it per step and `CyclePerformance.gas_heating_by_step` gives the heat needed to bring gas entering hotter than `Cycle(gas_supply_temperature=...)` (298.15 K by default) up to its inlet temperature, a hot purge or regeneration gas; `CyclePerformance.thermal_energy` adds both and gives the heat supplied per tonne of key component in the product (GJ/t). See `examples/07 Temperature swing.py`.

## Humid gas

`swingpy.humidity` gives the saturation pressure of water, relative humidity and mole fraction, the dew point, and `humid_composition(dry, relative_humidity, T, P)` for a feed. The data bank holds `zeolite-13x-humid`, a competitive dual-site Langmuir isotherm for CO<sub>2</sub>, N<sub>2</sub> and water on zeolite 13X: water takes up the sites CO<sub>2</sub> would use and pushes it off the adsorbent (`examples/08 Humid flue gas.py`).

Its water parameters are a fit to published isotherms below 60 % relative humidity (7 % RMS error). `AranovichDonohue(isotherm, component, d)` adds the multilayer rise of a vapour near saturation to any isotherm.

!!! tip "Order of the components"
    The model solves for every mole fraction but the last, which is one minus the others. Put the most abundant carrier gas last (N<sub>2</sub> in flue gas, H<sub>2</sub> in a hydrogen PSA). A trace component with a sharp front, such as water, must not be last: it would absorb the small errors of all the others and the integration would slow down badly.

Rate constants of the macropore LDF model are capped at `MacroporeLDF(max_rate=1000)` 1/s: a component that the solid barely holds (nitrogen next to adsorbed water) would otherwise get an enormous rate constant, which only means it is at equilibrium.

## Real gas behaviour

The column model treats the gas as ideal, as is usual for adsorption processes near atmospheric pressure. `swingpy.eos.PengRobinson` gives the compressibility factor and fugacity coefficients of a mixture (Peng and Robinson, *Ind. Eng. Chem. Fundam.* 15 (1976) 59), with critical constants from the reference equations of state and binary interaction parameters zero unless given. It agrees with the independent implementation in CoolProp to 10<sup>−10</sup>. Against the reference equations of state (GERG-2008 for mixtures), the ideal gas and Peng-Robinson compare as follows:

| Conditions | Z, reference | Error of the ideal gas | Error of Peng-Robinson |
|---|---|---|---|
| Flue gas near 1 bar (VSA, TSA) | 0.999 | 0.05 to 0.15 % | −0.03 % |
| CO<sub>2</sub>/H<sub>2</sub> 50/50, 25 °C, 15 bar | 0.983 | 1.7 % | −0.8 % |
| CO<sub>2</sub>/H<sub>2</sub> 50/50, 25 °C, 35 bar | 0.962 | 4.0 % | −1.8 % |
| CO<sub>2</sub>/H<sub>2</sub> 75/25, 25 °C, 25 bar | 0.927 | 7.8 % | −1.3 % |
| H<sub>2</sub>/CH<sub>4</sub> 70/30, 30 °C, 25 bar | 1.007 | 0.7 % | −1.0 % |

For CO<sub>2</sub> capture by vacuum or temperature swing near atmospheric pressure the ideal gas is exact to well within the other uncertainties of a model. At high pressure with much CO<sub>2</sub> it underestimates the gas density by several per cent; Peng-Robinson without interaction parameters halves that error but is not exact either.

This is also the practice of the reference work on pre-combustion capture: Casas et al. ([*Adsorption* 18 (2012) 143](https://doi.org/10.1007/s10450-012-9389-z)) simulate CO<sub>2</sub>/H<sub>2</sub> breakthroughs up to 35 bar with the ideal gas law and use Peng-Robinson only to check that the compressibility factor stays between 0.9 and 1. SwingPy does the same check for any simulation:

```python
from swingpy.analysis import compressibility_range
from swingpy.eos import PengRobinson

eos = PengRobinson(["CO2", "H2"])                       # or constants=[(Tc, pc, w), ...]
check = compressibility_range(model, run.results, eos, tolerance=0.05)
check["Z_min"], check["Z_max"], check["ideal"]          # extremes with step, time, position, p and T
report_run(model, run, "Output", eos=eos)               # adds an "Ideal gas check" section to Summary.md
```

In a specification the constants go in `gas.compressibility_check`, and `Problem.report` adds the check:

```yaml
gas:
  equation_of_state: ideal
  compressibility_check:
    model: peng_robinson
    source: Poling, Prausnitz and O'Connell, The Properties of Gases and Liquids, 5th ed., Appendix A
    critical_temperature: {CO2: 304.12 K, H2: 33.19 K}
    critical_pressure: {CO2: 73.74 bar, H2: 13.13 bar}
    acentric_factor: {CO2: 0.225, H2: -0.216}
    binary_interaction: {CO2-H2: 0.0}          # optional, zero by default
```

`PengRobinson` works on one composition or on arrays of points (compositions of shape components × points), and also gives the molar density and the residual enthalpy.
