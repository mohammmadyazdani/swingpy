# Valves, layers, dead volume and IAST

## Valves

`Valve(cv, pressure)` connects a column end to a reservoir with a linear valve, molar flow = cv × (upstream − downstream pressure), cv in mol/(s bar). The face pressure and the flow are solved together with the pressure drop over half a cell.

```python
Valve(0.035, 1.013, stream="waste")                                     # blowdown to the waste line
Valve(1.0, 3.0, direction="in", composition={"N2": 1.0}, stream="feed")   # filling from a 3 bar header
Valve(56.0, 14.997, extra_outflow=0.008, stream="product")               # product valve and a fixed side draw
```

The reservoir pressure can be a number, a function of time, or a `RecordedPressure`: the face pressure that another step recorded (`record_as`) in the previous cycle. This is how a bed equalizes with, or supplies purge to, another bed in a single-column simulation: the donor vents through its valve into the pressure history of the receiving step, the receiving step replays the donor's gas (`BufferInlet`), and at cyclic steady state the two are consistent.

A gas supplied from a line (a recorded product, a header) can be limited to the pressure of that line: `FlowInlet(..., max_pressure=15.0)`, `BufferInlet(..., max_pressure=...)`.

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

`ExtendedLangmuir(henry, affinity, dU)` has separate capacity and affinity constants, so a component can adsorb in proportion to its concentration without taking up sites (affinity zero), as a weakly adsorbed gas in its linear range. `PressureScaledLDF(rates, reference_pressure)` gives rate constants inversely proportional to pressure, the behaviour of macropore molecular diffusion.

## Resistances in series

`SeriesLDF(micropore_rates, activation_energies, macropore_diffusivities, film_coefficient, micropore_factor, macropore_factor, area_factor)` is one LDF rate with the film, macropore and micropore resistances added in series (Glueckauf; Ruthven 1984): 1/k_i = K_i/(a_p k_f) + K_i R_p<sup>2</sup>/(Ω_m ε_p D_p,i) + r_c<sup>2</sup>/(Ω_c D_c,i), with K_i = ρ_p q_i*/c_i and D_c/r_c<sup>2</sup> = (D_c0/r_c<sup>2</sup>) exp(−E_a/RT). It stands in for the bi-LDF model of the same papers, which keeps the macropore gas and the micropore loading as separate states; that model is not in SwingPy. The series form is exact for a linear isotherm at long times and an approximation otherwise.

## Heat through the wall

`ColumnModel.wall_heat(result)` gives the heat that crossed the wall in a step, from the outer heat transfer coefficient and the wall temperatures over time; `CyclePerformance.heat_by_step` lists it per step. An insulated column (`Column(outer_heat_transfer=0.0)`) exchanges none.

!!! tip "Order of the components"
    The model solves for every mole fraction but the last, which is one minus the others. Put the most abundant carrier gas last (N<sub>2</sub> in flue gas). A trace component with a sharp front must not be last: it would absorb the small errors of all the others and the integration would slow down badly.

Rate constants of the macropore LDF model are capped at `MacroporeLDF(max_rate=1000)` 1/s: a component that the solid barely holds would otherwise get an enormous rate constant, which only means it is at equilibrium.

## Ideal gas

The column model treats the gas as ideal, as is usual for pressure and vacuum swing processes near atmospheric pressure: for flue gas near 1 bar the compressibility factor is 0.999, an error of 0.05 to 0.15 % in the gas density, well within the other uncertainties of a model. At high pressure with much CO<sub>2</sub> the ideal gas underestimates the density by several per cent; real gas behaviour is one of the possible extensions of SwingPy.
