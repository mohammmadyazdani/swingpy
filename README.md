# SwingPy

**Finite volume simulation of pressure, vacuum and vacuum-pressure swing adsorption cycles (PSA, VSA, VPSA) in Python.**

[![tests](https://github.com/mohammmadyazdani/swingpy/actions/workflows/tests.yml/badge.svg)](https://github.com/mohammmadyazdani/swingpy/actions/workflows/tests.yml)
[![docs](https://github.com/mohammmadyazdani/swingpy/actions/workflows/docs.yml/badge.svg)](https://mohammmadyazdani.github.io/swingpy)

Developed by **Mohammad Yazdani** (Chemical Engineering, University of Tehran) · [ORCID 0009-0007-4661-957X](https://orcid.org/0009-0007-4661-957X) · [GitHub](https://github.com/mohammmadyazdani) · Documentation: https://mohammmadyazdani.github.io/swingpy

SwingPy simulates a packed adsorption column through any pressure swing cycle: feed, pressurization, blowdown, evacuation, purge, reflux and pressure equalization, from above atmospheric pressure (PSA) to vacuum (VSA) or both (VPSA). The model is nonisothermal and nonisobaric and handles any number of gas components. It takes a simulation from a single breakthrough to a full cycle at cyclic steady state, with its purity, recovery, energy and productivity. Its results are compared with published simulations and experiments (see [Comparison with published work](#comparison-with-published-work)). Every module is written to be read: equations are named after the articles they come from, and the examples double as tutorials.

![Purity and recovery predicted by SwingPy against the published values](docs/figures/Published%20comparison.png)

*Purity and recovery predicted by SwingPy against the values reported for VSA cycles of CO<sub>2</sub> capture (the article's model where it has one, the pilot plant or laboratory experiment otherwise). The grey band is ±3 points. The points outside it are the cases listed under [Comparison with published work](#comparison-with-published-work) as partial or not reproduced.*

## Install

```bash
pip install "swingpy[fast,plot] @ git+https://github.com/mohammmadyazdani/swingpy.git"
```

The extras are optional: `fast` adds numba (about 2.5 times faster), and `plot` adds matplotlib. To work on the code and run the tests:

```bash
git clone https://github.com/mohammmadyazdani/swingpy.git
cd swingpy
pip install -e ".[plot,test,fast]"
pytest -q
```

## Thirty-second tour

```python
from swingpy.spec import load_problem

problem = load_problem("examples/specs/VSA zeolite 13X flue gas.yaml")   # every input given, with units
run = problem.run(max_cycles=300)

print(run.performance.purity, run.performance.recovery, run.performance.specific_energy)
```

## How a simulation is put together

A simulation needs four kinds of objects. Each one is a small, documented class you can create yourself.

| Object | Describes | Example |
|---|---|---|
| `GasMixture` | Components and gas properties | `GasMixture(("CO2", "N2"))` |
| `Adsorbent` | Isotherm, heat of adsorption, particles, kinetics | `Adsorbent("13X", DualSiteLangmuir(...), ...)` |
| `Column` | Geometry, voidage, heat transfer, pressure drop model | `Column(length=1.0, inner_radius=0.1445)` |
| `Step` | A duration and a boundary condition at each end | `Step("BLOW", 30, product_end=PressureOutlet(...))` |

The key idea is that **a step is two boundary conditions**, one per column end. The same few classes cover every common step:

| Step | Feed end (Z = 0) | Product end (Z = 1) |
|---|---|---|
| Feed pressurization | `PressurizationInlet` | `Closed` |
| Adsorption | `FeedInlet` | `PressureOutlet` (back pressure) |
| Co-current blowdown | `Closed` | `PressureOutlet` (vacuum ramp) |
| Counter-current evacuation | `PressureOutlet` (vacuum ramp) | `Closed` |
| Light product pressurization | `Closed` | `BufferInlet` (raffinate record), stops at P_H |
| Light reflux (purge) | `PressureOutlet` (vacuum) | `BufferInlet` (raffinate record) |
| Heavy reflux | `TankInlet` (fraction of the extract) | `PressureOutlet` (vacuum) |
| Feed or purge from a mass flow controller | `FlowInlet` (mol/s, or `standard_litres_per_minute`) | either end |
| Pressure equalization, donor | `Closed` | `PressureOutlet` (no pump), recorded |
| Pressure equalization, receiver | `Closed` | `BufferInlet` (donor record) |

Steps that exchange gas are coupled through **recorded streams**: an outlet with `record_as="raffinate"` stores its flow, composition and temperature while it runs, and a `BufferInlet("raffinate")` or `TankInlet` replays it in a later step (the single-column "unibed" approach). A step can also end on a condition instead of a fixed time, for example `stop=PressureReached(1.0)` to pressurize until the column reaches 1 bar.

Each boundary carries a stream label (`"feed"`, `"product"`, `"raffinate"`, ...) and the machine that moves the gas (a compressor from a supply pressure, or a vacuum pump to a discharge pressure). The `Cycle` uses these to compute purity, recovery, specific energy and productivity for any cycle, without cycle-specific code.

See `examples/03 Build your own cycle.py` for a complete walk-through.

## Outputs and figures

`swingpy.report` writes everything a simulation produces to one folder: the numbers as CSV files (`swingpy.analysis`), the standard figures of adsorption processes (`swingpy.plotting`) and a readable `Summary.md`:

```python
from swingpy.report import report_breakthrough, report_run

report_breakthrough(model, result, "Output/Breakthrough")   # metrics, histories, fronts, isotherms
report_run(model, run, "Output/Cycle")                      # KPIs, step and stream tables, profiles, cycle diagram, ...
```

The figures cover equilibrium (isotherms), breakthrough (outlet curves with breakthrough times, temperatures, moving fronts) and the cycle at steady state (profiles at the end of every step, pressure diagram, outlet flows and compositions, step balances, heat maps, thermocouple histories, convergence, energy). See the gallery in the documentation (`docs/guide/figures.md`).

## Cyclic steady state

`Cycle.run` declares cyclic steady state when its test passes for five cycles in a row. Two tests are available and by default both must pass (`criterion="both"`):

- **mass balance**: the key component balance over a cycle closes to within `tolerance` (%), the rule used by Haghpanah et al. (2013) and Subraveti et al. (2021);
- **profiles**: no profile of the column changes by more than `profile_tolerance` over a cycle (dimensionless; 1e-3 is about 0.3 K or 0.1 % of the reference pressure).

The mass balance alone can close while the column is still slowly warming or loading. For IISERP MOF2 the profile change stays near 7e-3 for about 80 cycles before it falls, while the product end is still warming. Use `criterion="mass_balance"` to apply exactly the rule of an article; the examples report the cycle at which that rule alone would have stopped.


## Valves, layered beds, dead volume, IAST and acceleration

Steps can be driven by linear valves (`Valve`), to fixed or recorded pressures, so that pressure equalization and purge supply between beds are simulated as in an industrial unit. Beds can hold several adsorbent layers (`LayeredBed`, each with its own particle size and voidage) and dead volumes at their ends; mixture equilibrium can come from pure-component isotherms by IAST (`IAST`); `Cycle.run(accelerate="graves_morris")` (or another vector Steffensen method of Subraveti, Karimi and Gazzani 2026, or Anderson mixing) reaches cyclic steady state in fewer cycles; an automatic LSODA fallback handles steps where the gas is nearly at rest; uptake can be limited by film, macropore and micropore resistances in series (`SeriesLDF`); pressure equalization uses recorded streams replayed conservatively, or pure product in a single column (`BufferInlet(..., composition=...)`); and `ParticleDiffusion` resolves diffusion inside the particles, validated against the exact solutions of Crank (single particle) and of the linear column with particle diffusion.

## Gas properties and adsorbents

`swingpy.data` holds pure-component gas properties (CO<sub>2</sub>, N<sub>2</sub>, O<sub>2</sub>, Ar, CH<sub>4</sub>, CO, He), each kind with its source, and `data.gas(components, composition, temperature, pressure)` evaluates the mixture heat capacity, the Wilke viscosity and the Fuller diffusivity. SwingPy ships no adsorbents and no feeds: they come from you, in Python or in a specification file (`swingpy.spec`). The examples use adsorbents and setups reported in the literature, in `examples/setups` and `examples/specs`. Every model class checks its inputs and says which one is wrong and why.

## Examples

Each example runs from the repository root, for example `python "examples/02 Four step VSA.py"`, and writes its figures to the current folder.

| Example | What it shows |
|---|---|
| `01 Breakthrough.py` | a breakthrough of CO<sub>2</sub> through a column of zeolite 13X, with its mass and energy balances |
| `02 Four step VSA.py` | the four-step vacuum swing cycle for post-combustion capture, run to cyclic steady state |
| `03 Build your own cycle.py` | a tutorial: gas, adsorbent, column and steps built from scratch, run as a four-step VSA cycle |
| `04 Cycle configurations.py` | ready-made cycles with light product pressurization and pressure equalization |

## Development

```bash
pip install -e ".[test,fast,plot,docs]"
pytest --cov                           # the test suite, with coverage of the library
mkdocs serve                           # documentation at http://127.0.0.1:8000
python -m build                        # wheel and source package
python "tools/Export examples.py"      # write the ready-made cases as input files in examples/specs
python "tools/Make gallery.py"         # draw the figures of the documentation in docs/figures
```

Continuous integration (`.github/workflows`) runs the tests on Linux, Windows and macOS with Python 3.10 to 3.12, with and without numba, and builds and publishes the documentation. Changes are listed in `CHANGELOG.md`; `CITATION.cff` says how to cite SwingPy.

## What the model solves

Axially dispersed plug flow of an ideal gas through a packed bed, with:

- component and total mass balances in conservative finite volume form;
- uptake by linear driving force (macropore controlled, constant, pressure scaled, or film, macropore and micropore resistances in series) or by diffusion resolved inside the particles;
- multicomponent equilibrium: dual-site Langmuir, extended Sips and Toth (competitive or not, on a concentration or partial pressure basis, with temperature dependent heterogeneity and capacity), extended Langmuir and IAST, each with its temperature dependence;
- bed and wall energy balances with heat of adsorption and heat exchange with the surroundings through the wall;
- Darcy or Ergun pressure drop, dead volumes at the column ends and layered beds;
- UDS, van Leer, Superbee or WENO face reconstruction;
- stiff time integration (SciPy BDF, with an LSODA fallback) with a banded Jacobian pattern.

Eight running integrals per pair of ends record the component, enthalpy and work flows, so balances and KPIs are as accurate as the integration itself.

## Comparison with published work

SwingPy is compared with the simulations and experiments of the studies below. Every comparison uses only the data each article reports, nothing is fitted, and the result is stated plainly, including where the agreement is partial or absent. The author is grateful to the authors of these studies for publishing their models and data in enough detail to be reproduced.

| Study | What is compared | Outcome |
|---|---|---|
| R. Haghpanah, A. Majumder, R. Nilam, A. Rajendran, S. Farooq, I. A. Karimi, M. Amanullah, *Ind. Eng. Chem. Res.* 52 (2013) 4249, [doi:10.1021/ie302658y](https://doi.org/10.1021/ie302658y) | breakthrough of CO<sub>2</sub> on zeolite 13X, balance errors, accuracy of the discretization schemes, four-step VSA | breakthrough reproduced closely (CO<sub>2</sub> front at 535 s against about 520 s, balances closed to 10<sup>−4</sup> %); cycle trends reproduced, KPIs within about 10 % |
| R. Haghpanah, R. Nilam, A. Rajendran, S. Farooq, I. A. Karimi, *AIChE J.* 59 (2013) 4735, [doi:10.1002/aic.14192](https://doi.org/10.1002/aic.14192) | basic, light product pressurization and pressure equalization cycles | purity and recovery within 2 to 3 points at five operating points; the six-step cycle with light and heavy reflux is not reproduced |
| S. G. Subraveti, S. Roussanaly, R. Anantharaman, L. Riboldi, A. Rajendran, *Sep. Purif. Technol.* 256 (2021) 117832, [doi:10.1016/j.seppur.2020.117832](https://doi.org/10.1016/j.seppur.2020.117832) | three adsorbents: cycle, plant, column schedule and capture cost | purity and recovery within 1.5 points; capture cost of the published designs within 1.5 % (the plant, schedule and cost modules are not part of this release) |
| S. G. Subraveti, PhD thesis, University of Alberta (2021), Chapter 4 | isothermal three- and four-step cycles on IISERP MOF2 | partly: recovery within 1.7 points, purity 1.8 to 2.6 points lower |
| S. Krishnamurthy, V. R. Rao, S. Guntuka, P. Sharratt, R. Haghpanah, A. Rajendran, M. Amanullah, I. A. Karimi, S. Farooq, *AIChE J.* 60 (2014) 1830, [doi:10.1002/aic.14435](https://doi.org/10.1002/aic.14435) | pilot plant VSA on zeolite 13X | five runs within 1.2 points of purity and 0.7 of recovery; one partly; the two blowdown pressure runs not reproduced |
| T. T. T. Nguyen, G. K. H. Shimizu, A. Rajendran, *Chem. Eng. J.* 452 (2023) 139550, [doi:10.1016/j.cej.2022.139550](https://doi.org/10.1016/j.cej.2022.139550) | VSA experiments on structured CALF-20 | experiments II to V reproduced; VI to X partly (trends right, purity high) |
| S. G. Subraveti, K. Karimi, M. Gazzani, *Ind. Eng. Chem. Res.* 65 (2026) 9656, [doi:10.1021/acs.iecr.6c00407](https://doi.org/10.1021/acs.iecr.6c00407); S. G. Subraveti, K. Karimi, M. Gazzani, R. Anantharaman, *Syst. Control Trans.* 3 (2024) 215 | vector acceleration of the approach to cyclic steady state | steady states reproduced; numbers of cycles reproduced for the VSA test case |

The model is also checked against exact solutions: the uptake of a single particle (J. Crank, *The Mathematics of Diffusion*, 1975), within 0.5 %, and the breakthrough of a linear column with diffusion in the particles (D. M. Ruthven, *Principles of Adsorption and Adsorption Processes*, 1984), within 0.16 %.

## Package layout

```
src/swingpy/
    gas.py           GasMixture, adiabatic constant of the pumped gas
    isotherms.py     DualSiteLangmuir, Sips, Toth, ExtendedLangmuir, IAST
    adsorbent.py     Adsorbent, LayeredBed, MacroporeLDF, ConstantLDF, PressureScaledLDF, SeriesLDF,
                     ParticleDiffusion
    column.py        Column
    boundaries.py    inlets, outlets, vacuum pumps, valves, tanks and buffers
    steps.py         Step, PressureReached
    streams.py       StreamRecord: recorded streams that couple steps
    model.py         ColumnModel: equations, integration, diagnostics
    kernels.py       compiled right hand side (optional, numba)
    checks.py        checks of the values given by the user
    schemes.py       finite volume face reconstruction
    cycle.py         Cycle, cyclic steady state, performance indicators
    acceleration.py  vector acceleration (Graves-Morris and other Steffensen variants, Anderson)
    data/            data bank of gases, with their sources
    spec/            problem specifications: load_problem, to_spec, save_problem
    analysis.py      numbers of a simulation: breakthrough metrics, cycle tables, CSV export
    plotting.py      figures (optional matplotlib)
    report.py        everything a simulation produces, written to one folder
    reporting.py     console tables
examples/            tutorials, input files (specs/) and the setups they use (setups/)
tests/               physics, conservation and regression tests
```

## Possible extensions

- Real gas behaviour in the column equations, for high-pressure PSA with much CO<sub>2</sub>.
- Gas properties that vary with temperature and composition (the reference models all use constant ones).
- A compiled solver for valve faces.
- Hollow fibre and structured contactors.

## Feedback and contact

Bug reports, questions and suggestions are welcome:

- **Bugs and problems:** open an [issue](https://github.com/mohammmadyazdani/swingpy/issues/new/choose) with the version of SwingPy and Python, a short script that shows the problem, and what you expected.
- **New features and validation cases:** open an issue that names the article or data behind them; comparisons with published results are especially welcome.
- **Anything else** (collaboration, questions you prefer not to ask in public): write to Mohammad Yazdani, mohamad.yazdani7990@gmail.com.

## Citing

If you use SwingPy, please cite it:

```bibtex
@software{yazdani_swingpy_2026,
  author  = {Yazdani, Mohammad},
  title   = {{SwingPy}: simulation of pressure, vacuum and vacuum-pressure swing adsorption cycles},
  year    = {2026},
  version = {1.0.0},
  url     = {https://github.com/mohammmadyazdani/swingpy}
}
```

Please also cite the articles behind the setups and data you use: they are named in the headers of `examples/specs`, in each module of `examples/setups` and in `examples/setups/adsorbents.json`; `data.gas_sources()` names the sources of the gas properties.

## License

MIT, see `LICENSE`.
