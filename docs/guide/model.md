# The column model

SwingPy describes a packed column as a one-dimensional, axially dispersed plug flow of an ideal gas mixture through a bed of porous particles.

## Balances

For component \(i\) in the gas, with interstitial velocity \(v\), total concentration \(C = P/RT\) and bed voidage \(\varepsilon\):

$$
\frac{\partial (y_i C)}{\partial t} + \frac{\partial (v\, y_i C)}{\partial z}
= \frac{\partial}{\partial z}\left(D_L C \frac{\partial y_i}{\partial z}\right)
- \frac{1-\varepsilon}{\varepsilon}\,\frac{\partial q_i}{\partial t}
$$

with \(q_i\) the loading per volume of particles. Uptake follows the linear driving force model,

$$
\frac{\partial q_i}{\partial t} = k_i \left(q_i^* - q_i\right),
$$

where the equilibrium loading \(q_i^*\) comes from the isotherm and \(k_i\) is either constant or obtained from macropore diffusion, \(k_i = 15\,\varepsilon_p D_{m}/(\tau r_p^2)\, c_i/q_i^*\).

For uptake controlled by diffusion inside the particles (crystals or carbon molecular sieves, kinetic separations), `ParticleDiffusion(rates, shells=12)` resolves Fick's law in every particle on shells that thin towards the surface, with the surface at equilibrium with the gas; `rates` are the diffusional time constants D/r<sup>2</sup> (1/s).

The bed energy balance includes the heat capacities of the gas, the solid and the adsorbed phase, axial conduction, the heat of adsorption and heat transfer to the wall; the wall has its own energy balance with heat loss to the surroundings. The velocity follows from the local pressure gradient through Darcy's law or the Ergun equation.

## Boundary conditions

At an inlet, Danckwerts conditions set composition and temperature; at an outlet, gradients vanish. The flow at each end is set either by a velocity or molar flow (feed, purge, vacuum pump) or by a pressure (pressurization, blowdown, a back-pressure regulator), the other quantity following from the pressure drop over half a cell.

## Numerical method

The equations are written in dimensionless, conservative form and discretized on equal finite volumes. Face values on the upwind side are reconstructed with first-order upwind, van Leer, Superbee or WENO, choosing the upwind side face by face from the local velocity, so that flow reversal inside the column is handled. The resulting system is integrated by a variable-order BDF method with a sparse Jacobian. Running integrals of the component and enthalpy fluxes and of the machine work at both ends make purity, recovery, energy and balance errors as accurate as the integration itself.

The upwind side of each face follows the local velocity, so the equations are not smooth where the gas comes to rest. In a step where it is nearly at rest over part of the column (the blowdown of a short, very permeable bed, as in direct air capture) Newton's method in BDF then keeps failing; after 40 000 evaluations of the right hand side (`simulate(..., max_evaluations=...)`) the step is restarted with LSODA, which agrees with BDF to 10<sup>−8</sup> in the state and is far faster there.

## Cyclic steady state

`Cycle.run` repeats the cycle until steady state: by default the key component mass balance must close within a tolerance **and** the column profiles must stop changing, both over five cycles in a row. Either test alone can be met while the other is not, depending on the cycle.

### Faster approach to steady state

`Cycle.run(..., accelerate=True)` starts each cycle from an Anderson extrapolation of the last few cycles instead of the end of the last one. It works best when the approach to steady state is gradual: a Skarstrom cycle reaches steady state in 43 cycles instead of 107, with the same purity and recovery. When the slow part of the approach is a front travelling through the column (for example a bed that warms from one end over many cycles), extrapolation cannot predict it; the method then falls back to plain cycles and takes as many as without it. The results are always those of simulated cycles, so acceleration changes how fast steady state is reached, never where it is.

`Cycle.run(..., accelerate="graves_morris")` uses instead the vector extensions of Steffensen's method compared for adsorption cycles by Subraveti, Karimi and Gazzani ([Ind. Eng. Chem. Res. 65 (2026) 9656](https://doi.org/10.1021/acs.iecr.6c00407), Table 1): `lemarechal`, `sedogbo`, `a1`, `a4`, `irons_tuck`, `b1`, `zienkiewicz_lohner`, `graves_morris`, `c1`, `c2`, `macleod_m6` and `epsilon2`. Each extrapolation uses two successive plain cycles D → F(D) → F(F(D)); with u = F(D) − D, v = F(F(D)) − F(D) and w = v − u, Graves-Morris starts the next cycle from F(D) − (uᵀv / uᵀw) v. The schedule is that of the paper: three plain cycles, then an extrapolation every `acceleration_every` (3) cycles, and none once the residual is below `acceleration_stop` (10<sup>−4</sup>). The residual of the paper (its eq 2: over the steps, the largest 100 × sum of squares of the change of the dimensionless profiles from the previous cycle) is recorded in `CyclePerformance.residual`, and `criterion="residual"` applies the steady state test of the paper (residual below `residual_tolerance`, 10<sup>−5</sup>, and mass balance error below 0.5 %). A cycle that starts from an extrapolated state is run without the LSODA fallback: if one of its steps needs more than 40 000 evaluations it raises `StepTooStiff`, the extrapolation is discarded (counted in the accelerator's `rejected`) and the cycle is run again from the simulated state. On adiabatic pilot-scale columns some extrapolated states were otherwise integrated for more than 40 minutes. The conference paper that preceded the 2026 article (Subraveti et al., FOCAPD 2024) prints the Graves-Morris step as F(D) − (wᵀw / uᵀw) v; for a scalar linear sequence that form does not give the limit, while the 2026 form, used here, gives it exactly (Aitken's Δ<sup>2</sup>).

## Trace components

The model solves the mole fractions of all components but the last; the last is 1 minus the others. Components present at ppm levels (CO<sub>2</sub> from air, impurities in hydrogen) are nevertheless accurate in either position. For 1 ppm of CO<sub>2</sub> in N<sub>2</sub> fed to zeolite 13X for 2000 s, the CO<sub>2</sub> profile computed with the default solver tolerances differs from a reference solution (absolute tolerance 10<sup>−12</sup>) by 0.3 % of the feed mole fraction when CO<sub>2</sub> is solved directly and by 0.1 % when it is the last component, and the CO<sub>2</sub> balance closes within 0.01 % and 0.03 %. At 420 ppm the two orders differ by 0.13 % at most. The order of the components is therefore free; a test keeps this so (`tests/test_spec.py`).

