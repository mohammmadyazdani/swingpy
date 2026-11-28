# Steps and cycles

A step is two boundary conditions, one per column end, and a duration (or a stop condition). The same few classes cover every common step:

| Step | Feed end (Z = 0) | Product end (Z = 1) |
|---|---|---|
| Feed pressurization | `PressurizationInlet` | `Closed` |
| Adsorption | `FeedInlet` or `FlowInlet` | `PressureOutlet` |
| Blowdown (co-current) | `Closed` | `PressureOutlet` or `VacuumPump` |
| Evacuation (counter-current) | `PressureOutlet` or `VacuumPump` | `Closed` |
| Purge with a gas | `PressureOutlet` | `FlowInlet` |
| Light product pressurization or light reflux | `Closed` or `PressureOutlet` | `BufferInlet` |
| Heavy reflux | `TankInlet` | `PressureOutlet` |
| Adsorption with a recycle into the feed | `MixingInlet` | `PressureOutlet` |
| Any step through a valve | `Valve` | `Valve` |
| Equalization, purge supply between beds | `Valve` to a `RecordedPressure` (donor) | `BufferInlet` (receiver) |

## A family of VSA cycles

`swingpy.cycles.vsa_cycle` builds a single-column VSA cycle from optional steps, the building blocks of the cycle synthesis of Haghpanah et al., *AIChE J.* 59 (2013) 4735: adsorption (FEED), then blowdown (BLOW) or pressure equalization (EQDN), an optional heavy reflux at the intermediate pressure (HREF), evacuation (EVAC), an optional light reflux at the low pressure (LREF), the receiving side of the equalization (EQUP), and pressurization with light product (LPPR, completed by PRES) or with feed (PRES).

```python
from swingpy.cycles import vsa_cycle

cycle = vsa_cycle({"CO2": 0.15, "N2": 0.85}, t_ads=60.0, feed_velocity=0.3, high=1.0, low=0.03,
                  intermediate=0.07, t_blow=60.0, t_evac=100.0, t_lr=5.0)          # five steps, LPP and LR
cycle = vsa_cycle({"CO2": 0.15, "N2": 0.85}, t_ads=80.0, feed_velocity=0.5, high=1.0, low=0.03,
                  intermediate=0.474, depressurization="equalization", t_eq=60.0,
                  t_hr=5.0, reflux_ratio=0.5, t_evac=130.0)                          # PE, HR and LPP
```

Pressure changes are prescribed (an exponential approach at `rate` over `t_blow` or `t_evac`) or driven by vacuum pumps of given interstitial velocity (`blowdown_velocity`, `evacuation_velocity`), the step then ending when P_I or P_L is reached. The heavy reflux feeds back `reflux_ratio` of the recorded extract at constant molar flow (`TankInlet`) and is subtracted from the product; the light reflux uses the first `t_lr` seconds of the recorded raffinate and the light product pressurization the rest. The seven configurations of the article are presets in the same module (`HAGHPANAH_2013`, by name: `basic_four_step`, `four_step_lpp`, `five_step_lpp_lr`, `six_step_lpp_lr_hr`, `five_step_pe_fp`, `five_step_pe_lpp`, `six_step_pe_lpp_hr`) with the feed and pressures of the article and every step time and pressure as an argument; the column model is chosen independently of the cycle (`examples/setups/cycles.py` pairs them with the model of the article and its operating points).

## Coupled steps

In a plant the effluent of one column feeds another one at the same time. SwingPy simulates one column, so an outlet records its stream (`record_as="raffinate"`) and a later step replays it (`BufferInlet("raffinate")`); `TankInlet` replays a fraction of a recorded stream at its mean composition. A replayed stream can be limited to the pressure it was produced at (`max_pressure`). The replay is conservative: each sampling interval of the record delivers exactly the moles that left in it, so a sharp start (a pressure equalization opening onto an empty column) is replayed with the right amount. `BufferInlet(..., composition={"N2": 1.0})` replays only the recorded molar flow, with another gas: the single column treatment of pressure equalization with pure product. `PressurizationInlet` and `FlowInlet` accept the name of a record as their composition (with `default_composition` for the first cycle) and then use its mean gas, for example pressurization or purge with the product of the feed step of another column. `MixingInlet(molar_flow, composition, buffer="recycle")` adds a recorded stream, spread evenly over the step, to a fresh feed at fixed molar flow: the outlet of an earlier step sent back with the feed. The fresh moles count as the `feed` stream and the recycled ones as `recycle`, so recovery refers to the fresh feed only; label the outlet that fills the tank `stream="recycle"` and the two cancel in the cycle balance.

A feed inlet can also take its gas from a recorded stream, `FeedInlet(0.2, "light", default_composition={"N2": 1.0})`: the mean composition of the record, as from a well mixed tank (light product pressurization at a constant velocity).

## Stop conditions

`stop=PressureReached(0.1, location="feed", rising=False)` ends a step when the pressure at one end crosses a value; the duration is then the longest the step may last. With vacuum pumps of fixed capacity, the time to reach a vacuum is a result of the simulation.

## Performance

Every outlet and inlet carries a stream label. `Cycle(key_component="CO2")` computes the purity and recovery of the stream labelled `product`, the specific energy from the compressor and pump work of every step, and the productivity; `CyclePerformance.streams` gives the moles of every component in every stream.
