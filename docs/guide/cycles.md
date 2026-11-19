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
| Indirect heating or cooling (temperature swing) | any of the above, with `Step(..., ambient_temperature=420.0)` | |
| Any step through a valve | `Valve` | `Valve` |
| Equalization, purge supply between beds | `Valve` to a `RecordedPressure` (donor) | `BufferInlet` (receiver) |

## Coupled steps

In a plant the effluent of one column feeds another one at the same time. SwingPy simulates one column, so an outlet records its stream (`record_as="raffinate"`) and a later step replays it (`BufferInlet("raffinate")`); `TankInlet` replays a fraction of a recorded stream at its mean composition. A replayed stream can be limited to the pressure it was produced at (`max_pressure`). The replay is conservative: each sampling interval of the record delivers exactly the moles that left in it, so a sharp start (a pressure equalization opening onto an empty column) is replayed with the right amount. `BufferInlet(..., composition={"H2": 1.0})` replays only the recorded molar flow, with another gas: the single column treatment of pressure equalization with pure product of Ribeiro et al. (2008). `PressurizationInlet` and `FlowInlet` accept the name of a record as their composition (with `default_composition` for the first cycle) and then use its mean gas, for example pressurization or purge with the product of the feed step of another column. `MixingInlet(molar_flow, composition, buffer="heat1")` adds a recorded stream, spread evenly over the step, to a fresh feed at fixed molar flow: the outlet of a first heating step sent back with the feed, as in cycle D of Joss et al. (2017). The fresh moles count as the `feed` stream and the recycled ones as `recycle`, so recovery refers to the fresh feed only; label the outlet that fills the tank `stream="recycle"` and the two cancel in the cycle balance.

A temperature swing cycle sets the temperature outside the wall in each step (`ambient_temperature`), with the wall and its heat transfer coefficients in `Column`:

```python
heat = Step("HEA2", 600.0, feed_end=PressureOutlet(1.0, stream="product", check_valve=True),
            ambient_temperature=420.0)
ads = Step("FEED", 150.0,
           feed_end=MixingInlet(0.018, {"CO2": 0.12, "N2": 0.88}, buffer="heat1", temperature=300.0),
           product_end=PressureOutlet(1.3, stream="waste", check_valve=True), ambient_temperature=300.0)
```

## Instantaneous pressurization

When the pressurization is much faster than the uptake, as in direct air capture with step times of hours, it can be taken as instantaneous with the solid frozen: `instant_pressurization("PRES", {"CO2": 0.0004, "H2O": 0.0115, "N2": 0.9881}, 1.0)` is a step of zero duration after which the gas in the column has the given composition and pressure, the loadings and temperatures unchanged (Balasubramaniam et al., *Chem. Eng. J.* 485 (2024) 149568). The moles that enter count as the feed.

A feed inlet can also take its gas from a recorded stream, `FeedInlet(0.2, "light", default_composition={"N2": 1.0})`: the mean composition of the record, as from a well mixed tank (light product pressurization at a constant velocity).

## Stop conditions

`stop=PressureReached(0.1, location="feed", rising=False)` ends a step when the pressure at one end crosses a value; the duration is then the longest the step may last. With vacuum pumps of fixed capacity, the time to reach a vacuum is a result of the simulation.

## Performance

Every outlet and inlet carries a stream label. `Cycle(key_component="CO2")` computes the purity and recovery of the stream labelled `product`, the specific energy from the compressor and pump work of every step, and the productivity; `CyclePerformance.streams` gives the moles of every component in every stream.
