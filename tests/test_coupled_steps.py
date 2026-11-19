"""
Tests of recorded streams, stop conditions and coupled steps.
"""

import numpy as np
import pytest

from swingpy import BufferInlet, Closed, PressureOutlet, PressureReached, Step, TankInlet
from setups import cycles as aiche


@pytest.fixture(scope="module")
def lpp_cycle_results():
    model, cycle, state = aiche.example_case("four step with light product pressurization, high purity")
    results = cycle.run_once(model, state)
    results = cycle.run_once(model, results[-1].final_state)
    return model, cycle, state, results


def test_bookkeeping_matches_inventory_over_a_coupled_cycle(lpp_cycle_results):
    """Everything that entered minus everything that left equals the change held in the column."""
    model, cycle, _, results = lpp_cycle_results
    start = model.component_inventory(results[0].initial_state)
    end = model.component_inventory(results[-1].final_state)
    net = sum(r.moles_in["feed"] + r.moles_in["product"] for r in results)
    """Mol, against about 15 mol held."""
    assert np.allclose(end - start, net, rtol=1e-5, atol=1e-4)


def test_lpp_never_exceeds_the_supply_pressure(lpp_cycle_results):
    """
    The raffinate is produced at P_H, so the LPP face must never rise above
    it; the step ends at 0.99 P_H at the closed end and the top-up finishes
    the pressurization.
    """
    model, cycle, _, results = lpp_cycle_results
    lpp, top_up = results[-2], results[-1]
    assert np.max(model.end_history(lpp, "product")["pressure"]) <= 1.0 + 1e-9
    assert lpp.time[-1] < cycle.steps[-2].duration
    assert model.unpack(lpp.final_state)["P"][0] == pytest.approx(model.to_pressure(0.99), abs=1e-6)
    assert model.unpack(top_up.final_state)["P"][-1] >= model.to_pressure(0.999) - 1e-6


def test_buffer_replay_moves_the_recorded_amount(lpp_cycle_results):
    """The gas entering during LPP equals the recorded raffinate over the same time window."""
    model, cycle, _, results = lpp_cycle_results
    lpp = results[-2]
    recorded = cycle.buffers["raffinate"].total_moles(0.0, lpp.time[-1])
    """Atol: the CO2 in the raffinate is numerical noise (1e-8 mol, of either sign); replays clip it at zero."""
    assert np.allclose(lpp.moles_in["product"], recorded, rtol=2e-2, atol=1e-6)


def test_tank_inlet_feeds_the_requested_fraction():
    model = aiche.model(feed_velocity=0.2)
    state = model.initial_state(0.054, {"CO2": 0.9, "N2": 0.1})
    source = Step("SRCE", 20.0, feed_end=PressureOutlet(0.03, record_as="extract"))
    reflux = Step("HREF", 5.0, feed_end=TankInlet("extract", fraction=0.5), product_end=PressureOutlet(0.054))
    first = model.simulate(source, state, n_output=200)
    record = model.outflow_record(first, "feed")
    reflux.feed_end.attach(record, reflux.duration)
    second = model.simulate(reflux, first.final_state)
    assert second.moles_in["feed"].sum() == pytest.approx(0.5 * record.total_moles().sum(), rel=2e-2)


def test_empty_buffer_means_no_flow():
    model = aiche.model()
    step = Step("LPPR", 5.0, product_end=BufferInlet("missing"), feed_end=Closed())
    result = model.simulate(step, model.initial_state(0.5, {"N2": 1.0}))
    assert np.allclose(result.moles_in["product"], 0.0)


def test_satisfied_stop_condition_skips_step():
    model = aiche.model()
    step = Step("HOLD", 10.0, stop=PressureReached(0.5, location="mean", rising=True))
    result = model.simulate(step, model.initial_state(1.0, {"N2": 1.0}))
    assert result.time[-1] == 0.0


def test_pressure_equalization_moves_gas_from_donor_to_receiver():
    """The receiver takes in what the donor released and ends a little below the donor pressure."""
    model, cycle, _ = aiche.example_case("five step with pressure equalization")
    results = cycle.run_once(model, model.initial_state(1.0, {"N2": 1.0}))
    donor, receiver = results[1], results[3]
    released = -donor.moles_in["product"].sum()
    received = receiver.moles_in["product"].sum()
    assert received == pytest.approx(released, rel=1e-2)
    donor_pressure = model.unpack(donor.final_state)["P"].mean()
    receiver_pressure = model.unpack(receiver.final_state)["P"].mean()
    assert 0.0 < donor_pressure - receiver_pressure < model.to_pressure(0.1)


def test_flow_inlet_delivers_its_molar_flow_and_respects_its_pressure_cap():
    from swingpy import FlowInlet, standard_litres_per_minute
    from setups import skarstrom as dantas
    model = dantas.model(1)
    flow = standard_litres_per_minute(3.0)
    feed = Step("FEED", 30.0, feed_end=FlowInlet(flow, {"N2": 1.0}, 323.15), product_end=PressureOutlet(1.3))
    result = model.simulate(feed, model.initial_state(1.3, {"N2": 1.0}, 323.15))
    assert result.moles_in["feed"].sum() == pytest.approx(flow * 30.0, rel=1e-3)
    pressurize = Step("PRES", 60.0, feed_end=FlowInlet(flow, {"N2": 1.0}, 323.15, max_pressure=1.3))
    result = model.simulate(pressurize, model.initial_state(0.1, {"N2": 1.0}, 323.15))
    assert np.max(model.end_history(result, "feed")["pressure"]) <= 1.3 + 1e-9


def test_recorded_stream_total_is_exact_despite_a_sharp_start():
    """The first instants of an evacuation carry a spike in flow that sampling cannot resolve; the totals must not depend on it."""
    import setups as hg
    m = hg.model(n_cells=12)
    state = m.initial_state(1.0, {"CO2": 0.5, "N2": 0.5})
    step = Step("EVAC", 60.0, feed_end=PressureOutlet(0.05, stream="product"), product_end=Closed())
    result = m.simulate(step, state, n_output=30)
    record = m.outflow_record(result, "feed")
    assert np.allclose(record.total_moles(), -np.asarray(result.moles_in["feed"]), rtol=1e-9, atol=1e-12)
