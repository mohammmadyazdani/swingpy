"""The VSA cycle family of ``swingpy.cycles`` and its presets for the seven cycles of Haghpanah et al. (2013)."""
import numpy as np
import pytest

import setups.cycles as setup
import swingpy.cycles as article
from swingpy.cycles import vsa_cycle

FEED = {"CO2": 0.15, "N2": 0.85}
COMMON = dict(t_ads=60.0, feed_velocity=0.3, high=1.0, low=0.03, intermediate=0.1)

POINTS = {
    "basic four step (Figure 2)": dict(t_ads=74.6, t_bd=60.1, t_evac=85.4, intermediate=0.17, low=0.03,
                                       feed_velocity=0.27),
    "four step with LPP (Figure 6a)": dict(t_ads=53.15, t_bd=141.76, t_evac=131.21, intermediate=0.05, low=0.03,
                                           feed_velocity=0.33),
    "five step with LPP and LR (Figure 6b)": dict(t_ads=60.0, t_bd=60.0, t_evac=100.0, t_lr=5.0, intermediate=0.07,
                                                  low=0.03, feed_velocity=0.3),
    "six step with LPP, LR and HR (Figure 6c)": dict(t_ads=59.22, t_bd=35.19, t_hr=5.07, t_evac=104.61, t_lr=1.34,
                                                     intermediate=0.054, low=0.03, reflux_ratio=0.82,
                                                     feed_velocity=0.2),
    "five step with PE and FP (Figure 6d)": dict(t_ads=83.59, t_pe=60.0, t_evac=130.0, low=0.03, feed_velocity=0.56),
    "five step with PE and LPP (Figure 6e)": dict(t_ads=80.0, t_pe=60.0, t_evac=130.0, low=0.03, feed_velocity=0.5),
    "six step with PE, LPP and HR (Figure 6f)": dict(t_ads=80.0, t_pe=60.0, t_hr=5.0, t_evac=130.0, low=0.03,
                                                     reflux_ratio=0.5, feed_velocity=0.5),
}

STEPS = {
    "basic four step (Figure 2)": ["FEED", "BLOW", "EVAC", "PRES"],
    "four step with LPP (Figure 6a)": ["FEED", "BLOW", "EVAC", "LPPR", "PRES"],
    "five step with LPP and LR (Figure 6b)": ["FEED", "BLOW", "EVAC", "LREF", "LPPR", "PRES"],
    "six step with LPP, LR and HR (Figure 6c)": ["FEED", "BLOW", "HREF", "EVAC", "LREF", "LPPR", "PRES"],
    "five step with PE and FP (Figure 6d)": ["FEED", "EQDN", "EVAC", "EQUP", "PRES"],
    "five step with PE and LPP (Figure 6e)": ["FEED", "EQDN", "EVAC", "EQUP", "LPPR", "PRES"],
    "six step with PE, LPP and HR (Figure 6f)": ["FEED", "EQDN", "HREF", "EVAC", "EQUP", "LPPR", "PRES"],
}


def test_all_seven_configurations_are_listed():
    assert list(article.HAGHPANAH_2013) == list(STEPS)


@pytest.mark.parametrize("name", list(STEPS))
def test_step_sequence_follows_figure_6(name):
    cycle = article.HAGHPANAH_2013[name](**POINTS[name])
    assert [s.name for s in cycle.steps] == STEPS[name]


def test_equalization_pressure_is_eq_9():
    cycle = article.five_step_pe_lpp(t_ads=80.0, t_pe=60.0, t_evac=130.0, low=0.03, feed_velocity=0.5)
    donor = next(s for s in cycle.steps if s.name == "EQDN")
    assert donor.product_end.pressure(1.0e6) == pytest.approx(0.5 * (1.0 + 0.03) - 0.041)


def test_heavy_reflux_records_the_extract_and_is_subtracted_from_the_product():
    cycle = vsa_cycle(FEED, t_blow=30.0, t_evac=60.0, t_hr=5.0, reflux_ratio=0.5, **COMMON)
    evacuation = next(s for s in cycle.steps if s.name == "EVAC")
    reflux = next(s for s in cycle.steps if s.name == "HREF")
    assert evacuation.feed_end.record_as == "extract"
    assert reflux.feed_end.buffer == "extract" and reflux.feed_end.stream == "product"


def test_vacuum_pumps_end_steps_at_the_pressures():
    cycle = vsa_cycle(FEED, blowdown_velocity=1.0, evacuation_velocity=3.0, **COMMON)
    blowdown = next(s for s in cycle.steps if s.name == "BLOW")
    evacuation = next(s for s in cycle.steps if s.name == "EVAC")
    assert blowdown.stop.pressure == pytest.approx(0.1) and evacuation.stop.pressure == pytest.approx(0.03)


@pytest.mark.parametrize("kwargs, message", [
    (dict(depressurization="vent"), "depressurization"),
    (dict(pressurization="vent", t_blow=1.0, t_evac=1.0), "pressurization"),
    (dict(t_evac=1.0), "t_blow"),
    (dict(t_blow=1.0), "t_evac"),
    (dict(depressurization="equalization", t_evac=1.0), "t_eq"),
    (dict(t_blow=1.0, t_evac=1.0, t_lr=5.0, record_raffinate=False), "record_raffinate"),
])
def test_invalid_arguments_are_refused(kwargs, message):
    with pytest.raises(ValueError, match=message):
        vsa_cycle(FEED, **COMMON, **kwargs)


def test_pressures_must_be_ordered():
    with pytest.raises(ValueError, match="P_L < P_I < P_H"):
        vsa_cycle(FEED, t_ads=60.0, feed_velocity=0.3, high=1.0, low=0.2, intermediate=0.1, t_blow=1.0, t_evac=1.0)


@pytest.mark.parametrize("name", ["five step with PE and LPP (Figure 6e)", "six step with PE, LPP and HR (Figure 6f)"])
def test_new_configurations_reach_a_closed_mass_balance(name):
    model = setup.model(feed_velocity=POINTS[name]["feed_velocity"], n_cells=10)
    cycle = article.HAGHPANAH_2013[name](**POINTS[name])
    run = cycle.run(model, setup.initial_state(model), max_cycles=400, tolerance=0.5, criterion="mass_balance",
                    accelerate="anderson", verbose=False)
    assert run.converged
    assert abs(run.performance.mass_balance_error) < 0.5
    assert 0.0 < run.performance.purity < 100.0 and 0.0 < run.performance.recovery <= 100.5
    assert np.all(np.isfinite(run.final_state))


def test_example_setups_use_the_core_presets():
    assert setup.CYCLES is article.HAGHPANAH_2013
