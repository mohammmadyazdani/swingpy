"""
The compiled (numba) and numpy right hand sides must be the same equations.
"""

from dataclasses import replace

import numpy as np
import pytest

from swingpy import Column, ColumnModel, ConstantLDF
import setups as h
from setups import cycles as aiche

pytest.importorskip("numba")


def max_scaled_difference(a, b):
    return float(np.max(np.abs(a - b)) / np.max(np.abs(b)))


def test_backends_agree_on_every_step_of_a_coupled_cycle():
    name = "four step with light product pressurization, high purity"
    model, cycle, state = aiche.example_case(name)
    reference = aiche.model(feed_velocity=aiche.EXAMPLE_POINTS[name]["feed_velocity"], backend="numpy")
    assert model.backend == "numba"
    results = cycle.run_once(model, cycle.run_once(model, state)[-1].final_state)
    for result in results:
        for s in result.states.T[::5]:
            assert max_scaled_difference(model.rhs(0.3, s, result.step), reference.rhs(0.3, s, result.step)) < 1e-9


def test_backends_agree_with_ergun_and_constant_rates():
    adsorbent = replace(h.zeolite_13x(), kinetics=ConstantLDF((0.1, 0.5)))
    column = Column(length=1.0, inner_radius=0.1445, pressure_drop="ergun", inner_heat_transfer=8.6,
                    outer_heat_transfer=2.5)
    fast = ColumnModel(column, adsorbent, h.flue_gas(), backend="numba")
    slow = ColumnModel(column, adsorbent, h.flue_gas(), backend="numpy")
    step = h.four_step_vsa().steps[1]
    s = fast.simulate(step, fast.initial_state(1.0, {"CO2": 0.1, "N2": 0.9})).final_state
    assert max_scaled_difference(fast.rhs(0.1, s, step), slow.rhs(0.1, s, step)) < 1e-9
