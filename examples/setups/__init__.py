"""
Ready-made setups for the examples and tests: gases, adsorbents, columns,
models and cycles with the parameters of published studies. This is example
data, not part of the SwingPy library.

    setups                        flue gas on zeolite 13X (four-step VSA) and
                                  steam-methane-reformer flue gas (``smr_``)
    setups.cycles                 the cycle configurations of Haghpanah et al.
                                  (2013) with ready operating points
"""

from .vsa import (  # noqa: F401
    flue_gas, zeolite_13x, column, model, breakthrough_step, four_step_vsa, smr_adsorbent, smr_flue_gas, smr_column, smr_model, smr_four_step_lpp_cycle, FEED, SMR_FEED, SMR_HIGH_PRESSURE, SMR_DISCHARGE_PRESSURE, SMR_COMPRESSOR_EFFICIENCY, SMR_OPERATING_POINTS, smr_example_case)

from .adsorbents import adsorbent, adsorbent_sources, list_adsorbents, load_adsorbents  # noqa: F401

__all__ = ["adsorbent", "adsorbent_sources", "list_adsorbents", "load_adsorbents", 'flue_gas', 'zeolite_13x', 'column', 'model', 'breakthrough_step', 'four_step_vsa', 'smr_adsorbent', 'smr_flue_gas', 'smr_column', 'smr_model', 'smr_four_step_lpp_cycle', 'FEED', 'SMR_FEED', 'SMR_HIGH_PRESSURE', 'SMR_DISCHARGE_PRESSURE', 'SMR_COMPRESSOR_EFFICIENCY', 'SMR_OPERATING_POINTS', 'smr_example_case']
