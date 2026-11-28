"""
Problem specifications: the whole problem (components, gas, adsorbent, bed,
model options, streams, initial state and cycle) as a YAML or JSON file or a
dict, with units on every quantity and no hidden physical defaults.

    from swingpy.spec import load_problem
    problem = load_problem("My process.yaml")
    run = problem.run(max_cycles=300)
"""

from .builder import SPEC_VERSION, Problem, SpecError, load_problem
from .export import save_problem, to_spec
from .units import BASE_UNITS, UNITS, UnitError, quantity

__all__ = ["SPEC_VERSION", "Problem", "SpecError", "load_problem", "save_problem", "to_spec", "BASE_UNITS", "UNITS", "UnitError", "quantity"]
