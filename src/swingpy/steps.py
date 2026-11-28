"""
Cycle steps.

A step is the pair of boundary conditions at the two column ends and a
duration, with an optional condition that ends it early.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .boundaries import Boundary, Closed


@dataclass
class Step:
    """
    One step of a cycle.

    Attributes:
        name: Display name, for example "adsorption".
        duration: Step time, s.
        feed_end: Boundary condition at Z = 0.
        product_end: Boundary condition at Z = 1.
        stop: Optional condition that ends the step early, such as
            ``PressureReached``; ``duration`` is then the longest it may last.
            A step whose condition already holds at its start is skipped.
    """

    name: str
    duration: float
    feed_end: Boundary = None
    product_end: Boundary = None
    stop: object = None

    def __post_init__(self):
        if not self.duration >= 0.0:
            raise ValueError(f"Step {self.name}: duration must be zero or positive (s), got {self.duration!r}.")
        self.feed_end = self.feed_end or Closed()
        self.product_end = self.product_end or Closed()

    @property
    def direction(self) -> int:
        """
        +1 when the gas flows from the feed end towards the product end,
        -1 for the opposite direction. Used to pick the upwind side of the
        face reconstruction. A step with both ends closed counts as +1.
        """
        if self.feed_end.role == "outlet" or self.product_end.role == "inlet":
            return -1
        return 1


@dataclass
class PressureReached:
    """
    Ends a step when the pressure at a location reaches a target.

    Attributes:
        pressure: Target, bar.
        location: "feed" (first cell), "product" (last cell) or "mean".
        rising: True when the pressure rises to the target (pressurization),
            False when it falls to it (blowdown, evacuation).
    """

    pressure: float
    location: str = "feed"
    rising: bool = True

    def value(self, model, state):
        """Zero at the target, positive once it is reached."""
        P = model.unpack(state)["P"]
        current = {"feed": P[0], "product": P[-1]}.get(self.location, None)
        current = float(np.mean(P)) if current is None else float(current)
        difference = current - model.to_pressure(self.pressure)
        return difference if self.rising else -difference
