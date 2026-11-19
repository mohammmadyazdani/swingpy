"""
Cycle steps.

A step is the pair of boundary conditions at the two column ends, a
duration, and optionally the ambient temperature of the column wall during
the step (which turns a pressure swing cycle into a temperature swing one).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .boundaries import Boundary, Closed, InstantFill


@dataclass
class Step:
    """
    One step of a cycle.

    Attributes:
        name: Display name, for example "adsorption".
        duration: Step time, s.
        feed_end: Boundary condition at Z = 0.
        product_end: Boundary condition at Z = 1.
        ambient_temperature: Temperature outside the wall during this step, K.
            None keeps the column's own ambient temperature.
        stop: Optional condition that ends the step early, such as
            ``PressureReached``; ``duration`` is then the longest it may last.
            A step whose condition already holds at its start is skipped.
    """

    name: str
    duration: float
    feed_end: Boundary = None
    product_end: Boundary = None
    ambient_temperature: float = None
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


def instant_pressurization(name, composition, pressure, stream="feed"):
    """
    A pressurization taken as instantaneous with the solid frozen: the gas in
    the column becomes ``composition`` at ``pressure`` (bar) at once, with
    the loadings and temperatures unchanged. Used when the pressurization is
    much faster than the uptake, for example in direct air capture
    (Balasubramaniam et al., Chem. Eng. J. 485 (2024) 149568: "the column is
    considered to be pressurized and filled with ambient air instantaneously
    ... the solid phase is assumed to be frozen").

    The moles that enter count as the stream ``stream`` (the feed by
    default); no compression work is charged.
    """
    return Step(name, 0.0, feed_end=InstantFill(composition, pressure, stream=stream), product_end=Closed())
