"""
Recorded gas streams that couple one step of a cycle to another.

In a multi-bed plant the effluent of one column feeds another one at the
same time: the raffinate of the adsorption step pressurizes a column (light
product pressurization) or purges it (light reflux), and part of the
extract is sent back to the feed end (heavy reflux). SwingPy simulates a
single column, so such a stream is recorded while one step runs and played
back when the receiving step runs (the single-bed or "unibed" approach).
Records are replayed interval by interval, conserving the recorded moles.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class StreamRecord:
    """
    Time history of a gas stream leaving a column end.

    Attributes:
        time: Time from the start of the source step, s.
        molar_flow: Total molar flow, mol/s (positive).
        mole_fractions: Composition, shape (n_components, len(time)).
        temperature: K.
        pressure: Face pressure, bar (used when another step needs the
            pressure history of this one, for example a valve venting into it).
        cumulative: Moles of each component that have left since the start
            of the step, at each time, exactly as the model integrated them
            (shape components x times). When given, ``total_moles`` uses it
            rather than integrating the sampled flow, which misses or
            exaggerates sharp transients such as the first instants of an
            evacuation.
    """

    time: np.ndarray
    molar_flow: np.ndarray
    mole_fractions: np.ndarray
    temperature: np.ndarray
    pressure: np.ndarray = None
    cumulative: np.ndarray = None

    def pressure_at(self, time_s):
        """Face pressure at a time, bar (held at the ends of the record)."""
        return float(np.interp(time_s, self.time, self.pressure))

    def mean_composition(self):
        """Flow-weighted mean composition of the stream."""
        if self.cumulative is not None:
            moles = np.maximum(self.cumulative[:, -1] - self.cumulative[:, 0], 0.0)
            if moles.sum() > 0.0:
                return moles / moles.sum()
        weights = np.trapezoid(self.mole_fractions * self.molar_flow, self.time, axis=1) \
            if hasattr(np, "trapezoid") else np.trapz(self.mole_fractions * self.molar_flow, self.time, axis=1)
        total = weights.sum()
        return weights / total if total > 0.0 else self.mole_fractions[:, -1] / self.mole_fractions[:, -1].sum()

    @property
    def duration(self) -> float:
        return float(self.time[-1])

    def at(self, time_s):
        """
        Molar flow (mol/s), composition and temperature at a time; zero flow outside the record.

        With ``cumulative`` the flow is the mean over the sampling interval
        that contains the time (moles that left in the interval over its
        length), so that a replay returns exactly the recorded moles even
        when the source step starts with a sharp transient (a pressure
        equalization) that the sampled flow cannot resolve. Without it the
        sampled flow and composition are interpolated linearly.
        """
        if time_s < self.time[0] or time_s > self.time[-1]:
            return 0.0, self.mole_fractions[:, -1], float(self.temperature[-1])
        if self.cumulative is not None and len(self.time) > 1:
            k = min(max(int(np.searchsorted(self.time, time_s, side="right")) - 1, 0), len(self.time) - 2)
            moles = np.maximum(self.cumulative[:, k + 1] - self.cumulative[:, k], 0.0)
            span = self.time[k + 1] - self.time[k]
            temperature = float(np.interp(time_s, self.time, self.temperature))
            if moles.sum() <= 0.0 or span <= 0.0:
                return 0.0, self.mole_fractions[:, k] / self.mole_fractions[:, k].sum(), temperature
            return float(moles.sum() / span), moles / moles.sum(), temperature
        flow = float(np.interp(time_s, self.time, self.molar_flow))
        y = np.array([np.interp(time_s, self.time, row) for row in self.mole_fractions])
        return max(flow, 0.0), y / y.sum(), float(np.interp(time_s, self.time, self.temperature))

    def total_moles(self, start=0.0, end=None):
        """Moles of each component between two times, mol."""
        end = self.duration if end is None else end
        if self.cumulative is not None:
            at = lambda t: np.array([np.interp(t, self.time, row) for row in self.cumulative])
            return at(end) - at(start)
        mask = (self.time >= start) & (self.time <= end)
        flows = self.molar_flow[mask] * self.mole_fractions[:, mask]
        return np.trapezoid(flows, self.time[mask], axis=1) if mask.sum() > 1 else np.zeros(len(self.mole_fractions))
