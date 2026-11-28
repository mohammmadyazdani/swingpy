"""
Faster approach to cyclic steady state by vector acceleration of the cycle map.

Two families are available:

- ``SteffensenAccelerator``: the vector extensions of Steffensen's method
  compared for adsorption cycles by Subraveti, Karimi and Gazzani, Ind. Eng.
  Chem. Res. 65 (2026) 9656, https://doi.org/10.1021/acs.iecr.6c00407
  (Table 1), with their schedule: plain cycles first, then an extrapolation
  every few cycles, and plain cycles only once the residual is small. Their
  best method overall is Graves-Morris.
- ``AndersonAccelerator``: Anderson mixing of the last few cycles.

Anderson acceleration:

One cycle maps the state of the column at its start to the state at its
end, s -> F(s); cyclic steady state is the fixed point s = F(s). Repeating
the cycle converges linearly, and slow modes (a column that keeps warming
or loading a little each cycle) can take hundreds of cycles. Anderson
acceleration combines the last few iterates to extrapolate towards the
fixed point: with residuals g_k = F(s_k) - s_k it finds the weights that
minimise the norm of their combination and moves to the same combination
of the F(s_k).

Safeguards: the extrapolated state is projected back into the physical
range (mole fractions between 0 and 1, positive pressures, loadings and
temperatures), and whenever a step makes the residual grow markedly the
history is dropped and a plain cycle follows. The performance indicators and the
steady state tests are always those of a simulated cycle, never of an
extrapolated state.
"""

from __future__ import annotations

import numpy as np


def _project_state(model, size, profiles):
    """Keep an extrapolated state physical."""
    state = np.concatenate((profiles, np.zeros(model.state_size - size)))
    p = model.unpack(state)
    y = np.clip(p["y"], 0.0, 1.0)
    excess = y.sum(axis=0)
    y = np.where(excess > 1.0, y / np.maximum(excess, 1e-12), y)
    parts = [y.ravel(), np.maximum(p["P"], 1.0e-4), np.maximum(p["x"], 0.0).ravel(),
             np.clip(p["T"], 0.5, 2.0), np.clip(p["T_wall"], 0.5, 2.0)]
    if "shells" in p:
        parts.append(np.maximum(p["shells"], 0.0).ravel())
    return np.concatenate(parts)


def _dot(a, b):
    return float(np.dot(a, b))


"""
Table 1 of Subraveti, Karimi and Gazzani (2026). With D = the state at the start of a cycle, F = F(D) and FF = F(F(D))
the states after one and two cycles, u = F - D, v = FF - F, w = v - u, each method gives the new iterate.
"""
"""Class U of the table: the first four methods; class V: the next four; class W: the last four."""
STEFFENSEN_METHODS = {
    "lemarechal": lambda D, F, FF, u, v, w: D - _dot(u, w) / _dot(w, w) * u,
    "sedogbo": lambda D, F, FF, u, v, w: D - _dot(u, v) / _dot(v, w) * u,
    "a1": lambda D, F, FF, u, v, w: D - _dot(u, u) / _dot(u, w) * u,
    "a4": lambda D, F, FF, u, v, w: FF - _dot(v, v) / _dot(u, w) * u,
    "irons_tuck": lambda D, F, FF, u, v, w: F - _dot(u, w) / _dot(w, w) * v,
    "b1": lambda D, F, FF, u, v, w: D - _dot(u, u) / _dot(v, w) * v,
    "zienkiewicz_lohner": lambda D, F, FF, u, v, w: F - _dot(u, v) / _dot(v, w) * v,
    "graves_morris": lambda D, F, FF, u, v, w: F - _dot(u, v) / _dot(u, w) * v,
    "c1": lambda D, F, FF, u, v, w: D - _dot(u, u) / _dot(w, w) * w,
    "c2": lambda D, F, FF, u, v, w: F - _dot(u, v) / _dot(w, w) * w,
    "macleod_m6": lambda D, F, FF, u, v, w: FF - _dot(v, v) / _dot(w, w) * w,
    "epsilon2": lambda D, F, FF, u, v, w: F - _dot(u, u) / _dot(w, w) * w + _dot(v, v) / _dot(w, w) * u,
}


class SteffensenAccelerator:
    """
    Vector Steffensen extrapolation of the cycle map (Subraveti, Karimi and
    Gazzani 2026).

    The schedule is that of the paper: ``start`` plain cycles, an
    extrapolation at the next one, then an extrapolation after every
    ``every`` plain cycles; once the residual of a cycle falls below
    ``stop_residual`` no further extrapolation is made, so the last cycles
    to steady state are plain ones. Each extrapolation uses the last two
    plain cycles: D -> F(D) -> F(F(D)).

    Args:
        model: The ColumnModel (for the layout of the state).
        method: A key of ``STEFFENSEN_METHODS`` ("graves_morris" by default,
            the best method of the paper).
        start: Plain cycles before the first extrapolation (3 in the paper).
        every: Plain cycles between two extrapolations (3 in the paper, at
            least 2).
        stop_residual: Residual (``Cycle.residual``) below which no more
            extrapolations are made (1e-4 in the paper).
        max_jump: Safeguard, not in the paper: an extrapolation that moves
            the state more than ``max_jump`` times the change over the last
            plain cycle (|new - F(D)| > max_jump |u|) is discarded and the
            plain cycle kept. The scalar parameter of every variant is a
            ratio of inner products whose denominator can pass through zero
            (for Zienkiewicz-Lohner v.w): on the zeolite 13X VSA case one
            extrapolation had lambda = -1980, a jump of 1900 |u|, and the
            next cycle could not be integrated. A slow mode with a
            contraction factor a needs a jump of about 1 / (1 - a), so the
            default 100 keeps modes down to a = 0.99.
    """

    def __init__(self, model, method="graves_morris", start=3, every=3, stop_residual=1.0e-4, max_jump=100.0):
        if method not in STEFFENSEN_METHODS:
            raise ValueError(f"Unknown acceleration method '{method}'; available: {', '.join(STEFFENSEN_METHODS)}.")
        if every < 2 or start < 2:
            raise ValueError("At least two plain cycles are needed before each extrapolation (start, every >= 2).")
        self.model, self.method, self.start, self.every, self.stop_residual = model, method, start, every, stop_residual
        self.formula = STEFFENSEN_METHODS[method]
        self.size = model.state_size - 2 * (model.n_components + 2)
        self.plain, self.first = 0, True
        self.stopped = False
        self.max_jump = max_jump
        self.extrapolations, self.failures, self.rejected = 0, 0, 0
        self.inputs = []

    def next_state(self, state_in, state_out, residual=np.inf):
        """The state to start the next cycle from, given one cycle from ``state_in`` to ``state_out``."""
        out = np.asarray(state_out, dtype=float).copy()
        if residual < self.stop_residual:
            self.stopped = True
        self.plain += 1
        self.inputs = (self.inputs + [np.asarray(state_in, dtype=float)[: self.size]])[-2:]
        due = self.plain >= (self.start if self.first else self.every)
        if self.stopped or not due or len(self.inputs) < 2:
            return out
        D, F, FF = self.inputs[0], self.inputs[1], out[: self.size]
        u, v = F - D, FF - F
        w = v - u
        self.plain, self.first, self.inputs = 0, False, []
        with np.errstate(all="ignore"):
            new = self.formula(D, F, FF, u, v, w)
        if not np.all(np.isfinite(new)):
            self.failures += 1
            return out
        if np.linalg.norm(new - F) > self.max_jump * np.linalg.norm(u):
            self.rejected += 1
            return out
        self.extrapolations += 1
        out[: self.size] = _project_state(self.model, self.size, new)
        return out


class AndersonAccelerator:
    """
    Anderson mixing for the cycle map of one ColumnModel.

    Args:
        model: The ColumnModel (for the layout of the state).
        memory: Number of previous cycles combined (3 to 6 is typical).
        start: Plain cycles before the first extrapolation (lets the fast
            transients die out).
        damping: Fraction of the extrapolated step taken (1 = full step).
        growth: The history is dropped when the residual grows by more than
            this factor after an extrapolation (the residual of a cycle map
            is not monotonic, so a small growth is tolerated).
    """

    def __init__(self, model, memory=5, start=5, damping=1.0, growth=2.0):
        self.model, self.memory, self.start, self.damping, self.growth = model, memory, start, damping, growth
        """Profiles only, not the running integrals."""
        self.size = model.state_size - 2 * (model.n_components + 2)
        self.inputs, self.outputs = [], []
        self.cycles, self.last_norm = 0, np.inf
        self.restarts, self.extrapolations = 0, 0

    def next_state(self, state_in, state_out, residual=None):
        """The state to start the next cycle from, given one cycle from ``state_in`` to ``state_out``."""
        self.cycles += 1
        x, f = np.asarray(state_in, dtype=float)[: self.size], np.asarray(state_out, dtype=float)[: self.size]
        norm = float(np.max(np.abs(f - x)))
        if self.extrapolations and norm > self.growth * self.last_norm:
            """The last extrapolation did not help: forget the history and take a plain cycle."""
            self.inputs, self.outputs = [], []
            self.restarts += 1
        self.last_norm = norm
        self.inputs.append(x)
        self.outputs.append(f)
        self.inputs, self.outputs = self.inputs[-(self.memory + 1):], self.outputs[-(self.memory + 1):]
        if self.cycles < self.start or len(self.inputs) < 2:
            return np.asarray(state_out, dtype=float).copy()
        X = np.array(self.inputs).T
        Fv = np.array(self.outputs).T
        G = Fv - X
        dG = np.diff(G, axis=1)
        try:
            gamma, *_ = np.linalg.lstsq(dG, G[:, -1], rcond=None)
        except np.linalg.LinAlgError:
            return np.asarray(state_out, dtype=float).copy()
        new = Fv[:, -1] - np.diff(Fv, axis=1) @ gamma
        new = Fv[:, -1] + self.damping * (new - Fv[:, -1])
        self.extrapolations += 1
        out = np.asarray(state_out, dtype=float).copy()
        out[: self.size] = self._project(new)
        return out

    def _project(self, profiles):
        return _project_state(self.model, self.size, profiles)
