"""Input checks with clear error messages, shared by the model classes."""

from __future__ import annotations

import numpy as np


def positive(owner, name, value, allow_zero=False):
    """Raise ValueError unless value (or every element) is positive (or non-negative)."""
    values = np.atleast_1d(np.asarray(value, dtype=float))
    bad = values < 0.0 if allow_zero else values <= 0.0
    if np.any(bad) or not np.all(np.isfinite(values)):
        kind = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{owner}: {name} must be {kind} and finite, got {value!r}.")


def fraction(owner, name, value, open_interval=True):
    """Raise ValueError unless 0 < value < 1 (or 0 <= value <= 1)."""
    ok = (0.0 < value < 1.0) if open_interval else (0.0 <= value <= 1.0)
    if not ok:
        raise ValueError(f"{owner}: {name} must lie between 0 and 1, got {value!r}.")


def same_length(owner, **arrays):
    """Raise ValueError unless all arrays have the same length (one value per component)."""
    lengths = {name: len(np.atleast_1d(a)) for name, a in arrays.items() if a is not None}
    if len(set(lengths.values())) > 1:
        detail = ", ".join(f"{k} has {v}" for k, v in lengths.items())
        raise ValueError(f"{owner}: every parameter needs one value per component ({detail}).")
