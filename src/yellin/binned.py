"""
Binned-data utilities for the optimum interval method (paper Section IV).

Data are represented as counts in equal-width bins in F-space, where
F is the assumed signal CDF. Candidate intervals are one or more
consecutive bins.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from yellin.c_infinity import c_infinity
from yellin.transform import events_to_F


def _validate_counts(counts: np.ndarray) -> np.ndarray:
    """Validate 1D non-negative integer counts."""
    arr = np.asarray(counts)
    if arr.ndim != 1:
        raise ValueError("counts must be a 1D array")
    if arr.size == 0:
        raise ValueError("counts must contain at least one bin")
    if np.any(arr < 0):
        raise ValueError("counts must be non-negative")
    if not np.all(np.equal(arr, np.floor(arr))):
        raise ValueError("counts must contain integer values")
    return arr.astype(np.int64, copy=False)


def events_to_binned_counts(
    events: np.ndarray,
    spectrum_cdf,
    n_bins: int = 1000,
    clip: bool = True,
) -> np.ndarray:
    """
    Convert unbinned events into equal-width F-space bin counts.

    Args:
        events: 1D array of event values (e.g. energies).
        spectrum_cdf: Signal CDF F(s) -> [0, 1].
        n_bins: Number of equal-width bins in F-space.
        clip: Passed to events_to_F; if True, out-of-range values are clipped.

    Returns:
        Integer array of length n_bins with counts per F bin.
    """
    if n_bins < 1:
        raise ValueError("n_bins must be >= 1")

    F_values = events_to_F(events, spectrum_cdf, clip=clip)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    counts, _ = np.histogram(F_values, bins=edges)
    return counts.astype(np.int64)


def compute_CMax_binned(
    counts: np.ndarray,
    mu: float,
    fmin: float = 0.0,
    c_infinity_fn: Callable[[float, float], float] | None = None,
) -> float:
    """
    CMax for binned data using consecutive-bin intervals (paper Section IV).

    Args:
        counts: 1D non-negative integer counts in equal-width F bins.
        mu: Total expected events under the signal hypothesis.
        fmin: Minimum interval fraction.
        c_infinity_fn: Optional C∞(y, f) callable.

    Returns:
        Maximum C∞ over all consecutive-bin intervals with f > fmin.
    """
    if not (0.0 <= fmin < 1.0):
        raise ValueError("fmin must satisfy 0 <= fmin < 1")
    if mu <= 0:
        return 0.0

    arr = _validate_counts(counts)
    n_bins = arr.size
    c_inf_fn = c_infinity_fn or c_infinity

    prefix = np.concatenate(([0], np.cumsum(arr, dtype=np.int64)))
    min_width = max(1, int(np.floor(fmin * n_bins)) + 1)

    best = 0.0
    for width in range(min_width, n_bins + 1):
        f = width / n_bins
        x = mu * f
        if x <= 0:
            continue

        # For fixed width, C∞ is maximized by the interval with the smallest n.
        window_counts = prefix[width:] - prefix[:-width]
        n_min = float(np.min(window_counts))
        y = (n_min - x) / np.sqrt(x)
        c = c_inf_fn(y, f)
        if c > best:
            best = c

    return float(best)


__all__ = [
    "events_to_binned_counts",
    "compute_CMax_binned",
]

