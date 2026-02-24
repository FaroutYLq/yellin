"""
Candidate intervals and CMax for the optimum interval method.
"""

import numpy as np

from yellin.c_infinity import c_infinity
from yellin.transform import fm_from_F


def candidate_intervals(F_values: np.ndarray) -> list[tuple[int, float]]:
    """
    Find candidate (n, f) pairs: for each n, interval with n events
    and largest f. Intervals begin just after one event, end just before another.

    Returns:
        List of (n, f) where n=observed count, f=fraction of range.
    """
    F = np.concatenate([[0.0], np.sort(F_values), [1.0]])
    n_events = len(F_values)
    candidates: list[tuple[int, float]] = []

    for n in range(n_events + 1):
        # Interval from just after F[i] to just before F[j] has n events
        # when j = i + n + 1. f = F[j] - F[i].
        i_max = n_events - n + 1
        if i_max <= 0:
            break
        j_end = n + 1 + i_max
        if j_end > len(F):
            break
        f_vals = F[n + 1 : j_end] - F[:i_max]
        best_f = float(np.max(f_vals))
        if best_f > 0:
            candidates.append((n, best_f))

    return candidates


def compute_CMax(
    F_values: np.ndarray,
    mu: float,
    fmin: float,
    c_infinity_fn=None,
) -> float:
    """
    CMax = max of C∞(y; f) over candidate intervals with f > fmin.

    Uses effective fmin = max(fm, user_fmin) where fm = largest_gap/mu.
    """
    c_inf_fn = c_infinity_fn or c_infinity
    fm = fm_from_F(F_values) / mu if mu > 0 else 1.0
    eff_fmin = max(fm, fmin)

    candidates = candidate_intervals(F_values)
    best = 0.0
    for n, f in candidates:
        if f <= eff_fmin:
            continue
        x = mu * f
        if x <= 0:
            continue
        y = (n - x) / np.sqrt(x)
        c = c_inf_fn(y, f)
        if c > best:
            best = c
    return best
