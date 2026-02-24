"""
Upper limit solver for the optimum interval method.

Solves CMax(μ) = C̄Max(C, fmin, μ) for μ.
"""

import numpy as np
from scipy.optimize import brentq

from yellin.c_bar_max import c_bar_max
from yellin.intervals import compute_CMax
from yellin.transform import events_to_F


def upper_limit(
    events: np.ndarray,
    spectrum_cdf,
    C: float = 0.9,
    fmin: float = 0.0,
    known_background: float = 0.0,
) -> float:
    """
    Compute C-confidence upper limit on signal using high-statistics
    optimum interval method (arXiv:0709.2701).

    Args:
        events: 1D array of event values (e.g. energies).
        spectrum_cdf: Signal CDF F(s) -> [0,1].
        C: Confidence level (default 0.9).
        fmin: Minimum interval fraction (default 0).
        known_background: Known background to subtract from result.

    Returns:
        Upper limit on signal (total expected events - known_background).
    """
    F_values = events_to_F(events, spectrum_cdf)
    if F_values.size == 0:
        return 0.0

    def residual(mu: float) -> float:
        cmax = compute_CMax(F_values, mu, fmin)
        cbar = c_bar_max(C, fmin, mu)
        return cmax - cbar

    mu_low, mu_high = 1.0, 1e7
    r_high = residual(mu_high)
    r_low = residual(mu_low)
    while r_high <= 0 and mu_high < 1e10:
        mu_high *= 2
        r_high = residual(mu_high)
    while r_low >= 0 and mu_low > 1e-6:
        mu_low /= 2
        r_low = residual(mu_low)

    try:
        mu_total = brentq(residual, mu_low, mu_high, xtol=1e-4, maxiter=100)
    except ValueError:
        return 0.0

    return max(0.0, mu_total - known_background)
