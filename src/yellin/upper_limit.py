"""
Upper limit solver for the optimum interval method.

Solves CMax(μ) = C̄Max(C, fmin, μ) for μ.
"""

import numpy as np
from scipy.optimize import brentq

from yellin.binned import compute_CMax_binned
from yellin.c_bar_max import c_bar_max
from yellin.c_bar_max_binned import c_bar_max_binned
from yellin.intervals import compute_CMax
from yellin.transform import events_to_F


def upper_limit(
    events: np.ndarray,
    spectrum_cdf,
    C: float = 0.9,
    fmin: float = 0.0,
    known_background: float = 0.0,
    mu: float | None = None,
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
        mu: Optional model signal expectation on [s_min, s_max]. If provided,
            returns the upper limit on signal strength (mu_ul / mu).

    Returns:
        If mu is None: upper limit on signal (total expected events -
        known_background). If mu is provided: upper limit on signal strength.
    """
    if mu is not None and (not np.isfinite(mu) or mu <= 0):
        raise ValueError("mu must be a positive finite number when provided")

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

    ul_signal = max(0.0, mu_total - known_background)
    if mu is not None:
        return ul_signal / mu
    return ul_signal


def upper_limit_binned(
    counts: np.ndarray,
    C: float = 0.9,
    fmin: float = 0.0,
    known_background: float = 0.0,
    mu: float | None = None,
    binned_table_path=None,
    fallback_unbinned_cbar: bool = True,
) -> float:
    """
    Compute C-confidence upper limit using binned data (paper Section IV).

    Args:
        counts: 1D integer counts in equal-width F-space bins.
        C: Confidence level (default 0.9).
        fmin: Minimum interval fraction (default 0).
        known_background: Known background to subtract from result.
        mu: Optional model signal expectation in the analyzed region. If
            provided, returns the upper limit on signal strength (mu_ul / mu).
        binned_table_path: Optional path to binned C̄Max table.
        fallback_unbinned_cbar: If True, use unbinned C̄Max table when a
            dedicated binned table is unavailable.

    Returns:
        If mu is None: upper limit on signal (total expected events -
        known_background). If mu is provided: upper limit on signal strength.
    """
    if mu is not None and (not np.isfinite(mu) or mu <= 0):
        raise ValueError("mu must be a positive finite number when provided")

    counts = np.asarray(counts)
    if counts.ndim != 1:
        raise ValueError("counts must be a 1D array")
    if counts.size == 0:
        return 0.0
    if np.any(counts < 0):
        raise ValueError("counts must be non-negative")

    if np.sum(counts) == 0:
        return 0.0

    n_bins = counts.size

    def residual(mu: float) -> float:
        cmax = compute_CMax_binned(counts, mu, fmin)
        cbar = c_bar_max_binned(
            C,
            fmin,
            mu,
            n_bins=n_bins,
            table_path=binned_table_path,
            fallback_unbinned=fallback_unbinned_cbar,
        )
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

    ul_signal = max(0.0, mu_total - known_background)
    if mu is not None:
        return ul_signal / mu
    return ul_signal
