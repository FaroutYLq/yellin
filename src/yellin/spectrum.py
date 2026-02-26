"""
Spectrum CDF abstraction for the optimum interval method.

The signal cumulative distribution F(s) maps event variable s to [0,1],
representing the fraction of expected signal events below s.
"""

from typing import Callable

import numpy as np


def uniform_spectrum(s_min: float, s_max: float) -> Callable[[np.ndarray], np.ndarray]:
    """
    Uniform signal distribution over [s_min, s_max].

    Returns a callable F(s) = (s - s_min) / (s_max - s_min), clipped to [0, 1].
    """
    span = s_max - s_min
    if span <= 0:
        raise ValueError("s_max must be greater than s_min")

    def F(s: np.ndarray) -> np.ndarray:
        return np.clip((np.asarray(s, dtype=float) - s_min) / span, 0.0, 1.0)

    return F


def pdf_to_cdf(
    pdf: Callable[[np.ndarray], np.ndarray] | Callable[[float], float],
    s_min: float,
    s_max: float,
    n_grid: int = 4096,
    return_mu: bool = False,
    **pdf_kwargs,
) -> (
    Callable[[np.ndarray], np.ndarray]
    | tuple[Callable[[np.ndarray], np.ndarray], float]
):
    """
    Build a normalized CDF from a non-negative rate/PDF on [s_min, s_max].

    The input function can be any non-negative shape proportional to a PDF
    or a physical rate dN/ds. Absolute normalization is inferred numerically.
    If `return_mu=True`, the integrated rate
    `mu = integral_{s_min}^{s_max} pdf(s) ds` is returned together with the CDF.
    Additional keyword arguments are forwarded to `pdf`.
    """
    if s_max <= s_min:
        raise ValueError("s_max must be greater than s_min")
    if n_grid < 2:
        raise ValueError("n_grid must be >= 2")

    s_grid = np.linspace(s_min, s_max, n_grid)
    try:
        p_grid = np.asarray(pdf(s_grid, **pdf_kwargs), dtype=float)
        if p_grid.shape != s_grid.shape:
            p_grid = np.vectorize(
                lambda x: float(pdf(float(x), **pdf_kwargs)),
                otypes=[float],
            )(s_grid)
    except Exception:
        p_grid = np.vectorize(
            lambda x: float(pdf(float(x), **pdf_kwargs)),
            otypes=[float],
        )(s_grid)

    if not np.all(np.isfinite(p_grid)):
        raise ValueError("pdf must return finite values on [s_min, s_max]")

    # Allow tiny negative numerical noise only.
    neg_tol = 1e-12 * max(1.0, float(np.max(np.abs(p_grid))))
    if np.any(p_grid < -neg_tol):
        raise ValueError("pdf must be non-negative on [s_min, s_max]")
    p_grid = np.maximum(p_grid, 0.0)

    ds = np.diff(s_grid)
    increments = 0.5 * (p_grid[:-1] + p_grid[1:]) * ds
    cumulative = np.concatenate(([0.0], np.cumsum(increments)))
    total = float(cumulative[-1])
    if total <= 0:
        raise ValueError("pdf integral on [s_min, s_max] must be positive")

    cdf_grid = cumulative / total
    cdf_grid = np.maximum.accumulate(np.clip(cdf_grid, 0.0, 1.0))
    cdf_grid[0] = 0.0
    cdf_grid[-1] = 1.0

    def F(s: np.ndarray) -> np.ndarray:
        s_arr = np.asarray(s, dtype=float)
        return np.interp(s_arr, s_grid, cdf_grid, left=0.0, right=1.0)

    if return_mu:
        return F, total
    return F


class SpectrumCDF:
    """
    Callable spectrum CDF F(s) -> [0, 1].

    The cumulative distribution of the signal; F(s) is the fraction of
    expected signal events below s. Used to transform events to F-space.
    """

    def __init__(self, cdf: Callable[[np.ndarray], np.ndarray]) -> None:
        self._cdf = cdf

    def __call__(self, s: np.ndarray) -> np.ndarray:
        return self._cdf(np.asarray(s, dtype=float))
