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
