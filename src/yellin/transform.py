"""
Transform events to F-space for the optimum interval method.

F(s) = X(s)/mu is the CDF of the signal; events in F-space are in [0, 1].
"""

from typing import Callable

import numpy as np

from yellin.spectrum import SpectrumCDF


def events_to_F(
    events: np.ndarray,
    spectrum_cdf: SpectrumCDF | Callable[..., np.ndarray],
    clip: bool = True,
) -> np.ndarray:
    """
    Map event values to F-space [0, 1] using the signal CDF.

    Args:
        events: 1D array of event values (e.g. energies).
        spectrum_cdf: Callable F(s) -> [0,1], or SpectrumCDF instance.
        clip: If True, clip out-of-range F values to 0 or 1.
              If False, drop events outside valid range.

    Returns:
        Sorted F-values in [0, 1]. Empty array if no valid events.
    """
    events = np.asarray(events, dtype=float).ravel()
    if events.size == 0:
        return np.array([], dtype=float)

    cdf = spectrum_cdf._cdf if hasattr(spectrum_cdf, "_cdf") else spectrum_cdf
    F = np.clip(cdf(events), 0.0, 1.0) if clip else cdf(events)

    if not clip:
        mask = (F >= 0) & (F <= 1)
        F = F[mask]

    return np.sort(F.astype(float))


def fm_from_F(F_values: np.ndarray) -> float:
    """
    Largest gap between adjacent F-values (including boundaries 0, 1).

    Used to set minimum f: fm = largest_gap. Intervals with f < fm
    cannot be optimum (paper Section II).
    """
    if F_values.size == 0:
        return 1.0  # Entire range is gap

    F = np.concatenate([[0.0], np.sort(F_values), [1.0]])
    gaps = np.diff(F)
    return float(np.max(gaps))
