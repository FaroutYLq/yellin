"""
C̄Max calibration for the binned optimum interval method (paper Section IV).
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

from yellin.c_bar_max import c_bar_max


def _default_table_path() -> Path:
    pkl = Path(__file__).parent / "data" / "c_bar_max_binned_table.pkl"
    if pkl.exists():
        return pkl
    return Path(__file__).parent / "data" / "c_bar_max_binned_table.npz"


def c_bar_max_binned(
    C: float,
    fmin: float,
    mu: float,
    n_bins: int,
    table_path: Path | None = None,
    fallback_unbinned: bool = True,
) -> float:
    """
    C̄Max(C, fmin, mu) for binned intervals.

    Table keys are (n_bins, C, fmin). If the table is unavailable or the key is
    missing and fallback_unbinned=True, this falls back to the unbinned
    calibration c_bar_max(C, fmin, mu).
    """
    if n_bins < 1:
        raise ValueError("n_bins must be >= 1")
    if mu <= 0:
        raise ValueError("mu must be > 0")

    path = Path(table_path) if table_path else _default_table_path()
    key = (int(n_bins), round(C, 4), round(fmin, 4))

    if path.exists():
        if path.suffix == ".pkl":
            with open(path, "rb") as f:
                data = pickle.load(f)
            tables = data["tables"]
            fits = data["fits"]
        else:
            data = np.load(path, allow_pickle=True)
            tables = data["tables"].item()
            fits = data["fits"].item()

        if key in tables:
            mu_vals, cbar_vals = tables[key]
            if mu <= mu_vals[-1]:
                return float(np.interp(mu, mu_vals, cbar_vals))
            if key in fits:
                A, B = fits[key]
                return float(A + B / np.sqrt(mu))
            return float(np.interp(mu, mu_vals, cbar_vals))

    if fallback_unbinned:
        return c_bar_max(C, fmin, mu)

    raise FileNotFoundError(
        "Binned C̄Max table not found or missing key. "
        "Run scripts/generate_c_bar_max_binned_table.py or enable fallback_unbinned."
    )


__all__ = ["c_bar_max_binned"]

