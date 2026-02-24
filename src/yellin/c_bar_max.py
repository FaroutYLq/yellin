"""
C̄Max(C, fmin, μ) - calibration for optimum interval confidence level.

C̄Max is the value such that fraction C of experiments (no background)
have CMax < C̄Max. Tabulated and extrapolated for μ > 15310.
"""

import pickle
from pathlib import Path

import numpy as np


def _default_table_path() -> Path:
    pkl = Path(__file__).parent / "data" / "c_bar_max_table.pkl"
    if pkl.exists():
        return pkl
    return Path(__file__).parent / "data" / "c_bar_max_table.npz"


def c_bar_max(
    C: float,
    fmin: float,
    mu: float,
    table_path: Path | None = None,
) -> float:
    """
    C̄Max(C, fmin, μ) - calibration value for confidence level C.

    For μ > 15310 uses extrapolation A + B/sqrt(μ).

    Args:
        C: Confidence level (e.g. 0.9, 0.95).
        fmin: Minimum interval fraction.
        mu: Expected total events (signal + known background).
    """
    path = Path(table_path) if table_path else _default_table_path()
    if not path.exists():
        raise FileNotFoundError(
            f"C̄Max table not found. Run scripts/generate_c_bar_max_table.py"
        )
    if path.suffix == ".pkl":
        with open(path, "rb") as f:
            data = pickle.load(f)
        tables = data["tables"]
        fits = data["fits"]
    else:
        data = np.load(path, allow_pickle=True)
        tables = data["tables"].item()
        fits = data["fits"].item()

    key = (round(C, 4), round(fmin, 4))
    if key not in tables:
        raise ValueError(f"No table for C={C}, fmin={fmin}")

    mu_vals, cbar_vals = tables[key]
    if mu <= mu_vals[-1]:
        return float(np.interp(mu, mu_vals, cbar_vals))

    if key in fits:
        A, B = fits[key]
        return float(A + B / np.sqrt(mu))
    return float(np.interp(mu, mu_vals, cbar_vals))
