"""
C∞(y; f) - Gaussian/Brownian limit for optimum interval method.

C∞(y; f) = P(ymin > y) where ymin is the minimum of [w(t+f)-w(t)]/sqrt(f)
over t in [0, 1-f] for Wiener process w. Computed via Monte Carlo and
interpolation from a precomputed table.
"""

import math
from pathlib import Path

import numpy as np
from scipy.interpolate import RegularGridInterpolator


def _simulate_ymin(
    f: float,
    n_sub: int = 5000,
    n_trials: int = 100_000,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Monte Carlo: sample ymin for given f (Appendix A).

    Discretize [0,1] into n_sub intervals; each contributes N(0,1/n_sub).
    Sliding window of length f; y = (w(t+f)-w(t))/sqrt(f) has unit variance.
    """
    rng = rng or np.random.default_rng()
    k = max(1, int(round(f * n_sub)))
    ymin = np.empty(n_trials)

    for i in range(n_trials):
        inc = rng.standard_normal(n_sub) / math.sqrt(n_sub)
        w = np.concatenate([[0.0], np.cumsum(inc)])
        y_vals = (w[k:] - w[:-k]) / math.sqrt(f)
        ymin[i] = np.min(y_vals)

    return ymin


def _build_table(
    f_grid: np.ndarray,
    y_grid: np.ndarray,
    n_sub: int = 5000,
    n_trials_per_f: int = 50_000,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Build C∞ table: C[i,j] = P(ymin > y_grid[i]) at f_grid[j].
    """
    rng = rng or np.random.default_rng()
    C = np.zeros((len(y_grid), len(f_grid)))

    for j, f in enumerate(f_grid):
        if f <= 0 or f >= 1:
            C[:, j] = 0.0 if f >= 1 else 1.0
            continue
        ys = _simulate_ymin(f, n_sub, n_trials_per_f, rng)
        for i, y in enumerate(y_grid):
            C[i, j] = np.mean(ys > y)

    return C


def _default_table_path() -> Path:
    return Path(__file__).parent / "data" / "c_infinity_table.npz"


def _compute_and_save_table(
    table_path: Path | None = None,
    f_min: float = 0.01,
    f_max: float = 0.99,
    n_f: int = 30,
    y_min: float = -4.0,
    y_max: float = 4.0,
    n_y: int = 81,
    n_sub: int = 5000,
    n_trials: int = 30_000,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate C∞ table and save to npz. Returns (y_grid, f_grid, C_table)."""
    rng = np.random.default_rng(seed)
    f_grid = np.linspace(f_min, f_max, n_f)
    y_grid = np.linspace(y_min, y_max, n_y)
    C = _build_table(f_grid, y_grid, n_sub, n_trials, rng)

    path = table_path or _default_table_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, y_grid=y_grid, f_grid=f_grid, C=C)
    return y_grid, f_grid, C


def _load_table(table_path: Path | None = None) -> RegularGridInterpolator:
    """Load C∞ table and return interpolator."""
    path = table_path or _default_table_path()
    if not path.exists():
        raise FileNotFoundError(
            f"C∞ table not found at {path}. Run scripts/generate_c_infinity_table.py"
        )
    data = np.load(path)
    y_grid = data["y_grid"]
    f_grid = data["f_grid"]
    C = data["C"]
    interp = RegularGridInterpolator((y_grid, f_grid), C, bounds_error=False)
    return interp


_interpolator: RegularGridInterpolator | None = None


def c_infinity(y: float, f: float, table_path: Path | None = None) -> float:
    """
    C∞(y; f) = P(ymin > y) for Brownian minimum.

    Args:
        y: Scaled excess (n-x)/sqrt(x).
        f: Interval fraction x/mu in (0, 1).

    Returns:
        Probability in [0, 1].
    """
    global _interpolator
    if _interpolator is None:
        _interpolator = _load_table(table_path)

    pt = np.array([[np.clip(y, -4.0, 4.0), np.clip(f, 0.01, 0.99)]])
    result = float(_interpolator(pt)[0])
    return np.clip(result, 0.0, 1.0)


def c_infinity_mc(y: float, f: float, n_trials: int = 100_000) -> float:
    """
    Direct Monte Carlo for C∞ (slow, for testing/validation).
    """
    ys = _simulate_ymin(f, n_sub=5000, n_trials=n_trials)
    return float(np.mean(ys > y))
