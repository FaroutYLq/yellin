#!/usr/bin/env python3
"""
Generate C̄Max(C, fmin, μ) lookup tables via Monte Carlo.

Run once to produce src/yellin/data/c_bar_max_table.npz.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from yellin.c_infinity import c_infinity
from yellin.intervals import compute_CMax


def _simulate_cmax(mu: float, fmin: float, n_trials: int, rng) -> np.ndarray:
    """Simulate CMax distribution for given mu, fmin (no background)."""
    cmax_vals = np.empty(n_trials)
    for i in range(n_trials):
        n_evt = rng.poisson(mu)
        F = np.sort(rng.uniform(0, 1, n_evt)) if n_evt > 0 else np.array([])
        cmax_vals[i] = compute_CMax(F, mu, fmin, c_infinity)
    return cmax_vals


def _build_tables(
    C_vals: list[float],
    fmin_vals: list[float],
    mu_vals: list[float],
    n_trials: int = 5000,
    seed: int = 42,
) -> tuple[dict, dict]:
    """Build C̄Max tables and fits for A + B/sqrt(mu)."""
    rng = np.random.default_rng(seed)
    tables: dict[tuple[float, float], tuple[np.ndarray, np.ndarray]] = {}
    fits: dict[tuple[float, float], tuple[float, float]] = {}

    for C in C_vals:
        for fmin in fmin_vals:
            key = (C, fmin)
            cbar_list = []
            for mu in mu_vals:
                cmax = _simulate_cmax(mu, fmin, n_trials, rng)
                cbar = np.quantile(cmax, C)
                cbar_list.append(cbar)
            tables[key] = (np.array(mu_vals), np.array(cbar_list))

            # Fit C̄Max = A + B/sqrt(mu) for mu > 3500
            high = np.array(mu_vals) > 3500
            if np.sum(high) >= 3:
                mu_h = np.array(mu_vals)[high]
                cbar_h = np.array(cbar_list)[high]
                inv_sqrt = 1 / np.sqrt(mu_h)
                coef, const = np.polyfit(inv_sqrt, cbar_h, 1)
                fits[key] = (float(const), float(coef))  # A + B/sqrt(mu)

    return tables, fits


if __name__ == "__main__":
    out = Path(__file__).resolve().parents[1] / "src" / "yellin" / "data"
    out.mkdir(parents=True, exist_ok=True)

    C_vals = [0.9, 0.95]
    fmin_vals = [0.0, 0.02, 0.1, 0.2, 0.5]
    mu_vals = [55, 80, 120, 200, 350, 500, 800, 1200, 2000]

    print("Generating C̄Max tables (this may take 5-10 minutes)...")
    tables, fits = _build_tables(C_vals, fmin_vals, mu_vals, n_trials=200)

    import pickle
    with open(out / "c_bar_max_table.pkl", "wb") as f:
        pickle.dump({"tables": tables, "fits": fits}, f)
    print(f"Table saved to {out / 'c_bar_max_table.pkl'}")
    print("Fits (A, B) for C̄Max = A + B/sqrt(mu):")
    for k, (A, B) in fits.items():
        print(f"  C={k[0]}, fmin={k[1]}: A={A:.4f}, B={B:.4f}")
