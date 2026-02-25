#!/usr/bin/env python3
"""
Generate C̄Max(C, fmin, mu) table for binned-data optimum interval (Section IV).

The output is keyed by (n_bins, C, fmin) and saved to:
src/yellin/data/c_bar_max_binned_table.pkl
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from yellin.binned import compute_CMax_binned
from yellin.c_infinity import c_infinity


def _simulate_cmax_binned(
    mu: float,
    fmin: float,
    n_bins: int,
    n_trials: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Simulate CMax distribution for binned data (no background)."""
    cmax_vals = np.empty(n_trials, dtype=float)
    lam = mu / n_bins
    for i in range(n_trials):
        counts = rng.poisson(lam=lam, size=n_bins)
        cmax_vals[i] = compute_CMax_binned(counts, mu, fmin, c_infinity)
    return cmax_vals


def _build_tables(
    C_vals: list[float],
    fmin_vals: list[float],
    mu_vals: list[float],
    n_bins: int,
    n_trials: int,
    seed: int,
) -> tuple[dict, dict]:
    """Build binned C̄Max tables and high-mu fits A + B/sqrt(mu)."""
    rng = np.random.default_rng(seed)
    tables: dict[tuple[int, float, float], tuple[np.ndarray, np.ndarray]] = {}
    fits: dict[tuple[int, float, float], tuple[float, float]] = {}

    mu_arr = np.array(mu_vals, dtype=float)
    for C in C_vals:
        for fmin in fmin_vals:
            key = (int(n_bins), round(C, 4), round(fmin, 4))
            cbar_vals = np.empty_like(mu_arr, dtype=float)

            for i, mu in enumerate(mu_arr):
                cmax = _simulate_cmax_binned(mu, fmin, n_bins, n_trials, rng)
                cbar_vals[i] = np.quantile(cmax, C)

            tables[key] = (mu_arr.copy(), cbar_vals.copy())

            high = mu_arr > 500
            if np.sum(high) >= 3:
                inv = 1 / np.sqrt(mu_arr[high])
                coef, const = np.polyfit(inv, cbar_vals[high], 1)
                fits[key] = (float(const), float(coef))

    return tables, fits


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-bins", type=int, default=1000)
    parser.add_argument("--n-trials", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--mu-grid",
        type=str,
        default="55,100,200,400,700,1200,2000",
        help="Comma-separated mu values.",
    )
    args = parser.parse_args()

    if args.n_bins < 1:
        raise ValueError("--n-bins must be >= 1")
    if args.n_trials < 1:
        raise ValueError("--n-trials must be >= 1")

    mu_vals = [float(x.strip()) for x in args.mu_grid.split(",") if x.strip()]
    if not mu_vals:
        raise ValueError("--mu-grid must contain at least one value")

    C_vals = [0.9, 0.95]
    fmin_vals = [0.0, 0.02, 0.1, 0.2, 0.5]
    print(
        f"Generating binned C̄Max table for n_bins={args.n_bins}, "
        f"n_trials={args.n_trials}..."
    )
    tables, fits = _build_tables(
        C_vals=C_vals,
        fmin_vals=fmin_vals,
        mu_vals=mu_vals,
        n_bins=args.n_bins,
        n_trials=args.n_trials,
        seed=args.seed,
    )

    out = Path(__file__).resolve().parents[1] / "src" / "yellin" / "data"
    out.mkdir(parents=True, exist_ok=True)
    out_path = out / "c_bar_max_binned_table.pkl"
    with open(out_path, "wb") as f:
        pickle.dump({"tables": tables, "fits": fits}, f)

    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()

