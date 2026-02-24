#!/usr/bin/env python3
"""Quick C̄Max table - minimal for testing. Run generate_c_bar_max_table.py for full."""
import sys
from pathlib import Path
import pickle
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from yellin.c_infinity import c_infinity
from yellin.intervals import compute_CMax

def main():
    rng = np.random.default_rng(42)
    C_vals, fmin_vals = [0.9, 0.95], [0.0, 0.02, 0.1, 0.2, 0.5]
    mu_vals = [55, 100, 200, 400, 700, 1200, 2000]
    tables, fits = {}, {}

    for C in C_vals:
        for fmin in fmin_vals:
            key = (C, fmin)
            cbar_list = []
            for mu in mu_vals:
                cmax_vals = np.empty(80)
                for i in range(80):
                    n = rng.poisson(mu)
                    F = np.sort(rng.uniform(0, 1, n)) if n > 0 else np.array([])
                    cmax_vals[i] = compute_CMax(F, mu, fmin, c_infinity)
                cbar_list.append(np.quantile(cmax_vals, C))
            tables[key] = (np.array(mu_vals), np.array(cbar_list))
            high = np.array(mu_vals) > 500
            if np.sum(high) >= 2:
                inv = 1/np.sqrt(np.array(mu_vals)[high])
                cbar_h = np.array(cbar_list)[high]
                coef, const = np.polyfit(inv, cbar_h, 1)
                fits[key] = (float(const), float(coef))

    out = Path(__file__).resolve().parents[1] / "src" / "yellin" / "data"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "c_bar_max_table.pkl", "wb") as f:
        pickle.dump({"tables": tables, "fits": fits}, f)
    print(f"Quick table saved to {out / 'c_bar_max_table.pkl'}")

if __name__ == "__main__":
    main()
