#!/usr/bin/env python3
"""
Generate C∞(y; f) lookup table via Monte Carlo (Appendix A).

Run once to produce src/yellin/data/c_infinity_table.npz.
"""

import sys
from pathlib import Path

# Add project root for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from yellin.c_infinity import _compute_and_save_table

if __name__ == "__main__":
    out = Path(__file__).resolve().parents[1] / "src" / "yellin" / "data"
    print("Generating C∞ table (this may take several minutes)...")
    _compute_and_save_table(
        table_path=out / "c_infinity_table.npz",
        f_min=0.01,
        f_max=0.99,
        n_f=20,
        y_min=-4.0,
        y_max=4.0,
        n_y=50,
        n_sub=2000,
        n_trials=15_000,
        seed=42,
    )
    print(f"Table saved to {out / 'c_infinity_table.npz'}")
