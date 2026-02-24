# yellin

Python implementation of the **high-statistics optimum interval method** from

> S. Yellin, "Extending the Optimum Interval Method," arXiv:0709.2701 (2007)

for setting frequentist upper limits on signals in the presence of unknown
background. Applicable when expected event counts μ > 54.5.

## Installation

```bash
pip install .
```

## Usage

```python
import numpy as np
from yellin import upper_limit, SpectrumCDF, uniform_spectrum

# Event values (e.g. energies) and signal CDF
events = np.array([...])  # Your observed events
spectrum_cdf = SpectrumCDF(uniform_spectrum(s_min=0, s_max=100))

# 90% confidence upper limit
ul = upper_limit(events, spectrum_cdf, C=0.9)
print(f"90% CL upper limit: {ul}")

# With known background subtraction
ul_signal = upper_limit(
    events, spectrum_cdf, C=0.9, known_background=10.0
)
```

## Lookup tables

The method relies on two precomputed tables. Both must exist under
`src/yellin/data/` (or be generated there) for `upper_limit` to work.

### C∞ table (`c_infinity_table.npz`)

- **Role:** C∞(y; f) = P(ymin > y) in the Gaussian/Brownian limit (paper
  Section II, Appendix A). For an interval with fraction f and scaled excess
  y = (n − x)/√x, C∞ gives the probability that the data reject the assumed
  signal at least this strongly.
- **Content:** A 2D grid over (y, f): C∞ is tabulated for y in [−4, 4] and
  f in [0.01, 0.99], then used via interpolation. Values outside the grid
  are clipped.
- **Format:** NumPy `.npz` with arrays `y_grid`, `f_grid`, and `C`.
- **Generation:** Run once; the result is independent of your data.

  ```bash
  PYTHONPATH=src python scripts/generate_c_infinity_table.py
  ```

  Defaults: 20 f points, 50 y points, 15 000 Monte Carlo trials per f, ~1–2
  minutes. The script writes `src/yellin/data/c_infinity_table.npz`.

### C̄Max table (`c_bar_max_table.pkl`)

- **Role:** C̄Max(C, fmin, μ) is the calibration value such that a fraction
  C of experiments (no unknown background) have CMax < C̄Max. The upper limit
  is the μ for which observed CMax = C̄Max(C, fmin, μ) (paper Section II).
- **Content:** For each (C, fmin) pair, the table stores C̄Max at a discrete
  set of μ values. Supported (C, fmin): C ∈ {0.9, 0.95}, fmin ∈ {0, 0.02,
  0.1, 0.2, 0.5}. For μ larger than the tabulated range, C̄Max is
  extrapolated as A + B/√μ (paper Fig. 2).
- **Format:** Python pickle with a dict: `{"tables": {...}, "fits": {...}}`.
  Keys in `tables` are (C, fmin); values are `(mu_array, cbar_array)`.
  Keys in `fits` are (C, fmin); values are (A, B) for extrapolation.
- **Generation:** Two options:

  1. **Full table** (better for production; slower):

     ```bash
     PYTHONPATH=src python scripts/generate_c_bar_max_table.py
     ```

     Uses μ = 55, 80, 120, 200, 350, 500, 800, 1200, 2000, 200 trials per
     (C, fmin, μ). Can take tens of minutes depending on hardware.

  2. **Quick table** (faster; coarser for testing/CI):

     ```bash
     PYTHONPATH=src python scripts/generate_c_bar_max_quick.py
     ```

     Fewer μ points and 80 trials per point; typically ~1–2 minutes. Adequate
     for tests and quick checks; for publication-quality limits, use the full
     script and consider increasing `n_trials` and the μ grid in the script.

- **Location:** The library loads `src/yellin/data/c_bar_max_table.pkl` by
  default. If that file is missing, `c_bar_max` and thus `upper_limit` raise
  `FileNotFoundError` with instructions to run one of the scripts above.

### Summary

| Table        | File                     | Purpose              | Typical runtime |
|-------------|--------------------------|----------------------|------------------|
| C∞          | `c_infinity_table.npz`   | Brownian limit C∞(y; f) | ~1–2 min     |
| C̄Max (full) | `c_bar_max_table.pkl`   | Calibration C̄Max(C, fmin, μ) | 10+ min   |
| C̄Max (quick)| `c_bar_max_table.pkl`   | Same, fewer points   | ~1–2 min        |

Commit the generated files under `src/yellin/data/` so that CI and other
users get consistent results without regenerating.

## References

- arXiv:0709.2701 – Extending the Optimum Interval Method (this implementation)
- Phys. Rev. D 66, 032005 (2002) – Original optimum interval method (low stats)
