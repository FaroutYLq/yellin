# yellin

Python implementation of the **high-statistics optimum interval method** from

> S. Yellin, "Extending the Optimum Interval Method," arXiv:0709.2701 (2007)

for setting frequentist upper limits on signals in the presence of unknown
background. Applicable when expected event counts $\mu > 54.5$.

## Installation

```bash
pip install .
```

## Usage

Provide your observed events and the **signal** cumulative distribution
$F(s)$ (fraction of expected signal below $s$). Use `uniform_spectrum` only when
the signal is uniform in your observable range; otherwise use your signal’s CDF.

```python
import numpy as np
from yellin import upper_limit, SpectrumCDF, uniform_spectrum

# Event values (e.g. energies) and signal CDF
events = np.array([...])  # Your observed events
# Use the CDF of your signal of interest. uniform_spectrum is for a uniform
# signal in [s_min, s_max]; replace with your signal's CDF if different.
spectrum_cdf = SpectrumCDF(uniform_spectrum(s_min=0, s_max=100))

# 90% confidence upper limit
ul = upper_limit(events, spectrum_cdf, C=0.9)
print(f"90% CL upper limit: {ul}")

# With known background subtraction
ul_signal = upper_limit(
    events, spectrum_cdf, C=0.9, known_background=10.0
)
```

### Custom spectrum

The spectrum CDF $F(s)$ must be **normalized**: $F(s)$ must be non-decreasing
and map your observable range to $[0, 1]$, i.e. $F(s\_{\mathrm{min}}) = 0$ and
$F(s\_{\mathrm{max}}) = 1$ (or the appropriate limits for your support). The method
uses only the shape of the CDF, not the absolute normalization of the signal.

Example: signal PDF proportional to $s^2$ on $[s\_{\mathrm{min}}, s\_{\mathrm{max}}]$. The CDF is
$F(s) = (s^3 - s\_{\mathrm{min}}^3) / (s\_{\mathrm{max}}^3 - s\_{\mathrm{min}}^3)$, normalized to
$[0, 1]$ over that interval:

```python
import numpy as np
from yellin import upper_limit, SpectrumCDF

def power_cdf(s_min: float, s_max: float, index: float = 2.0):
    """CDF for PDF(s) ∝ s^index on [s_min, s_max]. Normalized to [0, 1]."""
    a = index + 1.0
    low = s_min**a
    span = s_max**a - low
    def F(s: np.ndarray) -> np.ndarray:
        s = np.asarray(s, dtype=float)
        return np.clip((s**a - low) / span, 0.0, 1.0)
    return F

events = np.array([...])
spectrum_cdf = SpectrumCDF(power_cdf(s_min=0.0, s_max=100.0, index=2.0))
ul = upper_limit(events, spectrum_cdf, C=0.9)
```

## Lookup tables

The method relies on two precomputed tables. Both must exist under
`src/yellin/data/` (or be generated there) for `upper_limit` to work.

### $C\_\infty$ table (`c_infinity_table.npz`)

- **Role:** $C\_\infty(y; f) = P(y\_{\mathrm{min}} > y)$ in the Gaussian/Brownian limit
  (paper Section II, Appendix A). For an interval with fraction $f$ and scaled
  excess $y = (n - x)/\sqrt{x}$, $C\_\infty$ gives the probability that the data
  reject the assumed signal at least this strongly.
- **Content:** A 2D grid over $(y, f)$: $C\_\infty$ is tabulated for $y$ in
  $[-4, 4]$ and $f$ in $[0.01, 0.99]$, then used via interpolation. Values
  outside the grid are clipped.
- **Format:** NumPy `.npz` with arrays `y_grid`, `f_grid`, and `C`.
- **Generation:** Run once; the result is independent of your data.

  ```bash
  PYTHONPATH=src python scripts/generate_c_infinity_table.py
  ```

  Defaults: 20 f points, 50 y points, 15 000 Monte Carlo trials per f, ~1–2
  minutes. The script writes `src/yellin/data/c_infinity_table.npz`.

### $\bar{C}\_{\mathrm{max}}$ table (`c_bar_max_table.pkl`)

- **Role:** $\bar{C}\_{\mathrm{max}}(C, f\_{\mathrm{min}}, \mu)$ is the calibration value such
  that a fraction $C$ of experiments (no unknown background) have
  $C\_{\mathrm{max}} < \bar{C}\_{\mathrm{max}}$. The upper limit is the $\mu$ for which
  observed $C\_{\mathrm{max}} = \bar{C}\_{\mathrm{max}}(C, f\_{\mathrm{min}}, \mu)$ (paper Section II).
- **Content:** For each $(C, f\_{\mathrm{min}})$ pair, the table stores
  $\bar{C}\_{\mathrm{max}}$ at a discrete set of $\mu$ values. Supported $(C,
  f\_{\mathrm{min}})$: $C \in \{0.9, 0.95\}$, $f\_{\mathrm{min}} \in \{0, 0.02, 0.1, 0.2,
  0.5\}$. For $\mu$ larger than the tabulated range, $\bar{C}\_{\mathrm{max}}$ is
  extrapolated as $A + B/\sqrt{\mu}$ (paper Fig. 2).
- **Format:** Python pickle with a dict: `{"tables": {...}, "fits": {...}}`.
  Keys in `tables` are $(C, f\_{\mathrm{min}})$; values are `(mu_array, cbar_array)`.
  Keys in `fits` are $(C, f\_{\mathrm{min}})$; values are $(A, B)$ for extrapolation.
- **Generation:** Two options:

  1. **Full table** (better for production; slower):

     ```bash
     PYTHONPATH=src python scripts/generate_c_bar_max_table.py
     ```

     Uses $\mu = 55, 80, 120, \ldots, 2000$, 200 trials per
     $(C, f\_{\mathrm{min}}, \mu)$. Can take tens of minutes depending on hardware.

  2. **Quick table** (faster; coarser for testing/CI):

     ```bash
     PYTHONPATH=src python scripts/generate_c_bar_max_quick.py
     ```

     Fewer $\mu$ points and 80 trials per point; typically ~1–2 minutes.
     Adequate for tests and quick checks; for publication-quality limits, use
     the full script and consider increasing `n_trials` and the $\mu$ grid.

- **Location:** The library loads `src/yellin/data/c_bar_max_table.pkl` by
  default. If that file is missing, `c_bar_max` and thus `upper_limit` raise
  `FileNotFoundError` with instructions to run one of the scripts above.

### Summary

| Table        | File                     | Purpose              | Typical runtime |
|-------------|--------------------------|----------------------|------------------|
| $C\_\infty$  | `c_infinity_table.npz`   | Brownian $C\_\infty(y; f)$ | ~1–2 min   |
| $\bar{C}\_{\mathrm{max}}$ (full) | `c_bar_max_table.pkl` | $\bar{C}\_{\mathrm{max}}(C,f\_{\mathrm{min}},\mu)$ | 10+ min |
| $\bar{C}\_{\mathrm{max}}$ (quick)| `c_bar_max_table.pkl` | Same, fewer points | ~1–2 min  |

Commit the generated files under `src/yellin/data/` so that CI and other
users get consistent results without regenerating.

## References

- arXiv:0709.2701 – Extending the Optimum Interval Method (this implementation)
- Phys. Rev. D 66, 032005 (2002) – Original optimum interval method (low stats)
