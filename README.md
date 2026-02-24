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

## Table generation

Before first use, generate the lookup tables (or use the pre-generated ones in
`src/yellin/data/`):

```bash
PYTHONPATH=src python scripts/generate_c_infinity_table.py
PYTHONPATH=src python scripts/generate_c_bar_max_table.py   # Full (slow)
# or
PYTHONPATH=src python scripts/generate_c_bar_max_quick.py   # Quick (testing)
```

## References

- arXiv:0709.2701 – Extending the Optimum Interval Method (this implementation)
- Phys. Rev. D 66, 032005 (2002) – Original optimum interval method (low stats)
