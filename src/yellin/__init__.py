"""
Yellin high-statistics optimum interval method.

Implements the method from arXiv:0709.2701 for setting frequentist upper
limits on signals in the presence of unknown background (mu > 54.5).
"""

from yellin.spectrum import SpectrumCDF, uniform_spectrum
from yellin.c_infinity import c_infinity
from yellin.c_bar_max import c_bar_max
from yellin.upper_limit import upper_limit

__all__ = [
    "SpectrumCDF",
    "uniform_spectrum",
    "c_infinity",
    "c_bar_max",
    "upper_limit",
]
