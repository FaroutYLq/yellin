"""
Yellin high-statistics optimum interval method.

Implements the method from arXiv:0709.2701 for setting frequentist upper
limits on signals in the presence of unknown background (mu > 54.5).
"""

from yellin.spectrum import SpectrumCDF, uniform_spectrum, pdf_to_cdf
from yellin.binned import events_to_binned_counts, compute_CMax_binned
from yellin.c_infinity import c_infinity
from yellin.c_bar_max import c_bar_max
from yellin.c_bar_max_binned import c_bar_max_binned
from yellin.upper_limit import upper_limit, upper_limit_binned

__all__ = [
    "SpectrumCDF",
    "uniform_spectrum",
    "pdf_to_cdf",
    "events_to_binned_counts",
    "compute_CMax_binned",
    "c_infinity",
    "c_bar_max",
    "c_bar_max_binned",
    "upper_limit",
    "upper_limit_binned",
]
