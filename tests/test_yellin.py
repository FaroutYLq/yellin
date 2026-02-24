"""Tests for yellin high-statistics optimum interval method."""

import numpy as np
import pytest

from yellin import SpectrumCDF, uniform_spectrum, upper_limit, c_infinity
from yellin.transform import events_to_F, fm_from_F
from yellin.intervals import candidate_intervals, compute_CMax


class TestTransform:
    """Test events_to_F and fm_from_F."""

    def test_events_to_F_uniform(self):
        spec = SpectrumCDF(uniform_spectrum(0, 1))
        events = np.array([0.2, 0.5, 0.8])
        F = events_to_F(events, spec)
        np.testing.assert_array_almost_equal(F, [0.2, 0.5, 0.8])

    def test_events_to_F_clip(self):
        spec = SpectrumCDF(uniform_spectrum(0, 1))
        events = np.array([-0.1, 0.5, 1.2])
        F = events_to_F(events, spec, clip=True)
        np.testing.assert_array_almost_equal(F, [0.0, 0.5, 1.0])

    def test_fm_from_F_empty(self):
        assert fm_from_F(np.array([])) == 1.0

    def test_fm_from_F_single(self):
        F = np.array([0.5])
        assert fm_from_F(F) == 0.5  # max of [0.5, 0.5]

    def test_fm_from_F_spread(self):
        F = np.array([0.2, 0.5, 0.9])
        # Gaps: 0.2, 0.3, 0.4, 0.1
        assert fm_from_F(F) == 0.4


class TestCandidateIntervals:
    """Test candidate_intervals."""

    def test_simple(self):
        F = np.array([0.25, 0.5, 0.75])
        cands = candidate_intervals(F)
        assert len(cands) == 4  # n=0,1,2,3
        for n, f in cands:
            assert 0 < f <= 1
            assert 0 <= n <= 3

    def test_n0_largest_gap(self):
        F = np.array([0.1, 0.2, 0.9])
        cands = candidate_intervals(F)
        n0 = next(c for c in cands if c[0] == 0)
        assert n0[1] == 0.7  # gap between 0.2 and 0.9


class TestCInfinity:
    """Test c_infinity."""

    def test_decreasing_in_y(self):
        f = 0.5
        c_lo = c_infinity(-2.0, f)
        c_hi = c_infinity(2.0, f)
        assert c_hi < c_lo

    def test_decreasing_in_f(self):
        y = 0.0
        c_high_f = c_infinity(y, 0.5)
        c_low_f = c_infinity(y, 0.1)
        assert c_low_f < c_high_f

    def test_bounds(self):
        c = c_infinity(0.0, 0.5)
        assert 0 <= c <= 1


class TestUpperLimit:
    """Test upper_limit."""

    def test_empty_events(self):
        spec = SpectrumCDF(uniform_spectrum(0, 1))
        assert upper_limit(np.array([]), spec) == 0.0

    def test_uniform_data(self):
        np.random.seed(42)
        events = np.random.uniform(0, 1, 100)
        spec = SpectrumCDF(uniform_spectrum(0, 1))
        ul = upper_limit(events, spec, C=0.9)
        assert ul > 0
        assert ul < 500

    def test_known_background(self):
        np.random.seed(43)
        events = np.random.uniform(0, 1, 50)
        spec = SpectrumCDF(uniform_spectrum(0, 1))
        ul = upper_limit(events, spec, C=0.9, known_background=10.0)
        ul_total = upper_limit(events, spec, C=0.9)
        assert ul < ul_total
