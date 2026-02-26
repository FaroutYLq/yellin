"""Tests for yellin high-statistics optimum interval method."""

import numpy as np
import pytest

from yellin import (
    SpectrumCDF,
    c_bar_max,
    c_bar_max_binned,
    c_infinity,
    pdf_to_cdf,
    uniform_spectrum,
    upper_limit,
    upper_limit_binned,
)
from yellin.binned import compute_CMax_binned, events_to_binned_counts
from yellin.transform import events_to_F, fm_from_F
from yellin.intervals import candidate_intervals


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


class TestSpectrumHelpers:
    """Test spectrum helper utilities."""

    def test_pdf_to_cdf_scaled_shape(self):
        cdf = pdf_to_cdf(lambda s: 5.0 * np.asarray(s, dtype=float) ** 2, 0.0, 1.0)
        x = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        np.testing.assert_allclose(cdf(x), x**3, atol=2e-3, rtol=0)

    def test_pdf_to_cdf_scale_invariant(self):
        cdf1 = pdf_to_cdf(lambda s: np.asarray(s, dtype=float) ** 2, 0.0, 1.0)
        cdf2 = pdf_to_cdf(lambda s: 17.0 * np.asarray(s, dtype=float) ** 2, 0.0, 1.0)
        x = np.linspace(0.0, 1.0, 50)
        np.testing.assert_allclose(cdf1(x), cdf2(x), atol=1e-8, rtol=0)

    def test_pdf_to_cdf_accepts_scalar_pdf(self):
        def scalar_pdf(s: float) -> float:
            return s * s if 0.0 <= s <= 1.0 else 0.0

        cdf = pdf_to_cdf(scalar_pdf, 0.0, 1.0)
        assert cdf(np.array([0.5]))[0] == pytest.approx(0.125, abs=2e-3)

    def test_pdf_to_cdf_negative_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            pdf_to_cdf(lambda s: np.asarray(s, dtype=float) - 0.5, 0.0, 1.0)

    def test_pdf_to_cdf_zero_integral_raises(self):
        with pytest.raises(ValueError, match="must be positive"):
            pdf_to_cdf(lambda s: np.zeros_like(np.asarray(s, dtype=float)), 0.0, 1.0)


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


class TestBinnedSupport:
    """Test binned-data support (paper Section IV)."""

    def test_events_to_binned_counts_uniform(self):
        spec = SpectrumCDF(uniform_spectrum(0, 1))
        events = np.array([0.1, 0.2, 0.85])
        counts = events_to_binned_counts(events, spec, n_bins=5)
        np.testing.assert_array_equal(counts, np.array([1, 1, 0, 0, 1]))

    def test_compute_cmax_binned_matches_bruteforce(self):
        counts = np.array([3, 0, 2, 1, 4], dtype=int)
        mu = 20.0
        fmin = 0.2

        def fake_c_inf(y: float, f: float) -> float:
            return float(-y + 0.5 * f)

        cmax = compute_CMax_binned(counts, mu, fmin=fmin, c_infinity_fn=fake_c_inf)

        n_bins = counts.size
        best = 0.0
        for i in range(n_bins):
            for j in range(i + 1, n_bins + 1):
                f = (j - i) / n_bins
                if f <= fmin:
                    continue
                n = float(np.sum(counts[i:j]))
                x = mu * f
                y = (n - x) / np.sqrt(x)
                c = fake_c_inf(y, f)
                if c > best:
                    best = c

        assert cmax == pytest.approx(best)

    def test_c_bar_max_binned_fallback(self):
        mu = 200.0
        c_ref = c_bar_max(0.9, 0.0, mu)
        c_bin = c_bar_max_binned(0.9, 0.0, mu, n_bins=100, fallback_unbinned=True)
        assert c_bin == pytest.approx(c_ref)

    def test_upper_limit_binned(self):
        rng = np.random.default_rng(44)
        events = rng.uniform(0, 1, 120)
        spec = SpectrumCDF(uniform_spectrum(0, 1))
        counts = events_to_binned_counts(events, spec, n_bins=100)
        ul = upper_limit_binned(counts, C=0.9)
        assert ul > 0
        assert ul < 500
