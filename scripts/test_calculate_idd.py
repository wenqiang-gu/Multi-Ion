#!/usr/bin/env python3
"""Unit tests for the IDD smoothing and fluctuation calculations."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import calculate_idd

try:
    import numpy as np
except ModuleNotFoundError:
    np = None


@unittest.skipUnless(np is not None, "NumPy is not installed")
class FluctuationTests(unittest.TestCase):
    def test_local_quadratic_reproduces_noiseless_quadratic(self) -> None:
        depths = np.linspace(0.0, 4.0, 41)
        values = 20.0 + 2.0 * depths + depths**2

        smoothed, window_bins = calculate_idd.smooth_local_quadratic(
            depths.tolist(), values.tolist(), 1.0
        )

        self.assertEqual(window_bins, 11)
        np.testing.assert_allclose(smoothed, values, rtol=0, atol=1e-10)

    def test_pointwise_percent_fluctuation(self) -> None:
        smooth = np.full(5, 50.0)
        expected = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        raw = smooth * (1.0 + expected / 100.0)

        residual, metrics = calculate_idd.calculate_fluctuation(
            raw.tolist(), smooth, 10.0
        )

        np.testing.assert_allclose(residual, expected, atol=1e-12)
        self.assertEqual(metrics["analyzed_bins"], 5)
        self.assertAlmostEqual(metrics["max_absolute_percent"], 2.0)

    def test_low_dose_bins_are_excluded(self) -> None:
        smooth = np.array([5.0, 9.0, 10.0, 100.0])
        raw = smooth.copy()

        residual, metrics = calculate_idd.calculate_fluctuation(
            raw.tolist(), smooth, 10.0
        )

        self.assertTrue(np.isnan(residual[0]))
        self.assertTrue(np.isnan(residual[1]))
        self.assertEqual(residual[2], 0.0)
        self.assertEqual(residual[3], 0.0)
        self.assertEqual(metrics["analyzed_bins"], 2)

    def test_smoothing_window_is_capped_at_curve_length(self) -> None:
        depths = np.linspace(0.0, 0.6, 7)
        values = 4.0 + depths + 0.5 * depths**2

        smoothed, window_bins = calculate_idd.smooth_local_quadratic(
            depths.tolist(), values.tolist(), 100.0
        )

        self.assertEqual(window_bins, 7)
        self.assertTrue(np.all(np.isfinite(smoothed)))
        np.testing.assert_allclose(smoothed, values, rtol=0, atol=1e-10)


if __name__ == "__main__":
    unittest.main()
