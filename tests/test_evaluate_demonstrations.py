"""Tests that unavailable aggregate values are not plotted as zero."""

from pathlib import Path
import tempfile
import unittest
from unittest import mock

import matplotlib.pyplot as plt

from scripts.evaluate_demonstrations import _plot_aggregate


class AggregatePlotTests(unittest.TestCase):
    def test_missing_or_undefined_metrics_are_labeled_not_zero(self):
        metrics = {
            "coverage_fraction": {"mean": None, "std": None},
            "jerk_rms_mps3": {"mean": None, "std": None},
            "mean_position_error_m": {"mean": 0.02, "std": None},
            "ik_nonconverged_count": {"mean": 0.0, "std": 0.0},
        }
        aggregate = {"methods": {
            "direct": {"metrics": metrics},
            "confidence-aware": {"metrics": {**metrics,
                "coverage_fraction": {"mean": 1.0, "std": 0.0}}},
        }}
        with tempfile.TemporaryDirectory() as directory:
            with mock.patch("scripts.evaluate_demonstrations.plt.close"):
                _plot_aggregate(aggregate, Path(directory) / "aggregate.png")
            figure = plt.gcf()
            axes = figure.axes
            self.assertTrue(any(text.get_text() == "N/A" for text in axes[0].texts))
            self.assertEqual(len(axes[0].patches), 1)
            self.assertTrue(any(text.get_text() == "N/A" for text in axes[1].texts))
            self.assertEqual(len(axes[1].patches), 0)
            plt.close(figure)


if __name__ == "__main__":
    unittest.main()
