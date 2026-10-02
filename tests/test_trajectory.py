"""Tests for Cartesian waypoint interpolation and smoothing."""

import unittest

import numpy as np

from simulation.trajectory_player import resample_waypoints, smooth_trajectory


class ResampleWaypointsTests(unittest.TestCase):
    def test_samples_by_cumulative_distance_at_configured_speed(self):
        waypoints = np.array(
            [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.1, 0.1, 0.0]]
        )

        times, samples = resample_waypoints(
            waypoints, speed=0.1, sample_period=0.5
        )

        np.testing.assert_allclose(times, [0.0, 0.5, 1.0, 1.5, 2.0])
        np.testing.assert_allclose(
            samples,
            [
                [0.0, 0.0, 0.0],
                [0.05, 0.0, 0.0],
                [0.1, 0.0, 0.0],
                [0.1, 0.05, 0.0],
                [0.1, 0.1, 0.0],
            ],
        )

    def test_ignores_repeated_interior_waypoint(self):
        waypoints = np.array(
            [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.1, 0.0, 0.0], [0.1, 0.1, 0.0]]
        )

        times, samples = resample_waypoints(
            waypoints, speed=0.1, sample_period=0.5
        )

        np.testing.assert_allclose(times, [0.0, 0.5, 1.0, 1.5, 2.0])
        np.testing.assert_allclose(samples[2], [0.1, 0.0, 0.0])

    def test_rejects_waypoints_with_wrong_shape(self):
        with self.assertRaisesRegex(ValueError, "shape"):
            resample_waypoints(np.array([[0.0, 1.0], [1.0, 2.0]]), 0.1, 0.02)

    def test_rejects_non_finite_waypoints(self):
        with self.assertRaisesRegex(ValueError, "finite"):
            resample_waypoints(
                np.array([[0.0, 0.0, 0.0], [np.inf, 0.0, 0.0]]), 0.1, 0.02
            )

    def test_rejects_fewer_than_two_waypoints(self):
        with self.assertRaisesRegex(ValueError, "two"):
            resample_waypoints(np.array([[0.0, 0.0, 0.0]]), 0.1, 0.02)

    def test_rejects_zero_length_path(self):
        with self.assertRaisesRegex(ValueError, "distance"):
            resample_waypoints(
                np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]), 0.1, 0.02
            )

    def test_rejects_nonpositive_speed_or_sample_period(self):
        path = np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]])
        for speed, period in ((0.0, 0.02), (0.1, 0.0), (0.1, -0.02)):
            with self.subTest(speed=speed, period=period):
                with self.assertRaisesRegex(ValueError, "speed|sample_period"):
                    resample_waypoints(path, speed, period)


class SmoothTrajectoryTests(unittest.TestCase):
    def test_smooths_interior_and_preserves_endpoints(self):
        samples = np.array(
            [[0.0, 0.0, 0.0], [1.0, 2.0, 0.0], [2.0, 0.0, 0.0],
             [3.0, 2.0, 0.0], [4.0, 0.0, 0.0]]
        )

        smoothed = smooth_trajectory(samples, window_size=3)

        np.testing.assert_array_equal(smoothed[0], samples[0])
        np.testing.assert_array_equal(smoothed[-1], samples[-1])
        self.assertAlmostEqual(smoothed[1, 1], 2.0 / 3.0)
        self.assertEqual(smoothed.shape, samples.shape)

    def test_caps_oversized_window_to_available_odd_length(self):
        samples = np.array(
            [[0.0, 0.0, 0.0], [1.0, 3.0, 0.0], [2.0, 0.0, 0.0]]
        )

        smoothed = smooth_trajectory(samples, window_size=9)

        np.testing.assert_array_equal(smoothed[0], samples[0])
        np.testing.assert_array_equal(smoothed[-1], samples[-1])
        self.assertEqual(smoothed.shape, samples.shape)
        self.assertAlmostEqual(smoothed[1, 1], 1.0)

    def test_window_one_disables_smoothing(self):
        samples = np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 0.0]])
        np.testing.assert_array_equal(smooth_trajectory(samples, 1), samples)

    def test_rejects_invalid_window_size(self):
        samples = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 0.0]])
        for window in (0, 2, -1):
            with self.subTest(window=window):
                with self.assertRaisesRegex(ValueError, "window_size"):
                    smooth_trajectory(samples, window)


if __name__ == "__main__":
    unittest.main()
