import tempfile
import unittest
from pathlib import Path

import numpy as np

from evaluation.trajectory_plot import plot_trajectory
from simulation.trajectory_player import TrajectoryLog


class TrajectoryPlotTests(unittest.TestCase):
    def test_saves_plot_under_nested_output_directory(self):
        log = TrajectoryLog(
            timestamps=np.array([0.02, 0.04, 0.06]),
            target_xyz=np.array([[0.4, 0.1, 0.5], [0.41, 0.1, 0.5], [0.42, 0.1, 0.5]]),
            actual_xyz=np.array([[0.4, 0.1, 0.49], [0.405, 0.1, 0.5], [0.418, 0.1, 0.5]]),
            ik_converged=np.ones(3, dtype=bool),
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nested" / "tracking.png"
            result = plot_trajectory(log, path)
            self.assertEqual(result, path)
            self.assertTrue(path.is_file())
            self.assertGreater(path.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
