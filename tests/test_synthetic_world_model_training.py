"""Preassigned synthetic scenario partitions must survive world-model loading."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.train_synthetic_world_model import (
    evaluate_rollout_baselines, load_synthetic_partitions,
)


class SyntheticWorldModelTrainingTests(unittest.TestCase):
    def test_reports_zero_velocity_and_contact_persistence_on_same_windows(self):
        states = np.zeros((4, 37), dtype=np.float32)
        next_states = states.copy()
        states[:, 20] = next_states[:, 20] = 1.0
        states[:, 24:27] = ((1, 0, 0), (1, 0, 0), (3, 0, 0), (3, 0, 0))
        next_states[:, 24:27] = ((1, 0, 0), (2, 0, 0), (3, 0, 0), (4, 0, 0))
        states[0, 32] = 0
        states[2, 32] = 1
        next_states[1, 32] = 0
        next_states[3, 32] = 0
        data = {"states": states, "next_states": next_states,
                "episode_ids": np.full(4, "episode-a")}
        metrics = evaluate_rollout_baselines(data, horizon=2)
        self.assertEqual(metrics["rollouts"], 2)
        self.assertAlmostEqual(metrics["contact_persistence_accuracy"], .75)
        self.assertAlmostEqual(metrics["package_linear_velocity"]["zero_velocity_rmse"], np.sqrt(10 / 3))
        self.assertAlmostEqual(metrics["package_linear_velocity"]["persistence_rmse"], np.sqrt(1 / 3))

    def _archive(self, folder: Path, name: str, group: str, split: str) -> None:
        states = np.zeros((4, 37), dtype=np.float32)
        states[:, 20] = 1.0
        following = states.copy()
        np.savez_compressed(
            folder / f"{name}.npz", states=states, actions=np.zeros((4, 4), np.float32),
            next_states=following, group_ids=np.full(4, group),
            episode_ids=np.full(4, name), source_kinds=np.full(4, "simulation_randomized"),
            splits=np.full(4, split), transition_dt_seconds=np.full(4, .02),
        )

    def test_loads_preassigned_partitions_without_crossing_families(self):
        with tempfile.TemporaryDirectory() as temporary:
            episodes = Path(temporary) / "episodes"
            episodes.mkdir()
            for split in ("train", "validation", "test"):
                self._archive(episodes, f"episode-{split}", f"family-{split}", split)
            parts = load_synthetic_partitions(Path(temporary))
            self.assertEqual(set(parts), {"train", "validation", "test"})
            self.assertEqual(len(parts["train"]["states"]), 4)
            self.assertEqual(set(parts["validation"]["group_ids"]), {"family-validation"})
            self.assertAlmostEqual(parts["test"]["transition_dt_seconds"], .02)

    def test_rejects_families_crossing_preassigned_partitions(self):
        with tempfile.TemporaryDirectory() as temporary:
            episodes = Path(temporary) / "episodes"
            episodes.mkdir()
            self._archive(episodes, "episode-train", "shared-family", "train")
            self._archive(episodes, "episode-test", "shared-family", "test")
            self._archive(episodes, "episode-validation", "validation-family", "validation")
            with self.assertRaisesRegex(ValueError, "cross partitions"):
                load_synthetic_partitions(Path(temporary))


if __name__ == "__main__":
    unittest.main()
