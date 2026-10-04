"""Tests for private episode alignment and held-out model-fidelity records."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np


class EpisodeOutcomeLoadingTests(unittest.TestCase):
    def _write_summary(
        self, root: Path, label: str, clip_id: str, method: str, robot_success: bool
    ) -> None:
        folder = root / "per_clip" / label / clip_id
        folder.mkdir(parents=True, exist_ok=True)
        (folder / f"{method}_summary.json").write_text(
            json.dumps({
                "clip_id": clip_id,
                "method": method,
                "human_label": label,
                "robot_success": robot_success,
            }),
            encoding="utf-8",
        )

    def test_keeps_human_label_distinct_from_simulated_robot_outcome(self):
        from world_model.fidelity import load_episode_outcomes

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_summary(root, "riusciti", "demo_a", "direct", False)

            outcomes = load_episode_outcomes(root)

        self.assertEqual(outcomes[("demo_a", "direct")]["human_label"], "riusciti")
        self.assertIs(outcomes[("demo_a", "direct")]["robot_success"], False)

    def test_rejects_duplicate_clip_method_outcomes(self):
        from world_model.fidelity import load_episode_outcomes

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_summary(root, "riusciti", "demo_a", "direct", True)
            self._write_summary(root, "falliti", "demo_a", "direct", False)

            with self.assertRaisesRegex(ValueError, "duplicate.*demo_a.*direct"):
                load_episode_outcomes(root)

    def test_rejects_missing_or_extra_episode_outcomes(self):
        from world_model.fidelity import validate_episode_outcomes

        outcomes = {("demo_a", "direct"): {"robot_success": True}}
        expected = [("demo_a", "direct"), ("demo_b", "confidence-aware")]

        with self.assertRaisesRegex(ValueError, "missing.*demo_b"):
            validate_episode_outcomes(outcomes, expected)

        with self.assertRaisesRegex(ValueError, "unexpected.*demo_extra"):
            validate_episode_outcomes(
                {**outcomes, ("demo_extra", "direct"): {"robot_success": False}},
                [("demo_a", "direct")],
            )


class WorldModelArchiveTests(unittest.TestCase):
    def test_pair_loader_preserves_episode_labels_and_phases(self):
        from scripts.evaluate_world_model_cv import _load_pair

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "direct.npz"
            states = np.zeros((2, 37), dtype=np.float32)
            states[:, 20] = 1.0
            np.savez_compressed(
                path,
                states=states,
                actions=np.zeros((2, 4), dtype=np.float32),
                next_states=states.copy(),
                clip_ids=np.asarray(["demo_a", "demo_a"]),
                episode_ids=np.asarray(["demo_a:direct", "demo_a:direct"]),
                human_labels=np.asarray(["riusciti", "riusciti"]),
                phases=np.asarray(["approach", "settle"]),
                transition_dt_seconds=np.full(2, 0.1),
            )

            data = _load_pair(path, "direct")

        np.testing.assert_array_equal(data["episode_ids"], ["demo_a:direct"] * 2)
        np.testing.assert_array_equal(data["human_labels"], ["riusciti"] * 2)
        np.testing.assert_array_equal(data["phases"], ["approach", "settle"])

    def test_pair_loader_rejects_episode_ids_for_another_method(self):
        from scripts.evaluate_world_model_cv import _load_pair

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "direct.npz"
            states = np.zeros((2, 37), dtype=np.float32)
            states[:, 20] = 1.0
            np.savez_compressed(
                path,
                states=states,
                actions=np.zeros((2, 4), dtype=np.float32),
                next_states=states.copy(),
                clip_ids=np.asarray(["demo_a", "demo_a"]),
                episode_ids=np.asarray(["demo_a:confidence-aware"] * 2),
                transition_dt_seconds=np.full(2, 0.1),
            )

            with self.assertRaisesRegex(ValueError, "episode_ids do not match"):
                _load_pair(path, "direct")

    def test_outer_fold_assignment_keeps_both_methods_for_each_clip_together(self):
        from world_model.fidelity import assign_outer_folds

        clips = np.asarray(["demo_a", "demo_b", "demo_a", "demo_b", "demo_c", "demo_c"])
        methods = np.asarray(["direct", "direct", "confidence-aware", "confidence-aware", "direct", "confidence-aware"])

        folds = assign_outer_folds(clips, fold_count=3, seed=4)

        for clip in np.unique(clips):
            self.assertEqual(len(np.unique(folds[clips == clip])), 1)
        self.assertEqual(len(folds), len(methods))
        self.assertEqual(set(np.unique(folds)), {0, 1, 2})


if __name__ == "__main__":
    unittest.main()
