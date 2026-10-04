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


class ModelFidelityScoringTests(unittest.TestCase):
    def setUp(self):
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("optional PyTorch runtime is not installed")

    def _normalizer(self):
        from world_model.state_dynamics import DynamicsNormalizer, STATE_DIM, ACTION_DIM

        return DynamicsNormalizer(
            state_mean=np.zeros(STATE_DIM, dtype=np.float32),
            state_scale=np.ones(STATE_DIM, dtype=np.float32),
            action_mean=np.zeros(ACTION_DIM, dtype=np.float32),
            action_scale=np.ones(ACTION_DIM, dtype=np.float32),
            delta_mean=np.zeros(STATE_DIM, dtype=np.float32),
            delta_scale=np.ones(STATE_DIM, dtype=np.float32),
        )

    def _action_sensitive_model(self):
        from world_model.state_dynamics import StateDynamicsEnsemble, STATE_DIM
        import torch

        model = StateDynamicsEnsemble(count=2, hidden_dim=1)
        with torch.no_grad():
            for member in model.members:
                for parameter in member.parameters():
                    parameter.zero_()
                member.trunk[0].weight[0, STATE_DIM] = 1.0
                member.trunk[2].weight[0, 0] = 1.0
                member.trunk[4].weight[0, 0] = 1.0
                member.continuous_head.weight[0, 0] = 1.0
        return model

    def _initial_state(self):
        from world_model.state_dynamics import STATE_DIM

        state = np.zeros(STATE_DIM, dtype=np.float32)
        state[20] = 1.0
        state[17:20] = (0.55, 0.14, 0.42)
        state[34:37] = (0.55, 0.14, 0.416)
        return state

    def test_rollout_is_action_conditioned_and_returns_all_states_and_uncertainties(self):
        from world_model.fidelity import rollout_episode

        model = self._action_sensitive_model()
        initial = self._initial_state()
        actions = np.zeros((3, 4), dtype=np.float32)
        actions[0, 0] = 1.0

        predicted, disagreement = rollout_episode(model, self._normalizer(), initial, actions)

        self.assertEqual(predicted.shape, (4, 37))
        self.assertEqual(disagreement.shape, (3,))
        self.assertGreater(predicted[1, 0], initial[0])
        self.assertEqual(float(disagreement.max()), 0.0)

    def test_rollout_accepts_one_step_and_longer_action_sequences(self):
        from world_model.fidelity import rollout_episode

        model = self._action_sensitive_model()
        normalizer = self._normalizer()
        one_step, one_uncertainty = rollout_episode(
            model, normalizer, self._initial_state(), np.zeros((1, 4), dtype=np.float32)
        )
        longer, longer_uncertainty = rollout_episode(
            model, normalizer, self._initial_state(), np.zeros((7, 4), dtype=np.float32)
        )

        self.assertEqual(one_step.shape, (2, 37))
        self.assertEqual(one_uncertainty.shape, (1,))
        self.assertEqual(longer.shape, (8, 37))
        self.assertEqual(longer_uncertainty.shape, (7,))
        self.assertTrue(np.isfinite(longer).all())
        np.testing.assert_allclose(np.linalg.norm(longer[:, 20:24], axis=1), 1.0, atol=1e-6)
        self.assertTrue(np.isin(longer[:, 32:34], (0.0, 1.0)).all())

    def test_rollout_uncertainty_increases_when_ensemble_members_disagree(self):
        from world_model.fidelity import rollout_episode
        import torch

        model = self._action_sensitive_model()
        with torch.no_grad():
            model.members[1].continuous_head.bias[0] = 1.0
        _, disagreement = rollout_episode(
            model, self._normalizer(), self._initial_state(), np.zeros((2, 4), dtype=np.float32)
        )

        self.assertTrue(np.all(disagreement > 0.0))

    def test_rollout_rejects_invalid_initial_state_and_actions(self):
        from world_model.fidelity import rollout_episode

        model = self._action_sensitive_model()
        normalizer = self._normalizer()
        bad_quaternion = self._initial_state()
        bad_quaternion[20:24] = 0.0
        with self.assertRaisesRegex(ValueError, "quaternion must have unit norm"):
            rollout_episode(model, normalizer, bad_quaternion, np.zeros((1, 4)))
        with self.assertRaisesRegex(ValueError, "finite"):
            rollout_episode(model, normalizer, self._initial_state(), np.full((1, 4), np.nan))

    def test_terminal_score_reflects_lift_tray_support_and_stability(self):
        from world_model.fidelity import score_terminal_state

        states = np.stack((self._initial_state(), self._initial_state()))
        states[0, 17:20] = (0.55, 0.14, 0.42)
        states[-1, 17:20] = (0.55, 0.14, 0.48)
        states[1, 32] = 1.0
        states[-1, 24:30] = 0.0
        states[-1, 33] = 1.0
        config = {
            "tray_inner_half_size_xy": 0.11,
            "package_half_size_xy": (0.025, 0.020),
            "max_linear_speed": 0.02,
            "max_angular_speed": 0.5,
            "minimum_lift_m": 0.05,
        }

        score = score_terminal_state(states, config)
        states[-1, 33] = 0.0
        unsupported_score = score_terminal_state(states, config)

        self.assertEqual(score, 1.0)
        self.assertLess(unsupported_score, score)

    def test_terminal_score_rewards_a_gripper_contact_during_the_episode(self):
        from world_model.fidelity import score_terminal_state

        states = np.stack((self._initial_state(), self._initial_state(), self._initial_state()))
        states[0, 17:20] = (0.55, 0.14, 0.42)
        states[1, 17:20] = (0.55, 0.14, 0.48)
        states[2, 17:20] = (0.55, 0.14, 0.48)
        states[1, 32] = 1.0
        states[2, 33] = 1.0
        states[2, 24:30] = 0.0
        config = {
            "tray_inner_half_size_xy": 0.11,
            "package_half_size_xy": (0.025, 0.020),
            "max_linear_speed": 0.02,
            "max_angular_speed": 0.5,
            "minimum_lift_m": 0.05,
        }

        grasped_score = score_terminal_state(states, config)
        states[:, 32] = 0.0
        never_grasped_score = score_terminal_state(states, config)

        self.assertGreater(grasped_score, never_grasped_score)

    def test_candidate_comparison_counts_correct_tied_and_uninformative_pairs(self):
        from world_model.fidelity import compare_predicted_outcomes

        predictions = [
            {"clip_id": "a", "method": "direct", "score": 0.8},
            {"clip_id": "a", "method": "confidence-aware", "score": 0.2},
            {"clip_id": "b", "method": "direct", "score": 0.5},
            {"clip_id": "b", "method": "confidence-aware", "score": 0.5},
            {"clip_id": "c", "method": "direct", "score": 0.7},
            {"clip_id": "c", "method": "confidence-aware", "score": 0.4},
        ]
        outcomes = {
            ("a", "direct"): {"robot_success": True},
            ("a", "confidence-aware"): {"robot_success": False},
            ("b", "direct"): {"robot_success": False},
            ("b", "confidence-aware"): {"robot_success": True},
            ("c", "direct"): {"robot_success": True},
            ("c", "confidence-aware"): {"robot_success": True},
        }

        report = compare_predicted_outcomes(predictions, outcomes)

        self.assertEqual(report["matched_episode_count"], 6)
        self.assertEqual(report["informative_pair_count"], 2)
        self.assertEqual(report["pairwise_tie_count"], 1)
        self.assertEqual(report["pairwise_ranking_accuracy"], 0.75)
        self.assertAlmostEqual(report["mean_selection_regret"], 1 / 6)


if __name__ == "__main__":
    unittest.main()
