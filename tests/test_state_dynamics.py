"""Tests for clip-held-out action-conditioned dynamics prediction."""

import importlib.util
import unittest

import numpy as np

from world_model.state_dynamics import (
    ACTION_DIM, CONTACT_STATE_INDICES, STATE_DIM, StateDynamicsModel,
    StateDynamicsFit, split_by_clip, validate_transitions,
)


class StateDynamicsDataTests(unittest.TestCase):
    def test_declares_ordered_state_and_action_dimensions(self):
        self.assertEqual(STATE_DIM, 37)
        self.assertEqual(ACTION_DIM, 4)
        self.assertEqual(CONTACT_STATE_INDICES, (32, 33))

    def test_validates_finite_state_action_transition_shapes(self):
        states = np.zeros((8, STATE_DIM), dtype=np.float32)
        actions = np.zeros((8, ACTION_DIM), dtype=np.float32)
        next_states = np.zeros_like(states)
        states[:, 20] = next_states[:, 20] = 1.0
        checked = validate_transitions(states, actions, next_states)
        self.assertEqual(checked[0].shape, (8, STATE_DIM))
        actions[0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "finite"):
            validate_transitions(states, actions, next_states)

    def test_splits_whole_clip_groups_deterministically(self):
        clip_ids = np.repeat(np.asarray([f"clip-{i}" for i in range(8)]), 10)
        first = split_by_clip(clip_ids, validation_fraction=0.25, seed=13)
        second = split_by_clip(clip_ids, validation_fraction=0.25, seed=13)
        np.testing.assert_array_equal(first.train_mask, second.train_mask)
        np.testing.assert_array_equal(first.validation_mask, second.validation_mask)
        self.assertFalse(set(clip_ids[first.train_mask]) & set(clip_ids[first.validation_mask]))
        self.assertEqual(len(first.validation_clip_ids), 2)

    def test_keeps_both_replay_methods_for_a_source_video_in_one_fold(self):
        source_clip_ids = np.repeat(np.asarray(["video-a", "video-b", "video-c", "video-d"]), 2)
        replay_methods = np.tile(np.asarray(["direct", "confidence-aware"]), 4)
        fit = split_by_clip(source_clip_ids, validation_fraction=0.5, seed=4)
        for clip_id in np.unique(source_clip_ids):
            rows = source_clip_ids == clip_id
            self.assertTrue(np.all(fit.train_mask[rows]) or np.all(fit.validation_mask[rows]))
            self.assertEqual(set(replay_methods[rows]), {"direct", "confidence-aware"})

    def test_held_out_evaluator_rejects_clip_leakage(self):
        from world_model.state_dynamics import evaluate_state_dynamics

        states = np.zeros((2, STATE_DIM), dtype=np.float32)
        states[:, 20] = 1.0
        actions = np.zeros((2, ACTION_DIM), dtype=np.float32)
        fit = StateDynamicsFit(None, None, ("seen-video",), ("validation-video",), {})
        with self.assertRaisesRegex(ValueError, "overlap"):
            evaluate_state_dynamics(fit, states, actions, states.copy(), np.asarray(["seen-video"] * 2))


@unittest.skipUnless(importlib.util.find_spec("torch"), "optional PyTorch runtime is not installed")
class StateDynamicsModelTests(unittest.TestCase):
    def test_predicts_next_state_with_the_declared_shape(self):
        from world_model.state_dynamics import StateDynamicsModel
        import torch

        model = StateDynamicsModel()
        states = torch.zeros((5, STATE_DIM))
        actions = torch.zeros((5, ACTION_DIM))
        state_delta, contact_logits = model(states, actions)
        self.assertEqual(tuple(state_delta.shape), (5, 32))
        self.assertEqual(tuple(contact_logits.shape), (5, 2))
        self.assertTrue(torch.isfinite(state_delta).all())
        self.assertTrue(torch.isfinite(contact_logits).all())

    def test_training_improves_held_out_error_over_persistence(self):
        from world_model.state_dynamics import fit_state_dynamics

        rng = np.random.default_rng(7)
        clip_ids = np.repeat(np.asarray([f"clip-{i}" for i in range(10)]), 60)
        states = np.zeros((len(clip_ids), STATE_DIM), dtype=np.float32)
        actions = rng.normal(0, 0.5, size=(len(clip_ids), ACTION_DIM)).astype(np.float32)
        next_states = np.zeros_like(states)
        for clip_index in range(10):
            current = np.zeros(STATE_DIM, dtype=np.float32)
            current[:2] = rng.normal(0, 0.5, size=2)
            current[20] = 1.0
            for step in range(60):
                index = clip_index * 60 + step
                states[index] = current
                following = current.copy()
                following[0] = 0.90 * current[0] + 0.05 * current[1] + 0.20 * actions[index, 0]
                following[1] = 0.90 * current[1] - 0.05 * current[0] + 0.20 * actions[index, 1]
                following[32] = float(actions[index, 2] > 0.0)
                next_states[index] = following
                current = following
        fit = fit_state_dynamics(
            states, actions, next_states, clip_ids,
            epochs=90, batch_size=16, seed=5, validation_fraction=0.2,
            patience=12, ensemble_size=2,
        )
        self.assertLess(fit.metrics["groups"]["ee_position"]["one_step_rmse"],
                        fit.metrics["groups"]["ee_position"]["persistence_one_step_rmse"])
        self.assertLess(fit.metrics["groups"]["ee_position"]["world_model_rmse"],
                        fit.metrics["groups"]["ee_position"]["persistence_rmse"])
        self.assertFalse(set(fit.train_clip_ids) & set(fit.validation_clip_ids))
        self.assertEqual(fit.metrics["state_dim"], STATE_DIM)
        self.assertIn("world_model_rmse", fit.metrics["groups"]["ee_position"])
        self.assertIn("persistence_rmse", fit.metrics["groups"]["ee_position"])
        self.assertIn("contact_f1", fit.metrics)

    def test_normalization_does_not_scale_binary_contact_flags(self):
        from world_model.state_dynamics import DynamicsNormalizer

        states = np.zeros((8, STATE_DIM), dtype=np.float32)
        states[:, 32] = np.arange(8) % 2
        states[:, 33] = np.arange(8) % 2
        actions = np.ones((8, ACTION_DIM), dtype=np.float32)
        states[:, 20] = 1.0
        normalizer = DynamicsNormalizer.fit(states, actions, states.copy())
        self.assertEqual(float(normalizer.state_mean[32]), 0.0)
        self.assertEqual(float(normalizer.state_scale[33]), 1.0)
        self.assertGreaterEqual(float(normalizer.state_scale[0]), 0.0199)
        self.assertGreaterEqual(float(normalizer.delta_scale[0]), 0.0099)

    def test_prediction_preserves_task_context_and_valid_contact_probabilities(self):
        from world_model.state_dynamics import DynamicsNormalizer, StateDynamicsEnsemble, predict_next_state

        states = np.zeros((8, STATE_DIM), dtype=np.float32)
        states[:, 20] = 1.0
        states[:, 34:37] = (0.55, 0.14, 0.416)
        actions = np.zeros((8, ACTION_DIM), dtype=np.float32)
        normalizer = DynamicsNormalizer.fit(states, actions, states.copy())
        model = StateDynamicsEnsemble(count=2)
        predicted, contacts, _ = predict_next_state(model, normalizer, states[:2], actions[:2])
        np.testing.assert_allclose(predicted[:, 34:37], states[:2, 34:37])
        self.assertTrue(np.all((contacts >= 0.0) & (contacts <= 1.0)))
        np.testing.assert_allclose(np.linalg.norm(predicted[:, 20:24], axis=1), 1.0, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
