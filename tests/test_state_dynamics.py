"""Tests for clip-held-out action-conditioned dynamics prediction."""

import importlib.util
import unittest

import numpy as np

from world_model.state_dynamics import ACTION_DIM, STATE_DIM, split_by_clip, validate_transitions


class StateDynamicsDataTests(unittest.TestCase):
    def test_declares_ordered_state_and_action_dimensions(self):
        self.assertEqual(STATE_DIM, 19)
        self.assertEqual(ACTION_DIM, 4)

    def test_validates_finite_state_action_transition_shapes(self):
        states = np.zeros((8, STATE_DIM), dtype=np.float32)
        actions = np.zeros((8, ACTION_DIM), dtype=np.float32)
        next_states = np.zeros_like(states)
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


@unittest.skipUnless(importlib.util.find_spec("torch"), "optional PyTorch runtime is not installed")
class StateDynamicsModelTests(unittest.TestCase):
    def test_predicts_next_state_with_the_declared_shape(self):
        from world_model.state_dynamics import StateDynamicsModel
        import torch

        model = StateDynamicsModel()
        states = torch.zeros((5, STATE_DIM))
        actions = torch.zeros((5, ACTION_DIM))
        prediction = model(states, actions)
        self.assertEqual(tuple(prediction.shape), (5, STATE_DIM))
        self.assertTrue(torch.isfinite(prediction).all())

    def test_training_improves_held_out_error_over_persistence(self):
        from world_model.state_dynamics import fit_state_dynamics

        rng = np.random.default_rng(7)
        clip_ids = np.repeat(np.asarray([f"clip-{i}" for i in range(10)]), 60)
        states = rng.normal(0, 0.5, size=(len(clip_ids), STATE_DIM)).astype(np.float32)
        actions = rng.normal(0, 0.5, size=(len(clip_ids), ACTION_DIM)).astype(np.float32)
        next_states = states.copy()
        next_states[:, 0] += 0.45 * states[:, 1] + 0.35 * actions[:, 0]
        next_states[:, 1] += -0.25 * states[:, 0] + 0.25 * actions[:, 1]
        fit = fit_state_dynamics(
            states, actions, next_states, clip_ids,
            epochs=180, batch_size=128, seed=5, validation_fraction=0.2,
        )
        self.assertLess(fit.metrics["one_step_rmse"], fit.metrics["persistence_one_step_rmse"])
        self.assertFalse(set(fit.train_clip_ids) & set(fit.validation_clip_ids))
        self.assertEqual(fit.metrics["state_dim"], STATE_DIM)
        self.assertIn("rollout_rmse", fit.metrics)
        self.assertIn("persistence_rollout_rmse", fit.metrics)

    def test_normalization_does_not_scale_binary_contact_flags(self):
        from world_model.state_dynamics import DynamicsNormalizer

        states = np.zeros((8, STATE_DIM), dtype=np.float32)
        states[:, 17] = np.arange(8) % 2
        states[:, 18] = np.arange(8) % 2
        actions = np.ones((8, ACTION_DIM), dtype=np.float32)
        normalizer = DynamicsNormalizer.fit(states, actions, states.copy())
        self.assertEqual(float(normalizer.state_mean[17]), 0.0)
        self.assertEqual(float(normalizer.state_scale[18]), 1.0)


if __name__ == "__main__":
    unittest.main()
