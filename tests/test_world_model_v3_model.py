"""Model behavior and checkpoint contracts for V3 dynamics."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

try:
    import torch  # noqa: F401
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "optional PyTorch runtime is not installed")
class DynamicsV3ModelTests(unittest.TestCase):
    def setUp(self):
        from world_model.v3.contracts import DynamicsConfig
        from world_model.v3.model import DynamicsNormalizerV3, DynamicsV3

        self.config = DynamicsConfig(action_mode="actuator8", ensemble_size=2, hidden_dim=16)
        self.model = DynamicsV3(self.config, seed=17, normalizer=DynamicsNormalizerV3(
            np.zeros(37), np.ones(37), np.zeros(8), np.ones(8), np.zeros(32), np.ones(32),
        ))

    def test_supports_both_action_dimensions_and_action_changes_prediction(self):
        from world_model.v3.contracts import DynamicsConfig
        from world_model.v3.model import DynamicsNormalizerV3, DynamicsV3

        for mode, dim in (("cartesian4", 4), ("actuator8", 8)):
            model = DynamicsV3(
                DynamicsConfig(mode, ensemble_size=1, hidden_dim=16), seed=4,
                normalizer=DynamicsNormalizerV3(
                    np.zeros(37), np.ones(37), np.zeros(dim), np.ones(dim),
                    np.zeros(32), np.ones(32),
                ),
            )
            state = torch.zeros((1, 37), dtype=torch.float32)
            state[:, 20] = 1.0
            first = torch.zeros((1, dim), dtype=torch.float32)
            second = first.clone()
            second[:, 0] = 1.0
            pred_a, _ = model.predict_member(0, state, first)
            pred_b, _ = model.predict_member(0, state, second)
            self.assertFalse(torch.allclose(pred_a, pred_b))

    def test_rollout_preserves_tray_normalizes_quaternion_and_keeps_members_separate(self):
        from world_model.v3.contracts import RolloutBatch

        initial = np.zeros((2, 37), dtype=np.float32)
        initial[:, 20] = 1.0
        initial[:, 34:37] = [[0.4, -0.2, 0.1], [0.5, 0.3, 0.0]]
        actions = np.zeros((2, 5, 8), dtype=np.float32)
        rollout = self.model.rollout(initial, actions)
        self.assertIsInstance(rollout, RolloutBatch)
        self.assertEqual(rollout.states.shape, (2, 2, 6, 37))
        self.assertEqual(rollout.contact_probs.shape, (2, 2, 5, 2))
        self.assertTrue(np.allclose(rollout.states[..., 34:37], initial[None, :, None, 34:37]))
        self.assertTrue(np.allclose(np.linalg.norm(rollout.states[..., 20:24], axis=-1), 1.0))
        self.assertEqual(rollout.valid.shape, (2, 2))

    def test_relative_features_are_deterministic_and_checkpoint_checks_contract(self):
        from world_model.v3.contracts import DynamicsConfig
        from world_model.v3.model import DynamicsNormalizerV3, DynamicsV3

        relative = DynamicsV3(
            DynamicsConfig("actuator8", relative_features=True, ensemble_size=1, hidden_dim=16),
            seed=3, normalizer=DynamicsNormalizerV3(
                np.zeros(43), np.ones(43), np.zeros(8), np.ones(8),
                np.zeros(32), np.ones(32),
            ),
        )
        state = np.zeros((1, 37), dtype=np.float32)
        state[:, 17] = 0.3
        state[:, 20] = 1.0
        state[:, 34] = 0.5
        self.assertEqual(relative.input_features(state).shape, (1, 43))
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "model.pt"
            self.model.save(checkpoint, {"data_hash": "example", "action_mode": "actuator8"})
            loaded = type(self.model).load(checkpoint, expected={"action_mode": "actuator8"})
            self.assertEqual(loaded.config.action_mode, "actuator8")
            with self.assertRaisesRegex(ValueError, "action_mode"):
                type(self.model).load(checkpoint, expected={"action_mode": "cartesian4"})


if __name__ == "__main__":
    unittest.main()
