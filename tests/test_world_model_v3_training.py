"""Small deterministic CPU training lifecycle contracts for V3."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

try:
    import torch  # noqa: F401
except ImportError:
    torch = None

from world_model.v3.contracts import EpisodeBatch


def make_batch(group: str, *, episodes: int = 1, steps: int = 27, action_dim: int = 8):
    count = episodes * steps
    states = np.zeros((count, 37), dtype=np.float32)
    states[:, 0] = np.linspace(0.0, 0.2, count)
    states[:, 17] = states[:, 0] * 0.5
    states[:, 20] = 1.0
    states[:, 24] = 0.01
    actions = np.zeros((count, action_dim), dtype=np.float32)
    actions[:, 0] = np.linspace(-0.5, 0.5, count)
    following = states.copy()
    for episode in range(episodes):
        lo, hi = episode * steps, (episode + 1) * steps
        if hi - lo > 1:
            following[lo:hi - 1] = states[lo + 1:hi]
    starts = np.concatenate([np.arange(steps) * 0.02 + i for i in range(episodes)])
    return EpisodeBatch(
        states=states, actions=actions, next_states=following,
        episode_ids=np.repeat([f"{group}-episode-{i}" for i in range(episodes)], steps),
        group_ids=np.repeat([f"{group}-family-{i}" for i in range(episodes)], steps),
        phases=np.full(count, "carry"), start_times=starts, end_times=starts + 0.02,
        metadata={"source_kind": "simulation_randomized", "observation_dt": 0.02,
                  "action_mode": "actuator8" if action_dim == 8 else "cartesian4"},
    )


@unittest.skipIf(torch is None, "optional PyTorch runtime is not installed")
class DynamicsV3TrainingTests(unittest.TestCase):
    def test_known_linear_action_dynamics_integrates_correctly(self):
        from world_model.v3.training import _step

        class LinearMember:
            def __call__(self, normalized_state, normalized_action):
                delta = torch.zeros((len(normalized_state), 32), dtype=normalized_state.dtype)
                delta[:, 0] = 0.5 * normalized_action[:, 0]
                contact_logits = torch.zeros((len(normalized_state), 2), dtype=normalized_state.dtype)
                return delta, contact_logits

        state = torch.zeros((2, 37), dtype=torch.float32)
        state[:, 0] = torch.tensor([0.2, -0.1])
        state[:, 20] = 1.0
        action = torch.zeros((2, 8), dtype=torch.float32)
        action[:, 0] = torch.tensor([0.4, -0.6])
        params = {
            "state_mean": torch.zeros(37), "state_scale": torch.ones(37),
            "action_mean": torch.zeros(8), "action_scale": torch.ones(8),
            "delta_mean": torch.zeros(32), "delta_scale": torch.ones(32),
        }
        predicted, _, _, _ = _step(
            LinearMember(), state, action, {"relative_features": False}, params,
        )
        np.testing.assert_allclose(predicted[:, 0].numpy(), [0.4, -0.4], atol=1e-7)
        self.assertTrue(torch.allclose(predicted[:, 20], torch.ones(2)))

    def test_normalizer_uses_training_only_and_floors_constant_dimensions(self):
        from world_model.v3.training import fit_normalizer

        train = make_batch("train")
        altered = make_batch("validation")
        altered.states[:, 0] += 100.0
        altered.next_states[:, 0] += 100.0
        first = fit_normalizer(train, relative_features=True)
        second = fit_normalizer(train, relative_features=True)
        validation_only = fit_normalizer(altered, relative_features=True)
        self.assertTrue(np.array_equal(first.state_mean, second.state_mean))
        self.assertFalse(np.array_equal(first.state_mean, validation_only.state_mean))
        self.assertTrue(np.all(first.state_scale > 0.0))

    def test_fit_is_reproducible_saves_provenance_and_uses_family_split(self):
        from world_model.v3.training import fit_dynamics

        train, validation = make_batch("train"), make_batch("validation")
        config = {
            "action_mode": "actuator8", "ensemble_size": 1, "hidden_dim": 8,
            "seed": 17, "epochs": 1, "batch_size": 2, "learning_rate": 5e-4,
            "rollout_horizon": 25, "patience": 1, "relative_features": False,
        }
        with tempfile.TemporaryDirectory() as temporary:
            first_dir = Path(temporary) / "first"
            second_dir = Path(temporary) / "second"
            first = fit_dynamics(train, validation, config=config, output_dir=first_dir)
            second = fit_dynamics(train, validation, config=config, output_dir=second_dir)
            self.assertEqual(first.training_group_ids, ("train-family-0",))
            self.assertEqual(first.validation_group_ids, ("validation-family-0",))
            self.assertTrue(np.array_equal(
                next(first.network.members[0].parameters()).detach().numpy(),
                next(second.network.members[0].parameters()).detach().numpy(),
            ))
            self.assertEqual((first_dir / "best.pt").is_file(), True)
            self.assertEqual((first_dir / "last.pt").is_file(), True)
            self.assertEqual(json_status(first_dir), "complete")

    def test_rejects_family_leakage_and_nonempty_output_without_resume(self):
        from world_model.v3.training import fit_dynamics

        train = make_batch("shared")
        validation = make_batch("shared-validation")
        validation.group_ids[:] = train.group_ids[0]
        config = {"action_mode": "actuator8", "ensemble_size": 1, "hidden_dim": 8,
                  "seed": 2, "epochs": 1, "batch_size": 2, "rollout_horizon": 25}
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "run"
            with self.assertRaisesRegex(ValueError, "overlap|family"):
                fit_dynamics(train, validation, config=config, output_dir=output)
            output.mkdir()
            (output / "keep.txt").write_text("existing")
            validation = make_batch("validation")
            with self.assertRaisesRegex(ValueError, "non-empty|resume"):
                fit_dynamics(train, validation, config=config, output_dir=output)

    def test_resume_restores_compatible_epoch_boundary_checkpoint(self):
        import world_model.v3.training as training

        train, validation = make_batch("train"), make_batch("validation")
        config = {"action_mode": "actuator8", "ensemble_size": 1, "hidden_dim": 8,
                  "seed": 3, "epochs": 1, "batch_size": 2, "rollout_horizon": 25}
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "interrupted"
            original_loss = training._window_loss
            try:
                training._window_loss = lambda *args, **kwargs: (_ for _ in ()).throw(
                    RuntimeError("simulated interruption"),
                )
                with self.assertRaisesRegex(RuntimeError, "simulated interruption"):
                    training.fit_dynamics(train, validation, config=config, output_dir=output)
            finally:
                training._window_loss = original_loss
            resumed = training.fit_dynamics(
                train, validation, config=config, output_dir=output, resume=True,
            )
            self.assertEqual(json_status(output), "complete")
            self.assertEqual(resumed.training_group_ids, ("train-family-0",))
            incompatible = dict(config, batch_size=3)
            with self.assertRaisesRegex(ValueError, "config"):
                training.fit_dynamics(
                    train, validation, config=incompatible, output_dir=output, resume=True,
                )


def json_status(run_dir: Path) -> str:
    import json

    return json.loads((run_dir / "status.json").read_text())["status"]


if __name__ == "__main__":
    unittest.main()
