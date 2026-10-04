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
    def test_family_episode_sampler_balances_groups_and_event_windows(self):
        from world_model.v3.training import _sample_training_windows

        batch = make_batch("sample", episodes=4, steps=30)
        for episode in range(4):
            lo, hi = episode * 30, (episode + 1) * 30
            if episode == 0:
                batch.states[lo + 10:hi, 32] = 1.0
                batch.next_states[lo + 9:hi - 1, 32] = 1.0
            batch.episode_ids[lo:hi] = f"sample-episode-{episode}"
            batch.group_ids[lo:hi] = f"sample-family-{episode}"
        windows, diagnostics = _sample_training_windows(
            batch, horizon=25, stride=1, count=4000,
            rng=np.random.default_rng(17), event_fraction=0.5,
        )
        family_counts = {}
        sampled_events = 0
        for episode, family, rows in windows:
            family_counts[family] = family_counts.get(family, 0) + 1
            self.assertEqual(len(set(batch.episode_ids[rows])), 1)
            self.assertTrue(np.all(np.diff(rows) == 1))
            has_event = bool(np.any(
                batch.states[rows, 32] != batch.next_states[rows, 32]
            ))
            sampled_events += has_event
        self.assertEqual(set(family_counts), {f"sample-family-{i}" for i in range(4)})
        self.assertLess(max(family_counts.values()) - min(family_counts.values()), 150)
        self.assertEqual(diagnostics["requested_event_windows"], 2000)
        self.assertGreater(sampled_events, 400)

    def test_sampler_falls_back_to_uniform_when_contact_events_are_absent(self):
        from world_model.v3.training import _sample_training_windows

        batch = make_batch("no-events", episodes=2, steps=30)
        windows, diagnostics = _sample_training_windows(
            batch, horizon=25, stride=1, count=20,
            rng=np.random.default_rng(2), event_fraction=0.5,
        )
        self.assertEqual(len(windows), 20)
        self.assertEqual(diagnostics["sampled_event_windows"], 0)
        self.assertEqual(diagnostics["uniform_fallback_windows"], 10)

    def test_weighted_state_loss_is_quaternion_sign_invariant(self):
        from world_model.v3.training import _weighted_state_loss

        truth = torch.zeros((2, 37), dtype=torch.float32)
        truth[:, 20] = 1.0
        prediction = truth.clone()
        prediction[1, 20:24] *= -1.0
        loss = _weighted_state_loss(prediction, truth, torch.ones(32))
        self.assertAlmostEqual(float(loss.detach()), 0.0, places=7)

    def test_one_step_loss_restarts_from_each_observed_state(self):
        from unittest.mock import patch
        from world_model.v3.training import _window_loss

        states = torch.zeros((1, 5, 37), dtype=torch.float32)
        states[0, :, 0] = torch.arange(5, dtype=torch.float32)
        states[0, :, 20] = 1.0
        following = states.clone()
        following[0, :, 0] += 1.0
        actions = torch.zeros((1, 5, 8), dtype=torch.float32)
        observed_inputs = []

        def controlled_step(member, state, action, config, params):
            observed_inputs.append(state[:, 0].detach().clone())
            predicted = state.clone()
            predicted[:, 0] += 1.0
            logits = torch.zeros((len(state), 2), dtype=state.dtype)
            return predicted, torch.zeros((len(state), 32)), logits, torch.sigmoid(logits)

        params = {"state_scale": torch.ones(32), "contact_pos_weight": torch.ones(2)}
        with patch("world_model.v3.training._step", side_effect=controlled_step):
            _window_loss(None, states, actions, following,
                         {"relative_features": False}, params, train=True)
        teacher_forced_inputs = torch.stack(observed_inputs[::2])[:, 0]
        torch.testing.assert_close(teacher_forced_inputs, states[0, :, 0])

    def test_contact_class_weights_use_training_counts_and_handle_missing_class(self):
        from world_model.v3.training import _contact_class_weights

        train = make_batch("contact-weights", steps=30)
        train.next_states[:, 32] = 0.0
        train.next_states[:3, 32] = 1.0
        train.next_states[:, 33] = 1.0
        weights, report = _contact_class_weights(train)
        np.testing.assert_allclose(weights, [9.0, 1.0])
        self.assertEqual(report["gripper_contact"]["positives"], 3)
        self.assertEqual(report["support_contact"]["positive_weight"], 1.0)
        self.assertEqual(report["support_contact"]["missing_class"], "negative")

    def test_gradients_remain_finite_at_25_50_and_100_steps(self):
        from world_model.v3.training import _window_loss
        from world_model.v3.model import DynamicsMemberV3

        batch = make_batch("gradients", steps=101)
        member = DynamicsMemberV3(43, 8, 8)
        state_mean = torch.zeros(43)
        state_scale = torch.ones(43)
        params = {"state_mean": state_mean, "state_scale": state_scale,
                  "action_mean": torch.zeros(8), "action_scale": torch.ones(8),
                  "delta_mean": torch.zeros(32), "delta_scale": torch.ones(32),
                  "contact_pos_weight": torch.ones(2)}
        for horizon in (25, 50, 100):
            states = torch.as_tensor(batch.states[:horizon][None])
            actions = torch.as_tensor(batch.actions[:horizon][None])
            following = torch.as_tensor(batch.next_states[:horizon][None])
            member.zero_grad(set_to_none=True)
            loss = _window_loss(
                member, states, actions, following,
                {"relative_features": True}, params, train=True,
            )
            loss.backward()
            self.assertTrue(torch.isfinite(loss))
            self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all()
                                for p in member.parameters()))

    def test_curriculum_horizons_cover_25_50_and_100_steps(self):
        from world_model.v3.training import _horizon_for_epoch

        schedule = ((25, 15), (50, 15), (100, 10))
        self.assertEqual([_horizon_for_epoch(schedule, e) for e in (0, 14, 15, 29, 30, 39)],
                         [25, 25, 50, 50, 100, 100])

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

    def test_fit_advances_through_each_curriculum_stage_and_records_physical_horizons(self):
        import json
        from world_model.v3.training import fit_dynamics

        train = make_batch("curriculum-train", steps=101)
        validation = make_batch("curriculum-validation", steps=101)
        config = {
            "action_mode": "actuator8", "ensemble_size": 1, "hidden_dim": 8,
            "seed": 19, "epochs": 3, "batch_size": 1, "learning_rate": 5e-4,
            "rollout_horizon": 25, "horizon_schedule": [
                {"horizon": 25, "epochs": 1},
                {"horizon": 50, "epochs": 1},
                {"horizon": 100, "epochs": 1},
            ],
        }
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "curriculum"
            fit_dynamics(train, validation, config=config, output_dir=output)
            rows = [json.loads(line) for line in (output / "training.jsonl").read_text().splitlines()]
            self.assertEqual([row["horizon"] for row in rows], [25, 50, 100])
            self.assertEqual([row["horizon_seconds"] for row in rows], [0.5, 1.0, 2.0])
            metrics = json.loads((output / "metrics.json").read_text())
            self.assertEqual(metrics["training_window_count_by_horizon"], {
                "25": 16, "50": 11, "100": 1,
            })

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
