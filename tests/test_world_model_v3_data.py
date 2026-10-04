"""V3 loaders must keep actions, transitions, timing and provenance aligned."""

import tempfile
import unittest
from pathlib import Path

import numpy as np


def valid_episode(*, episode: str, family: str, split: str, source: str,
                  steps: int = 8) -> dict[str, np.ndarray]:
    states = np.zeros((steps, 37), dtype=np.float32)
    states[:, 20] = 1.0
    for index in range(steps):
        states[index, 0] = index * 0.01
        states[index, 17] = index * 0.001
        states[index, 32] = float(index >= 2)
        states[index, 33] = float(index >= 5)
    next_states = states.copy()
    next_states[:-1] = states[1:]
    next_states[-1, 0] += 0.01
    next_states[-1, 17] += 0.001
    actions = np.arange(steps * 4, dtype=np.float32).reshape(steps, 4)
    actuator_controls = np.arange(steps * 8, dtype=np.float64).reshape(steps, 8)
    starts = np.arange(steps, dtype=np.float64) * 0.02
    ends = starts + 0.02
    return {
        "states": states, "actions": actions, "next_states": next_states,
        "actuator_controls": actuator_controls,
        "episode_ids": np.full(steps, episode),
        "group_ids": np.full(steps, family),
        "splits": np.full(steps, split),
        "source_kinds": np.full(steps, source),
        "phases": np.full(steps, "carry"),
        "simulation_start_times": starts,
        "simulation_end_times": ends,
        "transition_dt_seconds": ends - starts,
    }


def write_synthetic_episode(root: Path, **kwargs) -> dict:
    arrays = valid_episode(**kwargs)
    destination = root / "episodes" / f"{kwargs['episode']}.npz"
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, **arrays)
    return arrays


class WorldModelV3DataTests(unittest.TestCase):
    def test_loads_cartesian_and_actuator_actions_without_shifting_transitions(self):
        from world_model.v3.data import load_partitions

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            expected = write_synthetic_episode(
                root, episode="ep-a", family="family-a", split="train",
                source="simulation_randomized",
            )
            write_synthetic_episode(
                root, episode="ep-b", family="family-b", split="validation",
                source="simulation_randomized",
            )
            cartesian = load_partitions(root, action_mode="cartesian4")
            actuator = load_partitions(root, action_mode="actuator8")

            np.testing.assert_array_equal(cartesian["train"].actions, expected["actions"])
            np.testing.assert_array_equal(actuator["train"].actions, expected["actuator_controls"])
            np.testing.assert_array_equal(cartesian["train"].states, expected["states"])
            np.testing.assert_array_equal(cartesian["train"].next_states, expected["next_states"])
            np.testing.assert_array_equal(actuator["train"].episode_ids, expected["episode_ids"])
            self.assertEqual(actuator["train"].actions.shape, (8, 8))

    def test_rejects_missing_actuator_controls_without_falling_back_to_cartesian(self):
        from world_model.v3.data import load_partitions

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            arrays = write_synthetic_episode(
                root, episode="ep-a", family="family-a", split="train",
                source="simulation_randomized",
            )
            path = root / "episodes" / "ep-a.npz"
            del arrays["actuator_controls"]
            np.savez_compressed(path, **arrays)
            with self.assertRaisesRegex(ValueError, "actuator_controls"):
                load_partitions(root, action_mode="actuator8")

    def test_rejects_family_leakage(self):
        from world_model.v3.data import load_partitions

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for episode, split, source in (
                ("ep-a", "train", "simulation_randomized"),
                ("ep-b", "validation", "simulation_randomized"),
            ):
                write_synthetic_episode(root, episode=episode, family="family-a",
                                        split=split, source=source)
            with self.assertRaisesRegex(ValueError, "family.*partition|partition.*family"):
                load_partitions(root, action_mode="actuator8")

    def test_audit_counts_contact_events_per_transition_and_control_hold_durations(self):
        from world_model.v3.data import audit_actions, load_partitions

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            train = write_synthetic_episode(
                root, episode="ep-a", family="family-a", split="train",
                source="simulation_randomized",
            )
            train["actuator_controls"][:5] = 10.0
            train["actuator_controls"][5:] = 20.0
            np.savez_compressed(root / "episodes" / "ep-a.npz", **train)
            write_synthetic_episode(
                root, episode="ep-b", family="family-b", split="validation",
                source="simulation_randomized",
            )

            report = audit_actions(load_partitions(root, action_mode="actuator8"))
            result = report["partitions"]["train"]
            self.assertEqual(result["contact_changes"], {
                "gripper_contact": 1, "support_contact": 1,
            })
            self.assertEqual(result["phase_counts"], {"carry": 8})
            self.assertEqual(
                result["action_hold_run_length_observations"]["exactly_five_fraction"], 0.5
            )
            self.assertEqual(result["seconds"], 0.16)

    def test_rejects_temporal_gaps_interleaving_and_broken_state_chains(self):
        from world_model.v3.data import load_partitions

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            arrays = write_synthetic_episode(
                root, episode="ep-a", family="family-a", split="train",
                source="simulation_randomized",
            )
            path = root / "episodes" / "ep-a.npz"
            arrays["simulation_start_times"][4] += 0.01
            np.savez_compressed(path, **arrays)
            with self.assertRaisesRegex(ValueError, "gap|overlap|tim|contiguous"):
                load_partitions(root, action_mode="actuator8")

            arrays["simulation_start_times"] = np.arange(8) * 0.02
            arrays["simulation_end_times"] = arrays["simulation_start_times"] + 0.02
            arrays["next_states"][2, 0] += 1.0
            np.savez_compressed(path, **arrays)
            with self.assertRaisesRegex(ValueError, "chain|contiguous|next_states"):
                load_partitions(root, action_mode="actuator8")

    def test_rejects_invalid_quaternion_nonfinite_values_and_bad_action_mode(self):
        from world_model.v3.data import load_partitions

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            arrays = write_synthetic_episode(
                root, episode="ep-a", family="family-a", split="train",
                source="simulation_randomized",
            )
            path = root / "episodes" / "ep-a.npz"
            arrays["states"][0, 20:24] = 0.0
            np.savez_compressed(path, **arrays)
            with self.assertRaisesRegex(ValueError, "quaternion"):
                load_partitions(root, action_mode="actuator8")

            arrays["states"][0, 20:24] = (1.0, 0.0, 0.0, 0.0)
            arrays["actions"][0, 0] = np.nan
            np.savez_compressed(path, **arrays)
            with self.assertRaisesRegex(ValueError, "finite"):
                load_partitions(root, action_mode="cartesian4")

            with self.assertRaisesRegex(ValueError, "action_mode"):
                load_partitions(root, action_mode="torque7")

    def test_human_replay_loader_keeps_both_methods_in_the_source_clip_group(self):
        from world_model.v3.data import load_human_replays

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            shared = valid_episode(episode="video-a:direct", family="video-a",
                                   split="unassigned", source="human_replay")
            for method in ("direct", "confidence-aware"):
                arrays = {key: value.copy() for key, value in shared.items()}
                arrays["episode_ids"] = np.full(
                    len(arrays["states"]), f"video-a:{method}"
                )
                arrays["clip_ids"] = np.full(len(arrays["states"]), "video-a")
                arrays["actions"] += 1.0 if method == "confidence-aware" else 0.0
                path = root / f"{method}_transitions.npz"
                np.savez_compressed(path, **arrays)
            batch = load_human_replays(root, action_mode="actuator8")

            self.assertEqual(len(batch.states), 16)
            self.assertEqual(set(batch.episode_ids), {"video-a:direct", "video-a:confidence-aware"})
            self.assertEqual(set(batch.group_ids), {"video-a"})
            self.assertEqual(batch.metadata["source_kind"], "human_replay")
            self.assertEqual(batch.metadata["role"], "historical_diagnostic")

    def test_human_replay_loader_is_independent_of_optional_pytorch(self):
        from world_model.v3.contracts import EpisodeBatch

        self.assertEqual(EpisodeBatch.__module__, "world_model.v3.contracts")


if __name__ == "__main__":
    unittest.main()
