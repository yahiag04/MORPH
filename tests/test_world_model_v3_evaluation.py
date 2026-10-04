"""V3 evaluation scores contact events and action choices without leakage."""

import unittest

import numpy as np

from world_model.v3.contracts import EpisodeBatch, RolloutBatch


def make_episode(*, steps: int = 12, group: str = "family-a") -> EpisodeBatch:
    states = np.zeros((steps, 37), dtype=np.float32)
    states[:, 20] = 1.0
    states[:, 0] = np.arange(steps, dtype=np.float32) * 0.01
    states[:, 17] = states[:, 0]
    states[:, 32] = np.arange(steps) >= 3
    states[:, 33] = np.arange(steps) >= 8
    following = states.copy()
    following[:-1] = states[1:]
    actions = np.zeros((steps, 8), dtype=np.float32)
    actions[:-1, 0] = 0.01
    actions[-1, 0] = 0.0
    actions[:, 1] = following[:, 32]
    actions[:, 2] = following[:, 33]
    starts = np.arange(steps, dtype=np.float64) * 0.02
    ends = starts + 0.02
    return EpisodeBatch(
        states=states, actions=actions, next_states=following,
        episode_ids=np.full(steps, "episode-a"), group_ids=np.full(steps, group),
        phases=np.full(steps, "carry"), start_times=starts, end_times=ends,
        metadata={"source_kind": "simulation_randomized", "observation_dt": 0.02},
    )


class PerfectActionModel:
    """Small deterministic model implementing the public V3 rollout contract."""

    training_group_ids = ()

    def rollout(self, initial_states: np.ndarray, actions: np.ndarray) -> RolloutBatch:
        count, horizon = actions.shape[:2]
        states = np.empty((1, count, horizon + 1, 37), dtype=np.float32)
        states[0, :, 0] = initial_states
        contacts = np.empty((1, count, horizon, 2), dtype=np.float32)
        for step in range(horizon):
            state = states[0, :, step].copy()
            state[:, 0] += actions[:, step, 0]
            state[:, 17] += actions[:, step, 0]
            state[:, 32:34] = actions[:, step, 1:3]
            states[0, :, step + 1] = state
            contacts[0, :, step] = np.clip(actions[:, step, 1:3] * 0.98 + 0.01, 0.01, 0.99)
        return RolloutBatch(states=states, contact_probs=contacts,
                            valid=np.ones((1, count), dtype=bool))


class EarlyInvalidPerfectModel(PerfectActionModel):
    def rollout(self, initial_states: np.ndarray, actions: np.ndarray) -> RolloutBatch:
        result = super().rollout(initial_states, actions)
        result.valid[0] = initial_states[:, 0] >= 0.04
        return result


class EvaluationTests(unittest.TestCase):
    def test_action_conditioned_perfect_dynamics_beats_persistence(self):
        from world_model.v3.evaluation import BaselineDynamics, evaluate_dynamics

        report = evaluate_dynamics(PerfectActionModel(), make_episode(), horizons=(1, 5))
        one = report["horizons"]["1"]
        five = report["horizons"]["5"]
        self.assertAlmostEqual(one["groups"]["ee_position"]["model_rmse"], 0.0, places=7)
        self.assertGreater(one["groups"]["ee_position"]["persistence_rmse"], 0.0)
        self.assertEqual(
            one["family_cluster_bootstrap"]["package_position"]["family_count"], 1,
        )
        self.assertEqual(
            set(one["per_family_metrics"]["package_position"]), {"family-a"},
        )
        self.assertAlmostEqual(five["groups"]["package_position"]["model_rmse"], 0.0, places=7)
        self.assertGreater(five["groups"]["package_position"]["constant_velocity_rmse"], 0.0)
        self.assertEqual(five["orientation"]["model"]["endpoint_mean_radians"], 0.0)
        persistence = evaluate_dynamics(
            BaselineDynamics("persistence"), make_episode(), horizons=(5,),
        )
        self.assertEqual(
            persistence["horizons"]["5"]["groups"]["package_position"]["model_rmse"],
            persistence["horizons"]["5"]["groups"]["package_position"]["persistence_rmse"],
        )
        self.assertEqual(
            persistence["one_step_contacts"]["events"]["gripper_contact"]["f1"], 0.0,
        )

    def test_windowing_never_crosses_episode_boundaries_and_nulls_missing_horizons(self):
        from world_model.v3.evaluation import evaluate_dynamics

        batch = make_episode(steps=12)
        other = make_episode(steps=4, group="family-b")
        other.episode_ids[:] = "episode-b"
        other.start_times[:] += 3.0
        other.end_times[:] += 3.0
        joined = EpisodeBatch(
            states=np.concatenate((batch.states, other.states)),
            actions=np.concatenate((batch.actions, other.actions)),
            next_states=np.concatenate((batch.next_states, other.next_states)),
            episode_ids=np.concatenate((batch.episode_ids, other.episode_ids)),
            group_ids=np.concatenate((batch.group_ids, other.group_ids)),
            phases=np.concatenate((batch.phases, other.phases)),
            start_times=np.concatenate((batch.start_times, other.start_times)),
            end_times=np.concatenate((batch.end_times, other.end_times)),
            metadata=batch.metadata,
        )
        report = evaluate_dynamics(PerfectActionModel(), joined, horizons=(5, 20))
        self.assertEqual(report["horizons"]["5"]["window_count"], 2)
        self.assertIsNone(report["horizons"]["20"]["groups"]["package_position"]["model_rmse"])

    def test_quaternion_error_treats_q_and_negative_q_as_same_orientation(self):
        from world_model.v3.evaluation import quaternion_angular_error

        q = np.asarray([[1.0, 0.0, 0.0, 0.0]])
        self.assertEqual(float(quaternion_angular_error(q, -q)[0]), 0.0)

    def test_event_metric_uses_one_to_one_matching_with_tolerance(self):
        from world_model.v3.evaluation import match_contact_events

        truth = np.zeros(12, dtype=np.float32)
        truth[[3, 9]] = 1.0
        prediction = np.zeros(12, dtype=np.float32)
        prediction[[4, 9, 10]] = 0.9
        result = match_contact_events(prediction, truth, tolerance_steps=2)
        self.assertEqual((result["true_positive"], result["false_positive"],
                          result["false_negative"]), (2, 1, 0))
        self.assertAlmostEqual(result["f1"], 0.8)

    def test_episode_event_metrics_score_state_changes_on_uncompressed_timeline(self):
        from world_model.v3.evaluation import evaluate_dynamics

        batch = make_episode()
        batch.states[:, 32] = np.arange(len(batch.states)) >= 8
        batch.states[:, 33] = np.arange(len(batch.states)) >= 10
        batch.next_states[:] = batch.states
        batch.next_states[:-1] = batch.states[1:]
        batch.actions[:, 1] = batch.next_states[:, 32]
        batch.actions[:, 2] = batch.next_states[:, 33]
        report = evaluate_dynamics(EarlyInvalidPerfectModel(), batch, horizons=(1,))
        events = report["one_step_contacts"]["events"]
        self.assertEqual(events["gripper_contact"]["support"], 1)
        self.assertEqual(events["support_contact"]["support"], 1)
        self.assertEqual(events["gripper_contact"]["f1"], 1.0)
        self.assertEqual(events["support_contact"]["f1"], 1.0)
        self.assertAlmostEqual(events["gripper_contact"]["covered_transition_fraction"], 2 / 3)

    def test_empty_event_support_is_null_instead_of_a_perfect_f1(self):
        from world_model.v3.evaluation import match_contact_events

        result = match_contact_events(np.zeros(5), np.zeros(5))
        self.assertIsNone(result["f1"])
        self.assertEqual(result["support"], 0)

    def test_always_on_contact_prediction_is_penalized_for_event_scoring(self):
        from world_model.v3.evaluation import match_contact_events

        truth = np.zeros(20, dtype=np.float32)
        truth[[5, 15]] = 1.0
        result = match_contact_events(np.ones(20), truth)
        self.assertLess(result["f1"], 0.25)
        self.assertEqual(result["false_positive"], 18)

    def test_episode_evaluation_rejects_interleaved_rows_and_timing_gaps(self):
        from world_model.v3.evaluation import evaluate_dynamics

        batch = make_episode()
        batch.episode_ids[5] = "episode-b"
        with self.assertRaisesRegex(ValueError, "interleav"):
            evaluate_dynamics(PerfectActionModel(), batch, horizons=(1,))

        batch = make_episode()
        batch.start_times[4] += 0.01
        batch.end_times[4] += 0.01
        with self.assertRaisesRegex(ValueError, "gap|overlap|contigu"):
            evaluate_dynamics(PerfectActionModel(), batch, horizons=(1,))

    def test_held_out_evaluator_rejects_training_family_overlap(self):
        from world_model.v3.evaluation import evaluate_dynamics

        model = PerfectActionModel()
        model.training_group_ids = ("family-a",)
        with self.assertRaisesRegex(ValueError, "overlap|held.out|training"):
            evaluate_dynamics(model, make_episode(), horizons=(1,))

    def test_final_protocol_binds_hashes_and_exact_test_families(self):
        from world_model.v3.evaluation import validate_frozen_protocol

        protocol = {
            "schema_version": 1, "frozen": True, "action_mode": "actuator8",
            "dataset_manifest_sha256": "data-hash", "checkpoint_sha256": "model-hash",
            "test_group_ids": ["family-a", "family-b"],
        }
        validate_frozen_protocol(
            protocol, action_mode="actuator8", dataset_manifest_sha256="data-hash",
            checkpoint_sha256="model-hash", test_group_ids={"family-a", "family-b"},
        )
        with self.assertRaisesRegex(ValueError, "checkpoint hash"):
            validate_frozen_protocol(
                protocol, action_mode="actuator8", dataset_manifest_sha256="data-hash",
                checkpoint_sha256="changed", test_group_ids={"family-a", "family-b"},
            )
        with self.assertRaisesRegex(ValueError, "families"):
            validate_frozen_protocol(
                protocol, action_mode="actuator8", dataset_manifest_sha256="data-hash",
                checkpoint_sha256="model-hash", test_group_ids={"family-a"},
            )

    def test_candidate_ranking_pairs_only_different_outcomes_within_scene(self):
        from world_model.v3.evaluation import evaluate_candidates

        predictions = [
            {"group_id": "scene-a", "candidate_id": "good", "score": 0.8},
            {"group_id": "scene-a", "candidate_id": "bad", "score": 0.2},
            {"group_id": "scene-b", "candidate_id": "only", "score": 0.95},
        ]
        outcomes = [
            {"group_id": "scene-a", "candidate_id": "good", "success": True},
            {"group_id": "scene-a", "candidate_id": "bad", "success": False},
            {"group_id": "scene-b", "candidate_id": "only", "success": True},
        ]
        result = evaluate_candidates(predictions, outcomes)
        self.assertEqual(result["informative_pair_count"], 1)
        self.assertEqual(result["pairwise_ranking_accuracy"], 1.0)
        self.assertEqual(result["selection_regret"], 0.0)

    def test_candidate_ties_uninformative_pairs_and_scene_regret_are_reported(self):
        from world_model.v3.evaluation import evaluate_candidates

        result = evaluate_candidates(
            [
                {"group_id": "mixed", "candidate_id": "a", "score": 0.4},
                {"group_id": "mixed", "candidate_id": "b", "score": 0.4},
                {"group_id": "all-fail", "candidate_id": "c", "score": 0.2},
                {"group_id": "all-fail", "candidate_id": "d", "score": 0.8},
            ],
            [
                {"group_id": "mixed", "candidate_id": "a", "success": True},
                {"group_id": "mixed", "candidate_id": "b", "success": False},
                {"group_id": "all-fail", "candidate_id": "c", "success": False},
                {"group_id": "all-fail", "candidate_id": "d", "success": False},
            ],
        )
        self.assertEqual(result["informative_pair_count"], 1)
        self.assertEqual(result["pairwise_tie_count"], 1)
        self.assertEqual(result["pairwise_ranking_accuracy"], 0.5)
        self.assertEqual(result["uninformative_pair_count"], 1)
        self.assertEqual(result["selection_regret"], 0.25)

    def test_candidate_comparison_rejects_missing_duplicate_and_non_boolean_outcomes(self):
        from world_model.v3.evaluation import evaluate_candidates

        predicted = [{"group_id": "s", "candidate_id": "a", "score": 1.0}]
        with self.assertRaisesRegex(ValueError, "match|missing|candidate"):
            evaluate_candidates(predicted, [])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            evaluate_candidates(predicted * 2, [
                {"group_id": "s", "candidate_id": "a", "success": True}
            ])
        with self.assertRaisesRegex(ValueError, "boolean"):
            evaluate_candidates(predicted, [
                {"group_id": "s", "candidate_id": "a", "success": 1}
            ])

    def test_bootstrap_resamples_families_as_units_and_is_seeded(self):
        from world_model.v3.evaluation import bootstrap_families

        rows = [
            {"group_id": "family-a", "metrics": {"success": 1.0}},
            {"group_id": "family-a", "metrics": {"success": 1.0}},
            {"group_id": "family-b", "metrics": {"success": 0.0}},
        ]
        first = bootstrap_families(rows, seed=11, samples=200)
        second = bootstrap_families(rows, seed=11, samples=200)
        self.assertEqual(first, second)
        self.assertEqual(first["family_count"], 2)
        self.assertEqual(first["row_count"], 3)
        self.assertEqual(first["mean"]["success"], 0.5)
        self.assertEqual(first["confidence_interval_95"]["success"], [0.0, 1.0])


if __name__ == "__main__":
    unittest.main()
