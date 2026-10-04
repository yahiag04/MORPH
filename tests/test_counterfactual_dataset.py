"""Paired intervention families use one saved simulator initial condition."""

import tempfile
import unittest
from collections import defaultdict
from pathlib import Path
from unittest.mock import patch
import json

import mujoco
import numpy as np

from evaluation.counterfactual_dataset import (
    capture_family_snapshot,
    collect_counterfactual_dataset,
    load_family_snapshot,
    make_counterfactual_families,
    restore_family_snapshot,
    run_counterfactual_scenario,
)
from simulation.contact_manipulation_env import ContactManipulationEnv, ContactTaskConfig
from simulation.panda_env import DEFAULT_MODEL_PATH


class CounterfactualFamilyTests(unittest.TestCase):
    def test_seeded_eight_variants_share_one_family_and_split(self):
        first = make_counterfactual_families(families=4, seed=20261041, split="development")
        self.assertEqual(first, make_counterfactual_families(
            families=4, seed=20261041, split="development",
        ))
        self.assertEqual(len(first), 32)
        expected = {
            "nominal", "speed_1_5x", "speed_0_7x", "pickup_x_plus_15mm",
            "pickup_x_minus_15mm", "pickup_x_plus_30mm",
            "release_early", "release_late",
        }
        groups = defaultdict(list)
        for row in first:
            groups[row["group_id"]].append(row)
            self.assertNotIn("success", row)
        self.assertEqual(set(groups), {f"counterfactual-20261041-{i:03d}" for i in range(4)})
        self.assertEqual([len(groups[key]) for key in sorted(groups)], [8] * 4)
        self.assertEqual({row["split"] for row in first}, {"train", "validation"})
        for variants in groups.values():
            self.assertEqual({row["split"] for row in variants}.__len__(), 1)
            by_name = {row["variant"]: row for row in variants}
            nominal = by_name["nominal"]
            for name in ("speed_1_5x", "speed_0_7x"):
                row = by_name[name]
                self.assertEqual(row["pickup_xy"], nominal["pickup_xy"])
                self.assertEqual(row["dropoff_xy"], nominal["dropoff_xy"])
                self.assertEqual(row["release_open_fraction"], nominal["release_open_fraction"])
                self.assertNotEqual(row["cartesian_speed_mps"], nominal["cartesian_speed_mps"])
            self.assertEqual(
                by_name["release_early"]["release_open_fraction"], 0.2,
            )
            self.assertEqual(
                by_name["release_late"]["release_open_fraction"], 0.9,
            )
            for name in ("release_early", "release_late"):
                row = by_name[name]
                self.assertEqual(row["release_height_m"], nominal["release_height_m"])
                self.assertEqual(row["cartesian_speed_mps"], nominal["cartesian_speed_mps"])
            for name, offset in (("pickup_x_plus_15mm", 0.015),
                                 ("pickup_x_minus_15mm", -0.015),
                                 ("pickup_x_plus_30mm", 0.030)):
                row = by_name[name]
                self.assertEqual(row["pickup_xy"], nominal["pickup_xy"])
                self.assertEqual(row["dropoff_xy"], nominal["dropoff_xy"])
                self.assertEqual(row["cartesian_speed_mps"], nominal["cartesian_speed_mps"])
                self.assertEqual(row["release_open_fraction"], nominal["release_open_fraction"])
                self.assertAlmostEqual(
                    row["controller_pickup_xy"][0] - nominal["controller_pickup_xy"][0], offset,
                )

    def test_test_split_assigns_all_families_to_test(self):
        scenarios = make_counterfactual_families(families=5, seed=9, split="test")
        self.assertEqual({row["split"] for row in scenarios}, {"test"})

    def test_snapshot_restore_and_clone_leave_template_unchanged(self):
        scenario = make_counterfactual_families(families=4, seed=13, split="development")[0]
        with tempfile.TemporaryDirectory() as temporary:
            snapshot = capture_family_snapshot(scenario, output_dir=Path(temporary))
            stored = load_family_snapshot(snapshot["snapshot_path"])
            self.assertEqual(snapshot["compiled_model_sha256"], stored["compiled_model_sha256"])
            self.assertLess(Path(snapshot["snapshot_path"]).stat().st_size, 100_000)
            config = ContactTaskConfig(package_yaw_radians=scenario["package_yaw_radians"])
            env = ContactManipulationEnv(
                pickup_xyz=(*scenario["pickup_xy"], 0.412),
                dropoff_xyz=(*scenario["dropoff_xy"], 0.416), config=config,
            )
            initial_time = float(env.data.time)
            initial_qpos = env.data.qpos.copy()
            target = mujoco.MjData(env.model)
            restore_family_snapshot(env.model, target, stored)
            self.assertEqual(target.time, initial_time)
            np.testing.assert_array_equal(env.data.qpos, initial_qpos)
            np.testing.assert_array_equal(stored["qpos"], initial_qpos)
            self.assertTrue(np.array_equal(target.ctrl, stored["ctrl"]))
            self.assertTrue(np.array_equal(target.qacc_warmstart, stored["qacc_warmstart"]))
            before = {name: getattr(snapshot["initial_data"], name).copy()
                      for name in ("qpos", "qvel", "ctrl", "qacc_warmstart")}
            run_counterfactual_scenario(scenario, initial_snapshot=snapshot["initial_data"])
            for name, value in before.items():
                np.testing.assert_array_equal(getattr(snapshot["initial_data"], name), value)

    def test_development_and_test_seeds_use_disjoint_family_ids(self):
        development = make_counterfactual_families(families=100, seed=20261040,
                                                    split="development")
        test = make_counterfactual_families(families=100, seed=20261042, split="test")
        dev_ids = {row["group_id"] for row in development}
        test_ids = {row["group_id"] for row in test}
        self.assertFalse(dev_ids & test_ids)
        self.assertEqual(len(dev_ids), 100)
        self.assertEqual(len(test_ids), 100)

    def test_collection_errors_are_recorded_and_output_is_not_overwritten(self):
        scenarios = make_counterfactual_families(families=4, seed=27, split="development")
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "collection"
            with patch(
                "evaluation.counterfactual_dataset.run_counterfactual_scenario",
                side_effect=ValueError("controlled collection failure"),
            ):
                summary = collect_counterfactual_dataset(output, scenarios)
            self.assertEqual(summary["collection_errors"], 32)
            manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            self.assertTrue(all("collection_error" in item for item in manifest["episodes"]))
            with self.assertRaisesRegex(ValueError, "must be empty"):
                collect_counterfactual_dataset(output, scenarios)


if __name__ == "__main__":
    unittest.main()
