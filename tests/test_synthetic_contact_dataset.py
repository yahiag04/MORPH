"""Synthetic collection must preserve provenance and independent scenario groups."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from evaluation.synthetic_contact_dataset import (
    make_scenarios, collect_dataset, run_scenario, audit_transition_archive,
)
from evaluation.contact_demo_evaluator import write_transition_archive


class SyntheticContactDatasetTests(unittest.TestCase):
    def test_model_snapshot_captures_changes_in_included_xml(self):
        from evaluation.synthetic_contact_dataset import snapshot_robot_model
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            scene = root / 'scene.xml'
            included = root / 'robot.xml'
            scene.write_text('<mujoco><include file="robot.xml"/></mujoco>')
            included.write_text('<mujoco><worldbody><geom type="sphere" size=".1"/></worldbody></mujoco>')
            first = snapshot_robot_model(scene, root / 'first.mjb')
            included.write_text('<mujoco><worldbody><geom type="sphere" size=".2"/></worldbody></mujoco>')
            second = snapshot_robot_model(scene, root / 'second.mjb')
            self.assertEqual(first['scene_xml_sha256'], second['scene_xml_sha256'])
            self.assertNotEqual(first['compiled_model_sha256'], second['compiled_model_sha256'])
            self.assertTrue((root / 'first.mjb').is_file())

    def test_seeded_families_keep_all_interventions_in_one_partition(self):
        first = make_scenarios(families=60, seed=23)
        self.assertEqual(first, make_scenarios(families=60, seed=23))
        self.assertNotEqual(first, make_scenarios(families=60, seed=24))
        self.assertEqual(len(first), 240)
        self.assertEqual(len({s['episode_id'] for s in first}), 240)
        family_splits = {}
        for scenario in first:
            family_splits.setdefault(scenario['group_id'], set()).add(scenario['split'])
        self.assertTrue(all(len(splits) == 1 for splits in family_splits.values()))
        self.assertEqual({s: sum(row['split'] == s for row in first) for s in ['train','validation','test']},
                         {'train':168, 'validation':36, 'test':36})
        self.assertEqual({s['variant'] for s in first}, {'nominal','fast','grasp_offset','release_shift'})
        self.assertTrue(all(s['source_kind'] == 'simulation_randomized' for s in first))

    def test_one_episode_exports_real_commands_and_valid_sampling(self):
        scenario = make_scenarios(families=4, seed=23)[0]
        run = run_scenario(scenario)
        self.assertEqual(run['human_label'], 'not_applicable')
        self.assertEqual(run['source_kind'], 'simulation_randomized')
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'episode.npz'
            write_transition_archive([run], path)
            audit = audit_transition_archive(path)
            self.assertGreater(audit['transitions'], 100)
            self.assertAlmostEqual(audit['transition_dt_seconds'], .02)
            self.assertGreater(audit['phase_counts']['settle'], 0)
            with np.load(path, allow_pickle=False) as data:
                self.assertEqual(set(data['group_ids']), {scenario['group_id']})
                self.assertEqual(set(data['splits']), {scenario['split']})
                changed = np.any(np.diff(data['actuator_controls'],axis=0) != 0, axis=1)
                # Five observation intervals per held actuator command.
                self.assertTrue(np.all((np.flatnonzero(changed)+1)%5 == 0))

    def test_existing_outputs_and_repository_paths_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            (path/'sentinel').write_text('keep')
            with self.assertRaisesRegex(ValueError, 'empty'):
                collect_dataset(path, families=4, seed=23)
            self.assertEqual((path/'sentinel').read_text(), 'keep')
        with self.assertRaisesRegex(ValueError, 'repository'):
            collect_dataset(Path(__file__).resolve().parents[1]/'test-collection', families=4, seed=23)


if __name__ == '__main__':
    unittest.main()
