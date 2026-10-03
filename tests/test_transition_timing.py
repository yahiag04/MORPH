"""Rollout seconds must come from archive metadata or an explicit legacy interval."""

import unittest
import tempfile
from pathlib import Path
import numpy as np

from world_model.transition_timing import transition_interval


class TransitionTimingTests(unittest.TestCase):
    def test_video_loaders_do_not_resplit_reserved_synthetic_test_data(self):
        from scripts.train_world_model import _load
        from scripts.evaluate_world_model_cv import _load_pair
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'test-data.npz'
            np.savez(path, states=np.zeros((2,37)), actions=np.zeros((2,4)), next_states=np.zeros((2,37)),
                     clip_ids=np.array(['a','b']), transition_dt_seconds=np.full(2,.02),
                     splits=np.array(['test','test']))
            for loader in (_load, _load_pair):
                with self.assertRaisesRegex(ValueError,'partition'):
                    loader(path,'direct')

    def test_uses_observed_interval_and_requires_legacy_override(self):
        self.assertAlmostEqual(25*transition_interval(np.full(4,.1),count=4),2.5)
        self.assertAlmostEqual(25*transition_interval(np.full(4,.02),count=4),.5)
        with self.assertRaisesRegex(ValueError,'legacy'):
            transition_interval(None,count=4)
        self.assertAlmostEqual(transition_interval(None,count=4,legacy_interval=.1),.1)

    def test_rejects_mixed_nonfinite_or_conflicting_intervals(self):
        for intervals in (np.array([.1,.02]), np.array([.1,np.nan]), np.array([0.,0.]), np.ones(3)):
            with self.subTest(intervals=intervals), self.assertRaises(ValueError):
                transition_interval(intervals,count=2)
        with self.assertRaisesRegex(ValueError,'disagrees'):
            transition_interval(np.full(4,.1),count=4,legacy_interval=.02)


if __name__ == '__main__':
    unittest.main()
