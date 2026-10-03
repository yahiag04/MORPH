"""Explicit simulation timing for state-transition archives."""

import numpy as np


def transition_interval(intervals: np.ndarray | None, *, count: int,
                        legacy_interval: float | None = None) -> float:
    """Resolve one fixed observation interval; never infer timing from row count."""
    if legacy_interval is not None and (not np.isfinite(legacy_interval) or legacy_interval <= 0):
        raise ValueError('legacy transition interval must be finite and positive')
    if intervals is None:
        if legacy_interval is None:
            raise ValueError('archive lacks timing; supply an explicit legacy --transition-dt-seconds')
        return float(legacy_interval)
    intervals = np.asarray(intervals,dtype=np.float64)
    if count < 1 or intervals.shape != (count,):
        raise ValueError('transition_dt_seconds must contain one interval per transition')
    if not np.isfinite(intervals).all() or np.any(intervals <= 0):
        raise ValueError('transition intervals must be finite and positive')
    if not np.allclose(intervals,intervals[0],rtol=1e-7,atol=1e-9):
        raise ValueError('mixed transition intervals require separate datasets')
    interval = float(np.mean(intervals))
    if legacy_interval is not None and not np.isclose(legacy_interval,interval,rtol=1e-7,atol=1e-9):
        raise ValueError('explicit transition interval disagrees with archive timing')
    return interval
