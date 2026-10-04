#!/usr/bin/env python3
"""Collect paired, replayable MuJoCo intervention families outside the repository."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from evaluation.counterfactual_dataset import (  # noqa: E402
    collect_counterfactual_dataset,
    make_counterfactual_families,
)
from simulation.panda_env import DEFAULT_MODEL_PATH  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--families", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--split", choices=("development", "test"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--skip-replay-verification", action="store_true")
    args = parser.parse_args()
    try:
        scenarios = make_counterfactual_families(
            families=args.families, seed=args.seed, split=args.split,
        )
        summary = collect_counterfactual_dataset(
            args.output_dir, scenarios, model_path=args.model_path,
            progress=lambda line: print(line, flush=True),
            verify_replay=not args.skip_replay_verification,
        )
    except (ValueError, OSError) as error:
        parser.error(str(error))
    print(json.dumps(summary, indent=2))
    return 1 if summary["collection_errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
