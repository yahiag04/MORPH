#!/usr/bin/env python3
"""Collect timed, randomized MuJoCo episodes outside the repository."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from evaluation.synthetic_contact_dataset import collect_dataset
from simulation.panda_env import DEFAULT_MODEL_PATH


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--families', type=int, default=60, help='four variants per family; default 240 episodes')
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--model-path', type=Path, default=DEFAULT_MODEL_PATH)
    args = parser.parse_args()
    try:
        summary = collect_dataset(args.output_dir, families=args.families, seed=args.seed,
                                  model_path=args.model_path, progress=lambda line: print(line,flush=True))
    except (ValueError, OSError) as error:
        parser.error(str(error))
    print(json.dumps(summary,indent=2))
    return 1 if summary['collection_errors'] else 0


if __name__ == '__main__':
    raise SystemExit(main())
