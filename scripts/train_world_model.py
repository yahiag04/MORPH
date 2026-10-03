#!/usr/bin/env python3
"""Train and evaluate action-conditioned dynamics on recorded-demo simulations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from world_model.state_dynamics import fit_state_dynamics, save_checkpoint


def _load(path: Path, method: str) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        required = ("states", "actions", "next_states", "clip_ids")
        missing = set(required) - set(archive.files)
        if missing:
            raise ValueError(f"{path} is missing arrays: {sorted(missing)}")
        data = {key: archive[key] for key in required}
    clips = data["clip_ids"].astype(str)
    data["clip_ids"] = clips
    data["episode_ids"] = np.asarray([f"{method}:{clip}" for clip in clips])
    return data


def _plot(metrics: dict, destination: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.7), constrained_layout=True)
    names = ["EE position", "Arm position", "Package position"]
    learned = [metrics["groups"][key]["world_model_rmse"]
               for key in ("ee_position", "arm_position", "package_position")]
    persistence = [metrics["groups"][key]["persistence_rmse"]
                   for key in ("ee_position", "arm_position", "package_position")]
    x = np.arange(len(names))
    width = 0.36
    axes[0].bar(x - width / 2, learned, width, label="World model", color="#3568a8")
    axes[0].bar(x + width / 2, persistence, width, label="Persistence", color="#b5bec8")
    axes[0].set_xticks(x, names)
    axes[0].set_ylabel("Position RMSE (m)")
    axes[0].set_title("0.5 s free rollout")
    axes[0].legend(frameon=False)
    axes[1].bar(["Accuracy", "F1"],
                [metrics["rollout_contact_accuracy"], metrics["rollout_contact_f1"]],
                color="#5f9e79")
    axes[1].set_ylim(0, 1)
    axes[1].set_title("Contact prediction at rollout end")
    for ax in axes:
        ax.grid(axis="y", alpha=0.25)
        ax.set_axisbelow(True)
    fig.suptitle("Held-out clips: physical groups and contact classification")
    fig.savefig(destination, dpi=180)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--direct-transitions", type=Path, required=True)
    parser.add_argument("--confidence-transitions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir == ROOT or ROOT in output_dir.parents:
        parser.error("--output-dir must be outside the repository")
    output_dir.mkdir(parents=True, exist_ok=True)

    direct = _load(args.direct_transitions.expanduser(), "direct")
    confidence = _load(args.confidence_transitions.expanduser(), "confidence-aware")
    states = np.concatenate((direct["states"], confidence["states"]))
    actions = np.concatenate((direct["actions"], confidence["actions"]))
    next_states = np.concatenate((direct["next_states"], confidence["next_states"]))
    clips = np.concatenate((direct["clip_ids"], confidence["clip_ids"]))
    episodes = np.concatenate((direct["episode_ids"], confidence["episode_ids"]))
    fit = fit_state_dynamics(states, actions, next_states, clips, episode_ids=episodes,
                             epochs=args.epochs, seed=args.seed)
    save_checkpoint(fit, str(output_dir / "state_dynamics.pt"))
    (output_dir / "training_metrics.json").write_text(
        json.dumps(fit.metrics, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    public_metrics = {key: value for key, value in fit.metrics.items() if key != "per_clip"}
    public_metrics.update({
        "dataset": "MuJoCo transitions driven by 16 recorded human demonstrations",
        "training_clips": fit.metrics["train_clip_count"],
        "held_out_clips": fit.metrics["validation_clip_count"],
        "normalization": "state, action, and state-delta scales fit on training clips only; binary contact flags are left unscaled",
        "baseline": "persistence: predict state(t+1) = state(t)",
        "limitations": "Simulation-derived transitions; proxy for task dynamics, not physical-robot validation.",
    })
    (output_dir / "aggregate_metrics.json").write_text(
        json.dumps(public_metrics, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    _plot(public_metrics, output_dir / "world_model_comparison.png")
    print(json.dumps(public_metrics, indent=2))
    print(f"Local checkpoint and detailed metrics: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
