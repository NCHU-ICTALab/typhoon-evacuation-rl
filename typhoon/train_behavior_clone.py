"""Phase 12: clone the near-optimal value-density teacher into a soft-MoE.

Phase 11 proved value-density is within ~1-3% of the strict optimum and is
preference-monotone (0/60 endpoint inversions) while still differentiating by
preference (51/60 distinct schedules). This module behavior-clones value-density
into the four soft-MoE experts -- each expert imitates value-density at its own
centroid preference -- then distills the router. The result is a single servable
checkpoint that matches value-density, keeps continuous-preference control and a
Pareto set, and applies the same hard safety mask.

It never overwrites existing checkpoints: experts and the assembled soft-MoE are
written to fresh directories.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .baselines import select_action
from .build_scenario import DEFAULT_DB
from .decomposed_ppo import DecomposedActorCritic, ROUTING_PREFERENCES
from .env import TyphoonEvacuationEnv
from .soft_moe import SoftMoEActorCritic
from .train_rl import split_scenarios
from .train_soft_moe import (
    centroid_routing,
    continuous_routing_samples,
    pretrain_preference_router,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_EXPERT_DIR = ROOT / "models" / "vd_bc_experts"
DEFAULT_OUTPUT_DIR = ROOT / "models" / "soft_moe_vd_bc"
_PROFILE_ORDER = ("count", "balanced", "gt", "risk")


def collect_teacher_dataset(
    scenarios: list[dict],
    preference: tuple[float, float, float],
    *,
    episodes: int,
    seed: int,
) -> dict[str, np.ndarray]:
    """Roll out value-density at a fixed preference and record its decisions."""
    env = TyphoonEvacuationEnv(scenarios, randomize=True)
    rng = np.random.default_rng(seed)
    observations, masks, actions = [], [], []
    for episode in range(episodes):
        options = {
            "scenario_index": int(rng.integers(0, len(scenarios))),
            "preference": preference,
            "closure_hour": float(rng.uniform(*env.closure_range)),
            "tug_capacity": int(
                rng.integers(env.tug_range[0], env.tug_range[1] + 1)
            ),
            "demand_compression": float(rng.uniform(*env.compression_range)),
            "jitter": True,
        }
        obs, _ = env.reset(seed=seed + episode, options=options)
        done = False
        while not done:
            mask = env.action_masks()
            action = select_action(env, "value_density")
            observations.append(obs.astype(np.float32))
            masks.append(mask.astype(bool))
            actions.append(int(action))
            obs, _, done, _, _ = env.step(action)
    return {
        "observations": np.asarray(observations, dtype=np.float32),
        "action_masks": np.asarray(masks, dtype=bool),
        "actions": np.asarray(actions, dtype=np.int64),
    }


def train_expert(
    observation_space,
    n_actions: int,
    dataset: dict[str, np.ndarray],
    *,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    device: str = "cpu",
) -> tuple[DecomposedActorCritic, dict[str, float]]:
    """Supervised masked cross-entropy imitation of the teacher's actions."""
    torch.manual_seed(seed)
    expert = DecomposedActorCritic(
        observation_space, n_actions, actor_routing="shared", device=device
    )
    optimizer = torch.optim.Adam(expert.policy_parameters(), lr=learning_rate)
    obs = torch.as_tensor(dataset["observations"], device=device)
    masks = torch.as_tensor(dataset["action_masks"], device=device)
    actions = torch.as_tensor(dataset["actions"], device=device)
    count = len(actions)
    rng = np.random.default_rng(seed)
    loss_fn = torch.nn.CrossEntropyLoss()
    epoch_accuracy = []
    last_loss = 0.0
    expert.train()
    for _ in range(epochs):
        order = rng.permutation(count)
        correct = 0
        for start in range(0, count, batch_size):
            batch = order[start : start + batch_size]
            features = expert._features(obs[batch])
            logits = expert._policy_logits(features, obs[batch])
            logits = logits.masked_fill(~masks[batch], -1e9)
            loss = loss_fn(logits, actions[batch])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            last_loss = float(loss.detach().cpu())
            with torch.no_grad():
                correct += int((logits.argmax(-1) == actions[batch]).sum())
        epoch_accuracy.append(correct / count)
    expert.eval()
    return expert, {
        "final_loss": last_loss,
        "final_accuracy": epoch_accuracy[-1],
        "samples": int(count),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--expert-dir", type=Path, default=DEFAULT_EXPERT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episodes", type=int, default=600)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--router-steps", type=int, default=4000)
    parser.add_argument("--router-learning-rate", type=float, default=1e-3)
    parser.add_argument("--router-temperature", type=float, default=0.02)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_dates, test_dates, train_scenarios, _ = split_scenarios(args.db)
    probe = TyphoonEvacuationEnv(train_scenarios, randomize=False)
    observation_space = probe.observation_space
    n_actions = int(probe.action_space.n)

    experts = []
    expert_metrics = {}
    for index, profile in enumerate(_PROFILE_ORDER):
        preference = ROUTING_PREFERENCES[index]
        dataset = collect_teacher_dataset(
            train_scenarios, preference, episodes=args.episodes, seed=args.seed + index
        )
        expert, metrics = train_expert(
            observation_space,
            n_actions,
            dataset,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            seed=args.seed + index,
            device=args.device,
        )
        expert_dir = args.expert_dir / profile / f"seed-{args.seed}"
        expert.save(
            expert_dir / "final.pt",
            metadata={
                "profile": profile,
                "preference": list(preference),
                "teacher": "value_density",
                "behavior_cloning": metrics,
            },
        )
        experts.append(expert)
        expert_metrics[profile] = metrics
        print(
            f"expert {profile}: acc={metrics['final_accuracy']:.4f} "
            f"loss={metrics['final_loss']:.4f} samples={metrics['samples']}",
            flush=True,
        )

    model = SoftMoEActorCritic(
        observation_space, n_actions, experts=experts, device=args.device
    )
    router_metrics = pretrain_preference_router(
        model,
        steps=args.router_steps,
        batch_size=256,
        learning_rate=args.router_learning_rate,
        temperature=args.router_temperature,
        seed=args.seed,
    )
    output_dir = args.output_dir / f"seed-{args.seed}"
    manifest = {
        "phase": "12_value_density_behavior_clone",
        "seed": args.seed,
        "teacher": "value_density",
        "profiles": list(_PROFILE_ORDER),
        "train_dates": train_dates,
        "test_dates": test_dates,
        "expert_behavior_cloning": expert_metrics,
        "router_distillation": router_metrics,
        "centroid_routing": centroid_routing(model),
        "continuous_routing_samples": continuous_routing_samples(model),
        "hyperparameters": {
            "episodes": args.episodes,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "router_steps": args.router_steps,
            "router_temperature": args.router_temperature,
        },
    }
    model_path = model.save(output_dir / "final.pt", metadata=manifest)
    (output_dir / "training_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "router": router_metrics,
                "centroid_routing": manifest["centroid_routing"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    print(f"model: {model_path}")


if __name__ == "__main__":
    main()
