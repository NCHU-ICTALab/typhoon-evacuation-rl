"""Reference policies and common episode runner for the RL environment."""

from __future__ import annotations

import numpy as np


def select_action(env, policy: str) -> int:
    mask = env.action_masks()
    feasible = np.flatnonzero(mask[:-1]).tolist()
    if not feasible:
        return env.wait_action

    if policy == "fcfs":
        return min(
            feasible, key=lambda i: (float(env.vessels[i]["ready_hour"]), i)
        )
    if policy == "risk_aware":
        return min(
            feasible,
            key=lambda i: (
                env.closure_hour
                - env.now
                - float(env.vessels[i]["transit_hour"]),
                -int(env.vessels[i]["risk_points"]),
                -float(env.vessels[i]["gross_tonnage"]),
                i,
            ),
        )
    if policy == "value_density":
        return max(
            feasible,
            key=lambda i: (
                env.dispatch_value(env.vessels[i])
                / float(env.vessels[i]["transit_hour"]),
                -float(env.vessels[i]["ready_hour"]),
            ),
        )
    raise ValueError(f"Unknown policy: {policy}")


def run_baseline(env, policy: str, *, seed: int, options: dict) -> dict:
    env.reset(seed=seed, options=options)
    terminated = truncated = False
    episode_return = 0.0
    last_info = {}
    while not (terminated or truncated):
        action = select_action(env, policy)
        _, reward, terminated, truncated, last_info = env.step(action)
        episode_return += reward
    return {
        "policy": policy,
        "episode_return": round(episode_return, 6),
        **last_info["kpi"],
    }
