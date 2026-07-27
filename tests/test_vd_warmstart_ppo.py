import copy

import numpy as np
import torch
from gymnasium import spaces

from typhoon.decomposed_ppo import DecomposedActorCritic
from typhoon.soft_moe import SoftMoEActorCritic
from typhoon.train_vd_warmstart_ppo import (
    assemble_from_warm_experts,
    clone_trainable_expert,
    reset_value_head,
    service_gate,
    soften_action_logits,
)


def _space():
    return spaces.Box(low=-2.0, high=5.0, shape=(344,), dtype=np.float32)


def _summary(*, utility=None, differences=10, cases=12, margin=0.002):
    return {
        "utility": utility
        or {"count": 0.40, "balanced": 0.38, "gt": 0.367, "risk": 0.40},
        "mean_specialization_margin": margin,
        "cases_with_preference_action_difference": differences,
        "paired_base_cases": cases,
        "safety_violations": 0,
        "rejected_actions": 0,
    }


def test_clone_trainable_expert_preserves_weights_and_unfreezes():
    source = DecomposedActorCritic(_space(), 31)
    for parameter in source.parameters():
        parameter.requires_grad_(False)
    clone = clone_trainable_expert(source)
    assert all(parameter.requires_grad for parameter in clone.parameters())
    assert all(
        torch.equal(source.state_dict()[key], value)
        for key, value in clone.state_dict().items()
    )


def test_soften_logits_preserves_action_order_and_scales_output():
    expert = DecomposedActorCritic(_space(), 31)
    observations = torch.rand(4, 344)
    with torch.no_grad():
        features = expert._features(observations)
        before = expert._policy_logits(features, observations)
    soften_action_logits(expert, 3.0)
    with torch.no_grad():
        features = expert._features(observations)
        after = expert._policy_logits(features, observations)
    assert torch.equal(before.argmax(dim=-1), after.argmax(dim=-1))
    assert torch.allclose(after, before / 3.0, atol=1e-6)


def test_soften_logits_rejects_temperature_below_one():
    expert = DecomposedActorCritic(_space(), 31)
    with np.testing.assert_raises(ValueError):
        soften_action_logits(expert, 0.5)


def test_reset_value_head_preserves_actor_and_zeros_values():
    expert = DecomposedActorCritic(_space(), 31)
    actor_before = {
        key: value.clone() for key, value in expert.actor.state_dict().items()
    }
    reset_value_head(expert)
    observations = torch.rand(4, 344)
    assert torch.equal(expert.values(observations), torch.zeros(4, 3))
    assert all(
        torch.equal(actor_before[key], value)
        for key, value in expert.actor.state_dict().items()
    )


def test_assembly_preserves_phase12_router_and_installs_warm_experts():
    initial = SoftMoEActorCritic(_space(), 31)
    with torch.no_grad():
        initial.preference_router[-1].bias.copy_(
            torch.tensor([1.0, 2.0, 3.0, 4.0])
        )
    experts = [
        clone_trainable_expert(expert) for expert in initial.experts
    ]
    with torch.no_grad():
        experts[0].actor[-1].bias[7] = 99.0
    assembled = assemble_from_warm_experts(initial, experts)
    assert torch.equal(
        assembled.preference_router[-1].bias,
        initial.preference_router[-1].bias,
    )
    assert float(assembled.experts[0].actor[-1].bias[7]) == 99.0
    assert all(
        not parameter.requires_grad
        for expert in assembled.experts
        for parameter in expert.parameters()
    )


def test_service_gate_compares_phase10_and_does_not_require_diagonal():
    reference = _summary(differences=56, cases=60, margin=0.002287)
    passing = _summary(
        utility={
            "count": 0.403,
            "balanced": 0.390,
            "gt": 0.3651,
            "risk": 0.405,
        },
        differences=55,
        cases=60,
        margin=0.001851,
    )
    assert not service_gate(reference, passing, pilot=True)[
        "accepted_for_service"
    ]
    gate = service_gate(reference, passing, pilot=False)
    assert gate["accepted_for_service"]
    assert "all_four_diagonal" not in gate["conditions"]

    failing = copy.deepcopy(passing)
    failing["utility"]["gt"] = 0.360
    gate = service_gate(reference, failing, pilot=False)
    assert not gate["accepted_for_service"]
    assert not gate["conditions"]["no_material_profile_utility_regression"]
