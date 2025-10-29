import math
import torch
import pytest

from spark_trainer.models.gating import TopKRouter, RouterlessMoE, MoELoRA, MixtureOfDepths, GatingMetrics


def _rand_hidden(batch=2, seq=4, d_model=16, seed=42):
    torch.manual_seed(seed)
    return torch.randn(batch, seq, d_model)


def test_topk_router_shapes_and_metrics():
    hidden = _rand_hidden()
    router = TopKRouter(d_model=hidden.size(-1), num_experts=4, num_selected=2, enable_jitter=False)
    router.eval()

    expert_indices, expert_weights, metrics = router(hidden)

    assert expert_indices.shape == (hidden.size(0), hidden.size(1), 2)
    assert expert_weights.shape == (hidden.size(0), hidden.size(1), 2)

    # Weights should be normalized per token
    sums = expert_weights.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-4)

    # Metrics sanity checks
    assert isinstance(metrics, GatingMetrics)
    assert metrics.expert_utilization.numel() == 4
    assert 0.0 <= metrics.capacity_overflow <= 100.0
    assert metrics.gate_entropy is None or metrics.gate_entropy >= 0.0


def test_routerless_moe_forward_and_metrics():
    hidden = _rand_hidden()
    moe = RouterlessMoE(d_model=hidden.size(-1), d_ff=32, num_experts=4)
    moe.eval()

    out, metrics = moe(hidden)

    assert out.shape == hidden.shape
    assert isinstance(metrics, GatingMetrics)
    assert metrics.capacity_overflow == 0.0


def test_moelora_forward_small_dims():
    hidden = _rand_hidden()
    moe = MoELoRA(d_model=hidden.size(-1), d_ff=32, num_experts=4, num_selected=2, lora_rank=2, lora_alpha=4)
    moe.eval()

    out, metrics = moe(hidden)

    assert out.shape == hidden.shape
    assert isinstance(metrics, GatingMetrics)


def test_mixture_of_depths_continue_mask_and_metrics():
    hidden = _rand_hidden()
    mod = MixtureOfDepths(d_model=hidden.size(-1), threshold=0.5, min_layers=1)

    # Layer 0: below min_layers, must continue all tokens
    mask0, m0 = mod.should_continue(hidden, layer_idx=0)
    assert mask0.shape == hidden.shape[:2]
    assert mask0.all()
    assert 'avg_depth' in m0 and 'exit_rate' in m0

    # After min layer, some tokens should exit (depending on threshold)
    mask1, m1 = mod.should_continue(hidden, layer_idx=1)
    assert mask1.shape == hidden.shape[:2]
    assert 'avg_depth' in m1 and 'exit_rate' in m1
