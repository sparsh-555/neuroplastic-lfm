import torch
import pytest
from src.cluster import CfCCluster
from src.registry import ClusterRegistry
from src.merge import _trim, _elect_and_merge, ties_merge


def test_trim_zeros_small_magnitudes():
    τ = torch.tensor([0.1, 0.5, 0.01, 0.8, 0.02])
    result = _trim(τ, trim_p=0.4)
    assert result[2].item() == pytest.approx(0.0, abs=1e-6)
    assert result[4].item() == pytest.approx(0.0, abs=1e-6)
    assert result[1].item() == pytest.approx(0.5, abs=1e-6)
    assert result[3].item() == pytest.approx(0.8, abs=1e-6)


def test_trim_zero_threshold_keeps_all():
    τ = torch.tensor([0.1, 0.5, 0.01])
    assert torch.allclose(_trim(τ, trim_p=0.0), τ)


def test_trim_preserves_sign():
    τ = torch.tensor([-0.9, 0.1, -0.05])
    result = _trim(τ, trim_p=0.4)
    assert result[0].item() == pytest.approx(-0.9, abs=1e-6)
    assert result[1].item() == pytest.approx(0.1, abs=1e-6)


def test_elect_agreeing_signs_are_averaged():
    τ_a = torch.tensor([0.3, 0.5])
    τ_b = torch.tensor([0.1, 0.2])
    result = _elect_and_merge([τ_a, τ_b])
    assert result[0].item() == pytest.approx(0.2, abs=1e-5)
    assert result[1].item() == pytest.approx(0.35, abs=1e-5)


def test_elect_positive_wins_when_sum_positive():
    τ_a = torch.tensor([0.3])
    τ_b = torch.tensor([-0.1])
    result = _elect_and_merge([τ_a, τ_b])
    assert result[0].item() == pytest.approx(0.3, abs=1e-5)


def test_elect_negative_wins_when_sum_negative():
    τ_a = torch.tensor([0.1])
    τ_b = torch.tensor([-0.4])
    result = _elect_and_merge([τ_a, τ_b])
    assert result[0].item() == pytest.approx(-0.4, abs=1e-5)


def test_elect_both_zero_returns_zero():
    τ_a = torch.tensor([0.0])
    τ_b = torch.tensor([0.0])
    result = _elect_and_merge([τ_a, τ_b])
    assert result[0].item() == pytest.approx(0.0, abs=1e-6)


def test_ties_merge_produces_cfc_cluster():
    registry = ClusterRegistry()
    registry.spawn("task_a")
    registry.spawn("task_b")
    merged = ties_merge(registry, "task_a", "task_b", "merged")
    assert isinstance(merged, CfCCluster)


def test_ties_merge_registers_under_new_id():
    registry = ClusterRegistry()
    registry.spawn("task_a")
    registry.spawn("task_b")
    ties_merge(registry, "task_a", "task_b", "merged")
    assert "merged" in registry.clusters


def test_ties_merge_frozen_result():
    registry = ClusterRegistry()
    registry.spawn("task_a")
    registry.spawn("task_b")
    merged = ties_merge(registry, "task_a", "task_b", "merged")
    assert all(not p.requires_grad for p in merged.parameters())


def test_ties_merge_untrained_clusters_returns_near_base_init():
    registry = ClusterRegistry()
    registry.spawn("task_a")
    registry.spawn("task_b")
    merged = ties_merge(registry, "task_a", "task_b", "merged")
    for k, v in registry.base_init.items():
        diff = (merged.state_dict()[k].float() - v.float()).abs().max().item()
        assert diff < 1e-4, f"Key '{k}': merged deviates from base_init by {diff}"
