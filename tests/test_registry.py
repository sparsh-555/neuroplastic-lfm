import torch
import pytest
from src.cluster import CfCCluster
from src.registry import ClusterRegistry


def test_spawn_returns_cfc_cluster():
    registry = ClusterRegistry()
    cluster = registry.spawn("task_a")
    assert isinstance(cluster, CfCCluster)


def test_spawn_stores_base_init_on_first_spawn():
    registry = ClusterRegistry()
    registry.spawn("task_a")
    assert registry.base_init is not None
    assert len(registry.base_init) > 0


def test_base_init_matches_fresh_cluster():
    registry = ClusterRegistry()
    registry.spawn("task_a")
    torch.manual_seed(ClusterRegistry.CLUSTER_SEED)
    fresh = CfCCluster(seed=ClusterRegistry.CLUSTER_SEED)
    for k, v in registry.base_init.items():
        expected = fresh.state_dict()[k].float()
        actual   = v.float()
        assert torch.allclose(actual, expected, atol=1e-5), f"Mismatch in base_init key '{k}'"


def test_first_cluster_trainable():
    registry = ClusterRegistry()
    c_a = registry.spawn("task_a")
    assert all(p.requires_grad for p in c_a.parameters())


def test_spawn_freezes_prior_clusters():
    registry = ClusterRegistry()
    c_a = registry.spawn("task_a")
    registry.spawn("task_b")
    assert all(not p.requires_grad for p in c_a.parameters()), \
        "cluster_a should be frozen after cluster_b is spawned"


def test_second_cluster_trainable_after_spawn():
    registry = ClusterRegistry()
    registry.spawn("task_a")
    c_b = registry.spawn("task_b")
    assert all(p.requires_grad for p in c_b.parameters())


def test_get_returns_correct_cluster():
    registry = ClusterRegistry()
    c_a = registry.spawn("task_a")
    c_b = registry.spawn("task_b")
    assert registry.get("task_a") is c_a
    assert registry.get("task_b") is c_b


def test_get_unknown_task_raises_key_error():
    registry = ClusterRegistry()
    with pytest.raises(KeyError, match="nonexistent"):
        registry.get("nonexistent")


def test_three_sequential_spawns_freeze_all_prior():
    registry = ClusterRegistry()
    c_a = registry.spawn("a")
    c_b = registry.spawn("b")
    c_c = registry.spawn("c")
    assert all(not p.requires_grad for p in c_a.parameters())
    assert all(not p.requires_grad for p in c_b.parameters())
    assert all(p.requires_grad for p in c_c.parameters())
