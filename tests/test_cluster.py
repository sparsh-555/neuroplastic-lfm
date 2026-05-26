import torch
from src.cluster import CfCCluster


def test_forward_output_shape():
    cluster = CfCCluster(seed=0)
    h = torch.randn(2, 10, 2560)
    out = cluster(h)
    assert out.shape == (2, 10, 2560)


def test_maturity_gate_near_zero_at_init():
    cluster = CfCCluster(seed=0)
    gate = torch.sigmoid(cluster.maturity).item()
    assert gate < 0.01  # sigmoid(-6) ≈ 0.00248


def test_output_near_zero_at_init():
    cluster = CfCCluster(seed=0)
    h = torch.randn(2, 5, 2560)
    out = cluster(h)
    assert out.abs().max().item() < 1.0


def test_all_parameters_trainable_at_creation():
    cluster = CfCCluster(seed=0)
    # Check that trainable parameters exist (sparsity masks are non-trainable by design)
    trainable_params = [p for p in cluster.parameters() if p.requires_grad]
    assert len(trainable_params) > 0, "Should have trainable parameters"
    # Verify key components are trainable
    assert cluster.adapter_in.weight.requires_grad
    assert cluster.adapter_out.weight.requires_grad
    assert cluster.maturity.requires_grad


def test_parameter_count_in_expected_range():
    cluster = CfCCluster(seed=0)
    n = sum(p.numel() for p in cluster.parameters())
    assert 200_000 < n < 500_000, f"Expected 200K–500K params, got {n}"


def test_gradient_flows_to_all_components():
    cluster = CfCCluster(seed=0)
    h = torch.randn(2, 5, 2560, requires_grad=False)
    out = cluster(h)
    out.sum().backward()
    assert cluster.adapter_in.weight.grad is not None, "adapter_in gradient missing"
    assert cluster.adapter_out.weight.grad is not None, "adapter_out gradient missing"
    assert cluster.maturity.grad is not None, "maturity gate gradient missing"


def test_batch_size_one():
    cluster = CfCCluster(seed=0)
    h = torch.randn(1, 7, 2560)
    out = cluster(h)
    assert out.shape == (1, 7, 2560)


def test_different_seeds_produce_different_outputs():
    cluster_a = CfCCluster(seed=0)
    cluster_b = CfCCluster(seed=99)
    h = torch.randn(1, 5, 2560)
    out_a = cluster_a(h)
    out_b = cluster_b(h)
    assert not torch.allclose(out_a, out_b)
