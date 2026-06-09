import torch
from src.cluster import CfCCluster, MLPCluster

# ── CfCCluster ─────────────────────────────────────────────────────────────────


def test_forward_output_shape():
    cluster = CfCCluster(seed=0)
    h = torch.randn(2, 10, 2048)
    out = cluster(h)
    assert out.shape == (2, 10, 2048)


def test_maturity_gate_half_at_init():
    cluster = CfCCluster(seed=0)
    gate = torch.sigmoid(cluster.maturity).item()
    assert abs(gate - 0.5) < 1e-4


def test_output_near_zero_at_init():
    # adapter_out uses std=1e-3 init: output is small but non-zero.
    # This keeps initial noise negligible (~0.01% of residual stream magnitude)
    # while preserving gradient flow to the CfC and its τ projections.
    cluster = CfCCluster(seed=0)
    h = torch.randn(2, 5, 2048)
    out = cluster(h)
    # gate=0.5, adapter_out std=1e-3 → output should be well under 0.1
    assert out.abs().max().item() < 0.1


def test_gradient_flows_to_cfc_tau():
    # The critical test: time_a weights inside CfCCells must receive non-zero
    # gradients.  With zero-init adapter_out this failed (grad=0 through the chain).
    cluster = CfCCluster(seed=0)
    h = torch.randn(2, 5, 2048)
    out = cluster(h)
    out.sum().backward()
    cell = cluster.cfc.rnn_cell
    for i in range(cell.num_layers):
        layer = getattr(cell, f"layer_{i}")
        assert layer.time_a.weight.grad is not None, f"layer_{i} time_a grad missing"
        assert layer.time_a.weight.grad.norm().item() > 0, f"layer_{i} time_a grad is zero"


def test_all_parameters_trainable_at_creation():
    cluster = CfCCluster(seed=0)
    trainable_params = [p for p in cluster.parameters() if p.requires_grad]
    assert len(trainable_params) > 0
    assert cluster.adapter_in.weight.requires_grad
    assert cluster.adapter_out.weight.requires_grad
    assert cluster.maturity.requires_grad


def test_parameter_count_in_expected_range():
    cluster = CfCCluster(seed=0)
    n = sum(p.numel() for p in cluster.parameters())
    assert 100_000 < n < 500_000, f"Expected 100K–500K params, got {n}"


def test_gradient_flows_to_adapter_in():
    cluster = CfCCluster(seed=0)
    h = torch.randn(2, 5, 2048)
    out = cluster(h)
    out.sum().backward()
    assert cluster.adapter_in.weight.grad is not None
    assert cluster.adapter_out.weight.grad is not None
    assert cluster.maturity.grad is not None


def test_batch_size_one():
    cluster = CfCCluster(seed=0)
    h = torch.randn(1, 7, 2048)
    out = cluster(h)
    assert out.shape == (1, 7, 2048)


def test_different_seeds_produce_different_outputs():
    cluster_a = CfCCluster(seed=0)
    cluster_b = CfCCluster(seed=99)
    h = torch.randn(1, 5, 2048)
    out_a = cluster_a(h)
    out_b = cluster_b(h)
    assert not torch.allclose(out_a, out_b)


def test_cfc_respects_base_dim_4096():
    cluster = CfCCluster(seed=0, base_dim=4096)
    h = torch.randn(1, 5, 4096)
    out = cluster(h)
    assert out.shape == (1, 5, 4096)


# ── MLPCluster ─────────────────────────────────────────────────────────────────


def test_mlp_forward_output_shape():
    cluster = MLPCluster(seed=0)
    h = torch.randn(2, 10, 2048)
    out = cluster(h)
    assert out.shape == (2, 10, 2048)


def test_mlp_output_near_zero_at_init():
    cluster = MLPCluster(seed=0)
    h = torch.randn(2, 5, 2048)
    out = cluster(h)
    assert out.abs().max().item() < 0.1


def test_mlp_maturity_gate_half_at_init():
    cluster = MLPCluster(seed=0)
    gate = torch.sigmoid(cluster.maturity).item()
    assert abs(gate - 0.5) < 1e-4


def test_mlp_gradient_flows_to_all_layers():
    cluster = MLPCluster(seed=0)
    h = torch.randn(2, 5, 2048)
    out = cluster(h)
    out.sum().backward()
    assert cluster.adapter_in.weight.grad is not None
    assert cluster.ff1.weight.grad is not None
    assert cluster.ff2.weight.grad is not None
    assert cluster.adapter_out.weight.grad is not None
    assert cluster.maturity.grad is not None
    # All gradients must be non-zero (not blocked)
    assert cluster.adapter_in.weight.grad.norm().item() > 0
    assert cluster.ff1.weight.grad.norm().item() > 0


def test_mlp_parameter_count_comparable_to_cfc():
    cfc = CfCCluster(seed=0)
    mlp = MLPCluster(seed=0)
    n_cfc = sum(p.numel() for p in cfc.parameters())
    n_mlp = sum(p.numel() for p in mlp.parameters())
    assert 100_000 < n_mlp < 500_000
    assert 100_000 < n_cfc < 500_000


def test_mlp_respects_base_dim_4096():
    cluster = MLPCluster(seed=0, base_dim=4096)
    h = torch.randn(1, 5, 4096)
    out = cluster(h)
    assert out.shape == (1, 5, 4096)
