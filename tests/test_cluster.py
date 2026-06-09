import torch
from src.cluster import CfCCluster, MLPCluster


def test_forward_output_shape():
    cluster = CfCCluster(seed=0)
    h = torch.randn(2, 10, 2048)
    out = cluster(h)
    assert out.shape == (2, 10, 2048)


def test_maturity_gate_half_at_init():
    # Gate logit is 0.0 → sigmoid(0) = 0.5; adapter_out is zero-init so
    # output is exactly 0 regardless.  This init gives max gradient (σ'(0)=0.25).
    cluster = CfCCluster(seed=0)
    gate = torch.sigmoid(cluster.maturity).item()
    assert abs(gate - 0.5) < 1e-4


def test_output_exactly_zero_at_init():
    # adapter_out is zero-initialised (LoRA-style): no random noise at step 0.
    cluster = CfCCluster(seed=0)
    h = torch.randn(2, 5, 2048)
    out = cluster(h)
    assert out.abs().max().item() == 0.0


def test_all_parameters_trainable_at_creation():
    cluster = CfCCluster(seed=0)
    trainable_params = [p for p in cluster.parameters() if p.requires_grad]
    assert len(trainable_params) > 0, "Should have trainable parameters"
    assert cluster.adapter_in.weight.requires_grad
    assert cluster.adapter_out.weight.requires_grad
    assert cluster.maturity.requires_grad


def test_parameter_count_in_expected_range():
    cluster = CfCCluster(seed=0)
    n = sum(p.numel() for p in cluster.parameters())
    assert 100_000 < n < 500_000, f"Expected 100K–500K params, got {n}"


def test_gradient_flows_to_all_components():
    cluster = CfCCluster(seed=0)
    h = torch.randn(2, 5, 2048, requires_grad=False)
    out = cluster(h)
    out.sum().backward()
    assert cluster.adapter_in.weight.grad is not None, "adapter_in gradient missing"
    assert cluster.adapter_out.weight.grad is not None, "adapter_out gradient missing"
    assert cluster.maturity.grad is not None, "maturity gate gradient missing"


def test_batch_size_one():
    cluster = CfCCluster(seed=0)
    h = torch.randn(1, 7, 2048)
    out = cluster(h)
    assert out.shape == (1, 7, 2048)


def test_different_seeds_produce_different_internal_features():
    # Output is always zero at init (zero-init adapter_out), but internal features differ.
    cluster_a = CfCCluster(seed=0)
    cluster_b = CfCCluster(seed=99)
    h = torch.randn(1, 5, 2048)
    feat_a = cluster_a.adapter_in(h.float())
    feat_b = cluster_b.adapter_in(h.float())
    assert not torch.allclose(feat_a, feat_b)


# ── MLPCluster ablation baseline ───────────────────────────────────────────────


def test_mlp_forward_output_shape():
    cluster = MLPCluster(seed=0)
    h = torch.randn(2, 10, 2048)
    out = cluster(h)
    assert out.shape == (2, 10, 2048)


def test_mlp_output_exactly_zero_at_init():
    cluster = MLPCluster(seed=0)
    h = torch.randn(2, 5, 2048)
    out = cluster(h)
    assert out.abs().max().item() == 0.0


def test_mlp_maturity_gate_half_at_init():
    cluster = MLPCluster(seed=0)
    gate = torch.sigmoid(cluster.maturity).item()
    assert abs(gate - 0.5) < 1e-4


def test_mlp_gradient_flows():
    cluster = MLPCluster(seed=0)
    h = torch.randn(2, 5, 2048)
    out = cluster(h)
    out.sum().backward()
    assert cluster.adapter_in.weight.grad is not None
    assert cluster.adapter_out.weight.grad is not None
    assert cluster.maturity.grad is not None


def test_mlp_parameter_count_comparable_to_cfc():
    cfc = CfCCluster(seed=0)
    mlp = MLPCluster(seed=0)
    n_cfc = sum(p.numel() for p in cfc.parameters())
    n_mlp = sum(p.numel() for p in mlp.parameters())
    # MLP is a leaner ablation; both should be in the same order of magnitude (~100-500K)
    assert 100_000 < n_mlp < 500_000
    assert 100_000 < n_cfc < 500_000
