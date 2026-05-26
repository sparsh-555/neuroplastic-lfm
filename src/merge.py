import torch
import torch.nn as nn
from typing import List
from src.cluster import CfCCluster
from src.registry import ClusterRegistry


def _trim(τ: torch.Tensor, trim_p: float) -> torch.Tensor:
    """Zero out entries of τ below the trim_p-th percentile of absolute magnitude."""
    if trim_p == 0.0:
        return τ.clone()
    threshold = torch.quantile(τ.abs().float().flatten(), trim_p)
    return torch.where(τ.abs() >= threshold, τ, torch.zeros_like(τ))


def _elect_and_merge(task_vectors: List[torch.Tensor]) -> torch.Tensor:
    """
    TIES elect step: for each parameter position, elect the majority sign via
    sum-based tie-breaking, then average only the non-conflicting values.
    """
    stacked = torch.stack([τ.float() for τ in task_vectors], dim=0)  # (K, ...)
    total   = stacked.sum(dim=0)

    elected_sign                    = torch.sign(total)
    elected_sign[elected_sign == 0] = 1.0  # tie → positive

    signs  = torch.sign(stacked)
    keep   = (signs == elected_sign.unsqueeze(0)) | (stacked == 0)
    masked = stacked * keep.float()

    count  = (masked != 0).float().sum(dim=0).clamp(min=1.0)
    return masked.sum(dim=0) / count


def ties_merge(
    registry: ClusterRegistry,
    task_a: str,
    task_b: str,
    new_task_id: str,
    trim_p: float = 0.2,
) -> CfCCluster:
    """Merge two trained clusters into a new cluster using TIES-Merging."""
    base_init = registry.base_init
    state_a   = registry.get(task_a).state_dict()
    state_b   = registry.get(task_b).state_dict()

    # Identify learnable parameters (vs. fixed buffers like NCP sparsity masks)
    param_keys = {name for name, _ in CfCCluster(seed=0).named_parameters()}

    merged_state = {}
    for key in base_init:
        θ_init = base_init[key].float()
        if key not in param_keys:
            # Fixed buffer — identical across clusters, copy from base
            merged_state[key] = θ_init.to(base_init[key].dtype)
        else:
            τ_a      = _trim(state_a[key].float() - θ_init, trim_p)
            τ_b      = _trim(state_b[key].float() - θ_init, trim_p)
            merged_τ = _elect_and_merge([τ_a, τ_b])
            merged_state[key] = (θ_init + merged_τ).to(base_init[key].dtype)

    merged_cluster = CfCCluster(seed=0)
    merged_cluster.load_state_dict(merged_state)

    for param in merged_cluster.parameters():
        param.requires_grad = False

    registry.clusters[new_task_id] = merged_cluster
    return merged_cluster
