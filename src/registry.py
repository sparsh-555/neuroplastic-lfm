import torch
import torch.nn as nn
from typing import Dict, Optional, Type
from src.cluster import CfCCluster


class ClusterRegistry(nn.Module):
    def __init__(self, seed: int = 0, cluster_cls: Type[nn.Module] = CfCCluster):
        super().__init__()
        self.clusters: nn.ModuleDict = nn.ModuleDict()
        self.base_init: Optional[Dict[str, torch.Tensor]] = None
        # All clusters within a run share one seed so TIES task vectors are comparable
        # (task vector = trained_params - base_init requires identical starting point).
        # Different runs use different seeds to measure initialization variance.
        self._cluster_seed = seed
        self._cluster_cls = cluster_cls

    def spawn(self, task_id: str) -> nn.Module:
        # Freeze all existing clusters
        for cluster in self.clusters.values():
            for param in cluster.parameters():
                param.requires_grad = False

        # Reset RNG to the shared per-run seed so all clusters start identically
        torch.manual_seed(self._cluster_seed)
        cluster = self._cluster_cls(seed=self._cluster_seed)

        # Store base_init on first spawn (before any training)
        if self.base_init is None:
            self.base_init = {
                k: v.clone().detach() for k, v in cluster.state_dict().items()
            }

        # Ensure new cluster is trainable
        for param in cluster.parameters():
            param.requires_grad = True

        self.clusters[task_id] = cluster
        return cluster

    def get(self, task_id: str) -> nn.Module:
        if task_id not in self.clusters:
            raise KeyError(
                f"No cluster for task '{task_id}'. "
                f"Available: {list(self.clusters.keys())}"
            )
        return self.clusters[task_id]
