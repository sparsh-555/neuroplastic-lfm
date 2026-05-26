import torch
import torch.nn as nn
from typing import Dict, Optional
from src.cluster import CfCCluster


class ClusterRegistry(nn.Module):
    # All clusters start from the same init so TIES task vectors are comparable.
    CLUSTER_SEED = 0

    def __init__(self):
        super().__init__()
        self.clusters: nn.ModuleDict = nn.ModuleDict()
        self.base_init: Optional[Dict[str, torch.Tensor]] = None

    def spawn(self, task_id: str) -> CfCCluster:
        # Freeze all existing clusters
        for cluster in self.clusters.values():
            for param in cluster.parameters():
                param.requires_grad = False

        # Reset RNG to ensure deterministic initialization
        torch.manual_seed(self.CLUSTER_SEED)
        cluster = CfCCluster(seed=self.CLUSTER_SEED)

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

    def get(self, task_id: str) -> CfCCluster:
        if task_id not in self.clusters:
            raise KeyError(
                f"No cluster for task '{task_id}'. "
                f"Available: {list(self.clusters.keys())}"
            )
        return self.clusters[task_id]
