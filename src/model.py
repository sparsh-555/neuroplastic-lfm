import torch
import torch.nn as nn
from typing import Optional, Type
from src.cluster import CfCCluster
from src.registry import ClusterRegistry

# After layer 8 (3rd attention block at index 2,5,8), 7 layers + embedding_norm remain.
# The cluster correction propagates through 3 more attention blocks before lm_head,
# giving the model's own attention mechanism a chance to route it into place.
INJECT_AT = 8


class NeuroplasticLFM(nn.Module):
    def __init__(
        self,
        base_model: nn.Module,
        inject_at: int = INJECT_AT,
        seed: int = 0,
        cluster_cls: Type[nn.Module] = CfCCluster,
        base_dim: int = 2048,
    ):
        super().__init__()
        self.base      = base_model
        self.lm_head   = base_model.lm_head
        self.registry  = ClusterRegistry(seed=seed, cluster_cls=cluster_cls, base_dim=base_dim)
        self.inject_at = inject_at
        self._freeze_base()

    def _freeze_base(self) -> None:
        for param in self.base.parameters():
            param.requires_grad = False

    def spawn_cluster(self, task_id: str):
        cluster = self.registry.spawn(task_id)
        device = next(self.base.parameters()).device
        cluster.to(device)
        return cluster

    def forward(
        self,
        input_ids: torch.Tensor,
        task_id: Optional[str] = None,
        **kwargs,
    ) -> torch.Tensor:
        handle = None
        if task_id is not None:
            cluster = self.registry.get(task_id)
            def hook(module, input, output):
                # cluster.forward() handles float32↔backbone-dtype casting internally
                return output + cluster(output)
            handle = self.base.model.layers[self.inject_at].register_forward_hook(hook)

        try:
            out = self.base.model(input_ids, **kwargs)
        finally:
            if handle is not None:
                handle.remove()

        return self.lm_head(out.last_hidden_state)
