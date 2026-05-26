import torch
import torch.nn as nn
from typing import Optional
from src.registry import ClusterRegistry


class NeuroplasticLFM(nn.Module):
    def __init__(self, base_model: nn.Module):
        super().__init__()
        self.base     = base_model
        self.lm_head  = base_model.lm_head
        self.registry = ClusterRegistry()
        self._freeze_base()

    def _freeze_base(self) -> None:
        for param in self.base.parameters():
            param.requires_grad = False

    def spawn_cluster(self, task_id: str):
        return self.registry.spawn(task_id)

    def forward(
        self,
        input_ids: torch.Tensor,
        task_id: Optional[str] = None,
        **kwargs,
    ) -> torch.Tensor:
        out = self.base(input_ids, output_hidden_states=True, **kwargs)
        h   = out.hidden_states[-1]  # (B, L, hidden_size)

        if task_id is not None:
            h = h + self.registry.get(task_id)(h)

        return self.lm_head(h)  # (B, L, vocab_size)
