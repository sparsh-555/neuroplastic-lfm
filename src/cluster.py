import torch
import torch.nn as nn
from ncps.torch import CfC
from ncps.wirings import AutoNCP


class CfCCluster(nn.Module):
    BASE_DIM    = 2560
    CLUSTER_DIM = 64
    MOTOR_DIM   = 16

    def __init__(self, seed: int = 0):
        super().__init__()
        wiring           = AutoNCP(units=64, output_size=16, sparsity_level=0.5, seed=seed)
        self.adapter_in  = nn.Linear(self.BASE_DIM, self.CLUSTER_DIM)
        self.cfc         = CfC(self.CLUSTER_DIM, wiring, batch_first=True, return_sequences=True)
        self.adapter_out = nn.Linear(self.MOTOR_DIM, self.BASE_DIM)
        # Zero-init gating from LLaMA-Adapter (Zhang et al., 2023).
        # sigmoid(-6) ≈ 0.002: cluster has negligible influence at spawn.
        self.maturity    = nn.Parameter(torch.full((1,), -6.0))

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        # hidden: (B, L, BASE_DIM) — may be float16 from base model
        h      = hidden.float()
        x      = self.adapter_in(h)                                    # (B, L, CLUSTER_DIM)
        h0     = torch.zeros(x.size(0), self.CLUSTER_DIM,
                             device=x.device, dtype=x.dtype)
        out, _ = self.cfc(x, h0)                                       # (B, L, MOTOR_DIM)
        delta  = self.adapter_out(out)                                 # (B, L, BASE_DIM)
        return (torch.sigmoid(self.maturity) * delta).to(hidden.dtype)
