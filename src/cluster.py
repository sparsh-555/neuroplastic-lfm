import torch
import torch.nn as nn
from ncps.torch import CfC
from ncps.wirings import AutoNCP


class CfCCluster(nn.Module):
    BASE_DIM    = 2048
    CLUSTER_DIM = 64
    MOTOR_DIM   = 16

    def __init__(self, seed: int = 0):
        super().__init__()
        wiring           = AutoNCP(units=64, output_size=16, sparsity_level=0.5, seed=seed)
        self.adapter_in  = nn.Linear(self.BASE_DIM, self.CLUSTER_DIM)
        # pos_proj maps normalised position [0,1] → CLUSTER_DIM and adds it to the
        # CfC input before each forward pass.  Because time_a and time_b inside each
        # CfCCell receive x = content_features + pos_embedding, the effective time
        # constant τ = sigmoid(time_a(x)*1 + time_b(x)) becomes jointly conditioned
        # on token content AND sequence position — the functional neuroplasticity signal.
        # ncps' timespans API cannot be used here: it calls .squeeze() on the per-step
        # slice which collapses (B,1)→(B,) and then fails to broadcast with (B,n_neurons).
        self.pos_proj    = nn.Linear(1, self.CLUSTER_DIM, bias=False)
        self.cfc         = CfC(self.CLUSTER_DIM, wiring, batch_first=True, return_sequences=True)
        self.adapter_out = nn.Linear(self.MOTOR_DIM, self.BASE_DIM)
        # Zero-init adapter_out (LoRA-style): cluster contributes exactly 0 at step 0,
        # preventing random noise from disrupting the base model before the CfC has
        # learned anything useful.  Gradient still flows because cfc_out is non-zero.
        nn.init.zeros_(self.adapter_out.weight)
        nn.init.zeros_(self.adapter_out.bias)
        # Gate initialized to sigmoid(0) = 0.5.  At step 0 this is irrelevant because
        # adapter_out=0 forces delta=0.  Once adapter_out learns, sigmoid'(0)=0.25
        # gives ~5× better gradient flow than the previous sigmoid(-3) init.
        self.maturity    = nn.Parameter(torch.zeros(1))

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        # hidden: (B, L, BASE_DIM) — may be bfloat16 from base model
        h = hidden.float()
        x = self.adapter_in(h)                                       # (B, L, CLUSTER_DIM)
        B, L, _ = x.shape
        pos = torch.linspace(0.0, 1.0, L, device=x.device, dtype=x.dtype).view(1, L, 1)
        x = x + self.pos_proj(pos)                                   # (B, L, CLUSTER_DIM)
        h0 = torch.zeros(B, self.CLUSTER_DIM, device=x.device, dtype=x.dtype)
        out, _ = self.cfc(x, h0)                                     # (B, L, MOTOR_DIM)
        delta = self.adapter_out(out)                                 # (B, L, BASE_DIM)
        return (torch.sigmoid(self.maturity) * delta).to(hidden.dtype)
