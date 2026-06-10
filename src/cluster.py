import torch
import torch.nn as nn
from ncps.torch import CfC
from ncps.wirings import AutoNCP


class CfCCluster(nn.Module):
    """
    Recurrent adapter using Closed-form Continuous-time (CfC) cells wired by
    AutoNCP.  The ODE-derived τ dynamics condition the effective time constant
    on token content and sequence position jointly.

    base_dim must match the hidden dimension of the base model (2048 for LFM2.5-1.2B,
    4096 for LLaMA-3-8B, etc.).

    Initialisation strategy (critical):
      adapter_out uses small-random init (std=1e-3) rather than zero-init.
      Zero-init would block gradients: adapter_out.weight.T @ grad = 0, so the CfC
      and its τ projections receive no gradient signal.  std=1e-3 keeps initial output
      noise at ~0.01% of residual stream magnitude (negligible) while preserving the
      full gradient path to time_a / time_b inside each CfCCell.
    """
    CLUSTER_DIM = 64
    MOTOR_DIM   = 16

    def __init__(self, seed: int = 0, base_dim: int = 2048):
        super().__init__()
        self.base_dim    = base_dim
        wiring           = AutoNCP(units=64, output_size=16, sparsity_level=0.5, seed=seed)
        self.adapter_in  = nn.Linear(self.base_dim, self.CLUSTER_DIM)
        # pos_proj maps normalised position [0,1] → CLUSTER_DIM and adds it to the
        # CfC input before each forward pass.  time_a and time_b inside each CfCCell
        # receive x = content_features + pos_embedding, so τ is conditioned jointly
        # on token content AND sequence position.
        # ncps' timespans API cannot be used here: it calls .squeeze() on the per-step
        # slice which collapses (B,1)→(B,) and fails to broadcast with (B,n_neurons).
        self.pos_proj    = nn.Linear(1, self.CLUSTER_DIM, bias=False)
        self.cfc         = CfC(self.CLUSTER_DIM, wiring, batch_first=True, return_sequences=True)
        self.adapter_out = nn.Linear(self.MOTOR_DIM, self.base_dim)
        nn.init.normal_(self.adapter_out.weight, std=1e-3)
        nn.init.zeros_(self.adapter_out.bias)
        # Input-conditioned gate: per-token routing based on the hidden state.
        # Weight zero-init + large negative bias → gate starts nearly closed
        # (sigmoid(-4) ≈ 0.018) so the cluster earns its contribution gradually.
        # Unlike a scalar maturity gate, different tokens get different gate values,
        # so gradient signal continues even after average loss ≈ 0: the gate learns
        # which token positions the cluster can improve vs. leave to the base model.
        self.gate_proj   = nn.Linear(self.base_dim, 1, bias=True)
        nn.init.zeros_(self.gate_proj.weight)
        nn.init.constant_(self.gate_proj.bias, -4.0)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        # hidden: (B, L, base_dim) — may be bfloat16 from base model
        h    = hidden.float()
        gate = torch.sigmoid(self.gate_proj(h))                      # (B, L, 1) per-token
        x    = self.adapter_in(h)                                    # (B, L, CLUSTER_DIM)
        B, L, _ = x.shape
        pos  = torch.linspace(0.0, 1.0, L, device=x.device, dtype=x.dtype).view(1, L, 1)
        x    = x + self.pos_proj(pos)                                # (B, L, CLUSTER_DIM)
        h0   = torch.zeros(B, self.CLUSTER_DIM, device=x.device, dtype=x.dtype)
        out, _ = self.cfc(x, h0)                                     # (B, L, MOTOR_DIM)
        delta  = self.adapter_out(out)                               # (B, L, base_dim)
        return (gate * delta).to(hidden.dtype)


class MLPCluster(nn.Module):
    """
    Ablation baseline for CfCCluster.  Replaces the recurrent CfC block with a
    2-layer feed-forward network of identical input/output dimensions.  Both
    classes share: adapter_in, pos_proj, adapter_out, and input-conditioned gate — so the
    only variable is the processing block (CfC ODE dynamics vs. static MLP).

    Uses the same small-random adapter_out init as CfCCluster so both start with
    near-zero output and identical gradient flow conditions.

    Approximate param count (base_dim=2048): ~173K vs CfCCluster ~189K.
    """
    CLUSTER_DIM = 64
    MOTOR_DIM   = 16

    def __init__(self, seed: int = 0, base_dim: int = 2048):
        super().__init__()
        self.base_dim    = base_dim
        self.adapter_in  = nn.Linear(self.base_dim, self.CLUSTER_DIM)
        self.pos_proj    = nn.Linear(1, self.CLUSTER_DIM, bias=False)
        self.ff1         = nn.Linear(self.CLUSTER_DIM, self.CLUSTER_DIM)
        self.ff2         = nn.Linear(self.CLUSTER_DIM, self.MOTOR_DIM)
        self.adapter_out = nn.Linear(self.MOTOR_DIM, self.base_dim)
        nn.init.normal_(self.adapter_out.weight, std=1e-3)
        nn.init.zeros_(self.adapter_out.bias)
        self.gate_proj   = nn.Linear(self.base_dim, 1, bias=True)
        nn.init.zeros_(self.gate_proj.weight)
        nn.init.constant_(self.gate_proj.bias, -4.0)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        h    = hidden.float()
        gate = torch.sigmoid(self.gate_proj(h))                      # (B, L, 1) per-token
        x    = self.adapter_in(h)                                    # (B, L, CLUSTER_DIM)
        B, L, _ = x.shape
        pos  = torch.linspace(0.0, 1.0, L, device=x.device, dtype=x.dtype).view(1, L, 1)
        x    = x + self.pos_proj(pos)
        x    = torch.nn.functional.gelu(self.ff1(x))                # (B, L, CLUSTER_DIM)
        out  = self.ff2(x)                                           # (B, L, MOTOR_DIM)
        delta = self.adapter_out(out)                                # (B, L, base_dim)
        return (gate * delta).to(hidden.dtype)
