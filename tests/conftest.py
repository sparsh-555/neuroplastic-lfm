import torch
import torch.nn as nn
import pytest


class _IdentityLayer(nn.Module):
    """Stands in for Lfm2DecoderLayer. No params; forward hooks fire correctly."""
    def forward(self, hidden_states: torch.Tensor, **kwargs) -> torch.Tensor:
        return hidden_states


class MockLFMBase(nn.Module):
    """Minimal mock of Lfm2ForCausalLM for unit testing without loading the 1.2B model.

    Mirrors the real two-level structure:
      self.model          — backbone (_Backbone), has .layers ModuleList
      self.lm_head        — Linear head
    The backbone iterates through identity layers so register_forward_hook fires
    at the correct index, matching NeuroplasticLFM.inject_at behaviour.
    """

    HIDDEN_SIZE = 2048
    VOCAB_SIZE  = 100
    N_LAYERS    = 16

    class _Backbone(nn.Module):
        def __init__(self, hidden_size: int, n_layers: int) -> None:
            super().__init__()
            self.hidden_size = hidden_size
            self.layers = nn.ModuleList(
                [_IdentityLayer() for _ in range(n_layers)]
            )

        def forward(self, input_ids: torch.Tensor, **kwargs):
            B, L = input_ids.shape
            h = torch.randn(B, L, self.hidden_size, device=input_ids.device)
            for layer in self.layers:
                h = layer(h)
            return type("Output", (), {"last_hidden_state": h})()

    def __init__(self):
        super().__init__()
        self.model   = self._Backbone(self.HIDDEN_SIZE, self.N_LAYERS)
        self.lm_head = nn.Linear(self.HIDDEN_SIZE, self.VOCAB_SIZE, bias=False)
        self.config  = type("Config", (), {"hidden_size": self.HIDDEN_SIZE})()

    def forward(self, input_ids: torch.Tensor, **kwargs):
        out    = self.model(input_ids)
        logits = self.lm_head(out.last_hidden_state)
        return type("Output", (), {"logits": logits})()


@pytest.fixture
def mock_base():
    return MockLFMBase()
