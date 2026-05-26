import torch
import torch.nn as nn
import pytest


class MockLFMBase(nn.Module):
    """Minimal mock of Lfm2ForCausalLM for unit testing without loading the 1.2B model.

    Mirrors the real structure: self.model (backbone) + self.lm_head.
    self.model.forward() returns an object with last_hidden_state (like Lfm2Model).
    self.forward()       returns an object with logits (like Lfm2ForCausalLM).
    """

    HIDDEN_SIZE = 2048
    VOCAB_SIZE  = 100

    class _Backbone(nn.Module):
        def __init__(self, hidden_size: int) -> None:
            super().__init__()
            self.hidden_size = hidden_size

        def forward(self, input_ids: torch.Tensor, **kwargs):
            B, L = input_ids.shape
            h = torch.randn(B, L, self.hidden_size, device=input_ids.device)
            return type("Output", (), {"last_hidden_state": h})()

    def __init__(self):
        super().__init__()
        self.model   = self._Backbone(self.HIDDEN_SIZE)
        self.lm_head = nn.Linear(self.HIDDEN_SIZE, self.VOCAB_SIZE, bias=False)
        self.config  = type("Config", (), {"hidden_size": self.HIDDEN_SIZE})()

    def forward(self, input_ids: torch.Tensor, **kwargs):
        out    = self.model(input_ids)
        logits = self.lm_head(out.last_hidden_state)
        return type("Output", (), {"logits": logits})()


@pytest.fixture
def mock_base():
    return MockLFMBase()
