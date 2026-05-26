import torch
import torch.nn as nn
import pytest


class MockLFMBase(nn.Module):
    """Minimal mock of LFM2 for unit testing without loading the 1.2B model."""

    HIDDEN_SIZE = 2560
    VOCAB_SIZE  = 100

    def __init__(self):
        super().__init__()
        self.lm_head = nn.Linear(self.HIDDEN_SIZE, self.VOCAB_SIZE, bias=False)
        self.config  = type("Config", (), {"hidden_size": self.HIDDEN_SIZE})()

    def forward(self, input_ids, output_hidden_states=False, **kwargs):
        B, L = input_ids.shape
        h      = torch.randn(B, L, self.HIDDEN_SIZE, device=input_ids.device)
        logits = self.lm_head(h)
        if output_hidden_states:
            return type("Output", (), {"hidden_states": (h,), "logits": logits})()
        return type("Output", (), {"logits": logits})()


@pytest.fixture
def mock_base():
    return MockLFMBase()
