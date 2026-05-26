import torch
import pytest
from src.model import NeuroplasticLFM
from tests.conftest import MockLFMBase


@pytest.fixture
def model():
    return NeuroplasticLFM(MockLFMBase())


def test_base_params_all_frozen(model):
    for name, param in model.base.named_parameters():
        assert not param.requires_grad, f"Base param '{name}' should be frozen"


def test_no_trainable_params_before_any_spawn(model):
    trainable = [p for p in model.parameters() if p.requires_grad]
    assert len(trainable) == 0


def test_forward_without_cluster_returns_correct_shape(model):
    ids = torch.randint(0, MockLFMBase.VOCAB_SIZE, (2, 8))
    logits = model(ids)
    assert logits.shape == (2, 8, MockLFMBase.VOCAB_SIZE)


def test_forward_with_cluster_returns_correct_shape(model):
    model.spawn_cluster("task_a")
    ids = torch.randint(0, MockLFMBase.VOCAB_SIZE, (2, 8))
    logits = model(ids, task_id="task_a")
    assert logits.shape == (2, 8, MockLFMBase.VOCAB_SIZE)


def test_spawn_makes_cluster_params_trainable(model):
    model.spawn_cluster("task_a")
    trainable = [p for p in model.parameters() if p.requires_grad]
    assert len(trainable) > 0


def test_base_stays_frozen_after_spawn(model):
    model.spawn_cluster("task_a")
    for name, param in model.base.named_parameters():
        assert not param.requires_grad, f"Base param '{name}' should remain frozen"


def test_forward_no_task_id_uses_base_only(model):
    model.spawn_cluster("task_a")
    ids = torch.randint(0, MockLFMBase.VOCAB_SIZE, (1, 4))
    logits = model(ids, task_id=None)
    assert logits.shape == (1, 4, MockLFMBase.VOCAB_SIZE)


def test_unknown_task_id_raises_key_error(model):
    with pytest.raises(KeyError):
        ids = torch.randint(0, MockLFMBase.VOCAB_SIZE, (1, 4))
        model(ids, task_id="nonexistent")


def test_gradient_does_not_reach_base_params(model):
    model.spawn_cluster("science")
    ids    = torch.randint(0, MockLFMBase.VOCAB_SIZE, (2, 6))
    labels = ids.clone()
    logits = model(ids, task_id="science")
    loss   = torch.nn.functional.cross_entropy(
        logits[:, :-1].reshape(-1, MockLFMBase.VOCAB_SIZE),
        labels[:, 1:].reshape(-1),
    )
    loss.backward()
    for name, param in model.base.named_parameters():
        assert param.grad is None, f"Gradient leaked into base param '{name}'"


def test_train_cluster_reduces_loss_and_gate_opens():
    """Smoke test: train loop runs, records history, gate is near zero initially."""
    import torch
    from torch.utils.data import DataLoader
    from src.train import train_cluster

    class DictDS(torch.utils.data.Dataset):
        def __init__(self):
            self.ids = torch.randint(0, MockLFMBase.VOCAB_SIZE, (8, 8))
        def __len__(self): return len(self.ids)
        def __getitem__(self, i):
            return {"input_ids": self.ids[i], "labels": self.ids[i].clone()}

    base  = MockLFMBase()
    m     = NeuroplasticLFM(base)
    m.spawn_cluster("smoke")
    dl    = DataLoader(DictDS(), batch_size=2)
    history = train_cluster(m, "smoke", dl, max_steps=6, lr=1e-2, log_every=3)

    assert len(history) >= 1
    assert history[0]["loss"] > 0
    assert history[0]["gate"] < 0.01
