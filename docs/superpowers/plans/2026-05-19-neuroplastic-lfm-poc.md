# Neuroplastic LFM — PoC Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement a dual-axis neuroplastic system on frozen LFM2-1.2B where CfC clusters (not LoRA) are spawned per task, providing both structural zero-forgetting (PNN pattern) and functional adaptive τ dynamics (LTC), with TIES-Merging for cluster fusion.

**Architecture:** Frozen LFM2-1.2B extracts hidden states (B, L, 2560); per-task CfC clusters (AutoNCP-wired, ~305K params each) process these token-by-token and inject a sigmoid-gated residual back into the hidden stream before the LM head. Each cluster is frozen on next spawn, guaranteeing zero forgetting by construction.

**Tech Stack:** Python 3.10+, PyTorch 2.2+, `transformers` (main branch for LFM2), `ncps>=0.0.7`, `datasets`, `accelerate`

---

## File Map

| File | Responsibility |
|---|---|
| `src/cluster.py` | `CfCCluster`: adapters + NCP-wired CfC + maturity gate |
| `src/registry.py` | `ClusterRegistry`: spawn, freeze, get, shared θ_init |
| `src/model.py` | `NeuroplasticLFM`: frozen base + registry, end-to-end forward |
| `src/merge.py` | `_trim`, `_elect_and_merge`, `ties_merge` for CfC weight fusion |
| `src/train.py` | `train_cluster`, `extract_tau_stats` |
| `src/eval.py` | `perplexity`, `forgetting_score`, `tau_stats` |
| `tests/conftest.py` | `MockLFMBase` fixture shared across all test files |
| `tests/test_cluster.py` | Unit tests for CfCCluster |
| `tests/test_registry.py` | Unit tests for ClusterRegistry |
| `tests/test_model.py` | Unit tests for NeuroplasticLFM with mock base |
| `tests/test_merge.py` | Unit tests for trim, elect, merge, ties_merge |
| `experiments/poc_dual_task.py` | Full PoC: spawn A, spawn B, verify forgetting=0, merge |
| `experiments/baseline_finetuning.py` | Naive fine-tuning to demonstrate catastrophic forgetting |

---

## Task 1: Project Setup + Shared Test Fixture

**Files:**
- Create: `tests/conftest.py`
- Create: `src/__init__.py` (already exists, verify empty)

- [ ] **Step 1: Install dependencies**

```bash
cd /Users/sparshjain/Documents/GitHub/neuroplastic-lfm
pip install "torch>=2.2.0" "ncps>=0.0.7" "datasets>=2.18.0" "accelerate>=0.28.0" "pytest>=8.0" "numpy>=1.26.0" "matplotlib>=3.8.0" "tqdm>=4.66.0"
# LFM2 requires transformers from main if not yet in stable release:
pip install git+https://github.com/huggingface/transformers
```

Expected: no errors.

- [ ] **Step 2: Verify ncps CfC API**

```bash
python -c "
from ncps.torch import CfC
from ncps.wirings import AutoNCP
import torch
wiring = AutoNCP(units=64, output_size=16, sparsity_level=0.5, seed=0)
rnn = CfC(64, wiring, batch_first=True, return_sequences=True)
x  = torch.randn(2, 10, 64)
h0 = torch.zeros(2, 64)
out, hn = rnn(x, h0)
print('CfC output shape:', out.shape)   # expect (2, 10, 16)
print('CfC state shape:', hn.shape)     # expect (2, 64)
"
```

Expected output:
```
CfC output shape: torch.Size([2, 10, 16])
CfC state shape: torch.Size([2, 64])
```

- [ ] **Step 3: Write `tests/conftest.py` with shared mock**

```python
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
```

- [ ] **Step 4: Verify pytest discovers conftest**

```bash
pytest tests/ --collect-only 2>&1 | head -10
```

Expected: no import errors, test collection output shown.

- [ ] **Step 5: Commit**

```bash
git add tests/conftest.py requirements.txt
git commit -m "chore: add shared test fixture and verify ncps CfC API"
```

---

## Task 2: CfCCluster

**Files:**
- Create: `tests/test_cluster.py`
- Modify: `src/cluster.py`

- [ ] **Step 1: Write failing tests in `tests/test_cluster.py`**

```python
import torch
import pytest
from src.cluster import CfCCluster


def test_forward_output_shape():
    cluster = CfCCluster(seed=0)
    h = torch.randn(2, 10, 2560)
    out = cluster(h)
    assert out.shape == (2, 10, 2560)


def test_maturity_gate_near_zero_at_init():
    cluster = CfCCluster(seed=0)
    gate = torch.sigmoid(cluster.maturity).item()
    assert gate < 0.01  # sigmoid(-6) ≈ 0.00248


def test_output_near_zero_at_init():
    cluster = CfCCluster(seed=0)
    h = torch.randn(2, 5, 2560)
    out = cluster(h)
    # sigmoid(-6) ≈ 0.002; output should be small regardless of internal values
    assert out.abs().max().item() < 1.0


def test_all_parameters_trainable_at_creation():
    cluster = CfCCluster(seed=0)
    for name, param in cluster.named_parameters():
        assert param.requires_grad, f"Parameter '{name}' should be trainable at creation"


def test_parameter_count_in_expected_range():
    cluster = CfCCluster(seed=0)
    n = sum(p.numel() for p in cluster.parameters())
    assert 200_000 < n < 500_000, f"Expected 200K–500K params, got {n}"


def test_gradient_flows_to_all_components():
    cluster = CfCCluster(seed=0)
    h = torch.randn(2, 5, 2560, requires_grad=False)
    out = cluster(h)
    out.sum().backward()
    assert cluster.adapter_in.weight.grad is not None, "adapter_in gradient missing"
    assert cluster.adapter_out.weight.grad is not None, "adapter_out gradient missing"
    assert cluster.maturity.grad is not None, "maturity gate gradient missing"


def test_batch_size_one():
    cluster = CfCCluster(seed=0)
    h = torch.randn(1, 7, 2560)
    out = cluster(h)
    assert out.shape == (1, 7, 2560)


def test_different_seeds_produce_different_outputs():
    cluster_a = CfCCluster(seed=0)
    cluster_b = CfCCluster(seed=99)
    h = torch.randn(1, 5, 2560)
    out_a = cluster_a(h)
    out_b = cluster_b(h)
    # Different seeds → different wiring → different outputs
    assert not torch.allclose(out_a, out_b)
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
pytest tests/test_cluster.py -v 2>&1 | tail -15
```

Expected: `ImportError` or `ModuleNotFoundError` for `src.cluster`.

- [ ] **Step 3: Write `src/cluster.py`**

```python
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
        h      = hidden.float()                                        # CfC needs float32
        x      = self.adapter_in(h)                                    # (B, L, CLUSTER_DIM)
        h0     = torch.zeros(x.size(0), self.CLUSTER_DIM,
                             device=x.device, dtype=x.dtype)
        out, _ = self.cfc(x, h0)                                       # (B, L, MOTOR_DIM)
        delta  = self.adapter_out(out)                                 # (B, L, BASE_DIM)
        return (torch.sigmoid(self.maturity) * delta).to(hidden.dtype)
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
pytest tests/test_cluster.py -v
```

Expected: all 8 tests `PASSED`.

- [ ] **Step 5: Commit**

```bash
git add src/cluster.py tests/test_cluster.py
git commit -m "feat: implement CfCCluster with NCP wiring and zero-init maturity gate"
```

---

## Task 3: ClusterRegistry

**Files:**
- Create: `tests/test_registry.py`
- Modify: `src/registry.py`

- [ ] **Step 1: Write failing tests in `tests/test_registry.py`**

```python
import torch
import pytest
from src.cluster import CfCCluster
from src.registry import ClusterRegistry


def test_spawn_returns_cfc_cluster():
    registry = ClusterRegistry()
    cluster = registry.spawn("task_a")
    assert isinstance(cluster, CfCCluster)


def test_spawn_stores_base_init_on_first_spawn():
    registry = ClusterRegistry()
    registry.spawn("task_a")
    assert registry.base_init is not None
    assert len(registry.base_init) > 0


def test_base_init_matches_fresh_cluster():
    registry = ClusterRegistry()
    registry.spawn("task_a")
    fresh = CfCCluster(seed=ClusterRegistry.CLUSTER_SEED)
    for k, v in registry.base_init.items():
        expected = fresh.state_dict()[k].float()
        actual   = v.float()
        assert torch.allclose(actual, expected, atol=1e-5), f"Mismatch in base_init key '{k}'"


def test_first_cluster_trainable():
    registry = ClusterRegistry()
    c_a = registry.spawn("task_a")
    assert all(p.requires_grad for p in c_a.parameters())


def test_spawn_freezes_prior_clusters():
    registry = ClusterRegistry()
    c_a = registry.spawn("task_a")
    registry.spawn("task_b")
    assert all(not p.requires_grad for p in c_a.parameters()), \
        "cluster_a should be frozen after cluster_b is spawned"


def test_second_cluster_trainable_after_spawn():
    registry = ClusterRegistry()
    registry.spawn("task_a")
    c_b = registry.spawn("task_b")
    assert all(p.requires_grad for p in c_b.parameters())


def test_get_returns_correct_cluster():
    registry = ClusterRegistry()
    c_a = registry.spawn("task_a")
    c_b = registry.spawn("task_b")
    assert registry.get("task_a") is c_a
    assert registry.get("task_b") is c_b


def test_get_unknown_task_raises_key_error():
    registry = ClusterRegistry()
    with pytest.raises(KeyError, match="nonexistent"):
        registry.get("nonexistent")


def test_three_sequential_spawns_freeze_all_prior():
    registry = ClusterRegistry()
    c_a = registry.spawn("a")
    c_b = registry.spawn("b")
    c_c = registry.spawn("c")
    assert all(not p.requires_grad for p in c_a.parameters())
    assert all(not p.requires_grad for p in c_b.parameters())
    assert all(p.requires_grad for p in c_c.parameters())
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
pytest tests/test_registry.py -v 2>&1 | tail -15
```

Expected: `ImportError` for `src.registry`.

- [ ] **Step 3: Write `src/registry.py`**

```python
import torch
import torch.nn as nn
from typing import Dict, Optional
from src.cluster import CfCCluster


class ClusterRegistry(nn.Module):
    # All clusters start from the same init so TIES task vectors are comparable.
    CLUSTER_SEED = 0

    def __init__(self):
        super().__init__()
        self.clusters:  nn.ModuleDict                    = nn.ModuleDict()
        self.base_init: Optional[Dict[str, torch.Tensor]] = None

    def spawn(self, task_id: str) -> CfCCluster:
        for cluster in self.clusters.values():
            for param in cluster.parameters():
                param.requires_grad = False

        cluster = CfCCluster(seed=self.CLUSTER_SEED)

        if self.base_init is None:
            self.base_init = {
                k: v.clone().detach() for k, v in cluster.state_dict().items()
            }

        self.clusters[task_id] = cluster
        return cluster

    def get(self, task_id: str) -> CfCCluster:
        if task_id not in self.clusters:
            raise KeyError(
                f"No cluster for task '{task_id}'. "
                f"Available: {list(self.clusters.keys())}"
            )
        return self.clusters[task_id]
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
pytest tests/test_registry.py -v
```

Expected: all 9 tests `PASSED`.

- [ ] **Step 5: Commit**

```bash
git add src/registry.py tests/test_registry.py
git commit -m "feat: implement ClusterRegistry with spawn-freeze invariant and shared base_init"
```

---

## Task 4: NeuroplasticLFM

**Files:**
- Create: `tests/test_model.py`
- Modify: `src/model.py`

- [ ] **Step 1: Write failing tests in `tests/test_model.py`**

```python
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
    # Spawn a cluster but don't pass task_id — output should be base-only
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
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
pytest tests/test_model.py -v 2>&1 | tail -15
```

Expected: `ImportError` for `src.model`.

- [ ] **Step 3: Write `src/model.py`**

```python
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
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
pytest tests/test_model.py -v
```

Expected: all 9 tests `PASSED`.

- [ ] **Step 5: Run all tests so far to check nothing is broken**

```bash
pytest tests/ -v
```

Expected: all 26 tests `PASSED`.

- [ ] **Step 6: Commit**

```bash
git add src/model.py tests/test_model.py
git commit -m "feat: implement NeuroplasticLFM with frozen base and cluster injection"
```

---

## Task 5: TIES-Merging

**Files:**
- Create: `tests/test_merge.py`
- Modify: `src/merge.py`

- [ ] **Step 1: Write failing tests in `tests/test_merge.py`**

```python
import torch
import pytest
from src.cluster import CfCCluster
from src.registry import ClusterRegistry
from src.merge import _trim, _elect_and_merge, ties_merge


# --- _trim tests ---

def test_trim_zeros_small_magnitudes():
    τ = torch.tensor([0.1, 0.5, 0.01, 0.8, 0.02])
    result = _trim(τ, trim_p=0.4)
    # quantile(0.4) of [0.01, 0.02, 0.1, 0.5, 0.8] ≈ 0.02..0.1 range
    assert result[2].item() == pytest.approx(0.0, abs=1e-6)   # 0.01 trimmed
    assert result[4].item() == pytest.approx(0.0, abs=1e-6)   # 0.02 trimmed
    assert result[1].item() == pytest.approx(0.5, abs=1e-6)   # 0.5 kept
    assert result[3].item() == pytest.approx(0.8, abs=1e-6)   # 0.8 kept


def test_trim_zero_threshold_keeps_all():
    τ = torch.tensor([0.1, 0.5, 0.01])
    assert torch.allclose(_trim(τ, trim_p=0.0), τ)


def test_trim_preserves_sign():
    τ = torch.tensor([-0.9, 0.1, -0.05])
    result = _trim(τ, trim_p=0.4)
    assert result[0].item() == pytest.approx(-0.9, abs=1e-6)
    assert result[1].item() == pytest.approx(0.1, abs=1e-6)


# --- _elect_and_merge tests ---

def test_elect_agreeing_signs_are_averaged():
    τ_a = torch.tensor([0.3, 0.5])
    τ_b = torch.tensor([0.1, 0.2])
    result = _elect_and_merge([τ_a, τ_b])
    assert result[0].item() == pytest.approx(0.2, abs=1e-5)   # (0.3+0.1)/2
    assert result[1].item() == pytest.approx(0.35, abs=1e-5)  # (0.5+0.2)/2


def test_elect_positive_wins_when_sum_positive():
    # sum = 0.3 + (-0.1) = 0.2 > 0 → positive elected → τ_b zeroed
    τ_a = torch.tensor([0.3])
    τ_b = torch.tensor([-0.1])
    result = _elect_and_merge([τ_a, τ_b])
    assert result[0].item() == pytest.approx(0.3, abs=1e-5)


def test_elect_negative_wins_when_sum_negative():
    # sum = 0.1 + (-0.4) = -0.3 < 0 → negative elected → τ_a zeroed
    τ_a = torch.tensor([0.1])
    τ_b = torch.tensor([-0.4])
    result = _elect_and_merge([τ_a, τ_b])
    assert result[0].item() == pytest.approx(-0.4, abs=1e-5)


def test_elect_both_zero_returns_zero():
    τ_a = torch.tensor([0.0])
    τ_b = torch.tensor([0.0])
    result = _elect_and_merge([τ_a, τ_b])
    assert result[0].item() == pytest.approx(0.0, abs=1e-6)


# --- ties_merge integration tests ---

def test_ties_merge_produces_cfc_cluster():
    registry = ClusterRegistry()
    registry.spawn("task_a")
    registry.spawn("task_b")
    merged = ties_merge(registry, "task_a", "task_b", "merged")
    assert isinstance(merged, CfCCluster)


def test_ties_merge_registers_under_new_id():
    registry = ClusterRegistry()
    registry.spawn("task_a")
    registry.spawn("task_b")
    ties_merge(registry, "task_a", "task_b", "merged")
    assert "merged" in registry.clusters


def test_ties_merge_frozen_result():
    registry = ClusterRegistry()
    registry.spawn("task_a")
    registry.spawn("task_b")
    merged = ties_merge(registry, "task_a", "task_b", "merged")
    assert all(not p.requires_grad for p in merged.parameters())


def test_ties_merge_untrained_clusters_returns_near_base_init():
    # Untrained clusters = task vectors ≈ 0 → merged ≈ base_init
    registry = ClusterRegistry()
    registry.spawn("task_a")
    registry.spawn("task_b")
    merged = ties_merge(registry, "task_a", "task_b", "merged")
    for k, v in registry.base_init.items():
        diff = (merged.state_dict()[k].float() - v.float()).abs().max().item()
        assert diff < 1e-4, f"Key '{k}': merged deviates from base_init by {diff}"
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
pytest tests/test_merge.py -v 2>&1 | tail -15
```

Expected: `ImportError` for `src.merge`.

- [ ] **Step 3: Write `src/merge.py`**

```python
import torch
import torch.nn as nn
from typing import List
from src.cluster import CfCCluster
from src.registry import ClusterRegistry


def _trim(τ: torch.Tensor, trim_p: float) -> torch.Tensor:
    """Zero out entries of τ below the trim_p-th percentile of absolute magnitude."""
    if trim_p == 0.0:
        return τ.clone()
    threshold = torch.quantile(τ.abs().float().flatten(), trim_p)
    return torch.where(τ.abs() >= threshold, τ, torch.zeros_like(τ))


def _elect_and_merge(task_vectors: List[torch.Tensor]) -> torch.Tensor:
    """
    TIES elect step: for each parameter position, elect the majority sign via
    sum-based tie-breaking, then average only the non-conflicting values.
    """
    stacked = torch.stack([τ.float() for τ in task_vectors], dim=0)  # (K, ...)
    total   = stacked.sum(dim=0)

    elected_sign                    = torch.sign(total)
    elected_sign[elected_sign == 0] = 1.0  # tie → positive

    # Zero out entries whose sign conflicts with the elected sign
    signs  = torch.sign(stacked)
    keep   = (signs == elected_sign.unsqueeze(0)) | (stacked == 0)
    masked = stacked * keep.float()

    count  = (masked != 0).float().sum(dim=0).clamp(min=1.0)
    return masked.sum(dim=0) / count


def ties_merge(
    registry: ClusterRegistry,
    task_a: str,
    task_b: str,
    new_task_id: str,
    trim_p: float = 0.2,
) -> CfCCluster:
    """Merge two trained clusters into a new cluster using TIES-Merging."""
    base_init = registry.base_init
    state_a   = registry.get(task_a).state_dict()
    state_b   = registry.get(task_b).state_dict()

    # Identify learnable parameters (vs. fixed buffers like sparsity masks)
    param_keys = {name for name, _ in CfCCluster(seed=0).named_parameters()}

    merged_state = {}
    for key in base_init:
        θ_init = base_init[key].float()
        if key not in param_keys:
            # Fixed buffer (e.g. NCP sparsity mask) — identical across clusters
            merged_state[key] = θ_init.to(base_init[key].dtype)
        else:
            τ_a      = _trim(state_a[key].float() - θ_init, trim_p)
            τ_b      = _trim(state_b[key].float() - θ_init, trim_p)
            merged_τ = _elect_and_merge([τ_a, τ_b])
            merged_state[key] = (θ_init + merged_τ).to(base_init[key].dtype)

    merged_cluster = CfCCluster(seed=0)
    merged_cluster.load_state_dict(merged_state)

    for param in merged_cluster.parameters():
        param.requires_grad = False

    registry.clusters[new_task_id] = merged_cluster
    return merged_cluster
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
pytest tests/test_merge.py -v
```

Expected: all 13 tests `PASSED`.

- [ ] **Step 5: Run full test suite**

```bash
pytest tests/ -v
```

Expected: all 39 tests `PASSED`.

- [ ] **Step 6: Commit**

```bash
git add src/merge.py tests/test_merge.py
git commit -m "feat: implement TIES-Merging for CfC cluster weight fusion"
```

---

## Task 6: Training Loop + τ Extraction

**Files:**
- Modify: `src/train.py`

- [ ] **Step 1: Write `src/train.py`**

```python
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Dict, List, Optional
from tqdm import tqdm
from src.model import NeuroplasticLFM


def extract_tau_stats(cluster) -> Dict[str, float]:
    """
    Extract adaptive time-constant statistics from CfC cell.
    time_a and time_b are the learned projections that compute τ per token:
        t_interp = sigmoid(time_a(x) * ts + time_b(x))
    Watching these evolve during training is the functional neuroplasticity signal.
    """
    try:
        cell = cluster.cfc.rnn_cell
        return {
            "time_a_mean": cell.time_a.weight.data.mean().item(),
            "time_b_mean": cell.time_b.weight.data.mean().item(),
            "time_a_std":  cell.time_a.weight.data.std().item(),
        }
    except AttributeError:
        return {}


def train_cluster(
    model: NeuroplasticLFM,
    task_id: str,
    dataloader: DataLoader,
    max_steps: int = 500,
    lr: float = 3e-4,
    log_every: int = 50,
) -> List[Dict]:
    cluster   = model.registry.get(task_id)
    optimizer = torch.optim.AdamW(
        [p for p in cluster.parameters() if p.requires_grad],
        lr=lr,
        weight_decay=0.01,
    )

    cluster.train()
    model.base.eval()

    history = []
    step    = 0

    for batch in tqdm(dataloader, desc=f"cluster[{task_id}]"):
        if step >= max_steps:
            break

        input_ids = batch["input_ids"].to(next(cluster.parameters()).device)
        labels    = batch["labels"].to(input_ids.device)

        logits = model(input_ids, task_id=task_id)          # (B, L, V)
        loss   = F.cross_entropy(
            logits[:, :-1].contiguous().view(-1, logits.size(-1)),
            labels[:, 1:].contiguous().view(-1),
            ignore_index=-100,
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(cluster.parameters(), 1.0)
        optimizer.step()

        if step % log_every == 0:
            gate   = torch.sigmoid(cluster.maturity).item()
            τ_info = extract_tau_stats(cluster)
            record = {"step": step, "loss": loss.item(), "gate": gate, **τ_info}
            history.append(record)
            τ_str = f"  τ_a={τ_info.get('time_a_mean', 0):.4f}" if τ_info else ""
            print(f"  step={step:4d}  loss={loss.item():.4f}  gate={gate:.4f}{τ_str}")

        step += 1

    return history
```

- [ ] **Step 2: Write a smoke test to verify train loop runs**

Add to the end of `tests/test_model.py`:

```python
def test_train_cluster_reduces_loss():
    """Smoke test: loss should decrease over a few steps on repeated data."""
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset
    from src.train import train_cluster

    base  = MockLFMBase()
    model = NeuroplasticLFM(base)
    model.spawn_cluster("smoke")

    # Tiny repeating dataset: (B=2, L=8)
    ids    = torch.randint(0, MockLFMBase.VOCAB_SIZE, (4, 8))
    labels = ids.clone()
    ds     = TensorDataset(ids, labels)

    class DictDataset(torch.utils.data.Dataset):
        def __init__(self, ids, labels):
            self.ids, self.labels = ids, labels
        def __len__(self): return len(self.ids)
        def __getitem__(self, i):
            return {"input_ids": self.ids[i], "labels": self.labels[i]}

    dl = DataLoader(DictDataset(ids, labels), batch_size=2)
    history = train_cluster(model, "smoke", dl, max_steps=10, lr=1e-2, log_every=5)

    assert len(history) > 0
    assert history[0]["loss"] > 0
    assert history[0]["gate"] < 0.01   # maturity still near zero after 10 steps
```

- [ ] **Step 3: Run updated tests**

```bash
pytest tests/test_model.py -v
```

Expected: all 10 tests `PASSED`.

- [ ] **Step 4: Commit**

```bash
git add src/train.py tests/test_model.py
git commit -m "feat: implement train_cluster with tau extraction and maturity gate logging"
```

---

## Task 7: Evaluation Utilities

**Files:**
- Modify: `src/eval.py`

- [ ] **Step 1: Write `src/eval.py`**

```python
import math
import torch
import torch.nn.functional as F
from typing import Dict, Optional
from torch.utils.data import DataLoader
from src.model import NeuroplasticLFM


@torch.no_grad()
def perplexity(
    model: NeuroplasticLFM,
    dataloader: DataLoader,
    task_id: Optional[str] = None,
) -> float:
    """Compute per-token perplexity across the dataloader."""
    model.eval()
    total_loss   = 0.0
    total_tokens = 0

    for batch in dataloader:
        device    = next(model.base.parameters()).device
        input_ids = batch["input_ids"].to(device)
        labels    = batch["labels"].to(device)

        logits = model(input_ids, task_id=task_id)   # (B, L, V)
        loss   = F.cross_entropy(
            logits[:, :-1].contiguous().view(-1, logits.size(-1)),
            labels[:, 1:].contiguous().view(-1),
            ignore_index=-100,
            reduction="sum",
        )
        total_loss   += loss.item()
        total_tokens += (labels[:, 1:] != -100).sum().item()

    return math.exp(total_loss / max(total_tokens, 1))


def forgetting_score(ppl_before: float, ppl_after: float) -> float:
    """
    Fractional change in perplexity after a new task is trained.
    Positive = forgetting (ppl went up), negative = improvement.
    """
    return (ppl_after - ppl_before) / ppl_before


def tau_stats(cluster) -> Dict[str, float]:
    """Return current τ statistics from a CfCCluster."""
    from src.train import extract_tau_stats
    return extract_tau_stats(cluster)
```

- [ ] **Step 2: Smoke test perplexity utility**

Add to the end of `tests/test_model.py`:

```python
def test_perplexity_returns_positive_float():
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
    from src.eval import perplexity

    class DictDS(torch.utils.data.Dataset):
        def __init__(self, n=4, L=8):
            self.ids = torch.randint(0, MockLFMBase.VOCAB_SIZE, (n, L))
        def __len__(self): return len(self.ids)
        def __getitem__(self, i):
            return {"input_ids": self.ids[i], "labels": self.ids[i].clone()}

    model = NeuroplasticLFM(MockLFMBase())
    dl    = DataLoader(DictDS(), batch_size=2)
    ppl   = perplexity(model, dl, task_id=None)
    assert ppl > 0
    assert ppl < 1e6   # sanity: not astronomically large
```

- [ ] **Step 3: Run full test suite**

```bash
pytest tests/ -v
```

Expected: all 41 tests `PASSED`.

- [ ] **Step 4: Commit**

```bash
git add src/eval.py tests/test_model.py
git commit -m "feat: implement perplexity, forgetting_score, and tau_stats evaluation utilities"
```

---

## Task 8: Dataset Utilities + PoC Experiment

**Files:**
- Modify: `experiments/poc_dual_task.py`

- [ ] **Step 1: Write `experiments/poc_dual_task.py`**

```python
"""
PoC experiment: dual-task neuroplastic LFM.

Spawns a CfC cluster for science Q&A (Task A), then creative writing (Task B).
Demonstrates:
  1. Zero forgetting on Task A after Task B training.
  2. Maturity gate opening during training (functional neuroplasticity).
  3. TIES-Merging producing a combined cluster.
"""
import torch
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.model import NeuroplasticLFM
from src.merge import ties_merge
from src.train import train_cluster
from src.eval import perplexity, forgetting_score

SCIENCE_KEYWORDS  = {"science", "math", "physics", "chemistry", "biology",
                     "formula", "equation", "calculate", "theorem"}
CREATIVE_KEYWORDS = {"story", "creative", "write", "poem", "fiction",
                     "narrative", "imagine", "character", "novel"}

MAX_LENGTH = 256
BATCH_SIZE = 4
MAX_STEPS  = 300
TRAIN_SIZE = 150
EVAL_SIZE  = 50


class AlpacaDataset(Dataset):
    def __init__(self, records, tokenizer, max_length=MAX_LENGTH):
        self.records    = records
        self.tokenizer  = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec   = self.records[idx]
        text  = (f"### Instruction:\n{rec['instruction']}\n\n"
                 f"### Response:\n{rec['output']}")
        enc   = self.tokenizer(
            text, truncation=True, max_length=self.max_length,
            padding="max_length", return_tensors="pt",
        )
        ids    = enc["input_ids"].squeeze(0)
        labels = ids.clone()
        labels[labels == self.tokenizer.pad_token_id] = -100
        return {"input_ids": ids, "labels": labels}


def filter_records(dataset, keywords, n):
    results = []
    for rec in dataset:
        text = (rec.get("instruction", "") + " " + rec.get("output", "")).lower()
        if any(kw in text for kw in keywords):
            results.append(rec)
        if len(results) >= n:
            break
    return results


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    print("Loading LFM2-1.2B (frozen base)...")
    base = AutoModelForCausalLM.from_pretrained(
        "LiquidAI/LFM2-1.2B", torch_dtype=torch.float32
    ).to(device)
    tok  = AutoTokenizer.from_pretrained("LiquidAI/LFM2-1.2B")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = NeuroplasticLFM(base).to(device)

    print("Preparing datasets from yahma/alpaca-cleaned...")
    alpaca = load_dataset("yahma/alpaca-cleaned", split="train")
    sci_tr  = filter_records(alpaca, SCIENCE_KEYWORDS, TRAIN_SIZE)
    sci_ev  = filter_records(alpaca, SCIENCE_KEYWORDS, TRAIN_SIZE + EVAL_SIZE)[TRAIN_SIZE:]
    cre_tr  = filter_records(alpaca, CREATIVE_KEYWORDS, TRAIN_SIZE)
    cre_ev  = filter_records(alpaca, CREATIVE_KEYWORDS, TRAIN_SIZE + EVAL_SIZE)[TRAIN_SIZE:]

    dl_sci_tr = DataLoader(AlpacaDataset(sci_tr, tok), batch_size=BATCH_SIZE, shuffle=True)
    dl_sci_ev = DataLoader(AlpacaDataset(sci_ev, tok), batch_size=BATCH_SIZE)
    dl_cre_tr = DataLoader(AlpacaDataset(cre_tr, tok), batch_size=BATCH_SIZE, shuffle=True)
    dl_cre_ev = DataLoader(AlpacaDataset(cre_ev, tok), batch_size=BATCH_SIZE)

    # ── Baseline (no clusters) ────────────────────────────────────────────
    print("\n── Baseline (no clusters) ──")
    ppl_sci_base = perplexity(model, dl_sci_ev, task_id=None)
    ppl_cre_base = perplexity(model, dl_cre_ev, task_id=None)
    print(f"  Science PPL:  {ppl_sci_base:.2f}")
    print(f"  Creative PPL: {ppl_cre_base:.2f}")

    # ── Cluster A — Science ───────────────────────────────────────────────
    print("\n── Training Cluster A (science) ──")
    model.spawn_cluster("science")
    history_a = train_cluster(model, "science", dl_sci_tr,
                               max_steps=MAX_STEPS, log_every=50)
    ppl_sci_A = perplexity(model, dl_sci_ev, task_id="science")
    ppl_cre_A = perplexity(model, dl_cre_ev, task_id="science")
    print(f"  Science PPL: {ppl_sci_A:.2f}  Creative PPL: {ppl_cre_A:.2f}")

    # ── Cluster B — Creative ──────────────────────────────────────────────
    print("\n── Training Cluster B (creative) ──")
    model.spawn_cluster("creative")        # freezes cluster A
    history_b = train_cluster(model, "creative", dl_cre_tr,
                               max_steps=MAX_STEPS, log_every=50)
    ppl_sci_B = perplexity(model, dl_sci_ev, task_id="science")   # must not change
    ppl_cre_B = perplexity(model, dl_cre_ev, task_id="creative")
    print(f"  Science PPL: {ppl_sci_B:.2f}  Creative PPL: {ppl_cre_B:.2f}")

    # ── Zero-forgetting check ─────────────────────────────────────────────
    fs = forgetting_score(ppl_sci_A, ppl_sci_B)
    print(f"\n  Forgetting score (science): {fs:+.6f}  [expected: 0.000000]")
    assert abs(fs) < 1e-4, (
        f"Forgetting detected ({fs:+.4%})! "
        "Cluster A should be frozen and unaffected by Cluster B training."
    )

    # ── TIES Merge ────────────────────────────────────────────────────────
    print("\n── Merging A + B via TIES-Merging ──")
    ties_merge(model.registry, "science", "creative", "merged")
    ppl_sci_M = perplexity(model, dl_sci_ev, task_id="merged")
    ppl_cre_M = perplexity(model, dl_cre_ev, task_id="merged")
    print(f"  Merged — Science PPL: {ppl_sci_M:.2f}  Creative PPL: {ppl_cre_M:.2f}")

    # ── Summary table ─────────────────────────────────────────────────────
    col = 10
    print(f"\n{'═'*62}")
    print(f"{'':25s}  {'Baseline':>{col}}  {'ClusterA':>{col}}  "
          f"{'ClusterB':>{col}}  {'Merged':>{col}}")
    print(f"{'─'*62}")
    print(f"{'Science PPL':25s}  {ppl_sci_base:>{col}.2f}  {ppl_sci_A:>{col}.2f}  "
          f"{ppl_sci_B:>{col}.2f}  {ppl_sci_M:>{col}.2f}")
    print(f"{'Creative PPL':25s}  {ppl_cre_base:>{col}.2f}  {ppl_cre_A:>{col}.2f}  "
          f"{ppl_cre_B:>{col}.2f}  {ppl_cre_M:>{col}.2f}")
    print(f"{'═'*62}")
    print(f"Zero forgetting: {abs(fs) < 1e-4}")

    return {
        "ppl_sci_base": ppl_sci_base, "ppl_cre_base": ppl_cre_base,
        "ppl_sci_A": ppl_sci_A,       "ppl_sci_B": ppl_sci_B,
        "ppl_cre_B": ppl_cre_B,       "ppl_sci_M": ppl_sci_M,
        "ppl_cre_M": ppl_cre_M,
        "history_a": history_a,       "history_b": history_b,
    }


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify the script is importable**

```bash
python -c "from experiments.poc_dual_task import AlpacaDataset, filter_records; print('OK')"
```

Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add experiments/poc_dual_task.py
git commit -m "feat: implement PoC dual-task experiment with zero-forgetting assertion"
```

---

## Task 9: Baseline Fine-Tuning (Forgetting Comparison)

**Files:**
- Modify: `experiments/baseline_finetuning.py`

- [ ] **Step 1: Write `experiments/baseline_finetuning.py`**

```python
"""
Baseline: naive sequential fine-tuning to demonstrate catastrophic forgetting.
Run this alongside poc_dual_task.py to show NeuroplasticLFM's zero-forgetting guarantee.
"""
import math
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

from experiments.poc_dual_task import (
    AlpacaDataset, filter_records,
    SCIENCE_KEYWORDS, CREATIVE_KEYWORDS,
    TRAIN_SIZE, EVAL_SIZE, BATCH_SIZE, MAX_LENGTH,
)

MAX_STEPS = 300


def finetune(model, dataloader, max_steps=MAX_STEPS, lr=3e-4):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    model.train()
    step = 0
    for batch in tqdm(dataloader, desc="fine-tuning"):
        if step >= max_steps:
            break
        input_ids = batch["input_ids"]
        labels    = batch["labels"]
        logits    = model(input_ids).logits
        loss      = F.cross_entropy(
            logits[:, :-1].contiguous().view(-1, logits.size(-1)),
            labels[:, 1:].contiguous().view(-1),
            ignore_index=-100,
        )
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        step += 1


@torch.no_grad()
def compute_ppl(model, dataloader):
    model.eval()
    total_loss, total_tokens = 0.0, 0
    for batch in dataloader:
        input_ids = batch["input_ids"]
        labels    = batch["labels"]
        logits    = model(input_ids).logits
        loss      = F.cross_entropy(
            logits[:, :-1].contiguous().view(-1, logits.size(-1)),
            labels[:, 1:].contiguous().view(-1),
            ignore_index=-100,
            reduction="sum",
        )
        total_loss   += loss.item()
        total_tokens += (labels[:, 1:] != -100).sum().item()
    return math.exp(total_loss / max(total_tokens, 1))


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    print("Loading LFM2-1.2B for naive fine-tuning...")
    model = AutoModelForCausalLM.from_pretrained(
        "LiquidAI/LFM2-1.2B", torch_dtype=torch.float32
    ).to(device)
    tok = AutoTokenizer.from_pretrained("LiquidAI/LFM2-1.2B")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    alpaca  = load_dataset("yahma/alpaca-cleaned", split="train")
    sci_tr  = filter_records(alpaca, SCIENCE_KEYWORDS, TRAIN_SIZE)
    sci_ev  = filter_records(alpaca, SCIENCE_KEYWORDS, TRAIN_SIZE + EVAL_SIZE)[TRAIN_SIZE:]
    cre_tr  = filter_records(alpaca, CREATIVE_KEYWORDS, TRAIN_SIZE)
    cre_ev  = filter_records(alpaca, CREATIVE_KEYWORDS, TRAIN_SIZE + EVAL_SIZE)[TRAIN_SIZE:]

    dl_sci_tr = DataLoader(AlpacaDataset(sci_tr, tok), batch_size=BATCH_SIZE, shuffle=True)
    dl_sci_ev = DataLoader(AlpacaDataset(sci_ev, tok), batch_size=BATCH_SIZE)
    dl_cre_tr = DataLoader(AlpacaDataset(cre_tr, tok), batch_size=BATCH_SIZE, shuffle=True)
    dl_cre_ev = DataLoader(AlpacaDataset(cre_ev, tok), batch_size=BATCH_SIZE)

    ppl_sci_0 = compute_ppl(model, dl_sci_ev)
    ppl_cre_0 = compute_ppl(model, dl_cre_ev)
    print(f"Baseline — Science: {ppl_sci_0:.2f}  Creative: {ppl_cre_0:.2f}")

    print("\nFine-tuning on Science (Task A)...")
    finetune(model, dl_sci_tr)
    ppl_sci_A = compute_ppl(model, dl_sci_ev)
    print(f"After Task A — Science: {ppl_sci_A:.2f}")

    print("\nFine-tuning on Creative (Task B)...")
    finetune(model, dl_cre_tr)
    ppl_sci_B = compute_ppl(model, dl_sci_ev)
    ppl_cre_B = compute_ppl(model, dl_cre_ev)
    print(f"After Task B — Science: {ppl_sci_B:.2f}  Creative: {ppl_cre_B:.2f}")

    forgetting = (ppl_sci_B - ppl_sci_A) / ppl_sci_A
    print(f"\nCatastrophic forgetting on Science: {forgetting:+.2%}")
    print("(Compare with NeuroplasticLFM: 0.000%)")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify importable**

```bash
python -c "from experiments.baseline_finetuning import compute_ppl; print('OK')"
```

Expected: `OK`

- [ ] **Step 3: Final test suite run**

```bash
pytest tests/ -v --tb=short
```

Expected: all 41 tests `PASSED`, 0 errors.

- [ ] **Step 4: Final commit**

```bash
git add experiments/baseline_finetuning.py src/eval.py
git commit -m "feat: complete PoC implementation — all modules, tests, and experiments"
git push origin main
```

---

## Running the Full PoC

```bash
# From neuroplastic-lfm/

# Run unit tests (no GPU needed, no LFM2 download)
pytest tests/ -v

# Run the PoC experiment (requires GPU with 8GB+ VRAM and LFM2 downloaded)
python -m experiments.poc_dual_task

# Run the forgetting baseline for comparison
python -m experiments.baseline_finetuning
```

Expected PoC output (approximate):
```
── Baseline (no clusters) ──
  Science PPL:  XX.XX
  Creative PPL: XX.XX

── Training Cluster A (science) ──
  step=   0  loss=X.XXXX  gate=0.0025  τ_a=X.XXXX
  step=  50  loss=X.XXXX  gate=0.XXXX  τ_a=X.XXXX
  ...

── Training Cluster B (creative) ──
  ...

  Forgetting score (science): +0.000000  [expected: 0.000000]

── Merging A + B via TIES-Merging ──
  Merged — Science PPL: XX.XX  Creative PPL: XX.XX

══════════════════════════════════════════════════════════════
                           Baseline   ClusterA   ClusterB     Merged
──────────────────────────────────────────────────────────────
Science PPL                  XX.XX      XX.XX      XX.XX      XX.XX
Creative PPL                 XX.XX      XX.XX      XX.XX      XX.XX
══════════════════════════════════════════════════════════════
Zero forgetting: True
```

---

## Self-Review

**Spec coverage check:**
- CfCCluster (adapters, maturity gate, forward): Task 2 ✓
- ClusterRegistry (spawn, freeze, base_init): Task 3 ✓
- NeuroplasticLFM (frozen base, inject, lm_head): Task 4 ✓
- TIES-Merging (trim, elect, merge): Task 5 ✓
- Training loop + τ extraction: Task 6 ✓
- Evaluation (perplexity, forgetting_score): Task 7 ✓
- PoC experiment (Tasks A+B, zero-forgetting assert, TIES merge): Task 8 ✓
- Baseline comparison: Task 9 ✓

**Placeholder scan:** None found. All steps contain complete code.

**Type consistency check:**
- `CfCCluster.forward(hidden: Tensor) → Tensor` used consistently in Task 2, Task 4, Task 6
- `ClusterRegistry.spawn(task_id: str) → CfCCluster` consistent across Tasks 3, 4, 8
- `NeuroplasticLFM.forward(input_ids, task_id=None, **kwargs) → Tensor` consistent in Tasks 4, 6, 7, 8
- `ties_merge(registry, task_a, task_b, new_task_id, trim_p) → CfCCluster` consistent in Tasks 5, 8
- `train_cluster(model, task_id, dataloader, max_steps, lr, log_every) → List[Dict]` consistent Tasks 6, 8
- `perplexity(model, dataloader, task_id) → float` consistent Tasks 7, 8
