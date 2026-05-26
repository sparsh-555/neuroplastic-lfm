# Neuroplastic LFM — Implementation Plan

> Dual-axis neuroplasticity on LFM2-1.2B: structural cluster spawning (PNN-style) + functional adaptive τ dynamics (LTC-derived) with TIES-Merging for cluster fusion.

---

## Research Context

This PoC applies the established **expand-and-freeze paradigm** (SEMA, MoCL, HAM) to a novel substrate — CfC clusters instead of LoRA adapters — which unlocks a second axis of neuroplasticity (adaptive τ dynamics) that is structurally impossible in weight-decomposition methods. TIES-Merging is extended to the CfC weight domain for the first time.

### Novelty Gap (adversarially verified)

| Claim | Prior Art | Status |
|---|---|---|
| Spawn+freeze structural pattern | SEMA, MoCL, HAM (2023–2025) | **Known** — positioned as baseline paradigm |
| Zero-init maturity gate | LLaMA-Adapter (2023), HAM importance scalar | **Known** — adopted and cited, not a contribution |
| **CfC as the spawned module** (not LoRA) | None found | **Novel** |
| **Per-token recurrent processing with hidden state** | LoRA has no recurrence | **Novel** |
| **Adaptive τ per token** via CfC time_a/time_b (functional neuroplasticity) | Impossible in LoRA | **Novel** |
| **TIES-Merging on CfC weight vectors** | All prior merging targets transformer/LoRA | **Novel** |
| **Cross-substrate lateral connection** (gated-conv base ↔ CfC cluster) | No prior cross-substrate work | **Novel** |

### Correct Paper Framing

> "We apply the established expand-and-freeze paradigm to a CfC substrate, replacing LoRA adapters with NCP-wired CfC clusters. This substitution unlocks functional neuroplasticity — input-dependent adaptive time constants that evolve per token during sequence processing — which is structurally impossible in static linear decomposition methods. We further demonstrate the first application of TIES-Merging to CfC weight vectors, enabling principled cluster fusion with interference resolution."

**HAM** (ICLR 2026) is the closest existing system: frozen LLM + spawn LoRA-per-task + importance scalar + hierarchical merge. This project = HAM with CfC substrate. That framing is precise and will hold up under review.

---

## Architecture

### Base Model

- **Model**: `LiquidAI/LFM2-1.2B` (HuggingFace)
- **Parameters**: 1.17B, fully frozen during cluster training
- **Architecture**: 16 blocks — 10 double-gated LIV convolution blocks + 6 GQA blocks
- **Dimensions**: `hidden_size=2560`, `intermediate_size=12288`, `num_attention_heads=32`, `vocab_size=65536`
- **Context**: up to 32,768 tokens

### Forward Pass Blueprint

```
Input tokens
      ↓
╔══════════════════════════╗
║  LFM2-1.2B  [FROZEN]    ║  hidden_size = 2560
║  10× GatedConv           ║
║  6× GQA                  ║
╚══════════╤═══════════════╝
           │  hidden_states: (B, L, 2560)
           │
           ├─── adapter_in: Linear(2560 → 64)      ┐
           │         ↓                              │
           │    [CfC cluster — TRAINABLE]           │  per-task cluster
           │    AutoNCP(units=64, output=16)         │  ~305K params
           │    processes tokens recurrently         │
           │    adaptive τ per token (time_a, time_b)│
           │         ↓                              │
           │    adapter_out: Linear(16 → 2560)      │
           │    maturity_gate: scalar, init=-6.0    ┘
           │              │
           └── h + sigmoid(α) × cluster_output ──→ [LM Head] → logits
```

### CfC Cluster Specification

- **Wiring**: `AutoNCP(units=64, output_size=16, sparsity_level=0.5)`
  - 64 total neurons (inter + command + motor), sparse Erdos-Rényi connectivity
  - 16 motor neurons = cluster output dimension
- **Input adapter**: `Linear(2560, 64)` — projects base hidden state to CfC sensory input
- **Output adapter**: `Linear(16, 2560)` — projects CfC motor output to residual space
- **Maturity gate**: `nn.Parameter(torch.full((1,), -6.0))` → `sigmoid(-6) ≈ 0.002`
  - Adopted from LLaMA-Adapter (Zhang et al., 2023) zero-init gating mechanism
  - Cluster starts with negligible influence; gate opens as training progresses
- **Total trainable params per cluster**: ~305K (0.026% of base model)

### Functional Neuroplasticity (the τ dynamics)

Inside each CfC cell, per forward pass:
```
t_interp = sigmoid(time_a(x) * ts + time_b(x))   # input-dependent time constant
new_h    = ff1 * (1 - t_interp) + t_interp * ff2  # τ-gated state update
```

`time_a` and `time_b` are learned linear projections that produce a **different effective τ for each token, conditioned on the token's content**. This is the closed-form approximation of LTC's `τ_sys = 1 / (1/τ + f(x, I, t, θ))`. Watching these parameters evolve during training is the functional neuroplasticity signal.

### Zero-Forgetting Guarantee

Base model frozen + each cluster trained independently + prior clusters frozen on spawn = catastrophic forgetting is architecturally impossible. Task k accuracy is always served by base + cluster_k, which nothing touches after training.

---

## TIES-Merging for Cluster Fusion

Given two trained clusters A and B with shared initialization θ_init:

```
1. Task vectors:   τ_A = θ_A − θ_init,   τ_B = θ_B − θ_init
2. Trim:           zero entries below 20th percentile of |τ| (per-cluster)
3. Elect:          for each param, resolve sign conflicts by magnitude dominance
4. Merge:          θ_AB = θ_init + mean(τ_A', τ_B')  where signs agree
```

Applied to CfC weight vectors (`ff1`, `ff2`, `time_a`, `time_b`, `w_tau`, `A`). Sparsity masks are fixed (not `requires_grad`) and skipped in merge. This is the first application of TIES-Merging outside of transformer/LoRA weight spaces.

---

## Experiment Design

### Tasks

| | Task A | Task B |
|---|---|---|
| Domain | Science / math Q&A | Creative writing |
| Dataset | `yahma/alpaca-cleaned` (filtered) | `tatsu-lab/alpaca` (filtered) |
| Size | 200 train / 50 eval examples | 200 train / 50 eval examples |
| Metric | Perplexity on held-out split | Perplexity on held-out split |

### Experiment Sequence

```
1. Load frozen LFM2-1.2B
2. Baseline: perplexity on Task A and Task B (no clusters)
3. spawn_cluster("task_a") → train 500 steps → eval ppl_A, ppl_B
4. spawn_cluster("task_b") → train 500 steps → eval ppl_A, ppl_B
   → Key check: ppl_A unchanged (zero forgetting)
5. ties_merge("task_a", "task_b", "task_ab") → eval ppl_A, ppl_B
   → Key check: merged < individual on both tasks (or graceful degradation)
6. Baseline comparison: full fine-tune on Task B → show forgetting on Task A
```

### Metrics

| Metric | Measures |
|---|---|
| Perplexity on Task A (before/after Task B training) | Forgetting = 0 |
| Perplexity vs fine-tuned baseline | Plasticity |
| Perplexity of merged cluster vs individual clusters | Merge quality |
| `sigmoid(α)` over training steps | Maturity gate opening |
| `time_a`, `time_b` distribution shift over training | Functional neuroplasticity |
| Trainable params per cluster vs LoRA-r8 | Parameter efficiency |

### Baselines to Compare Against

1. **Full fine-tuning** — catastrophic forgetting baseline
2. **LoRA (r=8)** — parameter-efficient but still forgetting
3. **HAM** (if reproducible) — closest prior art, LoRA substrate

---

## File Structure

```
neuroplastic-lfm/
├── PLAN.md                    ← this file
├── requirements.txt
├── src/
│   ├── __init__.py
│   ├── cluster.py             # CfCCluster: wiring + CfC + adapters + maturity gate
│   ├── registry.py            # ClusterRegistry: spawn, get, freeze, save/load
│   ├── model.py               # NeuroplasticLFM: frozen LFM2 + registry
│   ├── merge.py               # TIES-Merging for CfC weight vectors
│   ├── train.py               # train_cluster(model, task_id, dataloader)
│   └── eval.py                # perplexity, forgetting_score, tau_stats
├── experiments/
│   ├── poc_dual_task.py       # main PoC: spawn A, spawn B, verify forgetting=0, merge
│   └── baseline_finetuning.py # naive fine-tune for forgetting comparison
├── tests/
│   ├── test_cluster.py        # forward shape, maturity gate init, gradient flow
│   ├── test_registry.py       # spawn+freeze invariant, no grad leakage to base
│   └── test_merge.py          # TIES correctness on toy weight vectors
└── notebooks/
    └── visualize_tau.ipynb    # τ dynamics, wiring diagram, forgetting charts
```

---

## Implementation Phases

### Phase 0 — Environment Setup `(Day 1)`

```bash
pip install transformers ncps torch datasets accelerate
# If LFM2 not yet in stable transformers:
pip install git+https://github.com/huggingface/transformers
```

Verification:
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained("LiquidAI/LFM2-1.2B", torch_dtype=torch.float16)
out = model(input_ids, output_hidden_states=True)
assert out.hidden_states[-1].shape[-1] == 2560
```

**Gate**: hidden states shape `(B, L, 2560)` confirmed → proceed.

---

### Phase 1 — Base Wrapper `(Day 1–2)`

**`src/model.py`** — `NeuroplasticLFM(nn.Module)`

- Load LFM2, freeze all base parameters
- `forward(input_ids, task_id=None)`:
  - Get `hidden_states[-1]` via `output_hidden_states=True`
  - If `task_id` active: `h = h + registry.get(task_id)(h)`
  - Pass through `lm_head`

**Gate**: frozen forward produces identical outputs to raw LFM2.

---

### Phase 2 — CfC Cluster `(Day 2–3)`

**`src/cluster.py`** — `CfCCluster(nn.Module)`

```python
from ncps.torch import CfC
from ncps.wirings import AutoNCP

class CfCCluster(nn.Module):
    BASE_DIM, CLUSTER_DIM, MOTOR_DIM = 2560, 64, 16

    def __init__(self, seed=42):
        wiring           = AutoNCP(64, 16, sparsity_level=0.5, seed=seed)
        self.adapter_in  = nn.Linear(self.BASE_DIM, self.CLUSTER_DIM)
        self.cfc         = CfC(self.CLUSTER_DIM, wiring, batch_first=True, return_sequences=True)
        self.adapter_out = nn.Linear(self.MOTOR_DIM, self.BASE_DIM)
        self.maturity    = nn.Parameter(torch.full((1,), -6.0))

    def forward(self, hidden):          # (B, L, 2560)
        x       = self.adapter_in(hidden)                     # (B, L, 64)
        h0      = torch.zeros(x.size(0), 64, device=x.device)
        out, _  = self.cfc(x, h0)                            # (B, L, 16)
        delta   = self.adapter_out(out)                       # (B, L, 2560)
        return torch.sigmoid(self.maturity) * delta
```

**Gate**: `cluster(h).shape == h.shape`, `sigmoid(maturity) ≈ 0.002`.

---

### Phase 3 — Cluster Registry `(Day 3)`

**`src/registry.py`** — `ClusterRegistry(nn.Module)`

- `spawn(task_id)`: create `CfCCluster`, store `θ_init`, freeze all existing clusters
- `get(task_id)`: return cluster
- `freeze_all()`: called before spawning a new cluster

**Gate**: after spawning cluster B, cluster A's parameters have `requires_grad=False`.

---

### Phase 4 — Training Loop `(Day 4)`

**`src/train.py`** — `train_cluster(model, task_id, dataloader, max_steps=500, lr=3e-4)`

- Only cluster parameters + adapters + maturity gate are optimized (base frozen)
- Log every 50 steps: loss, `sigmoid(α)`, τ statistics (`time_a.weight.mean()`)
- AdamW, `clip_grad_norm_=1.0`

**Gate**: loss decreasing, `sigmoid(α)` rising from ~0.002 toward ~0.3–0.7.

---

### Phase 5 — TIES Merge `(Day 5)`

**`src/merge.py`** — `ties_merge(registry, task_a, task_b, new_id, trim_p=0.2)`

- Compute task vectors τ_A, τ_B
- Trim, elect sign, average non-conflicting
- Materialize new `CfCCluster` with merged weights

**Gate**: merged cluster perplexity within 20% of individual clusters on both tasks.

---

### Phase 6 — Evaluation `(Day 6–7)`

**`src/eval.py`** — perplexity, forgetting score, τ shift

Run full experiment sequence, collect all metrics.

**Gate**: Task A perplexity unchanged after Task B training (forgetting = 0).

---

### Phase 7 — Visualization `(Day 7–8)`

**`notebooks/visualize_tau.ipynb`**

1. Forgetting comparison bar chart (Full FT vs LoRA vs NeuroplasticLFM)
2. τ dynamics line chart (time_a/time_b distribution at step 0, 250, 500)
3. Maturity gate α curve over training steps
4. NCP wiring diagram (`wiring.draw_graph()`)

---

## Hardware Requirements

| Setup | VRAM | Notes |
|---|---|---|
| LFM2-1.2B FP16 | ~2.5GB | Model weights |
| CfC cluster (×1) | ~5MB | 305K params |
| Inference batch=1 | ~4GB | With KV cache |
| Cluster training batch=4 | ~8GB | AdamW on 305K params only |
| **Minimum** | **8GB** | RTX 3070 / Colab T4 |
| **Comfortable** | **16GB** | RTX 3090 / A10 / Colab A100 |

Base model never gets optimizer states (frozen), so cluster training is memory-negligible.

---

## Key References

| Paper | Role |
|---|---|
| LFM2 Technical Report (Liquid AI, 2024) | Base model architecture |
| STAR: Synthesis of Tailored Architectures (arXiv:2411.17800) | Why LFM2 gated-conv blocks are LTC-derived |
| Closed-Form Continuous-Time Neural Networks (Nature MI, 2022) | CfC cluster unit |
| Neural Circuit Policies (Nature MI, 2020) | NCP sparse wiring for cluster topology |
| Progressive Neural Networks (Rusu et al., 2016) | Structural spawn+freeze+lateral pattern |
| Lifelong Learning with DEN (Yoon et al., 2018) | Expansion trigger criteria |
| TIES-Merging (NeurIPS 2023) | Cluster merge algorithm |
| Task Arithmetic (ICLR 2023) | Task vector math |
| Loss of Plasticity (Nature, 2024) | Problem statement |
| Neuroplastic Expansion (ICLR 2025, arXiv:2410.07994) | SOTA fix being compared against |
| LLaMA-Adapter (arXiv:2303.16199) | Source of zero-init gating mechanism |
| HAM (ICLR 2026 sub.) | Closest prior art (LoRA substrate, not CfC) |
| SEMA (UNSW/CSIRO, 2024) | Closest structural prior art |
