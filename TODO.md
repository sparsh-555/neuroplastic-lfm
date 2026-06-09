# NeuroplasticLFM — TODO

## Done

- [x] Mid-model injection at layer 8 via forward hook
- [x] Variable τ dynamics via `pos_proj` positional embedding
- [x] TIES-Merging on CfC task vectors
- [x] `baseline_finetuning.py` — catastrophic forgetting demo (+138%)
- [x] `lora_baseline.py` — lr sweep [1e-5, 5e-5, 1e-4, 3e-4], best lr=1e-5: sci 2.35, cre 3.70, forgetting -0.24%
- [x] `adaptive_spawning.py` — cooldown + buffer reset + recalibrating baseline; 2 spawns on creative
- [x] Multi-seed infrastructure — seed flows NeuroplasticLFM → ClusterRegistry → AutoNCP; confirmed real variance (sci 2.41 ± 0.01)
- [x] `sequential_tasks.py` — 3-way comparison: sequential LoRA vs per-task LoRA vs NeuroplasticLFM on 5 tasks
- [x] Run 007 results documented

---

## Venue Reality Check

**NeurIPS 2026 main track: CLOSED** (deadline was May 6, 2026).

Realistic targets:
| Venue | Deadline | Notes |
|---|---|---|
| EMNLP 2026 | ~June–July 2026 | Check ARR deadline — may still be open |
| NeurIPS 2026 Workshop | ~Sept–Oct 2026 | ContinualAI or LoRA workshops |
| ICLR 2027 | ~Oct 2026 | 4 months away — achievable with focused work |
| ICML 2027 | ~Feb 2027 | Most time, but field moves fast |

**Recommendation:** Target ICLR 2027. Use the next 4 months to fix the two blockers below.

---

## P0: BLOCKERS — Fix these before anything else

### 1. Evaluation pipeline rewrite (metric + benchmark)
**This single item blocks everything else.** Every 2025-2026 CL-LLM paper uses accuracy on NLP classification tasks. PPL cannot be compared against any existing work.

**Benchmark to use:** Long Sequence Benchmark (same as CaLoRA NeurIPS 2025)
- Task 1: Yelp (sentiment)
- Task 2: IMDB (sentiment)
- Task 3: BoolQA (question answering)
- Task 4: MultiRec (question answering)
- Task 5: DBpedia (topic classification)

Or TRACE-8 (8 tasks, more common in 2025 papers).

**Metrics to implement:**
- **AP** (Average Performance): `1/T × Σ Acc_{T,Ti}` — mean accuracy across all tasks after final task
- **F.Ra** (Forgetting Rate): `1/(T-1) × Σ (max_k Acc_{k,Ti} − Acc_{T,Ti})` — how much was forgotten
- **BWT** (Backward Transfer): `1/(T-1) × Σ (Acc_{T,Ti} − Acc_{Ti,Ti})` — net effect of later tasks on earlier
- **FWT** (Forward Transfer): `1/(T-1) × Σ (Acc_{Ti,Ti} − Acc_{0,Ti})` — whether earlier tasks helped later ones

**Impact:** Once metrics/benchmark are right, all other experiments (ablations, baselines, multi-seed) are run once and done correctly.

### 2. Run sequential_tasks.py on RunPod (ready, needs execution)
```bash
git pull && PYTHONPATH=/neuroplastic-lfm python experiments/sequential_tasks.py
```
Expected output: LoRA forgetting compounds over 5 tasks; per-task LoRA and NeuroplasticLFM stay at 0.
~50 min total (5 tasks × 3 methods × 500 steps each).

---

## P1: Core Architecture Validity

- [ ] **CfC vs MLP ablation** — most important ablation
  - Build a 187K MLP adapter (same `adapter_in → hidden → adapter_out`, same injection at layer 8)
  - Train on same tasks, same steps, compare AP / F.Ra / BWT
  - If MLP ≈ CfC on all metrics: the CfC contribution needs reframing
  - If CfC > MLP: this is the differentiating result

- [ ] **Maturity gate direction** — architectural coherence issue
  - Gate starts at sigmoid(-3) ≈ 0.047 and decreases to ~0.031 during training
  - This means cluster injection *weakens* as training proceeds (wrong direction for "maturity")
  - Either: rename to "dampening gate" and justify the correction-shrinking interpretation
  - Or: flip the gate design so it opens (sigmoid initialised to < 0 with positive gradient pressure)

- [ ] **Label-free spawning validation on 5+ tasks**
  - `sequential_tasks.py` manually calls `spawn_cluster(task_name)` — this requires knowing task boundaries
  - Need an end-to-end run where the model encounters a 5-task stream with no labels and adaptive_spawning drives all spawns
  - Science stream never triggered in 2-task demo; fix calibration for more domains first

---

## P2: Baselines (required for paper)

- [ ] **O-LoRA** — has zero-forgetting guarantee via orthogonal subspaces; most direct peer
  - Code: `github.com/cmnfriend/O-LoRA`
  - This is the baseline reviewers will ask about most; "sequential LoRA" will be seen as a strawman
- [ ] **CaLoRA** (NeurIPS 2025) — PEFT-based CL with *backward* knowledge transfer; current SOTA
  - Missed in original baseline list; now the paper to beat
- [ ] **GainLoRA** (NeurIPS 2025) — most direct structural competitor (gating on LoRA branches)
- [ ] **EWC-LoRA** (ICLR 2026) — `github.com/Yaoyz96/low-rank-cl`
- [ ] **InfLoRA** / **SD-LoRA** — NeurIPS 2025 orthogonal LoRA variants (brief comparison or mention in related work)

---

## P3: Generalization and Significance

- [ ] **Second base model** — test on LLaMA-3-8B or Mistral-7B
  - LFM2.5-1.2B is unknown outside LiquidAI; results need to transfer to a well-known model
- [ ] **5 seeds + statistical significance** — t-test or Wilcoxon across seeds
  - Current 3-seed run gives ±0.01 variance; good, but formal test needed
- [ ] **Inference latency analysis** — measure per-token latency with N=1, 3, 5 clusters active
  - N clusters = N forward hook evaluations; need to show overhead is acceptable

---

## P4: Ablations

- [ ] `inject_at` positions: 4, 6, 8, 10, 12 — justify layer 8 empirically
- [ ] Cluster sizes: CLUSTER_DIM 16, 32, 64, 128
- [ ] Gate initialization: -5.0, -3.0, -1.0
- [ ] With/without TIES merging (compare vs simple weight averaging)

---

## P5: Polish

- [ ] Visualization notebook (`notebooks/visualize_tau.ipynb`)
  - Forgetting bar chart: full FT / seq LoRA / per-task LoRA / O-LoRA / NeuroplasticLFM
  - Gate curve; τ dynamics; CfC wiring diagram
- [ ] Related work differentiation: CaLoRA vs NeuroplasticLFM; GainLoRA vs NeuroplasticLFM
- [ ] Paper writeup (method, results, related work sections)
- [ ] Replace PLAN.md — currently incorrect about layer types
- [ ] Document LoRA lr sweep results in `results/` (currently only in RunPod stdout)
