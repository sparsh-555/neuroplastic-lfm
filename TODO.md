# NeuroplasticLFM — TODO

## Done

- [x] Mid-model injection at layer 8 via forward hook
- [x] Variable τ dynamics via `pos_proj` positional embedding
- [x] TIES-Merging on CfC task vectors
- [x] `baseline_finetuning.py` — catastrophic forgetting demo (+138%)
- [x] `lora_baseline.py` — lr sweep [1e-5, 5e-5, 1e-4, 3e-4], best-of-sweep reported
  - Result: best lr=1e-5, sci 2.35, cre 3.70, forgetting -0.24%
  - LoRA at proper lr is better than NeuroplasticLFM on raw PPL (2.35 vs 2.42)
  - Paper narrative repositioned: architectural guarantee + 2.4× fewer params + label-free
- [x] `adaptive_spawning.py` — autonomous spawning with recalibrating baseline
  - Cooldown + buffer reset implemented; recalibration after each spawn
  - 2 spawns on creative (legitimate: sub-distribution diversity); creative 4.36→3.87
- [x] Multi-seed infrastructure — seed flows NeuroplasticLFM → ClusterRegistry → AutoNCP
- [x] Run 007 results documented

## P0: Needs RunPod re-run (code is ready, just needs execution)

- [ ] **Multi-seed re-run** — `git pull && python experiments/multi_seed_eval.py`
  - Seed fix pushed (4899f03); previous run gave ±0.00 because AutoNCP seed was hardcoded
  - Should now show real variance across seeds 0, 1, 2

## P1: Critical before professor meeting or any submission

- [ ] **Run sequential_tasks.py on RunPod** — code is ready, needs execution
  - `git pull && PYTHONPATH=/neuroplastic-lfm python experiments/sequential_tasks.py`
  - 5 tasks: science → creative → math → coding → history (Alpaca keyword filter)
  - Sequential LoRA: one shared adapter lr=1e-5, optimizer persists across tasks
  - NeuroplasticLFM: one new frozen CfC cluster per task
  - Metric: task-1 (science) PPL trajectory + full backward-transfer matrix
  - Expected: LoRA forgetting compounds; NeuroplasticLFM stays at exact 0
  - ~36 min total (5×2 runs × 500 steps)

- [ ] **Visualization notebook** (`notebooks/visualize_tau.ipynb`)
  - Forgetting comparison bar chart across methods (full FT / LoRA / NeuroplasticLFM)
  - Gate curve over training steps (0.047 → 0.031)
  - τ dynamics (pos_proj weights across sequence positions)
  - CfC wiring diagram

## P2: Required for NeurIPS full paper

### Baselines

- [ ] **GainLoRA** (NeurIPS 2025) — most direct competitor; uses gating on LoRA branches
  - Differentiator: GainLoRA gates weight-space LoRA updates; we gate CfC liquid dynamics
    injected mid-model with position-dependent time constants
  - Need either comparison results or a clear differentiation paragraph
- [ ] **O-LoRA / OPLoRA** (Oct 2025) — orthogonal projection LoRA, mathematical forgetting guarantee
  - Direct comparison to our architectural zero-forgetting claim
- [ ] **EWC-LoRA** (ICLR 2026) — EWC + LoRA, replaces raw EWC as regularization baseline
  - Code: github.com/Yaoyz96/low-rank-cl
- [ ] **ASO-LoRA** — multi-LoRA no task labels (comparable to our autonomous spawning)

### Benchmarks

- [ ] **Replace Alpaca keyword filtering with TRACE or Seq-GLUE**
  - All 2025 CL-for-LLM papers use TRACE (8-task) or Seq-GLUE; keyword splits will be
    the first reviewer comment
  - Needed for multi-task experiment above anyway

### Ablations

- [ ] `inject_at` positions: 4, 6, 8, 10, 12 — justify layer 8 empirically
- [ ] Cluster sizes: CLUSTER_DIM 16, 32, 64, 128
- [ ] Gate initialization: -5.0, -3.0, -1.0
- [ ] Statistical significance — t-test / Wilcoxon across 3 seeds (needs multi-seed re-run first)

## P3: Polish

- [ ] **Related work differentiation paragraph** — GainLoRA vs NeuroplasticLFM
- [ ] Fix adaptive spawning calibration — science stream never triggers; calibrate on
  neutral held-out examples so science also triggers a spawn
- [ ] Paper writeup (method, results, related work sections)
- [ ] Update `PLAN.md` — currently says 10 GatedConv + 6 GQA; actual is interleaved
- [ ] Document LoRA lr sweep results in `results/` (currently only in RunPod stdout)
