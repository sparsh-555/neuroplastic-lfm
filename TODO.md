# NeuroplasticLFM — TODO

## P0: Run 7 Fixes (done)

- [x] Add `cooldown_steps` to `AdaptiveSpawnController` — prevents over-spawning after each trigger
- [x] Clear `ExampleBuffer` after spawn — next cluster trains on post-spawn examples only
- [ ] Re-run `adaptive_spawning.py` on RunPod — verify single spawn per distribution

## P1: Critical for any submission

- [ ] **LoRA lr sweep** — rewrite `lora_baseline.py` to sweep [1e-5, 5e-5, 1e-4, 3e-4], report best-of-sweep
  - Current comparison at lr=3e-4 is not credible; LoRA on 1.2B needs ~1e-4 or lower
- [ ] **Multiple seeds** — run `poc_dual_task.py` with seeds 0, 1, 2; report mean ± std on all PPL numbers
- [ ] **Visualization notebook** (`notebooks/visualize_tau.ipynb`)
  - Gate curve over training steps
  - τ dynamics (pos_proj weights across sequence positions)
  - Forgetting comparison bar chart (full FT vs LoRA best vs NeuroplasticLFM)
  - CfC wiring diagram

## P2: Required for NeurIPS full paper

### Baselines — field has moved significantly (NeurIPS 2025 / ICLR 2026)

- [ ] **GainLoRA** (NeurIPS 2025) — uses gating to integrate LoRA branches, most direct competitor
  - Our maturity gate is conceptually related; need differentiation paragraph in related work
  - Differentiator: GainLoRA gates LoRA weight updates; we gate CfC liquid dynamics injected mid-model
- [ ] **O-LoRA / OPLoRA** (Oct 2025) — constrains LoRA updates to orthogonal complement of pre-trained weight singular vectors
  - Mathematical forgetting guarantee, direct comparison to our architectural zero-forgetting
- [ ] **EWC-LoRA** (ICLR 2026) — EWC + LoRA combined, replaces raw EWC as the regularization baseline
  - Code: github.com/Yaoyz96/low-rank-cl
- [ ] **ASO-LoRA** — multi-LoRA with soft orthogonality, no task labels (comparable to our autonomous spawning)
- [ ] Drop raw EWC, PackNet, A-GEM — superseded by LoRA-CL variants above

### Benchmarks

- [ ] **Replace Alpaca keyword filtering with TRACE or Seq-GLUE**
  - TRACE and Seq-GLUE are the standard benchmarks in all 2025 CL-for-LLM papers
  - Alpaca keyword splits will be the first reviewer criticism after GainLoRA
  - Seq-GLUE: sequential GLUE tasks (SST-2, MNLI, QQP, etc.)
  - TRACE: 8-task CL benchmark for instruction-following LLMs

### Ablations

- [ ] `inject_at` positions: 4, 6, 8, 10, 12 — justify layer 8 empirically
- [ ] Cluster sizes: CLUSTER_DIM 16, 32, 64, 128
- [ ] Gate initialization: -5.0, -3.0, -1.0
- [ ] Statistical significance — t-test or Wilcoxon across 3 seeds

## P3: Polish

- [ ] **Related work differentiation paragraph** — GainLoRA vs NeuroplasticLFM
  - GainLoRA: gated integration of LoRA branches (weight-space gating, no liquid dynamics)
  - NeuroplasticLFM: CfC clusters with liquid time constants inject at mid-model; PNN isolation = architectural (not learned) zero-forgetting guarantee
- [ ] Fix science stream calibration in `adaptive_spawning.py`
  - Calibrate on neutral held-out examples with lower loss, not mixed task data
  - OR lower threshold to 1.10 so science also triggers
- [ ] Paper writeup (method section, results section, related work)
- [ ] Update `PLAN.md` with correct layer_types (currently wrong — says 10 GatedConv + 6 GQA; actual is interleaved)
