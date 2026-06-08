# NeuroplasticLFM — TODO

## P0: Run 7 Fixes (immediate)

- [x] Add `cooldown_steps` to `AdaptiveSpawnController` — prevents over-spawning after each trigger
- [x] Clear `ExampleBuffer` after spawn — next cluster trains on post-spawn examples only
- [ ] Re-run `adaptive_spawning.py` and verify single spawn per distribution

## P1: Critical for any submission

- [ ] Fix LoRA baseline — sweep lr (1e-5, 5e-5, 1e-4, 3e-4), report best-of-sweep result
  - Current lr=3e-4 is wrong for LoRA on 1.2B model; comparison is not credible as-is
- [ ] Multiple seeds — run `poc_dual_task.py` with seeds 0, 1, 2; report mean ± std on PPL
- [ ] Build visualization notebook (`notebooks/visualize_tau.ipynb`)
  - Gate curve over training steps
  - τ dynamics (pos_proj weights across positions)
  - Forgetting comparison bar chart (full FT vs LoRA vs NeuroplasticLFM)
  - CfC wiring diagram

## P2: Required for NeurIPS full paper

- [ ] Replace Alpaca keyword filtering with proper CL benchmark
  - Option A: MMLU domain splits (science, humanities, social science, STEM)
  - Option B: Split the model across SuperGLUE tasks
  - Current "science vs creative by keyword" is too weak a task distinction
- [ ] Add proper continual learning baselines
  - EWC (Elastic Weight Consolidation)
  - PackNet (progressive weight pruning)
  - A-GEM (Averaged Gradient Episodic Memory)
- [ ] Ablation studies
  - `inject_at` positions: 4, 6, 8, 10, 12 — why layer 8?
  - Cluster sizes: CLUSTER_DIM 32, 64, 128
  - Gate initialization: -5.0, -3.0, -1.0
- [ ] Statistical significance — t-test or Wilcoxon across seeds

## P3: Polish

- [ ] Fix science stream calibration in `adaptive_spawning.py`
  - Calibrate on neutral "easy" examples rather than mixed task data
  - OR lower threshold to 1.10 so science also triggers
  - (Science not triggering is correct behavior but unclear for paper narrative)
- [ ] Paper writeup (method section, results section, related work)
- [ ] Update `PLAN.md` with correct layer_types (currently wrong — says 10 GatedConv + 6 GQA)
