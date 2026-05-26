# Run 001 — First End-to-End PoC

**Date:** 2026-05-27  
**Hardware:** RunPod NVIDIA L4 (22 GB VRAM)  
**Commit:** `31bf7cf`  
**Config:** `gate_init=-6.0`, `max_steps=300`, `batch_size=4`, `lr=3e-4`, `MAX_LENGTH=256`  
**Dataset:** `yahma/alpaca-cleaned` — 150 train / 50 eval per task

---

## Pipeline

```
Base: LiquidAI/LFM2.5-1.2B-Instruct (frozen, bfloat16)
Task A: science/math Q&A keywords
Task B: creative writing keywords
Merge: TIES-Merging (trim_p=0.20)
```

---

## Results

```
── Baseline (no clusters) ──
  Science PPL:  3.15
  Creative PPL: 3.98

── Training Cluster A (science) — 300 steps ──
  step=   0  loss=0.8555  gate=0.0025
  step=  50  loss=1.1797  gate=0.0025
  step= 100  loss=1.1406  gate=0.0026
  step= 150  loss=1.2734  gate=0.0026
  step= 200  loss=1.2969  gate=0.0027
  step= 250  loss=0.7500  gate=0.0028
  Science PPL: 3.15  Creative PPL: 3.98

── Training Cluster B (creative) — 300 steps ──
  step=   0  loss=2.0469  gate=0.0025
  step=  50  loss=1.7422  gate=0.0025
  step= 100  loss=1.8438  gate=0.0026
  step= 150  loss=1.3906  gate=0.0026
  step= 200  loss=1.0312  gate=0.0027
  step= 250  loss=1.5078  gate=0.0028
  Science PPL: 3.15  Creative PPL: 3.98

  Forgetting score (science): +0.000000  [expected: 0.000000]

── Merging A + B via TIES-Merging ──
  Merged — Science PPL: 3.15  Creative PPL: 3.98

══════════════════════════════════════════════════════════════
                             Baseline    ClusterA    ClusterB      Merged
──────────────────────────────────────────────────────────────
Science PPL                      3.15        3.15        3.15        3.15
Creative PPL                     3.98        3.98        3.98        3.98
══════════════════════════════════════════════════════════════
Zero forgetting: True
```

---

## Analysis

### What worked

- Full pipeline ran end-to-end without errors.
- Zero forgetting confirmed: Science PPL identical before and after Task B training (`+0.000000`). This is architecturally guaranteed — Cluster A is frozen before Cluster B spawns, and the base model is never modified.
- TIES-Merging completed successfully.

### What didn't work

PPL is unchanged from baseline across all conditions. The cluster had no measurable effect on either task.

**Root cause:** `gate_init=-6.0` creates a bootstrapping problem. `sigmoid(-6) ≈ 0.0025`, so the cluster residual contributes only 0.25% of its output to the hidden state. The gradient signal through the maturity gate is scaled by the same factor (`sigmoid'(-6) ≈ 0.0025`), making it extremely slow to open. Over 300 steps the gate moved from `0.0025` → `0.0028` — effectively no change. The cluster never escaped the suppression regime.

The loss trajectory (0.85 → 1.18 → 0.75) shows noisy fluctuation rather than consistent decrease, consistent with an optimizer hunting for signal through near-zero-scaled outputs.

---

## Follow-up Config (Run 002)

Adjusted in commit `a96cad1`:

| Parameter | Run 001 | Run 002 |
|---|---|---|
| `gate_init` | `-6.0` | `-3.0` |
| `max_steps` | `300` | `500` |

`sigmoid(-3) ≈ 0.047` gives ~20× stronger gradient signal through the gate while still starting small. 500 steps matches the original PLAN spec. Expected outcome: visible PPL reduction on trained task, measurable gate opening curve, τ distribution shift.
