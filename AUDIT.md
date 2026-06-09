# NeuroplasticLFM — Experimental Audit

**Date:** June 2026  
**Purpose:** Honest gap analysis before professor meeting / submission. Identifies corners cut, departures from field standard, and missing experiments as of June 2026.

---

## Summary Verdict

The architecture is real and the zero-forgetting property is demonstrated. The prototype works. But the paper is not submission-ready for three structural reasons: (1) we evaluate with perplexity, the field uses accuracy; (2) we test on keyword-filtered Alpaca, the field uses standardized NLP benchmarks; (3) the "label-free" headline claim is not validated end-to-end. Everything else is fixable without changing the architecture.

---

## 🔴 Showstoppers

### S1 — Metric: PPL only, no task accuracy

**What we do:** Measure perplexity on held-out Alpaca examples.

**What the field does (June 2026):**  
Every CL-LLM paper since 2024 — CaLoRA (NeurIPS 2025), O-LoRA, InfLoRA, GainLoRA, SAPT, TRACE — evaluates accuracy on downstream NLP classification tasks using four standard metrics:

| Metric | Formula | Meaning |
|---|---|---|
| AP (Average Performance) | `1/T × Σ Acc_{T,Ti}` | Mean accuracy across all tasks after final training |
| F.Ra (Forgetting Rate) | `1/(T-1) × Σ (max_k Acc_{k,Ti} − Acc_{T,Ti})` | How much was forgotten |
| BWT (Backward Transfer) | `1/(T-1) × Σ (Acc_{T,Ti} − Acc_{Ti,Ti})` | Net effect of later tasks on earlier |
| FWT (Forward Transfer) | `1/(T-1) × Σ (Acc_{Ti,Ti} − Acc_{0,Ti})` | Whether earlier tasks helped later ones |

**Why this matters:** Our PPL numbers (2.35, 2.42, etc.) cannot be put in a comparison table with any existing paper. They use accuracy (e.g., 88.3% on Yelp, 91.2% on IMDB). Reviewers will reject with "compare on standard metrics."

**Fix:** Replace PPL with accuracy-based evaluation on the Long Sequence Benchmark or TRACE.

---

### S2 — Dataset: Alpaca keyword filtering is not a benchmark

**What we do:** Filter Alpaca-Cleaned by keywords like `{"physics", "chemistry", "biology"}` to create pseudo-tasks.

**Problems:**
1. Not reproducible — keyword matching is fuzzy and order-dependent
2. Task overlap — SCIENCE_KEYWORDS includes "calculate" and "theorem" which also appear in MATH_KEYWORDS
3. Not used by any published paper — no comparison is possible
4. Alpaca itself is a synthetic instruction dataset, not a real task benchmark

**What the field uses:**

*Long Sequence Benchmark (used by CaLoRA NeurIPS 2025):*
- Yelp (sentiment) → IMDB (sentiment) → BoolQA (QA) → MultiRec (QA) → DBpedia (topic)

*TRACE-8 (used by many 2024-2025 papers):*
8 diverse tasks including math, coding, medical QA, summarisation

*Seq-GLUE:* Sequential version of standard GLUE tasks

**Fix:** Replace with Long Sequence Benchmark as the primary 5-task setup. TRACE-8 as the 8-task extension for the full paper.

---

### S3 — NeurIPS 2026 main track deadline has passed

**Deadline was:** May 6, 2026 (AoE). Today is June 2026.

**Realistic venues:**
| Venue | Deadline | Recommended? |
|---|---|---|
| EMNLP 2026 | ~June–July 2026 | Check now — might still be open |
| NeurIPS 2026 Workshop (ContinualAI) | ~Sept–Oct 2026 | Good for proof-of-concept |
| ICLR 2027 | ~Oct 2026 | **Best target** — 4 months to fix everything |
| ICML 2027 | ~Feb 2027 | Most time, but field moves fast |

---

## 🟠 High Severity

### H1 — Missed baseline: CaLoRA (NeurIPS 2025)

CaLoRA (Causal-Aware LoRA) was accepted at NeurIPS 2025 and is the current SOTA for PEFT-based CL. It introduces:
- **PaCA** (Parameter-level Counterfactual Attribution): estimates causal effect of LoRA parameters to identify which weights matter for each task
- **Backward knowledge transfer**: new tasks actively improve old task performance (not just avoid forgetting)

This is stronger than our claim (we only guarantee zero forgetting, CaLoRA claims positive backward transfer). Our related work section needs to engage with this directly. Our baseline comparisons must include it.

**Other NeurIPS 2025 papers we missed:**
- **InfLoRA** — constrains new task updates to orthogonal subspace of old task weights (similar to O-LoRA)
- **SD-LoRA** — decouples LoRA magnitude and direction for better continual learning
- **MoE-P** — mixture of experts for CL (relevant: NeuroplasticLFM is also a growing-architecture method)

---

### H2 — Maturity gate is going in the wrong direction

**Current behavior:** Gate logit starts at -3 (sigmoid ≈ 0.047) and decreases to -3.5 (sigmoid ≈ 0.031) over 1000 training steps.

**What this means:** The cluster injects *less* as training proceeds. The injection signal weakens by ~35% over training.

**The problem:** A gate named "maturity" should open as the cluster matures, not close. A decreasing gate suggests the cluster is learning to suppress its own contribution — possibly because the base model's residual stream already captures most of what's needed and the cluster's corrections become noise.

**Options:**
1. Rename to "dampening gate" and argue that a small, precision correction is better than a large, noisy one (the cluster learns to be minimally invasive)
2. Redesign the gate initialization so gradient pressure opens it: start at +3, let training decide direction
3. Investigate whether the cluster is learning at all or just passively decaying

---

### H3 — "Label-free" claim is unvalidated at 5+ tasks

**What `sequential_tasks.py` does:** Calls `model.spawn_cluster("science")`, `model.spawn_cluster("creative")`, etc. — it *knows* the task identity and manually triggers spawning.

**What the label-free claim requires:** The model encounters a data stream with no task labels, the adaptive spawn controller detects distribution shift, and spawns a new cluster autonomously.

**Current adaptive spawning validation:**
- Only tested on 2 tasks (science + creative)
- Science stream never triggered a spawn (only creative did, twice)
- The "trigger" was recalibrated but still failed to detect science as a new task

Until adaptive spawning correctly identifies 3+ distinct tasks from a label-free stream, the paper cannot claim "label-free" as its main differentiator without qualification.

---

### H4 — No CfC vs MLP ablation

**The gap:** We've never tested whether CfC liquid dynamics are doing anything beyond what a plain MLP of the same size would do.

A 187K MLP with:
- `Linear(2048 → 64)` (adapter_in)
- `ReLU()`
- `Linear(64 → 2048)` (adapter_out)
- Same maturity gate
- Same injection point

...would be much simpler, faster, and easier to explain. If it performs the same:
- The CfC contribution vanishes
- The paper needs reframing as "PNN-style growing architecture for LLMs" with CfC as a design option, not the core contribution

If CfC clearly outperforms MLP:
- This is the key differentiating result and should lead the ablation section
- The τ dynamics and liquid time constants are doing meaningful work

This ablation should be run before investing more in the CfC architecture.

---

## 🟡 Moderate Severity

### M1 — Sequential LoRA baseline is a strawman

Our "Sequential LoRA" is the weakest possible LoRA setup: one shared adapter, trained on all tasks continuously, no protection. Against proper CL-LoRA methods (CaLoRA, O-LoRA, InfLoRA), reviewers will say we chose the easiest baseline to beat.

**Fix:** Replace or supplement sequential LoRA with O-LoRA (zero-forgetting guarantee via orthogonal subspaces). This is the honest peer comparison. If NeuroplasticLFM matches O-LoRA on forgetting but with fewer parameters and no task labels, that's a publishable result.

---

### M2 — Only one base model

Results are only on LFM2.5-1.2B, which is obscure outside LiquidAI. All competitive papers test on at least LLaMA-7B or Mistral-7B. Testing on LLaMA-3-8B would take one RunPod run and significantly expand the claim's credibility.

---

### M3 — 3 seeds, no statistical significance

The multi-seed std is ±0.01 PPL, which is good. But with accuracy-based metrics the variance might be higher, and the gap between methods might be smaller. Standard is 5 seeds + t-test or Wilcoxon. This is easy to add once the eval pipeline is rewritten.

---

### M4 — Inference overhead unmeasured

Each CfC cluster adds a forward pass through 187K parameters at every layer-8 activation. With 5 tasks:
- Inference might require routing to the right cluster (needs task label) OR running all 5 clusters and summing (5× overhead)
- Neither approach is measured

This is a practical concern that will be raised in reviews. Measure it.

---

### M5 — Inject_at = 8 not empirically justified

The choice of layer 8 is motivated by "3 attention blocks before lm_head" but this is theoretical. Ablating injection positions 4, 6, 8, 10, 12 is standard for adapter-based papers and would both validate the choice and add a minor contribution.

---

### M6 — TIES-merging marginally hurts individual task PPL

Merged cluster PPL (2.42) is slightly worse than individual cluster PPL (2.41 science). This suggests TIES-merging is not beneficial for our setup. Either: (a) it needs a held-out validation set to tune the trim threshold, or (b) it should be dropped from the main experiments and moved to "future work."

---

## Field Coverage Check (June 2026)

### Papers we're tracking:
- ✅ GainLoRA (NeurIPS 2025) — gating on LoRA branches
- ✅ O-LoRA — orthogonal subspace LoRA
- ✅ EWC-LoRA (ICLR 2026) — EWC + LoRA
- ✅ ASO-LoRA — multi-LoRA, no task labels
- ❌ **CaLoRA (NeurIPS 2025)** — MISSED. Current SOTA. Backward knowledge transfer.
- ❌ **InfLoRA (NeurIPS 2025)** — MISSED. Orthogonal subspace, similar to O-LoRA.
- ❌ **SD-LoRA (NeurIPS 2025)** — MISSED. Magnitude/direction decomposition.
- ❌ **Sparse memory fine-tuning (Oct 2025)** — only 11% F1 drop vs 71% for LoRA. Different approach but competitive result.

### Benchmarks we're not using:
- ❌ Long Sequence Benchmark (CaLoRA's benchmark — Yelp/IMDB/BoolQA/MultiRec/DBpedia)
- ❌ TRACE-8 (most common 2024-2025 benchmark)
- ❌ Seq-GLUE
- ✅ Our keyword-filtered Alpaca — not standard, not reproducible

---

## Action Order

Fix in this order — each step unblocks the next:

```
1. [~1 week]  Rewrite eval pipeline: Long Sequence Benchmark + accuracy + AP/BWT/F.Ra
2. [~1 week]  Re-run all experiments with new eval (sequential_tasks, multi-seed, baselines)
3. [~3 days]  CfC vs MLP ablation — validates or changes the core claim
4. [~3 days]  Add O-LoRA as honest peer comparison
5. [~1 week]  Fix label-free spawning for 5+ tasks (adaptive_spawning end-to-end)
6. [~3 days]  Add CaLoRA to related work; compare if feasible
7. [~2 days]  Statistical significance (5 seeds + t-test)
8. [~1 week]  Second base model (LLaMA-3-8B or Mistral-7B)
9. [~2 days]  Measure inference latency
10. [ongoing] Write paper (method, experiments, related work, conclusion)
```

**Total realistic timeline to ICLR 2027 submission:** 6–8 weeks of focused work.
