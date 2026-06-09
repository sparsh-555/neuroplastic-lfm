# NeuroplasticLFM — Experimental Audit

**Date:** June 2026  
**Purpose:** Honest gap analysis before professor meeting / submission. Identifies corners cut, departures from field standard, and missing experiments as of June 2026.

---

## Summary Verdict

The architecture is real and the zero-forgetting property is demonstrated. The prototype works. But the paper is not submission-ready for four structural reasons: (1) we evaluate with perplexity, the field uses accuracy; (2) we test on keyword-filtered Alpaca, the field uses standardized NLP benchmarks; (3) the "label-free" headline claim is not validated end-to-end; (4) the paper is named "NeuroplasticLFM" but the architecture is fully model-agnostic and the LoRA comparison is architecturally incomplete on a hybrid model. The framing decision — general method vs. LFM-specific — must be resolved before writing anything, because it changes which experiments are the primary results.

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

### H4 — LoRA comparison is architecturally incomplete on LFM

**The structural problem:** LFM2.5-1.2B has 16 layers — 6 attention (indices 2,5,8,10,12,14) and 10 convolutional/SSM layers in between. Our LoRA baseline targets only `q_proj`, `k_proj`, `v_proj` in the 6 attention layers. The 10 conv layers are untouched by LoRA entirely.

On a pure transformer (Mistral-7B, LLaMA-3-8B), LoRA can adapt all 32 attention layers — the tool is being used as designed. On LFM, LoRA is adapting roughly 6/16 of the model's parameters, with the conv layers permanently frozen. This means our LoRA baseline on LFM is weaker than its true capability on the architectures it was designed for.

**The consistency note:** NeuroplasticLFM also only injects at an attention layer output (layer 8), so both methods equally ignore LFM's conv architecture. The head-to-head comparison between them is internally consistent — same handicap on both sides.

**Why it still matters:**
1. "LoRA on LFM" is not the same as "LoRA" — calling it simply "LoRA" in the paper without qualification will mislead readers
2. The conclusion "NeuroplasticLFM outperforms LoRA on LFM" is partly an artifact of the base model choice, not pure architectural superiority
3. A reviewer familiar with LoRA will catch this immediately

**Fix:** Add a sentence in the experimental section clarifying that LoRA targets only attention projections on LFM's hybrid architecture. Test on a second model (pure transformer) where LoRA is at full capability — if NeuroplasticLFM still wins, the result is robust.

---

### H5 — Architecture is model-agnostic but paper was named for LFM specificity ✅ RESOLVED

**Resolution (June 2026):** Web search confirmed no prior work has used CfC/LTC/NCP as
adapters for any LLM or transformer. The novelty is architecture-independent, so Path B
(LFM-specific narrative) has no competitive advantage over Path A and is now deprecated.

**Decision taken: Path A — rename to NeuroplasticLM.**
- Primary claim: first recurrent, ODE-derived adapter for CL in LLMs
- The τ dynamics and sparse NCP wiring are the differentiator from MLP adapters, not LFM-specificity
- HAM (arXiv 2509.13211, Sep 2025) is closest prior art: LoRA substrate, vision only, no NLP tasks
- LLaMA-3-8B is now a required secondary experiment (generalization evidence, not framing test)

**Remaining action:** rename the codebase/paper from NeuroplasticLFM → NeuroplasticLM
(low priority until results are stable — keep code names as-is for now).

---

### H6 — CfC vs MLP ablation — NEEDS VALID RE-RUN

**Run 010 (June 2026) is INVALIDATED.** The zero-init adapter_out introduced a gradient
blockage: `adapter_out.weight.T @ grad = 0.T @ grad = 0`, so all time_a / time_b weights
inside the CfCCells received zero gradient throughout the entire run.  The CfC operated
with fixed random τ — the liquid property never activated.  Run 010 compared a
"non-functioning CfC" vs MLP; the result is meaningless for the liquid dynamics question.

**Fix (commit f5e0c64):** `nn.init.normal_(adapter_out.weight, std=1e-3)` replaces
`nn.init.zeros_`.  Initial output noise is ~0.001 (0.01% of residual stream, negligible).
Gradient norms to time_a now 0.02–0.06 (were 0.000000).

**Framing pivot retracted:** The "NCP-wired growing adapter" pivot from run 010 is retracted.
CfC remains the primary substrate.  The valid CfC vs MLP comparison is Run 012 (pending).

**Run 011 result (fixed gradient flow):** NeuroplasticLFM AP = **0.830** vs per-task LoRA
AP = 0.838 — gap collapsed from 4.4 pts (run 009) to **0.8 pts**.  Zero-forgetting intact
(BWT=0, F.Ra=0).  NeuroplasticLFM wins boolq (0.810 vs 0.790) and dbpedia (0.990 vs 0.950).

**τ_a_std observation:** std was 0.1675 at step 0 and flat throughout all 5 tasks / 500 steps.
τ is NOT specialising at this training scale.  CfC runs as a fixed-τ recurrent feature extractor.
Gradients DO flow (norm 0.02–0.06) but divergence signal is too weak at 500 steps.

**Run 012 result (valid ablation):** CfC AP = **0.832** vs MLP AP = **0.828** — Δ = +0.004,
below the noise threshold.  CfC ≈ MLP at 500 steps.  Per-task: CfC wins boolq (+5 pts)
and dbpedia (+3 pts); MLP wins imdb (+4 pts) and multirc (+2 pts).  CfC has a structural
advantage on context-heavy tasks even with near-fixed τ.

**Revised primary claim:** The grow-and-freeze paradigm (not CfC specifically) is the
contribution.  CfC is the recommended substrate.  See run_012.md for full framing.

**One valid finding from run 010:** Training speed — CfC is ~4× slower than MLP per step
(5.2 vs 20.5 it/s).  This is a real cost regardless of whether τ learns.

---

## 🟡 Moderate Severity

### M1 — Sequential LoRA baseline is a strawman

Our "Sequential LoRA" is the weakest possible LoRA setup: one shared adapter, trained on all tasks continuously, no protection. Against proper CL-LoRA methods (CaLoRA, O-LoRA, InfLoRA), reviewers will say we chose the easiest baseline to beat.

**Fix:** Replace or supplement sequential LoRA with O-LoRA (zero-forgetting guarantee via orthogonal subspaces). This is the honest peer comparison. If NeuroplasticLFM matches O-LoRA on forgetting but with fewer parameters and no task labels, that's a publishable result.

---

### M2 — Only one base model (now elevated by framing decision)

Results are only on LFM2.5-1.2B, which is obscure outside LiquidAI. All competitive papers test on at least LLaMA-7B or Mistral-7B.

This was previously a "nice to have" but is now tightly coupled to the Path A vs Path B framing decision (H5). The LLaMA-3-8B experiment is not just for credibility — it's the empirical test that decides the paper's entire framing. The code change needed is minimal: `d_model = 4096`, `inject_at = 16` (proportional mid-model), adapter dims updated. Everything else — registry, spawning, TIES-merging — is unchanged.

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
0. [DONE]     Framing: Path A confirmed — NeuroplasticLM, general method
              Web search June 2026 found no prior CfC-as-adapter work anywhere.

1. [DONE]     Rewrite eval pipeline: Long Sequence Benchmark + accuracy + AP/BWT/F.Ra/FWT
              → src/benchmark.py, src/cl_metrics.py, experiments/cl_benchmark.py

2. [DONE]     Fix maturity gate — zero-init adapter_out + gate init 0 + remove clamp
              → Gate was stuck at 0.047 due to random adapter_out noise + vanishing gradient

3. [~1 week]  Re-run cl_benchmark.py on LFM with fixed gate; confirm AP improves
              → Baseline results: AP 0.808, BWT 0.000, F.Ra 0.000 (gate broken)

4. [INVALID]  CfC vs MLP ablation Run 010 — gradient blockage (zero-init adapter_out)
              → CfC τ received 0 gradient; comparison was "broken CfC vs MLP"
              → Framing pivot RETRACTED; fix in commit f5e0c64 (std=1e-3 init)
              → Re-run as Run 012 with fixed gradient flow

4b.[DONE]     CfC vs MLP ablation Run 012 — CfC AP 0.832 vs MLP AP 0.828, Δ=0.004 (noise)
              → CfC wins boolq/dbpedia; MLP wins imdb/multirc; grow-and-freeze is the claim

5. [~3 days]  Add O-LoRA as honest peer comparison (zero-forgetting guarantee)

6. [~2 days]  Run same core experiment on LLaMA-3-8B (d_model=4096, inject_at=16)
              → Required generalization evidence for Path A claim

7. [~1 week]  Fix label-free spawning for 5+ tasks (adaptive_spawning end-to-end)

8. [~3 days]  Add CaLoRA to related work; compare if feasible

9. [~2 days]  Statistical significance (5 seeds + t-test)

10. [~2 days]  Measure inference latency

11. [ongoing] Write paper (method, experiments, related work, conclusion)
              → Paper name: NeuroplasticLM (rename codebase when results are stable)
```

**Total realistic timeline to ICLR 2027 submission:** 6–8 weeks of focused work.
