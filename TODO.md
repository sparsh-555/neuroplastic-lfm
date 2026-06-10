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
- [x] Run 007 results documented (PPL era)
- [x] Long Sequence Benchmark eval pipeline — `src/benchmark.py`, `src/cl_metrics.py`, `experiments/cl_benchmark.py`
- [x] Gate fix — zero-init adapter_out + gate init 0 + remove clamp (LoRA-style, LLaMA-Adapter)
- [x] Run 008: first benchmark run (broken gate) — AP: seq 0.800 / pt 0.842 / nplm 0.808; BWT: nplm 0.000
- [x] Run 009: gate-fixed benchmark run — AP: seq 0.814 / pt 0.842 / nplm 0.798; BWT: nplm 0.000 (⚠ gradient blockage — superseded by Run 011)
- [x] Run 011: fixed gradient flow — AP: seq 0.810 / pt 0.838 / nplm **0.830**; BWT: nplm 0.000; gap vs per-task LoRA collapsed to 0.8 pts
- [x] Framing resolved — Path A (NeuroplasticLM general); Path B deprecated; no prior CfC-as-adapter work found
- [x] HAM (arXiv 2509.13211) confirmed as closest prior art — LoRA substrate, vision only
- [x] Run 010: CfC vs MLP ablation — INVALIDATED (zero-init adapter_out blocked all CfC τ gradients; broken CfC vs MLP, not liquid CfC vs MLP)
- [x] Run 012: valid CfC vs MLP ablation on LFM — CfC AP 0.832 vs MLP AP 0.828 (Δ=0.004, noise); CfC wins boolq+dbpedia; grow-and-freeze is the primary claim
- [x] Run 014: LLaMA-3-8B CfC vs MLP ablation — MLP AP 0.900 vs CfC AP 0.858 (Δ=−0.042); MLP beats per-task LoRA (0.888) at 15× fewer params; CfC advantage requires TRACE-8 temporal tasks
- [x] Intelligent gate — replaced scalar `maturity` with `gate_proj = nn.Linear(base_dim, 1)`, bias=-4.0; per-token routing, gradient signal continues after loss≈0
- [x] Gradient flow fix — nn.init.normal_(adapter_out.weight, std=1e-3) in both CfCCluster and MLPCluster; τ grad norms now 0.02–0.06
- [x] LLaMA hook fix — model.py hook now handles tuple layer output (LlamaDecoderLayer returns (hidden, ...) not plain tensor)

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

**Recommendation:** Target ICLR 2027 (Oct 2026 deadline). Two viable paths:

| | Path A — Focused | Path B — Full Vision |
|---|---|---|
| Scope | Grow-and-freeze + CfC on temporal tasks | + Variable cluster size + CKA merge |
| Differentiator from HAM | CfC substrate, NLP domain | All of Path A + principled merge/prune |
| Timeline | 4–5 weeks | 8–10 weeks |
| Paper strength | Solid if CfC > MLP on TRACE-8 | Stronger; recovers original system vision |
| Risk | Weak if CfC ≈ MLP on TRACE-8 | More implementation; HAM must be cited |

**Recommendation: Path B.** The original vision (self-managing neuroplasticity loop) is
the intellectual core of the project. Variable size + CKA merge are buildable in 2 weeks.
Path A is a fallback if TRACE-8 results are disappointing.

---

## Framing Decision — RESOLVED: Path A (NeuroplasticLM, general)

**Web search June 2026 confirmed: no prior work uses CfC/LTC/NCP as adapters for any LLM
or transformer.** The novelty is architecture-independent. Path B (LFM-specific narrative)
is deprecated — the "liquid + liquid" synergy has no empirical grounding yet and is
unnecessary when the general claim is already novel.

**Resolved direction:**
- Rename: `NeuroplasticLFM` → **`NeuroplasticLM`**
- Primary claim: first recurrent, ODE-derived adapter (CfC/NCP) for CL in LLMs —
  architectural zero-forgetting + adaptive τ dynamics per token, impossible in LoRA
- Test on LFM (done) + LLaMA-3-8B (done) — generalization confirmed on both models
- HAM (arXiv 2509.13211) is closest prior art: LoRA substrate, vision only, no NLP

**⚠️ HAM overlap warning (June 2026):** A newer HAM paper (ICLR 2026 submission) does
adapter similarity grouping + pruning + merging for vision CL. This directly overlaps with
the auto-dreamer concept. Differentiators: CfC substrate, NLP domain, CKA-based merge
(vs weight-norm), activation-importance pruning (keeps high-τ neurons specifically),
label-free inference. Must cite and explicitly differentiate if we implement merge/prune.
See AUDIT.md Field Coverage for full comparison.

**LoRA comparison note (still relevant):**
LFM2.5-1.2B has 16 layers — 6 attention + 10 conv. LoRA targets only `q_proj/k_proj/v_proj`
in the 6 attention layers; NeuroplasticLM also injects at an attention layer — both methods
equally ignore the conv layers, so the comparison is internally consistent. Still needs a
sentence in the experimental section clarifying this for LFM-unfamiliar reviewers.

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

Or **TRACE-8** (8 tasks, more common in 2024-2026 papers — arXiv 2310.06762, NeurIPS 2024 datasets track):

| # | Task | Domain | Why it's hard |
|---|---|---|---|
| 1 | ScienceQA | Multi-hop science QA (elementary/high school) | Multi-step reasoning chains, doesn't converge at step 100 |
| 2 | FOMC | Federal Reserve forward guidance prediction | Specialized financial vocabulary, domain shift |
| 3 | MeetingBank | Meeting summarization | Long contexts, temporal structure |
| 4 | C-STANCE | Chinese zero-shot stance detection (Sina Weibo) | Cross-lingual, novel token distribution |
| 5 | 20Minuten | German news summarization | Multilingual, long sequences |
| 6 | Py150 | Python code completion (150K programs from GitHub) | Long programs, strict sequential syntax dependencies |
| 7 | NumGLUE-T1 | Math word problems | Multi-step arithmetic, LLaMA2-chat 13B: 28.8%→2% after sequential training |
| 8 | NumGLUE-T2 | Math word problems (variant) | Same — catastrophic forgetting even on strong models |

TRACE key result: aligned LLMs exhibit **significant forgetting even with instruction tuning**.
LLaMA2-chat 13B gsm8k: 28.8% → 2% after sequential training on TRACE datasets.
This creates the sustained gradient signal that activates τ specialization in CfC.

**Why TRACE-8 is required for the CfC story:**
On current benchmark (classification), MLP beats CfC by 4.2 AP (Run 014). CfC's ODE τ
dynamics are overhead on position-invariant tasks. On Py150 (code) and NumGLUE (math),
sequential structure is strict — early tokens constrain valid continuations. CfC's per-position
time constants should provide measurable advantage here.

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

- [x] **Run 011 DONE** — NeuroplasticLFM AP=0.830 vs per-task LoRA AP=0.838 (gap=0.8 pts); zero-forgetting intact; wins boolq+dbpedia. τ_a_std flat (0.1675 throughout) — τ not specialising at 500 steps.

- [x] **Run 012 DONE** — CfC AP=0.832 vs MLP AP=0.828 on LFM (Δ=0.004, noise). Grow-and-freeze is primary claim.
- [x] **Run 014 DONE** — LLaMA-3-8B: MLP AP=0.900 beats per-task LoRA (0.888) at 15× fewer params. CfC AP=0.858. MLP > CfC on classification; TRACE-8 needed for CfC claim.
- [x] **Intelligent gate DONE** — `gate_proj = nn.Linear(base_dim, 1)` replaces scalar maturity. Per-token routing, starts nearly closed (sigmoid(-4)≈0.018), gradient persists after loss≈0.

- [ ] **Run 015** — rerun LLaMA benchmark with intelligent gate to confirm CfC regression fix
- [ ] **TRACE-8 pipeline + Run 016** — critical for CfC vs MLP on temporal tasks (Py150, NumGLUE)
  - If CfC > MLP: τ dynamics validated, CfC paper confirmed
  - If CfC ≈ MLP: grow-and-freeze paper; CfC is one substrate option among others

- [ ] **Label-free spawning validation on 5+ tasks**
  - `sequential_tasks.py` manually calls `spawn_cluster(task_name)` — this requires knowing task boundaries
  - Need an end-to-end run where the model encounters a 5-task stream with no labels and adaptive_spawning drives all spawns
  - Science stream never triggered in 2-task demo; fix calibration for more domains first

---

## P2: Baselines (required for paper)

- [x] **O-LoRA Run 013 DONE** — AP 0.846, BWT 0.000, FWT +0.185; beats per-task LoRA by 0.8 pts
  - NeuroplasticLFM gap: 1.6 pts AP, wins boolq (+2) and dbpedia (+5) vs O-LoRA
  - O-LoRA uses 2.4× more params and requires task label at inference
- [ ] **O-LoRA on LLaMA** — run `experiments/olora_baseline.py` with LLaMA loader; key question: does gap narrow on larger model?
- [ ] **CaLoRA** (NeurIPS 2025) — current SOTA for PEFT-based CL, backward knowledge transfer via PaCA (parameter-level causal attribution)
- [ ] **GainLoRA** (NeurIPS 2025) — most direct structural competitor (gating on LoRA branches); search GitHub before implementing
- [ ] **EWC-LoRA** (ICLR 2026) — `github.com/Yaoyz96/low-rank-cl`
- [ ] **OA-Adapter** (ICLR 2026) — dynamic orthogonal subspace adapter; related to O-LoRA but adaptive
- [ ] **InfLoRA** / **SD-LoRA** (NeurIPS 2025) — orthogonal LoRA variants; mention in related work is sufficient
- [ ] **ELLA** (NeurIPS 2025) — subspace decorrelation for adapters; related work mention

**2025–2026 benchmark landscape (from web search June 2026):**
- Std-CL 5: same as Long Sequence Benchmark (5 tasks, classification)
- Seq-GLUE 7: 7 GLUE tasks in sequence
- Long-CL 15: 15-task version of Seq-GLUE
- **TRACE-8**: hardest, 8 tasks including math+code — use this for CfC advantage experiments

---

## P2b: Self-Managing System (Original Vision — recovers architectural novelty)

These two features recover the core of the original design. Together they make NeuroplasticLM
a self-managing system rather than "frozen adapters that accumulate." Both are needed to
differentiate from HAM (ICLR 2026), which does similarity-based grouping + pruning for vision.

- [ ] **Variable cluster size**
  - Current: fixed `CLUSTER_DIM=64` always. Every task gets identical capacity.
  - Target: grow `AutoNCP(units, output_size=units//4)` with units ∈ {16, 32, 64, 128}
  - Algorithm: train at units=16, eval validation loss; if not converged, grow to 32; repeat
  - Stop when val loss plateaus or maximum size reached
  - Result: small tasks (yelp) get tiny clusters; hard tasks (NumGLUE math) get large ones
  - Paper claim: "task-adaptive capacity allocation" — each task gets exactly what it needs

- [ ] **CKA-based merge detection** (replaces TIES-merging)
  - Current: TIES-merging is a crude weight average. Not latent-space-based. Marginally hurts PPL.
  - Target: after each cluster freezes, compute Centered Kernel Alignment (CKA) between its
    64-dim internal representations and every existing frozen cluster on a 50-sample probe set
  - If CKA(cluster_new, cluster_existing) > 0.7: merge via distillation (not weight average)
    - Train a new merged cluster to reproduce outputs of both original clusters on their tasks
    - Merged cluster is smaller than two separate clusters — reduces total memory
  - If CKA < 0.3: fully novel knowledge, keep separate
  - If 0.3 ≤ CKA < 0.7: partial overlap — keep separate but note for later merge opportunity
  - This is the auto-dreamer's core function: intelligent, latent-space-based consolidation
  - Differentiator from HAM: CKA on CfC hidden states measures τ-dynamics alignment, not weight norms

---

## P3: Generalization and Significance

- [x] **LLaMA-3-8B run** — DONE (Run 014). MLP AP=0.900, CfC AP=0.858.
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
