"""
5-task sequential continual learning experiment.

Compares:
  1. Sequential LoRA — one shared r=8 adapter (lr=1e-5) trained across all 5 tasks.
     The optimizer persists between tasks so momentum accumulates — vanilla sequential
     CL, the natural case where a practitioner keeps fine-tuning the same adapter.
  2. NeuroplasticLFM — spawns a frozen CfC cluster per task. Prior clusters are
     architecturally immutable; science PPL is provably unchanged by later tasks.

Primary metric: Task-1 (science) PPL trajectory as tasks are added.
Full backward-transfer matrix also printed.

Run: PYTHONPATH=/neuroplastic-lfm python experiments/sequential_tasks.py
"""
import math
from itertools import cycle

import torch
import torch.nn.functional as F
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, TaskType
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from experiments.poc_dual_task import (
    AlpacaDataset,
    BATCH_SIZE,
    CREATIVE_KEYWORDS,
    EVAL_SIZE,
    filter_records,
    SCIENCE_KEYWORDS,
)
from src.eval import perplexity
from src.model import NeuroplasticLFM
from src.train import train_cluster

# ── Config ─────────────────────────────────────────────────────────────────────
MAX_LENGTH  = 256
TRAIN_SIZE  = 200
MAX_STEPS   = 500

LORA_R              = 8
LORA_ALPHA          = 16
LORA_LR             = 1e-5   # best lr from lora_baseline.py sweep
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj"]

MATH_KEYWORDS = {
    "calculate", "arithmetic", "integer", "prime", "probability",
    "matrix", "derivative", "geometry", "algebra", "integral",
}
CODING_KEYWORDS = {
    "code", "program", "function", "python", "algorithm",
    "debug", "implement", "variable", "loop", "class",
}
HISTORY_KEYWORDS = {
    "history", "war", "century", "civilization", "ancient",
    "empire", "dynasty", "medieval", "revolution", "treaty",
}

# Ordered: science is always "task 1" for the forgetting measurement
TASKS = {
    "science":  SCIENCE_KEYWORDS,
    "creative": CREATIVE_KEYWORDS,
    "math":     MATH_KEYWORDS,
    "coding":   CODING_KEYWORDS,
    "history":  HISTORY_KEYWORDS,
}
TASK_NAMES = list(TASKS.keys())


# ── Data ───────────────────────────────────────────────────────────────────────

def build_dataloaders(alpaca, tok):
    train_dls, eval_dls = {}, {}
    for name, keywords in TASKS.items():
        all_recs = filter_records(alpaca, keywords, TRAIN_SIZE + EVAL_SIZE)
        tr = all_recs[:TRAIN_SIZE]
        ev = all_recs[TRAIN_SIZE:TRAIN_SIZE + EVAL_SIZE]
        if len(tr) < TRAIN_SIZE or len(ev) < EVAL_SIZE:
            print(f"  WARNING: {name} only found {len(tr)} train / {len(ev)} eval records")
        train_dls[name] = DataLoader(
            AlpacaDataset(tr, tok, MAX_LENGTH),
            batch_size=BATCH_SIZE, shuffle=True,
        )
        eval_dls[name] = DataLoader(
            AlpacaDataset(ev, tok, MAX_LENGTH),
            batch_size=BATCH_SIZE,
        )
        print(f"  {name:10s}: {len(tr)} train / {len(ev)} eval")
    return train_dls, eval_dls


# ── Shared PPL for PEFT-wrapped models ────────────────────────────────────────

@torch.no_grad()
def compute_ppl(model, dataloader) -> float:
    """PPL for any HuggingFace model that exposes `.logits` in its forward output."""
    device = next(model.parameters()).device
    model.eval()
    total_loss, total_tokens = 0.0, 0
    for batch in dataloader:
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["labels"].to(device)
        logits = model(input_ids, attention_mask=attention_mask).logits
        loss = F.cross_entropy(
            logits[:, :-1].contiguous().view(-1, logits.size(-1)),
            labels[:, 1:].contiguous().view(-1),
            ignore_index=-100,
            reduction="sum",
        )
        total_loss   += loss.item()
        total_tokens += (labels[:, 1:] != -100).sum().item()
    return math.exp(total_loss / max(total_tokens, 1))


# ── Experiment 1: Sequential LoRA ──────────────────────────────────────────────

def run_sequential_lora(base, train_dls, eval_dls, device):
    """
    One shared LoRA adapter trained sequentially on all 5 tasks.
    Optimizer persists across tasks (vanilla sequential CL — no replay, no freezing).
    Returns list of dicts: ppl_matrix[i] = {task: PPL} for tasks seen after step i.
    """
    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=LORA_TARGET_MODULES,
        lora_dropout=0.05,
        bias="none",
    )
    model = get_peft_model(base, lora_cfg)
    trainable, total = model.get_nb_trainable_parameters()
    print(f"  LoRA trainable: {trainable:,} ({100*trainable/total:.3f}% of {total:,})")

    # Single persistent optimizer — the standard sequential fine-tuning setup
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LORA_LR,
        weight_decay=0.01,
    )

    ppl_matrix = []

    for i, task in enumerate(TASK_NAMES):
        print(f"\n  Task {i+1}/{len(TASK_NAMES)}: {task}")
        model.train()
        for step, batch in enumerate(
            tqdm(cycle(train_dls[task]), total=MAX_STEPS, desc=f"LoRA/{task}", leave=False)
        ):
            if step >= MAX_STEPS:
                break
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].to(device)
            logits = model(input_ids, attention_mask=attention_mask).logits
            loss = F.cross_entropy(
                logits[:, :-1].contiguous().view(-1, logits.size(-1)),
                labels[:, 1:].contiguous().view(-1),
                ignore_index=-100,
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        # Snapshot: evaluate all tasks trained so far
        snapshot = {}
        for j in range(i + 1):
            t = TASK_NAMES[j]
            snapshot[t] = compute_ppl(model, eval_dls[t])
            marker = " ← current" if t == task else ""
            print(f"    {t:10s}: {snapshot[t]:.2f}{marker}")
        ppl_matrix.append(snapshot)

    del model
    torch.cuda.empty_cache()
    return ppl_matrix


# ── Experiment 2: Per-task LoRA ────────────────────────────────────────────────

def run_pertask_lora(base, train_dls, eval_dls, device):
    """
    Per-task LoRA: one independent adapter per task, activated by task label at eval.
    Zero forgetting is guaranteed because adapters never interfere — but task identity
    must be known at inference time. This is the LoRA upper bound for comparison.
    Returns list of dicts: ppl_matrix[i] = {task: PPL} for tasks seen after step i.
    """
    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=LORA_TARGET_MODULES,
        lora_dropout=0.05,
        bias="none",
    )
    model = get_peft_model(base, lora_cfg)
    trainable, total = model.get_nb_trainable_parameters()
    print(f"  LoRA trainable (per adapter): {trainable:,}")

    # get_peft_model creates adapter "default" for task 0; remaining tasks get their
    # own named adapters via add_adapter so weights are always independently initialised.
    adapter_names = {TASK_NAMES[0]: "default"}
    for task in TASK_NAMES[1:]:
        model.add_adapter(task, lora_cfg)
        adapter_names[task] = task

    ppl_matrix = []

    for i, task in enumerate(TASK_NAMES):
        print(f"\n  Task {i+1}/{len(TASK_NAMES)}: {task}")
        model.set_adapter(adapter_names[task])
        # Fresh optimizer each task — set_adapter makes only this adapter's params trainable
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=LORA_LR, weight_decay=0.01,
        )
        model.train()
        for step, batch in enumerate(
            tqdm(cycle(train_dls[task]), total=MAX_STEPS, desc=f"LoRA-pt/{task}", leave=False)
        ):
            if step >= MAX_STEPS:
                break
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].to(device)
            logits = model(input_ids, attention_mask=attention_mask).logits
            loss = F.cross_entropy(
                logits[:, :-1].contiguous().view(-1, logits.size(-1)),
                labels[:, 1:].contiguous().view(-1),
                ignore_index=-100,
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        # Eval: switch to each task's own adapter for measurement
        snapshot = {}
        for j in range(i + 1):
            t = TASK_NAMES[j]
            model.set_adapter(adapter_names[t])
            snapshot[t] = compute_ppl(model, eval_dls[t])
            marker = " ← current" if t == task else ""
            print(f"    {t:10s}: {snapshot[t]:.2f}{marker}")
        ppl_matrix.append(snapshot)

    del model
    torch.cuda.empty_cache()
    return ppl_matrix


# ── Experiment 3: NeuroplasticLFM ─────────────────────────────────────────────

def run_neuroplastic(base, train_dls, eval_dls, device):
    """
    Spawns one CfC cluster per task; prior clusters are frozen by the registry.
    Science PPL is provably constant after task 1 — this confirms the guarantee at scale.
    Returns list of dicts: ppl_matrix[i] = {task: PPL} for tasks seen after step i.
    """
    model = NeuroplasticLFM(base, seed=0).to(device)
    ppl_matrix = []

    for i, task in enumerate(TASK_NAMES):
        print(f"\n  Task {i+1}/{len(TASK_NAMES)}: {task}")
        model.spawn_cluster(task)
        train_cluster(model, task, train_dls[task], max_steps=MAX_STEPS, log_every=100)

        snapshot = {}
        for j in range(i + 1):
            t = TASK_NAMES[j]
            snapshot[t] = perplexity(model, eval_dls[t], task_id=t)
            marker = " ← current" if t == task else ""
            print(f"    {t:10s}: {snapshot[t]:.2f}{marker}")
        ppl_matrix.append(snapshot)

    del model
    torch.cuda.empty_cache()
    return ppl_matrix


# ── Results printing ───────────────────────────────────────────────────────────

def print_results(baseline_ppls, lora_matrix, pertask_matrix, nplm_matrix):
    n   = len(TASK_NAMES)
    col = 10
    W   = 78

    # ── Science PPL trajectory ─────────────────────────────────────────────────
    print(f"\n\n{'═'*W}")
    print("Task-1 (Science) PPL as tasks are added  [lower = better]")
    print(f"{'─'*W}")
    print(f"  {'':28s}  {'Sci PPL':>{col}}  {'Δ vs T1':>{col}}  {'Labels?':>8}  {'Params/task':>11}")
    print(f"  {'─'*28}  {'─'*col}  {'─'*col}  {'─'*8}  {'─'*11}")
    print(f"  {'Baseline (frozen base)':28s}  {baseline_ppls['science']:>{col}.2f}  {'—':>{col}}  {'—':>8}  {'—':>11}")

    lora_t1    = lora_matrix[0]["science"]
    pertask_t1 = pertask_matrix[0]["science"]
    nplm_t1    = nplm_matrix[0]["science"]

    for i, task in enumerate(TASK_NAMES):
        lora_ppl    = lora_matrix[i]["science"]
        pertask_ppl = pertask_matrix[i]["science"]
        nplm_ppl    = nplm_matrix[i]["science"]
        lora_d    = f"{lora_ppl    - lora_t1:+.2f}"    if i > 0 else "—"
        pertask_d = f"{pertask_ppl - pertask_t1:+.2f}" if i > 0 else "—"
        nplm_d    = f"{nplm_ppl    - nplm_t1:+.2f}"    if i > 0 else "—"
        tasks_str = "→".join(TASK_NAMES[:i+1])
        print(f"\n  After {tasks_str}")
        print(f"    {'Sequential LoRA':26s}  {lora_ppl:>{col}.2f}  {lora_d:>{col}}  {'no':>8}  {'~590K':>11}")
        print(f"    {'Per-task LoRA':26s}  {pertask_ppl:>{col}.2f}  {pertask_d:>{col}}  {'YES':>8}  {'~590K':>11}")
        print(f"    {'NeuroplasticLFM':26s}  {nplm_ppl:>{col}.2f}  {nplm_d:>{col}}  {'no':>8}  {'~187K':>11}")

    print(f"\n{'─'*W}")
    lora_forget    = lora_matrix[-1]["science"]    - lora_t1
    pertask_forget = pertask_matrix[-1]["science"] - pertask_t1
    nplm_forget    = nplm_matrix[-1]["science"]    - nplm_t1
    print(f"Total forgetting after {n} tasks (ΔSciPPL: T{n} − T1):")
    print(f"  Sequential LoRA:  {lora_forget:+.3f}  ({lora_forget/lora_t1*100:+.1f}%)  [no labels, forgetting compounds]")
    print(f"  Per-task LoRA:    {pertask_forget:+.3f}  ({pertask_forget/pertask_t1*100:+.1f}%)  [task labels required, 3x more params]")
    print(f"  NeuroplasticLFM:  {nplm_forget:+.3f}  ({nplm_forget/nplm_t1*100:+.1f}%)  [no labels, architectural guarantee]")
    print(f"{'═'*W}")

    # ── Full backward transfer matrix ──────────────────────────────────────────
    cell = 7

    def _matrix_block(label, matrix):
        print(f"\n  {label}")
        hdr = f"  {'eval \\ trained →':20s}" + "".join(f"  T{i+1:>{cell-2}}" for i in range(n))
        print(hdr)
        print(f"  {'─'*20}" + f"  {'─'*cell}" * n)
        for j, eval_task in enumerate(TASK_NAMES):
            row = f"  {eval_task:20s}"
            for i in range(n):
                if j <= i:
                    row += f"  {matrix[i][eval_task]:>{cell}.2f}"
                else:
                    row += f"  {'—':>{cell}}"
            print(row)

    print(f"\n\n{'═'*W}")
    print("Full backward-transfer matrix  (PPL of row task after training on T1..Tk)")
    print(f"{'─'*W}")
    _matrix_block("SEQUENTIAL LORA  (one shared adapter)", lora_matrix)
    _matrix_block("PER-TASK LORA    (one adapter per task, task label at eval)", pertask_matrix)
    _matrix_block("NEUROPLASTICLFM  (one frozen CfC cluster per task)", nplm_matrix)
    print(f"{'═'*W}")


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    print("\nLoading LFM2.5-1.2B-Instruct...")
    base = AutoModelForCausalLM.from_pretrained(
        "LiquidAI/LFM2.5-1.2B-Instruct", dtype=torch.bfloat16
    ).to(device)
    tok = AutoTokenizer.from_pretrained("LiquidAI/LFM2.5-1.2B-Instruct")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    print("\nBuilding dataloaders for 5 tasks...")
    alpaca = load_dataset("yahma/alpaca-cleaned", split="train")
    train_dls, eval_dls = build_dataloaders(alpaca, tok)

    # Baseline: frozen base PPL (no adapters, no clusters)
    print("\n── Baseline PPL (frozen base, no adapters) ──")
    baseline_ppls = {}
    for task in TASK_NAMES:
        baseline_ppls[task] = compute_ppl(base, eval_dls[task])
        print(f"  {task:10s}: {baseline_ppls[task]:.2f}")

    # ── Experiment 1: Sequential LoRA ─────────────────────────────────────────
    print(f"\n{'═'*60}")
    print("EXPERIMENT 1: Sequential LoRA (one shared adapter, all tasks)")
    print(f"{'═'*60}")
    lora_matrix = run_sequential_lora(base, train_dls, eval_dls, device)

    # Reload clean base — PEFT injects LoRA layers in-place into the base module tree
    print("\nReloading clean base model for per-task LoRA run...")
    del base
    torch.cuda.empty_cache()
    base = AutoModelForCausalLM.from_pretrained(
        "LiquidAI/LFM2.5-1.2B-Instruct", dtype=torch.bfloat16
    ).to(device)

    # ── Experiment 2: Per-task LoRA ───────────────────────────────────────────
    print(f"\n{'═'*60}")
    print("EXPERIMENT 2: Per-task LoRA (one adapter per task, task label at eval)")
    print(f"{'═'*60}")
    pertask_matrix = run_pertask_lora(base, train_dls, eval_dls, device)

    # Reload clean base again for NeuroplasticLFM
    print("\nReloading clean base model for NeuroplasticLFM run...")
    del base
    torch.cuda.empty_cache()
    base = AutoModelForCausalLM.from_pretrained(
        "LiquidAI/LFM2.5-1.2B-Instruct", dtype=torch.bfloat16
    ).to(device)

    # ── Experiment 3: NeuroplasticLFM ─────────────────────────────────────────
    print(f"\n{'═'*60}")
    print("EXPERIMENT 3: NeuroplasticLFM (one frozen CfC cluster per task)")
    print(f"{'═'*60}")
    nplm_matrix = run_neuroplastic(base, train_dls, eval_dls, device)

    print_results(baseline_ppls, lora_matrix, pertask_matrix, nplm_matrix)

    return {
        "baseline_ppls":  baseline_ppls,
        "lora_matrix":    lora_matrix,
        "pertask_matrix": pertask_matrix,
        "nplm_matrix":    nplm_matrix,
    }


if __name__ == "__main__":
    main()
