"""
Continual learning benchmark: Long Sequence Benchmark (5 tasks).

Tasks: Yelp (sentiment) → IMDB (sentiment) → BoolQ (QA) →
       MultiRC (QA) → DBpedia-14 (topic classification)

Compares three methods with standard CL metrics (AP, BWT, F.Ra, FWT):
  1. Sequential LoRA  — one shared adapter, no forgetting protection
  2. Per-task LoRA    — one adapter per task (requires task label at inference)
  3. NeuroplasticLFM  — one frozen CfC cluster per task (label-free at inference)

Run: PYTHONPATH=/neuroplastic-lfm python experiments/cl_benchmark.py
     (estimated ~90 min on a single A100 for all three methods)
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from itertools import cycle
from typing import Dict, List

from peft import LoraConfig, get_peft_model, TaskType
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.benchmark import (
    TASK_ORDER,
    build_cl_dataloaders,
    evaluate_accuracy,
)
from src.cl_metrics import CLMetrics, compute_cl_metrics, format_metrics_table
from src.model import NeuroplasticLFM
from src.train import train_cluster

# ── Config ─────────────────────────────────────────────────────────────────────
MODEL_NAME          = "LiquidAI/LFM2.5-1.2B-Instruct"
N_TRAIN             = 200
N_EVAL              = 100
BATCH_SIZE          = 4
MAX_LENGTH          = 512
MAX_STEPS           = 500

LORA_R              = 8
LORA_ALPHA          = 16
LORA_LR             = 1e-5
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj"]


# ── Experiment 1: Sequential LoRA ──────────────────────────────────────────────

def run_sequential_lora(
    base, train_dls, eval_dls, label_tids, device
) -> List[Dict[str, float]]:
    """
    One shared LoRA adapter trained on all 5 tasks sequentially.
    Optimizer persists — vanilla CL baseline that compounds forgetting.
    """
    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_R, lora_alpha=LORA_ALPHA,
        target_modules=LORA_TARGET_MODULES,
        lora_dropout=0.05, bias="none",
    )
    model = get_peft_model(base, lora_cfg)
    trainable, total = model.get_nb_trainable_parameters()
    print(f"  LoRA trainable: {trainable:,} params ({100*trainable/total:.3f}% of {total:,})")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LORA_LR, weight_decay=0.01,
    )

    acc_matrix: List[Dict[str, float]] = []

    for i, task in enumerate(TASK_ORDER):
        print(f"\n  Task {i+1}/{len(TASK_ORDER)}: {task}")
        model.train()
        for step, batch in enumerate(
            tqdm(cycle(train_dls[task]), total=MAX_STEPS, desc=f"seq-lora/{task}", leave=False)
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

        snapshot: Dict[str, float] = {}
        for j in range(i + 1):
            t   = TASK_ORDER[j]
            acc = evaluate_accuracy(model, eval_dls[t], label_tids[t], device)
            snapshot[t] = acc
            marker = " ← current" if t == task else ""
            print(f"    {t:10s}: {acc:.3f}{marker}")
        acc_matrix.append(snapshot)

    del model
    torch.cuda.empty_cache()
    return acc_matrix


# ── Experiment 2: Per-task LoRA ────────────────────────────────────────────────

def run_pertask_lora(
    base, train_dls, eval_dls, label_tids, device
) -> List[Dict[str, float]]:
    """
    One independent LoRA adapter per task.  Zero forgetting by construction;
    task identity required at inference.  LoRA upper-bound for comparison.
    """
    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_R, lora_alpha=LORA_ALPHA,
        target_modules=LORA_TARGET_MODULES,
        lora_dropout=0.05, bias="none",
    )
    model = get_peft_model(base, lora_cfg)
    trainable, _ = model.get_nb_trainable_parameters()
    print(f"  LoRA trainable (per adapter): {trainable:,} params")

    adapter_names = {TASK_ORDER[0]: "default"}
    for task in TASK_ORDER[1:]:
        model.add_adapter(task, lora_cfg)
        adapter_names[task] = task

    acc_matrix: List[Dict[str, float]] = []

    for i, task in enumerate(TASK_ORDER):
        print(f"\n  Task {i+1}/{len(TASK_ORDER)}: {task}")
        model.set_adapter(adapter_names[task])
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=LORA_LR, weight_decay=0.01,
        )
        model.train()
        for step, batch in enumerate(
            tqdm(cycle(train_dls[task]), total=MAX_STEPS, desc=f"pt-lora/{task}", leave=False)
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

        snapshot: Dict[str, float] = {}
        for j in range(i + 1):
            t = TASK_ORDER[j]
            model.set_adapter(adapter_names[t])
            acc     = evaluate_accuracy(model, eval_dls[t], label_tids[t], device)
            snapshot[t] = acc
            marker  = " ← current" if t == task else ""
            print(f"    {t:10s}: {acc:.3f}{marker}")
        acc_matrix.append(snapshot)

    del model
    torch.cuda.empty_cache()
    return acc_matrix


# ── Experiment 3: NeuroplasticLFM ─────────────────────────────────────────────

def run_neuroplastic(
    base, train_dls, eval_dls, label_tids, device
) -> List[Dict[str, float]]:
    """
    Spawns one frozen CfC cluster per task.  Prior clusters are immutable;
    forgetting is architecturally impossible.  No task labels at inference.
    """
    model = NeuroplasticLFM(base, seed=0, base_dim=2048).to(device)
    acc_matrix: List[Dict[str, float]] = []

    for i, task in enumerate(TASK_ORDER):
        print(f"\n  Task {i+1}/{len(TASK_ORDER)}: {task}")
        model.spawn_cluster(task)
        train_cluster(model, task, train_dls[task], max_steps=MAX_STEPS, log_every=100)

        snapshot: Dict[str, float] = {}
        for j in range(i + 1):
            t   = TASK_ORDER[j]
            acc = evaluate_accuracy(
                model, eval_dls[t], label_tids[t], device,
                task_id=t, is_nplm=True,
            )
            snapshot[t] = acc
            marker = " ← current" if t == task else ""
            print(f"    {t:10s}: {acc:.3f}{marker}")
        acc_matrix.append(snapshot)

    del model
    torch.cuda.empty_cache()
    return acc_matrix


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    print(f"\nLoading {MODEL_NAME} …")
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16
    ).to(device)
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    print("\nBuilding Long Sequence Benchmark dataloaders …")
    train_dls, eval_dls, label_tids = build_cl_dataloaders(
        tok, n_train=N_TRAIN, n_eval=N_EVAL,
        batch_size=BATCH_SIZE, max_length=MAX_LENGTH,
    )

    # Zero-shot baseline: base model accuracy before any fine-tuning
    print("\n── Zero-shot baseline (frozen base, no adapters) ──")
    baseline_accs: Dict[str, float] = {}
    for task in TASK_ORDER:
        acc = evaluate_accuracy(base, eval_dls[task], label_tids[task], device)
        baseline_accs[task] = acc
        print(f"  {task:10s}: {acc:.3f}")

    # ── Sequential LoRA ───────────────────────────────────────────────────────
    print(f"\n{'═'*60}")
    print("EXPERIMENT 1: Sequential LoRA (one shared adapter, all tasks)")
    print(f"{'═'*60}")
    seq_lora_matrix = run_sequential_lora(base, train_dls, eval_dls, label_tids, device)

    # Reload clean base (PEFT modifies it in-place)
    print("\nReloading base model for per-task LoRA …")
    del base
    torch.cuda.empty_cache()
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16
    ).to(device)

    # ── Per-task LoRA ─────────────────────────────────────────────────────────
    print(f"\n{'═'*60}")
    print("EXPERIMENT 2: Per-task LoRA (one adapter per task, task label at eval)")
    print(f"{'═'*60}")
    pertask_lora_matrix = run_pertask_lora(base, train_dls, eval_dls, label_tids, device)

    # Reload clean base for NeuroplasticLFM
    print("\nReloading base model for NeuroplasticLFM …")
    del base
    torch.cuda.empty_cache()
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16
    ).to(device)

    # ── NeuroplasticLFM ───────────────────────────────────────────────────────
    print(f"\n{'═'*60}")
    print("EXPERIMENT 3: NeuroplasticLFM (one frozen CfC cluster per task)")
    print(f"{'═'*60}")
    nplm_matrix = run_neuroplastic(base, train_dls, eval_dls, label_tids, device)

    # ── Metrics ───────────────────────────────────────────────────────────────
    acc_matrices = {
        "Sequential LoRA":    seq_lora_matrix,
        "Per-task LoRA":      pertask_lora_matrix,
        "NeuroplasticLFM":    nplm_matrix,
    }
    cl_results: Dict[str, CLMetrics] = {
        name: compute_cl_metrics(mat, TASK_ORDER, baseline_accs)
        for name, mat in acc_matrices.items()
    }

    print("\n\n" + format_metrics_table(cl_results, TASK_ORDER, acc_matrices))

    print("\nBaseline (zero-shot):")
    for task in TASK_ORDER:
        print(f"  {task:10s}: {baseline_accs[task]:.3f}")

    return {
        "baseline_accs":      baseline_accs,
        "seq_lora_matrix":    seq_lora_matrix,
        "pertask_lora_matrix": pertask_lora_matrix,
        "nplm_matrix":        nplm_matrix,
        "cl_results":         cl_results,
    }


if __name__ == "__main__":
    main()
