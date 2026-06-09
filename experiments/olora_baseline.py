"""
O-LoRA baseline: Orthogonal Subspace Learning for continual learning.

Paper: "Orthogonal Subspace Learning for Language Model Continual Learning"
       Wang et al., EMNLP 2023 Findings. arXiv:2310.14152

Algorithm:
  - One LoRA adapter per task (same as per-task LoRA)
  - During task t training, adds orthogonality regularisation:
      L = CE + λ₁ * Σ_{i<t} ||A_i @ A_t.T||_F²
    where A_i, A_t are the LoRA A matrices (shape [r, k]) of past/current task
  - This soft constraint pushes each task's adapter into an orthogonal subspace,
    reducing interference between tasks
  - At inference: requires task identity (same as per-task LoRA)
  - Zero forgetting by construction (frozen past adapters)

Key difference from per-task LoRA: the orthogonality constraint theoretically
allows positive forward transfer (later tasks can reuse prior subspace directions).

Run: PYTHONPATH=/neuroplastic-lfm python experiments/olora_baseline.py
"""
from __future__ import annotations

from itertools import cycle
from typing import Dict, List

import torch
import torch.nn.functional as F
from peft import LoraConfig, TaskType, get_peft_model
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.benchmark import TASK_ORDER, build_cl_dataloaders, evaluate_accuracy
from src.cl_metrics import CLMetrics, compute_cl_metrics, format_metrics_table

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

LAMBDA1             = 0.5   # orthogonality weight (paper default)


# ── O-LoRA helpers ─────────────────────────────────────────────────────────────

def _collect_A_matrices(model, adapter_name: str) -> List[torch.Tensor]:
    """Return all lora_A weight tensors for the given adapter, in module order."""
    matrices = []
    for module in model.modules():
        if hasattr(module, "lora_A") and adapter_name in module.lora_A:
            matrices.append(module.lora_A[adapter_name].weight)  # [r, in_features]
    return matrices


def _orthogonal_loss(
    current_A: List[torch.Tensor],
    stored_A_per_task: List[List[torch.Tensor]],
) -> torch.Tensor:
    """
    Compute ||A_prev @ A_curr.T||_F² summed over all past tasks and all layers.
    Both A matrices are [r, k]; their product A_prev @ A_curr.T is [r, r].
    Gradients flow only through current_A — stored tensors are detached.
    Normalised by number of past tasks to keep λ₁ scale-invariant.
    """
    if not stored_A_per_task:
        return torch.tensor(0.0, device=current_A[0].device)

    loss = torch.tensor(0.0, device=current_A[0].device)
    for past_A in stored_A_per_task:
        for A_prev, A_curr in zip(past_A, current_A):
            O = A_prev @ A_curr.T   # [r, r]; gradients only through A_curr
            loss = loss + (O ** 2).sum()

    return loss / len(stored_A_per_task)


# ── O-LoRA experiment ──────────────────────────────────────────────────────────

def run_olora(
    base, train_dls, eval_dls, label_tids, device
) -> List[Dict[str, float]]:
    """
    O-LoRA: one adapter per task with orthogonal subspace constraint during training.
    Task identity required at inference (identical to per-task LoRA eval protocol).
    """
    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_R, lora_alpha=LORA_ALPHA,
        target_modules=LORA_TARGET_MODULES,
        lora_dropout=0.05, bias="none",
    )

    model = get_peft_model(base, lora_cfg)
    trainable, total = model.get_nb_trainable_parameters()
    print(f"  LoRA trainable (per adapter): {trainable:,} params")
    print(f"  O-LoRA λ₁ = {LAMBDA1}")

    adapter_names = {TASK_ORDER[0]: "default"}
    for task in TASK_ORDER[1:]:
        model.add_adapter(task, lora_cfg)
        adapter_names[task] = task

    stored_A: List[List[torch.Tensor]] = []   # stored A matrices per completed task
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
            tqdm(cycle(train_dls[task]), total=MAX_STEPS,
                 desc=f"olora/{task}", leave=False)
        ):
            if step >= MAX_STEPS:
                break

            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].to(device)

            logits = model(input_ids, attention_mask=attention_mask).logits
            ce_loss = F.cross_entropy(
                logits[:, :-1].contiguous().view(-1, logits.size(-1)),
                labels[:, 1:].contiguous().view(-1),
                ignore_index=-100,
            )

            current_A = _collect_A_matrices(model, adapter_names[task])
            orth_loss = _orthogonal_loss(current_A, stored_A)
            loss      = ce_loss + LAMBDA1 * orth_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            if step % 100 == 0:
                tqdm.write(
                    f"  step={step:4d}  ce={ce_loss.item():.4f}"
                    f"  orth={orth_loss.item():.6f}  total={loss.item():.4f}"
                )

        # Store detached A matrices for this task before moving on
        with torch.no_grad():
            task_A = [w.detach().clone() for w in _collect_A_matrices(model, adapter_names[task])]
        stored_A.append(task_A)

        snapshot: Dict[str, float] = {}
        for j in range(i + 1):
            t = TASK_ORDER[j]
            model.set_adapter(adapter_names[t])
            acc = evaluate_accuracy(model, eval_dls[t], label_tids[t], device)
            snapshot[t] = acc
            marker = " ← current" if t == task else ""
            print(f"    {t:10s}: {acc:.3f}{marker}")
        acc_matrix.append(snapshot)

    return acc_matrix


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    print(f"\nLoading {MODEL_NAME} …")
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.bfloat16
    ).to(device)
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    print("\nBuilding Long Sequence Benchmark dataloaders …")
    train_dls, eval_dls, label_tids = build_cl_dataloaders(
        tok, n_train=N_TRAIN, n_eval=N_EVAL,
        batch_size=BATCH_SIZE, max_length=MAX_LENGTH,
    )

    print("\n── Zero-shot baseline (frozen base, no adapters) ──")
    baseline_accs: Dict[str, float] = {}
    for task in TASK_ORDER:
        acc = evaluate_accuracy(base, eval_dls[task], label_tids[task], device)
        baseline_accs[task] = acc
        print(f"  {task:10s}: {acc:.3f}")

    print(f"\n{'═'*60}")
    print("O-LoRA: orthogonal subspace continual learning")
    print(f"{'═'*60}")
    olora_matrix = run_olora(base, train_dls, eval_dls, label_tids, device)

    acc_matrices = {"O-LoRA": olora_matrix}
    cl_results: Dict[str, CLMetrics] = {
        name: compute_cl_metrics(mat, TASK_ORDER, baseline_accs)
        for name, mat in acc_matrices.items()
    }

    print("\n\n" + format_metrics_table(cl_results, TASK_ORDER, acc_matrices))

    # Reference numbers from Run 011 for direct comparison
    print("\n── Reference (Run 011, same benchmark/config) ──")
    print("  Sequential LoRA : AP 0.810  BWT -0.028  F.Ra 0.038  FWT +0.163")
    print("  Per-task LoRA   : AP 0.838  BWT +0.000  F.Ra 0.000  FWT +0.170  (task label needed)")
    print("  NeuroplasticLFM : AP 0.830  BWT +0.000  F.Ra 0.000  FWT +0.163  (no task label)")

    print("\nBaseline (zero-shot):")
    for task in TASK_ORDER:
        print(f"  {task:10s}: {baseline_accs[task]:.3f}")


if __name__ == "__main__":
    main()
