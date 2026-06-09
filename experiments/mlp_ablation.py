"""
Ablation: CfCCluster vs MLPCluster on the Long Sequence Benchmark.

Both methods use identical hyperparameters, injection point, and training setup.
The only variable is the processing block inside the adapter:
  - CfCCluster: recurrent ODE-derived CfC cells wired by AutoNCP (~187K params)
  - MLPCluster:  2-layer feed-forward (adapter_in → GELU → ff2 → adapter_out) (~171K params)

Shared design: same adapter_in/out dims, same zero-init adapter_out, same maturity gate.

This answers: do the CfC τ dynamics contribute beyond what a static MLP provides?
  - If CfC > MLP on AP: τ dynamics are doing real work, CfC substrate is justified
  - If CfC ≈ MLP: paper pivots to "NCP-wired growing adapter" framing, CfC is one option

Run: PYTHONPATH=/neuroplastic-lfm python experiments/mlp_ablation.py
     (~60 min on a single A100)
"""
from __future__ import annotations

import torch
from typing import Dict, List

from transformers import AutoModelForCausalLM, AutoTokenizer

from src.benchmark import TASK_ORDER, build_cl_dataloaders, evaluate_accuracy
from src.cl_metrics import CLMetrics, compute_cl_metrics, format_metrics_table
from src.cluster import CfCCluster, MLPCluster
from src.model import NeuroplasticLFM
from src.train import train_cluster

# ── Config ─────────────────────────────────────────────────────────────────────
MODEL_NAME = "LiquidAI/LFM2.5-1.2B-Instruct"
N_TRAIN    = 200
N_EVAL     = 100
BATCH_SIZE = 4
MAX_LENGTH = 512
MAX_STEPS  = 500
CLUSTER_LR = 3e-4


def _count_params(model: NeuroplasticLFM) -> int:
    return sum(p.numel() for p in model.registry.parameters())


def run_neuroplastic_variant(
    base,
    train_dls,
    eval_dls,
    label_tids,
    device,
    cluster_cls,
    label: str,
) -> List[Dict[str, float]]:
    model = NeuroplasticLFM(base, seed=0, cluster_cls=cluster_cls).to(device)
    acc_matrix: List[Dict[str, float]] = []

    for i, task in enumerate(TASK_ORDER):
        print(f"\n  Task {i+1}/{len(TASK_ORDER)}: {task}")
        model.spawn_cluster(task)

        if i == 0:
            n_params = sum(p.numel() for p in model.registry.clusters[task].parameters())
            print(f"  {label} cluster params: {n_params:,}")

        train_cluster(model, task, train_dls[task], max_steps=MAX_STEPS, lr=CLUSTER_LR, log_every=100)

        snapshot: Dict[str, float] = {}
        for j in range(i + 1):
            t = TASK_ORDER[j]
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

    print("\n── Zero-shot baseline (frozen base, no adapters) ──")
    baseline_accs: Dict[str, float] = {}
    for task in TASK_ORDER:
        acc = evaluate_accuracy(base, eval_dls[task], label_tids[task], device)
        baseline_accs[task] = acc
        print(f"  {task:10s}: {acc:.3f}")

    # ── CfCCluster ────────────────────────────────────────────────────────────
    print(f"\n{'═'*60}")
    print("ABLATION A: NeuroplasticLFM with CfCCluster (recurrent ODE adapter)")
    print(f"{'═'*60}")
    cfc_matrix = run_neuroplastic_variant(
        base, train_dls, eval_dls, label_tids, device,
        cluster_cls=CfCCluster, label="CfCCluster",
    )

    # Reload clean base for MLPCluster run
    print("\nReloading base model for MLPCluster run …")
    del base
    torch.cuda.empty_cache()
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16
    ).to(device)

    # ── MLPCluster ────────────────────────────────────────────────────────────
    print(f"\n{'═'*60}")
    print("ABLATION B: NeuroplasticLFM with MLPCluster (feed-forward ablation)")
    print(f"{'═'*60}")
    mlp_matrix = run_neuroplastic_variant(
        base, train_dls, eval_dls, label_tids, device,
        cluster_cls=MLPCluster, label="MLPCluster",
    )

    # ── Metrics ───────────────────────────────────────────────────────────────
    acc_matrices = {
        "CfCCluster (NeuroplasticLFM)": cfc_matrix,
        "MLPCluster (ablation)":        mlp_matrix,
    }
    cl_results: Dict[str, CLMetrics] = {
        name: compute_cl_metrics(mat, TASK_ORDER, baseline_accs)
        for name, mat in acc_matrices.items()
    }

    print("\n\n" + format_metrics_table(cl_results, TASK_ORDER, acc_matrices))
    print("\nBaseline (zero-shot):")
    for task in TASK_ORDER:
        print(f"  {task:10s}: {baseline_accs[task]:.3f}")

    cfc_ap  = cl_results["CfCCluster (NeuroplasticLFM)"].ap
    mlp_ap  = cl_results["MLPCluster (ablation)"].ap
    delta   = cfc_ap - mlp_ap
    verdict = "CfC > MLP" if delta > 0.01 else ("MLP > CfC" if delta < -0.01 else "CfC ≈ MLP")
    print(f"\nAblation verdict: {verdict}  (ΔAP = {delta:+.3f})")
    if delta > 0.01:
        print("  → CfC τ dynamics contribute meaningfully; recurrent substrate is justified")
    elif delta < -0.01:
        print("  → MLP matches or beats CfC; consider NCP-wired growing adapter framing")
    else:
        print("  → CfC ≈ MLP within noise; τ dynamics provide no clear benefit at this scale")

    return {
        "baseline_accs": baseline_accs,
        "cfc_matrix":    cfc_matrix,
        "mlp_matrix":    mlp_matrix,
        "cl_results":    cl_results,
    }


if __name__ == "__main__":
    main()
