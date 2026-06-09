"""
Standard continual learning metrics over an accuracy matrix.

acc_matrix[i][task] = accuracy on `task` evaluated after training on tasks 0..i.
Lower-triangle structure: only indices where task was already introduced are populated.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class CLMetrics:
    ap:    float   # Average Performance: mean acc across all tasks at the end
    bwt:   float   # Backward Transfer: mean acc change on old tasks from new training
    f_ra:  float   # Forgetting Rate: mean max-acc drop (always ≥ 0)
    fwt:   float   # Forward Transfer: diagonal improvement vs. zero-shot baseline


def compute_cl_metrics(
    acc_matrix: List[Dict[str, float]],
    task_order: List[str],
    baseline_accs: Dict[str, float],
) -> CLMetrics:
    """
    Compute the four standard CL metrics.

    Args:
        acc_matrix    : acc_matrix[i][task] after training i+1 tasks (0-indexed)
        task_order    : ordered list of task names (length T)
        baseline_accs : accuracy of the frozen base model on each task (zero-shot)
                        used as the FWT denominator

    Formulas (T = number of tasks, i = 0-indexed):
        AP    = (1/T)   × Σ_{i=0}^{T-1}  Acc_{T-1, Ti}
        BWT   = (1/T-1) × Σ_{i=0}^{T-2}  (Acc_{T-1,Ti} − Acc_{i,Ti})
        F.Ra  = (1/T-1) × Σ_{i=0}^{T-2}  max(0, max_{k≥i} Acc_{k,Ti} − Acc_{T-1,Ti})
        FWT   = (1/T-1) × Σ_{i=1}^{T-1}  (Acc_{i,Ti} − baseline_accs[Ti])
    """
    T = len(task_order)
    assert len(acc_matrix) == T, f"expected {T} rows, got {len(acc_matrix)}"

    # ── AP ────────────────────────────────────────────────────────────────────
    ap = sum(acc_matrix[T - 1][t] for t in task_order) / T

    if T == 1:
        return CLMetrics(ap=ap, bwt=0.0, f_ra=0.0, fwt=0.0)

    # ── BWT ───────────────────────────────────────────────────────────────────
    # Negative = forgetting, positive = backward knowledge transfer
    bwt = sum(
        acc_matrix[T - 1][task_order[i]] - acc_matrix[i][task_order[i]]
        for i in range(T - 1)
    ) / (T - 1)

    # ── F.Ra ──────────────────────────────────────────────────────────────────
    # For each past task, how much did it drop from its best recorded accuracy?
    f_ra_vals: List[float] = []
    for i in range(T - 1):
        task = task_order[i]
        max_acc   = max(acc_matrix[k][task] for k in range(i, T))
        final_acc = acc_matrix[T - 1][task]
        f_ra_vals.append(max(0.0, max_acc - final_acc))
    f_ra = sum(f_ra_vals) / len(f_ra_vals)

    # ── FWT ───────────────────────────────────────────────────────────────────
    # For each non-first task, how much does seeing prior tasks help vs. zero-shot?
    fwt = sum(
        acc_matrix[i][task_order[i]] - baseline_accs.get(task_order[i], 0.0)
        for i in range(1, T)
    ) / (T - 1)

    return CLMetrics(ap=ap, bwt=bwt, f_ra=f_ra, fwt=fwt)


def format_metrics_table(
    results: Dict[str, CLMetrics],
    task_order: List[str],
    acc_matrices: Dict[str, List[Dict[str, float]]],
) -> str:
    """Render a compact summary table for all methods."""
    lines = []
    W = 82
    lines.append("═" * W)
    lines.append(
        f"  {'Method':26s}  {'AP':>7}  {'BWT':>7}  {'F.Ra':>7}  {'FWT':>7}"
    )
    lines.append("─" * W)
    for method, m in results.items():
        lines.append(
            f"  {method:26s}  {m.ap:7.3f}  {m.bwt:+7.3f}  {m.f_ra:7.3f}  {m.fwt:+7.3f}"
        )
    lines.append("═" * W)

    # Per-task accuracy at end (for each method)
    T = len(task_order)
    lines.append("")
    lines.append(f"Per-task final accuracy  (after all {T} tasks trained)")
    lines.append("─" * W)
    header = f"  {'Task':16s}" + "".join(f"  {m[:12]:>12s}" for m in results)
    lines.append(header)
    lines.append("─" * W)
    for task in task_order:
        row = f"  {task:16s}"
        for method in results:
            mat = acc_matrices[method]
            acc = mat[T - 1].get(task, float("nan"))
            row += f"  {acc:12.3f}"
        lines.append(row)
    lines.append("═" * W)

    return "\n".join(lines)
