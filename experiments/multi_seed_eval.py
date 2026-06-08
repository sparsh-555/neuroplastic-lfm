"""
Multi-seed evaluation for NeuroplasticLFM.

Runs poc_dual_task with seeds 0, 1, 2 and reports mean ± std on all PPL metrics.
Seeds affect CfC cluster weight initialisation and dataloader shuffle order.

Run: PYTHONPATH=/neuroplastic-lfm python experiments/multi_seed_eval.py
"""
import statistics
import torch

SEEDS = [0, 1, 2]

METRICS = [
    ("ppl_sci_base",  "Baseline science PPL"),
    ("ppl_cre_base",  "Baseline creative PPL"),
    ("ppl_sci_A",     "ClusterA science PPL"),
    ("ppl_cre_A",     "ClusterA creative PPL"),
    ("ppl_sci_B",     "ClusterB science PPL"),
    ("ppl_cre_B",     "ClusterB creative PPL"),
    ("ppl_sci_M",     "Merged science PPL"),
    ("ppl_cre_M",     "Merged creative PPL"),
]


def run_one_seed(seed: int) -> dict:
    """Run the full poc_dual_task experiment under a fixed seed."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Import here so each call gets a fresh module-level state
    from experiments.poc_dual_task import main as poc_main
    print(f"\n{'='*62}")
    print(f"SEED {seed}")
    print(f"{'='*62}")
    return poc_main()


def main() -> None:
    all_results = []

    for seed in SEEDS:
        results = run_one_seed(seed)
        # Drop non-numeric fields (history_a, history_b)
        all_results.append({k: v for k, v in results.items() if isinstance(v, float)})

    # Aggregate
    col_w = 26
    val_w = 14
    print(f"\n\n{'═'*(col_w + val_w*3 + 8)}")
    print(f"Multi-seed summary  (seeds: {SEEDS})")
    print(f"{'─'*(col_w + val_w*3 + 8)}")
    print(f"  {'Metric':{col_w}}  {'Seed 0':>{val_w}}  {'Seed 1':>{val_w}}  {'Seed 2':>{val_w}}  {'Mean ± Std':>{val_w}}")
    print(f"  {'─'*col_w}  {'─'*val_w}  {'─'*val_w}  {'─'*val_w}  {'─'*val_w}")

    for key, label in METRICS:
        vals = [r[key] for r in all_results]
        mean = statistics.mean(vals)
        std  = statistics.stdev(vals) if len(vals) > 1 else 0.0
        row_vals = "  ".join(f"{v:>{val_w}.2f}" for v in vals)
        print(f"  {label:{col_w}}  {row_vals}  {mean:>{val_w-5}.2f} ± {std:.2f}")

    print(f"{'═'*(col_w + val_w*3 + 8)}")

    # Forgetting is always 0.0 by construction — confirm
    for i, r in enumerate(all_results):
        fs = abs(r["ppl_sci_B"] - r["ppl_sci_A"])
        assert fs < 1e-4, f"Seed {SEEDS[i]}: forgetting detected ({fs:.6f})"
    print("Zero forgetting confirmed across all seeds.")


if __name__ == "__main__":
    main()
