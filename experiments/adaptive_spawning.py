"""
Autonomous cluster spawning demonstration.

A stream of examples arrives without task labels. The AdaptiveSpawnController
monitors loss and spawns clusters automatically when it detects distribution
shifts — no manual spawn_cluster("science") calls.

Sequence:
  1. Calibrate baseline loss on a small mixed sample
  2. Stream science examples → loss rises above threshold → cluster auto-spawns
  3. Stream creative examples → loss rises again → second cluster auto-spawns
  4. Evaluate: PPL with auto-spawned clusters vs baseline

Run: PYTHONPATH=/neuroplastic-lfm python experiments/adaptive_spawning.py
"""
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from experiments.poc_dual_task import (
    AlpacaDataset,
    BATCH_SIZE,
    CREATIVE_KEYWORDS,
    EVAL_SIZE,
    filter_records,
    SCIENCE_KEYWORDS,
    TRAIN_SIZE,
)
from src.eval import perplexity
from src.model import NeuroplasticLFM
from src.spawn_trigger import AdaptiveSpawnController


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    print("Loading LFM2.5-1.2B-Instruct (frozen)...")
    base = AutoModelForCausalLM.from_pretrained(
        "LiquidAI/LFM2.5-1.2B-Instruct", dtype=torch.bfloat16
    ).to(device)
    tok = AutoTokenizer.from_pretrained("LiquidAI/LFM2.5-1.2B-Instruct")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = NeuroplasticLFM(base).to(device)

    print("\nPreparing datasets...")
    alpaca = load_dataset("yahma/alpaca-cleaned", split="train")
    sci_tr = filter_records(alpaca, SCIENCE_KEYWORDS, TRAIN_SIZE)
    sci_ev = filter_records(alpaca, SCIENCE_KEYWORDS, TRAIN_SIZE + EVAL_SIZE)[TRAIN_SIZE:]
    cre_tr = filter_records(alpaca, CREATIVE_KEYWORDS, TRAIN_SIZE)
    cre_ev = filter_records(alpaca, CREATIVE_KEYWORDS, TRAIN_SIZE + EVAL_SIZE)[TRAIN_SIZE:]

    # Use a small random sample as calibration (simulates "normal" traffic)
    calib_records = sci_tr[:25] + cre_tr[:25]
    dl_calib = DataLoader(AlpacaDataset(calib_records, tok), batch_size=BATCH_SIZE, shuffle=True)
    dl_sci_stream = DataLoader(AlpacaDataset(sci_tr, tok), batch_size=BATCH_SIZE, shuffle=False)
    dl_cre_stream = DataLoader(AlpacaDataset(cre_tr, tok), batch_size=BATCH_SIZE, shuffle=False)
    dl_sci_ev = DataLoader(AlpacaDataset(sci_ev, tok), batch_size=BATCH_SIZE)
    dl_cre_ev = DataLoader(AlpacaDataset(cre_ev, tok), batch_size=BATCH_SIZE)

    # ── Baseline ─────────────────────────────────────────────────────────────
    ppl_sci_base = perplexity(model, dl_sci_ev, task_id=None)
    ppl_cre_base = perplexity(model, dl_cre_ev, task_id=None)
    print(f"\nBaseline — Science PPL: {ppl_sci_base:.2f}  Creative PPL: {ppl_cre_base:.2f}")

    # ── Calibrate controller ──────────────────────────────────────────────────
    controller = AdaptiveSpawnController(
        model,
        window=20,          # rolling window of 20 batches
        threshold=1.15,     # spawn when rolling loss > baseline * 1.15
        buffer_size=200,    # keep last 200 batches as training buffer
        train_steps=500,    # train each auto-spawned cluster for 500 steps
        min_buffer_to_spawn=40,
    )
    print("\nCalibrating loss baseline...")
    controller.calibrate(dl_calib, n_batches=20)

    # ── Stream science (novel distribution → triggers spawn) ──────────────────
    print("\nStreaming science examples (no task labels)...")
    sci_cluster_id = None
    for batch in dl_sci_stream:
        cluster_id = controller.process(batch)
        if cluster_id is not None and sci_cluster_id is None:
            sci_cluster_id = cluster_id

    print(f"Science stream complete. Active cluster: {sci_cluster_id!r}")

    # ── Stream creative (second novel distribution → triggers second spawn) ───
    print("\nStreaming creative examples (no task labels)...")
    cre_cluster_id = None
    for batch in dl_cre_stream:
        cluster_id = controller.process(batch)
        if cluster_id != sci_cluster_id and cluster_id is not None and cre_cluster_id is None:
            cre_cluster_id = cluster_id

    print(f"Creative stream complete. Active cluster: {cre_cluster_id!r}")

    # ── Evaluate auto-spawned clusters ────────────────────────────────────────
    print("\nEvaluating auto-spawned clusters...")
    ppl_sci_auto = perplexity(model, dl_sci_ev, task_id=sci_cluster_id)
    ppl_cre_auto = perplexity(model, dl_cre_ev, task_id=cre_cluster_id)

    col = 12
    print(f"\n{'═'*60}")
    print(f"{'':25s}  {'Baseline':>{col}}  {'Auto-cluster':>{col}}")
    print(f"{'─'*60}")
    print(f"{'Science PPL':25s}  {ppl_sci_base:>{col}.2f}  {ppl_sci_auto:>{col}.2f}")
    print(f"{'Creative PPL':25s}  {ppl_cre_base:>{col}.2f}  {ppl_cre_auto:>{col}.2f}")
    print(f"{'═'*60}")
    print(f"Clusters spawned: {controller._spawn_count}  (no task labels used)")
    print(f"Science cluster: '{sci_cluster_id}'  Creative cluster: '{cre_cluster_id}'")


if __name__ == "__main__":
    main()
