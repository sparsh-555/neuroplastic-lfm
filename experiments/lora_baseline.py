"""
LoRA r=8 baseline: lr sweep + sequential fine-tuning with parameter-efficient adapters.

Experiment flow:
  1. Sweep lr in [1e-5, 5e-5, 1e-4, 3e-4] — each run gets a fresh adapter via
     model.add_adapter(), so lora_A is Kaiming-reinitialised and lora_B=0 each time.
     No manual weight-copying needed.
  2. Select best lr = lowest science PPL after 1000 steps of science training.
  3. Sequential forgetting experiment at best lr: continue training the already-trained
     science adapter on creative data, then measure science PPL degradation.
  4. Report lr sweep table + head-to-head vs NeuroplasticLFM (Run 007).

Run: PYTHONPATH=/neuroplastic-lfm python experiments/lora_baseline.py
"""
import math
from itertools import cycle

import torch
import torch.nn.functional as F
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, TaskType
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

from experiments.poc_dual_task import (
    AlpacaDataset,
    BATCH_SIZE,
    CREATIVE_KEYWORDS,
    EVAL_SIZE,
    filter_records,
    SCIENCE_KEYWORDS,
    TRAIN_SIZE,
)

MAX_STEPS = 1000
LORA_R = 8
LORA_ALPHA = 16
# Effective lr scaling = lora_alpha / r = 2.0, so effective lr = lr * 2
LR_SWEEP = [1e-5, 5e-5, 1e-4, 3e-4]
# q/k/v only — LFM2 uses out_proj on conv layers too; o_proj would be safer but
# LFM2 attention uses q/k/v naming, not o_proj, so this targets attention only
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj"]


def train_lora(model, dataloader, max_steps: int = MAX_STEPS, lr: float = 1e-4) -> None:
    # Build optimizer over currently-active adapter's trainable params only.
    # set_adapter() sets requires_grad=True only on the active adapter's weights.
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
        weight_decay=0.01,
    )
    device = next(model.parameters()).device
    model.train()
    for step, batch in enumerate(tqdm(cycle(dataloader), total=max_steps, desc=f"lr={lr:.0e}")):
        if step >= max_steps:
            break
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
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


@torch.no_grad()
def compute_ppl(model, dataloader) -> float:
    device = next(model.parameters()).device
    model.eval()
    total_loss, total_tokens = 0.0, 0
    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        logits = model(input_ids, attention_mask=attention_mask).logits
        loss = F.cross_entropy(
            logits[:, :-1].contiguous().view(-1, logits.size(-1)),
            labels[:, 1:].contiguous().view(-1),
            ignore_index=-100,
            reduction="sum",
        )
        total_loss += loss.item()
        total_tokens += (labels[:, 1:] != -100).sum().item()
    return math.exp(total_loss / max(total_tokens, 1))


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    print("Loading LFM2.5-1.2B-Instruct...")
    base = AutoModelForCausalLM.from_pretrained(
        "LiquidAI/LFM2.5-1.2B-Instruct", dtype=torch.bfloat16
    ).to(device)
    tok = AutoTokenizer.from_pretrained("LiquidAI/LFM2.5-1.2B-Instruct")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=LORA_TARGET_MODULES,
        lora_dropout=0.05,
        bias="none",
    )
    # Create model with initial "default" adapter
    model = get_peft_model(base, lora_cfg)
    trainable, total = model.get_nb_trainable_parameters()
    print(f"LoRA trainable params (per adapter): {trainable:,}  ({100*trainable/total:.3f}% of {total:,})")
    print(f"Effective lr scaling: lora_alpha/r = {LORA_ALPHA}/{LORA_R} = {LORA_ALPHA/LORA_R:.1f}x")

    print("\nPreparing datasets...")
    alpaca = load_dataset("yahma/alpaca-cleaned", split="train")
    sci_tr = filter_records(alpaca, SCIENCE_KEYWORDS, TRAIN_SIZE)
    sci_ev = filter_records(alpaca, SCIENCE_KEYWORDS, TRAIN_SIZE + EVAL_SIZE)[TRAIN_SIZE:]
    cre_tr = filter_records(alpaca, CREATIVE_KEYWORDS, TRAIN_SIZE)
    cre_ev = filter_records(alpaca, CREATIVE_KEYWORDS, TRAIN_SIZE + EVAL_SIZE)[TRAIN_SIZE:]

    dl_sci_tr = DataLoader(AlpacaDataset(sci_tr, tok), batch_size=BATCH_SIZE, shuffle=True)
    dl_sci_ev = DataLoader(AlpacaDataset(sci_ev, tok), batch_size=BATCH_SIZE)
    dl_cre_tr = DataLoader(AlpacaDataset(cre_tr, tok), batch_size=BATCH_SIZE, shuffle=True)
    dl_cre_ev = DataLoader(AlpacaDataset(cre_ev, tok), batch_size=BATCH_SIZE)

    ppl_sci_base = compute_ppl(model, dl_sci_ev)
    ppl_cre_base = compute_ppl(model, dl_cre_ev)
    print(f"\nBaseline — Science: {ppl_sci_base:.2f}  Creative: {ppl_cre_base:.2f}")

    # ── LR sweep ─────────────────────────────────────────────────────────────
    # Each lr gets a fresh adapter: add_adapter() reinitialises lora_A (Kaiming)
    # and lora_B (zeros), so runs are fully independent.
    print(f"\n{'─'*58}")
    print("LR sweep — science task, 1000 steps each, fresh adapter per run")
    print(f"{'─'*58}")

    sweep_results = []
    for i, lr in enumerate(LR_SWEEP):
        adapter_name = "default" if i == 0 else f"sweep_{i}"
        if i > 0:
            model.add_adapter(adapter_name, lora_cfg)
        model.set_adapter(adapter_name)

        train_lora(model, dl_sci_tr, max_steps=MAX_STEPS, lr=lr)
        ppl_sci = compute_ppl(model, dl_sci_ev)
        ppl_cre = compute_ppl(model, dl_cre_ev)
        sweep_results.append((lr, adapter_name, ppl_sci, ppl_cre))
        print(f"  lr={lr:.0e} (eff {lr*LORA_ALPHA/LORA_R:.0e})  sci={ppl_sci:.2f}  cre={ppl_cre:.2f}")

    best = min(sweep_results, key=lambda x: x[2])
    best_lr, best_adapter, best_sci_sweep, best_cre_sweep = best
    print(f"\nBest lr: {best_lr:.0e}  adapter: '{best_adapter}'  sci={best_sci_sweep:.2f}  cre={best_cre_sweep:.2f}")

    # ── Sequential forgetting at best lr ─────────────────────────────────────
    # Reactivate the already-trained science adapter and continue on creative.
    # ppl_sci_A comes from the sweep run (same adapter, no retraining needed).
    print(f"\n{'─'*58}")
    print(f"Sequential forgetting experiment — best lr={best_lr:.0e}")
    print(f"{'─'*58}")

    model.set_adapter(best_adapter)
    ppl_sci_A = compute_ppl(model, dl_sci_ev)
    ppl_cre_A = compute_ppl(model, dl_cre_ev)
    print(f"After Science  — Science: {ppl_sci_A:.2f}  Creative: {ppl_cre_A:.2f}")

    print(f"\nContinuing LoRA on Creative ({MAX_STEPS} steps, same adapter)...")
    train_lora(model, dl_cre_tr, max_steps=MAX_STEPS, lr=best_lr)
    ppl_sci_B = compute_ppl(model, dl_sci_ev)
    ppl_cre_B = compute_ppl(model, dl_cre_ev)
    print(f"After Creative — Science: {ppl_sci_B:.2f}  Creative: {ppl_cre_B:.2f}")

    forgetting = (ppl_sci_B - ppl_sci_A) / ppl_sci_A

    # ── Results ───────────────────────────────────────────────────────────────
    col = 10
    print(f"\n{'═'*72}")
    print("LR sweep summary")
    print(f"{'─'*72}")
    print(f"  {'lr':>8}  {'eff lr':>8}  {'sci PPL':>10}  {'cre PPL':>10}")
    print(f"  {'─'*8}  {'─'*8}  {'─'*10}  {'─'*10}")
    for lr, name, psc, pcr in sweep_results:
        flag = "  ← best" if lr == best_lr else ""
        print(f"  {lr:>8.0e}  {lr*LORA_ALPHA/LORA_R:>8.0e}  {psc:>10.2f}  {pcr:>10.2f}{flag}")

    print(f"\n{'═'*72}")
    print(f"Sequential experiment at best lr={best_lr:.0e}  (eff {best_lr*LORA_ALPHA/LORA_R:.0e})")
    print(f"{'─'*72}")
    print(f"  {'':22s}  {'Baseline':>{col}}  {'After sci':>{col}}  {'After cre':>{col}}")
    print(f"  {'─'*22}  {'─'*col}  {'─'*col}  {'─'*col}")
    print(f"  {'Science PPL':22s}  {ppl_sci_base:>{col}.2f}  {ppl_sci_A:>{col}.2f}  {ppl_sci_B:>{col}.2f}")
    print(f"  {'Creative PPL':22s}  {ppl_cre_base:>{col}.2f}  {ppl_cre_A:>{col}.2f}  {ppl_cre_B:>{col}.2f}")
    print(f"{'═'*72}")
    print(f"LoRA forgetting (science): {forgetting:+.2%}")
    print(f"NeuroplasticLFM forgetting: 0.00%  (Run 007: sci 2.42, cre 3.82)")
    print(f"Trainable params — LoRA r=8: {trainable:,}  |  CfC cluster: ~187,000")


if __name__ == "__main__":
    main()
