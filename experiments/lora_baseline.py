"""
LoRA r=8 baseline: sequential fine-tuning with parameter-efficient adapters.

Measures:
  - Task-specific PPL improvement vs NeuroplasticLFM (CfC clusters)
  - Catastrophic forgetting under sequential training vs NeuroplasticLFM zero-forgetting
  - Trainable parameter count comparison

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
# q/k/v only — unambiguous in LFM2 (out_proj also exists on conv layers)
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj"]


def train_lora(model, dataloader, max_steps: int = MAX_STEPS, lr: float = 3e-4) -> None:
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
        weight_decay=0.01,
    )
    device = next(model.parameters()).device
    model.train()
    for step, batch in enumerate(tqdm(cycle(dataloader), total=max_steps, desc="lora")):
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
    model = get_peft_model(base, lora_cfg)
    trainable, total = model.get_nb_trainable_parameters()
    print(f"LoRA trainable params: {trainable:,}  ({100*trainable/total:.3f}% of {total:,})")

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

    ppl_sci_0 = compute_ppl(model, dl_sci_ev)
    ppl_cre_0 = compute_ppl(model, dl_cre_ev)
    print(f"\nBaseline  — Science: {ppl_sci_0:.2f}  Creative: {ppl_cre_0:.2f}")

    print(f"\nTraining LoRA on Science ({MAX_STEPS} steps)...")
    train_lora(model, dl_sci_tr)
    ppl_sci_A = compute_ppl(model, dl_sci_ev)
    ppl_cre_A = compute_ppl(model, dl_cre_ev)
    print(f"After Science — Science: {ppl_sci_A:.2f}  Creative: {ppl_cre_A:.2f}")

    print(f"\nContinuing LoRA on Creative ({MAX_STEPS} steps, same model)...")
    train_lora(model, dl_cre_tr)
    ppl_sci_B = compute_ppl(model, dl_sci_ev)
    ppl_cre_B = compute_ppl(model, dl_cre_ev)
    print(f"After Creative — Science: {ppl_sci_B:.2f}  Creative: {ppl_cre_B:.2f}")

    forgetting = (ppl_sci_B - ppl_sci_A) / ppl_sci_A
    col = 12
    print(f"\n{'═'*62}")
    print(f"{'':25s}  {'Baseline':>{col}}  {'LoRA(sci)':>{col}}  {'LoRA(sci+cre)':>{col}}")
    print(f"{'─'*62}")
    print(f"{'Science PPL':25s}  {ppl_sci_0:>{col}.2f}  {ppl_sci_A:>{col}.2f}  {ppl_sci_B:>{col}.2f}")
    print(f"{'Creative PPL':25s}  {ppl_cre_0:>{col}.2f}  {ppl_cre_A:>{col}.2f}  {ppl_cre_B:>{col}.2f}")
    print(f"{'═'*62}")
    print(f"LoRA forgetting (science): {forgetting:+.2%}")
    print(f"NeuroplasticLFM forgetting: 0.00%  (Run 006: sci 2.41, cre 3.85)")
    print(f"Trainable params — LoRA r=8: {trainable:,}  |  CfC cluster: ~187,000")


if __name__ == "__main__":
    main()
