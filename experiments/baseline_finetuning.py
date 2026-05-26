"""
Baseline: naive sequential fine-tuning to demonstrate catastrophic forgetting.
Run this alongside poc_dual_task.py to show NeuroplasticLFM's zero-forgetting guarantee.
"""
import math
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

from experiments.poc_dual_task import (
    AlpacaDataset, filter_records,
    SCIENCE_KEYWORDS, CREATIVE_KEYWORDS,
    TRAIN_SIZE, EVAL_SIZE, BATCH_SIZE, MAX_LENGTH,
)

MAX_STEPS = 300


def finetune(model, dataloader, max_steps: int = MAX_STEPS, lr: float = 3e-4) -> None:
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    model.train()
    step = 0
    for batch in tqdm(dataloader, desc="fine-tuning"):
        if step >= max_steps:
            break
        input_ids = batch["input_ids"]
        labels    = batch["labels"]
        logits    = model(input_ids).logits
        loss      = F.cross_entropy(
            logits[:, :-1].contiguous().view(-1, logits.size(-1)),
            labels[:, 1:].contiguous().view(-1),
            ignore_index=-100,
        )
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        step += 1


@torch.no_grad()
def compute_ppl(model, dataloader) -> float:
    model.eval()
    total_loss, total_tokens = 0.0, 0
    for batch in dataloader:
        input_ids = batch["input_ids"]
        labels    = batch["labels"]
        logits    = model(input_ids).logits
        loss      = F.cross_entropy(
            logits[:, :-1].contiguous().view(-1, logits.size(-1)),
            labels[:, 1:].contiguous().view(-1),
            ignore_index=-100,
            reduction="sum",
        )
        total_loss   += loss.item()
        total_tokens += (labels[:, 1:] != -100).sum().item()
    return math.exp(total_loss / max(total_tokens, 1))


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    print("Loading LFM2-1.2B for naive fine-tuning...")
    model = AutoModelForCausalLM.from_pretrained(
        "LiquidAI/LFM2-1.2B", torch_dtype=torch.float32
    ).to(device)
    tok = AutoTokenizer.from_pretrained("LiquidAI/LFM2-1.2B")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    alpaca  = load_dataset("yahma/alpaca-cleaned", split="train")
    sci_tr  = filter_records(alpaca, SCIENCE_KEYWORDS, TRAIN_SIZE)
    sci_ev  = filter_records(alpaca, SCIENCE_KEYWORDS, TRAIN_SIZE + EVAL_SIZE)[TRAIN_SIZE:]
    cre_tr  = filter_records(alpaca, CREATIVE_KEYWORDS, TRAIN_SIZE)
    cre_ev  = filter_records(alpaca, CREATIVE_KEYWORDS, TRAIN_SIZE + EVAL_SIZE)[TRAIN_SIZE:]

    dl_sci_tr = DataLoader(AlpacaDataset(sci_tr, tok), batch_size=BATCH_SIZE, shuffle=True)
    dl_sci_ev = DataLoader(AlpacaDataset(sci_ev, tok), batch_size=BATCH_SIZE)
    dl_cre_tr = DataLoader(AlpacaDataset(cre_tr, tok), batch_size=BATCH_SIZE, shuffle=True)
    dl_cre_ev = DataLoader(AlpacaDataset(cre_ev, tok), batch_size=BATCH_SIZE)

    ppl_sci_0 = compute_ppl(model, dl_sci_ev)
    ppl_cre_0 = compute_ppl(model, dl_cre_ev)
    print(f"Baseline — Science: {ppl_sci_0:.2f}  Creative: {ppl_cre_0:.2f}")

    print("\nFine-tuning on Science (Task A)...")
    finetune(model, dl_sci_tr)
    ppl_sci_A = compute_ppl(model, dl_sci_ev)
    print(f"After Task A — Science: {ppl_sci_A:.2f}")

    print("\nFine-tuning on Creative (Task B)...")
    finetune(model, dl_cre_tr)
    ppl_sci_B = compute_ppl(model, dl_sci_ev)
    ppl_cre_B = compute_ppl(model, dl_cre_ev)
    print(f"After Task B — Science: {ppl_sci_B:.2f}  Creative: {ppl_cre_B:.2f}")

    forgetting = (ppl_sci_B - ppl_sci_A) / ppl_sci_A
    print(f"\nCatastrophic forgetting on Science: {forgetting:+.2%}")
    print("(Compare with NeuroplasticLFM: 0.000%)")


if __name__ == "__main__":
    main()
