"""
PoC experiment: dual-task neuroplastic LFM.

Spawns a CfC cluster for science Q&A (Task A), then creative writing (Task B).
Demonstrates:
  1. Zero forgetting on Task A after Task B training.
  2. Maturity gate opening during training (functional neuroplasticity).
  3. TIES-Merging producing a combined cluster.
"""
import torch
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.model import NeuroplasticLFM
from src.merge import ties_merge
from src.train import train_cluster
from src.eval import perplexity, forgetting_score

SCIENCE_KEYWORDS  = {"science", "math", "physics", "chemistry", "biology",
                     "formula", "equation", "calculate", "theorem"}
CREATIVE_KEYWORDS = {"story", "creative", "write", "poem", "fiction",
                     "narrative", "imagine", "character", "novel"}

MAX_LENGTH = 256
BATCH_SIZE = 4
MAX_STEPS  = 2000
TRAIN_SIZE = 150
EVAL_SIZE  = 50


class AlpacaDataset(Dataset):
    def __init__(self, records, tokenizer, max_length=MAX_LENGTH):
        self.samples = []
        for rec in records:
            instruction = rec.get("instruction", "")
            inp         = rec.get("input", "")
            output      = rec.get("output", "")
            if inp:
                instruction = f"{instruction}\n\n{inp}"

            full_text = tokenizer.apply_chat_template(
                [{"role": "user",      "content": instruction},
                 {"role": "assistant", "content": output}],
                tokenize=False,
                add_generation_prompt=False,
            )
            prompt_text = tokenizer.apply_chat_template(
                [{"role": "user", "content": instruction}],
                tokenize=False,
                add_generation_prompt=True,
            )

            full_enc   = tokenizer(full_text,   add_special_tokens=False,
                                   truncation=True, max_length=max_length,
                                   padding="max_length", return_tensors="pt")
            prompt_enc = tokenizer(prompt_text, add_special_tokens=False,
                                   truncation=True, max_length=max_length,
                                   return_tensors="pt")

            ids           = full_enc["input_ids"].squeeze(0)
            attention_mask = full_enc["attention_mask"].squeeze(0)
            n_prompt      = min(prompt_enc["input_ids"].shape[1], max_length)
            labels        = ids.clone()
            labels[:n_prompt]                        = -100
            labels[labels == tokenizer.pad_token_id] = -100
            self.samples.append({
                "input_ids":      ids,
                "attention_mask": attention_mask,
                "labels":         labels,
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def filter_records(dataset, keywords, n):
    results = []
    for rec in dataset:
        text = (rec.get("instruction", "") + " " + rec.get("output", "")).lower()
        if any(kw in text for kw in keywords):
            results.append(rec)
        if len(results) >= n:
            break
    return results


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    print("Loading LFM2.5-1.2B-Instruct (frozen base)...")
    base = AutoModelForCausalLM.from_pretrained(
        "LiquidAI/LFM2.5-1.2B-Instruct", dtype=torch.bfloat16
    ).to(device)
    tok  = AutoTokenizer.from_pretrained("LiquidAI/LFM2.5-1.2B-Instruct")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = NeuroplasticLFM(base).to(device)

    print("Preparing datasets from yahma/alpaca-cleaned...")
    alpaca = load_dataset("yahma/alpaca-cleaned", split="train")
    sci_tr  = filter_records(alpaca, SCIENCE_KEYWORDS, TRAIN_SIZE)
    sci_ev  = filter_records(alpaca, SCIENCE_KEYWORDS, TRAIN_SIZE + EVAL_SIZE)[TRAIN_SIZE:]
    cre_tr  = filter_records(alpaca, CREATIVE_KEYWORDS, TRAIN_SIZE)
    cre_ev  = filter_records(alpaca, CREATIVE_KEYWORDS, TRAIN_SIZE + EVAL_SIZE)[TRAIN_SIZE:]

    dl_sci_tr = DataLoader(AlpacaDataset(sci_tr, tok), batch_size=BATCH_SIZE, shuffle=True)
    dl_sci_ev = DataLoader(AlpacaDataset(sci_ev, tok), batch_size=BATCH_SIZE)
    dl_cre_tr = DataLoader(AlpacaDataset(cre_tr, tok), batch_size=BATCH_SIZE, shuffle=True)
    dl_cre_ev = DataLoader(AlpacaDataset(cre_ev, tok), batch_size=BATCH_SIZE)

    # ── Baseline (no clusters) ────────────────────────────────────────────
    print("\n── Baseline (no clusters) ──")
    ppl_sci_base = perplexity(model, dl_sci_ev, task_id=None)
    ppl_cre_base = perplexity(model, dl_cre_ev, task_id=None)
    print(f"  Science PPL:  {ppl_sci_base:.2f}")
    print(f"  Creative PPL: {ppl_cre_base:.2f}")

    # ── Cluster A — Science ───────────────────────────────────────────────
    print("\n── Training Cluster A (science) ──")
    model.spawn_cluster("science")
    history_a = train_cluster(model, "science", dl_sci_tr,
                               max_steps=MAX_STEPS, log_every=50)
    ppl_sci_A = perplexity(model, dl_sci_ev, task_id="science")
    ppl_cre_A = perplexity(model, dl_cre_ev, task_id="science")
    print(f"  Science PPL: {ppl_sci_A:.2f}  Creative PPL: {ppl_cre_A:.2f}")

    # ── Cluster B — Creative ──────────────────────────────────────────────
    print("\n── Training Cluster B (creative) ──")
    model.spawn_cluster("creative")
    history_b = train_cluster(model, "creative", dl_cre_tr,
                               max_steps=MAX_STEPS, log_every=50)
    ppl_sci_B = perplexity(model, dl_sci_ev, task_id="science")
    ppl_cre_B = perplexity(model, dl_cre_ev, task_id="creative")
    print(f"  Science PPL: {ppl_sci_B:.2f}  Creative PPL: {ppl_cre_B:.2f}")

    # ── Zero-forgetting check ─────────────────────────────────────────────
    fs = forgetting_score(ppl_sci_A, ppl_sci_B)
    print(f"\n  Forgetting score (science): {fs:+.6f}  [expected: 0.000000]")
    assert abs(fs) < 1e-4, (
        f"Forgetting detected ({fs:+.4%})! "
        "Cluster A should be frozen and unaffected by Cluster B training."
    )

    # ── TIES Merge ────────────────────────────────────────────────────────
    print("\n── Merging A + B via TIES-Merging ──")
    ties_merge(model.registry, "science", "creative", "merged")
    ppl_sci_M = perplexity(model, dl_sci_ev, task_id="merged")
    ppl_cre_M = perplexity(model, dl_cre_ev, task_id="merged")
    print(f"  Merged — Science PPL: {ppl_sci_M:.2f}  Creative PPL: {ppl_cre_M:.2f}")

    # ── Summary table ─────────────────────────────────────────────────────
    col = 10
    print(f"\n{'═'*62}")
    print(f"{'':25s}  {'Baseline':>{col}}  {'ClusterA':>{col}}  "
          f"{'ClusterB':>{col}}  {'Merged':>{col}}")
    print(f"{'─'*62}")
    print(f"{'Science PPL':25s}  {ppl_sci_base:>{col}.2f}  {ppl_sci_A:>{col}.2f}  "
          f"{ppl_sci_B:>{col}.2f}  {ppl_sci_M:>{col}.2f}")
    print(f"{'Creative PPL':25s}  {ppl_cre_base:>{col}.2f}  {ppl_cre_A:>{col}.2f}  "
          f"{ppl_cre_B:>{col}.2f}  {ppl_cre_M:>{col}.2f}")
    print(f"{'═'*62}")
    print(f"Zero forgetting: {abs(fs) < 1e-4}")

    return {
        "ppl_sci_base": ppl_sci_base, "ppl_cre_base": ppl_cre_base,
        "ppl_sci_A": ppl_sci_A,       "ppl_cre_A": ppl_cre_A,
        "ppl_sci_B": ppl_sci_B,       "ppl_cre_B": ppl_cre_B,
        "ppl_sci_M": ppl_sci_M,       "ppl_cre_M": ppl_cre_M,
        "history_a": history_a,       "history_b": history_b,
    }


if __name__ == "__main__":
    main()
