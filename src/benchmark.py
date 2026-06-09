"""
Long Sequence Benchmark for continual learning evaluation.

Tasks (in order): Yelp → IMDB → BoolQ → MultiRC → DBpedia-14
Same benchmark used by CaLoRA (NeurIPS 2025) and related CL-LLM papers.

Each task is formatted as a completion prompt; accuracy is measured by
comparing log-probs of candidate label tokens at the last prompt position.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from functools import partial
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedTokenizerBase


@dataclass(frozen=True)
class TaskMeta:
    hf_name: str
    hf_config: Optional[str]
    train_split: str
    eval_split: str
    label_words: Tuple[str, ...]  # index → word


TASK_META: Dict[str, TaskMeta] = {
    "yelp": TaskMeta(
        "yelp_polarity", None, "train", "test",
        ("negative", "positive"),
    ),
    "imdb": TaskMeta(
        "imdb", None, "train", "test",
        ("negative", "positive"),
    ),
    "boolq": TaskMeta(
        "super_glue", "boolq", "train", "validation",
        ("no", "yes"),
    ),
    "multirc": TaskMeta(
        "super_glue", "multirc", "train", "validation",
        ("no", "yes"),
    ),
    "dbpedia": TaskMeta(
        "dbpedia_14", None, "train", "test",
        (
            "company", "education", "artist", "athlete", "office",
            "transport", "building", "nature", "village", "animal",
            "plant", "album", "film", "book",
        ),
    ),
}

TASK_ORDER: List[str] = ["yelp", "imdb", "boolq", "multirc", "dbpedia"]


# ── Formatters ─────────────────────────────────────────────────────────────────

def _fmt_yelp(ex: dict) -> Tuple[str, int]:
    return f"Review: {ex['text'][:400]}\nSentiment: ", int(ex["label"])


def _fmt_imdb(ex: dict) -> Tuple[str, int]:
    return f"Review: {ex['text'][:400]}\nSentiment: ", int(ex["label"])


def _fmt_boolq(ex: dict) -> Tuple[str, int]:
    return (
        f"Passage: {ex['passage'][:350]}\nQuestion: {ex['question']}\nAnswer: ",
        int(ex["label"]),
    )


def _fmt_multirc(ex: dict) -> Tuple[str, int]:
    return (
        f"Paragraph: {ex['paragraph'][:280]}\n"
        f"Question: {ex['question']}\n"
        f"Answer: {ex['answer']}\n"
        f"Is this answer correct? ",
        int(ex["label"]),
    )


def _fmt_dbpedia(ex: dict) -> Tuple[str, int]:
    content = (ex.get("content") or ex.get("abstract", ""))[:350]
    return f"Article: {content}\nCategory: ", int(ex["label"])


_FORMATTERS = {
    "yelp": _fmt_yelp,
    "imdb": _fmt_imdb,
    "boolq": _fmt_boolq,
    "multirc": _fmt_multirc,
    "dbpedia": _fmt_dbpedia,
}


# ── Dataset ────────────────────────────────────────────────────────────────────

class CLDataset(Dataset):
    """
    Classification dataset for CL experiments.

    Each item is (prompt + label_token).  During training the cross-entropy loss
    is restricted to the label token position only (all other labels are -100).
    During evaluation, prompt_len lets callers index the logit at the boundary.
    """

    def __init__(
        self,
        examples: List[Tuple[str, int]],
        tokenizer: PreTrainedTokenizerBase,
        label_token_ids: List[int],
        max_length: int = 512,
    ):
        self._examples = examples
        self._tok = tokenizer
        self._label_token_ids = label_token_ids
        self._max_length = max_length

    def __len__(self) -> int:
        return len(self._examples)

    def __getitem__(self, idx: int) -> dict:
        prompt, label_idx = self._examples[idx]
        label_tok = self._label_token_ids[label_idx]

        enc = self._tok(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self._max_length - 1,
            add_special_tokens=True,
        )
        prompt_ids = enc["input_ids"][0]
        prompt_len = int(prompt_ids.shape[0])

        # Full sequence: [prompt_tokens ..., label_token]
        full_ids = torch.cat([prompt_ids, torch.tensor([label_tok])], dim=0)

        # Training labels: -100 everywhere except the label position
        labels = torch.full_like(full_ids, -100)
        labels[-1] = label_tok

        return {
            "input_ids": full_ids,
            "attention_mask": torch.ones_like(full_ids),
            "labels": labels,
            "label_idx": label_idx,
            "prompt_len": prompt_len,
        }


def _collate_cl(batch: List[dict], pad_id: int) -> dict:
    max_len = max(item["input_ids"].shape[0] for item in batch)

    def _pad(t: torch.Tensor, val: int) -> torch.Tensor:
        n = max_len - t.shape[0]
        return t if n == 0 else F.pad(t, (0, n), value=val)

    return {
        "input_ids":      torch.stack([_pad(b["input_ids"],      pad_id) for b in batch]),
        "attention_mask": torch.stack([_pad(b["attention_mask"],       0) for b in batch]),
        "labels":         torch.stack([_pad(b["labels"],            -100) for b in batch]),
        "label_idx":      torch.tensor([b["label_idx"]  for b in batch]),
        "prompt_len":     torch.tensor([b["prompt_len"] for b in batch]),
    }


# ── Loader builder ─────────────────────────────────────────────────────────────

def _sample(rows: list, n: int, seed: int = 42) -> list:
    rng = random.Random(seed)
    return rng.sample(rows, min(n, len(rows)))


def build_cl_dataloaders(
    tokenizer: PreTrainedTokenizerBase,
    n_train: int = 200,
    n_eval: int = 100,
    batch_size: int = 4,
    max_length: int = 512,
    seed: int = 42,
) -> Tuple[
    Dict[str, DataLoader],
    Dict[str, DataLoader],
    Dict[str, List[int]],
]:
    """
    Load and tokenize all 5 Long Sequence Benchmark tasks.

    Returns:
        train_dls  — one DataLoader per task for training
        eval_dls   — one DataLoader per task for evaluation
        label_tids — {task: [token_id_per_label]} used in evaluate_accuracy
    """
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    train_dls: Dict[str, DataLoader] = {}
    eval_dls:  Dict[str, DataLoader] = {}
    label_tids: Dict[str, List[int]] = {}

    for task in TASK_ORDER:
        meta = TASK_META[task]
        fmt  = _FORMATTERS[task]

        # Compute label token IDs (take first token of each label word)
        tids: List[int] = []
        for word in meta.label_words:
            enc = tokenizer.encode(f" {word}", add_special_tokens=False)
            tids.append(int(enc[0]))
        label_tids[task] = tids

        print(f"  Loading {task} ({meta.hf_name}) …")
        if meta.hf_config:
            ds_train = load_dataset(meta.hf_name, meta.hf_config, split=meta.train_split)
            ds_eval  = load_dataset(meta.hf_name, meta.hf_config, split=meta.eval_split)
        else:
            ds_train = load_dataset(meta.hf_name, split=meta.train_split)
            ds_eval  = load_dataset(meta.hf_name, split=meta.eval_split)

        train_rows = _sample(list(ds_train), n_train, seed=seed)
        eval_rows  = _sample(list(ds_eval),  n_eval,  seed=seed + 1)

        train_examples = [fmt(r) for r in train_rows]
        eval_examples  = [fmt(r) for r in eval_rows]

        collate = partial(_collate_cl, pad_id=pad_id)

        train_dls[task] = DataLoader(
            CLDataset(train_examples, tokenizer, tids, max_length),
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate,
        )
        eval_dls[task] = DataLoader(
            CLDataset(eval_examples, tokenizer, tids, max_length),
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate,
        )
        n_tr = len(train_examples)
        n_ev = len(eval_examples)
        print(f"    {task:10s}: {n_tr} train / {n_ev} eval  labels={list(meta.label_words)}")

    return train_dls, eval_dls, label_tids


# ── Accuracy evaluation ────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_accuracy(
    model,
    eval_loader: DataLoader,
    label_token_ids: List[int],
    device: torch.device,
    task_id: Optional[str] = None,
    is_nplm: bool = False,
) -> float:
    """
    Classification accuracy via log-prob ranking at the last prompt token position.

    For each example the model scores each label token ID from its logit at
    position (prompt_len - 1); the argmax is the prediction.

    Args:
        model           : PEFT model (has .logits) or NeuroplasticLFM (raw tensor)
        eval_loader     : DataLoader produced by build_cl_dataloaders
        label_token_ids : list of vocab token IDs, one per class
        device          : torch device
        task_id         : required when is_nplm=True
        is_nplm         : True for NeuroplasticLFM, False for HF / PEFT models
    """
    model.eval()
    correct = 0
    total   = 0
    label_tids = torch.tensor(label_token_ids, device=device)

    for batch in eval_loader:
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        prompt_lens    = batch["prompt_len"]
        gold_labels    = batch["label_idx"]

        if is_nplm:
            logits = model(input_ids, task_id=task_id, attention_mask=attention_mask)
        else:
            out    = model(input_ids, attention_mask=attention_mask)
            logits = out.logits

        # logits: [B, T, V]
        for b in range(input_ids.shape[0]):
            pos    = int(prompt_lens[b]) - 1      # last prompt token position
            scores = logits[b, pos, label_tids]   # [num_labels]
            pred   = int(scores.argmax())
            if pred == int(gold_labels[b]):
                correct += 1
            total += 1

    return correct / total if total > 0 else 0.0
