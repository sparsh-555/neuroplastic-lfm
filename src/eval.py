import math
import torch
import torch.nn.functional as F
from typing import Dict, Optional
from torch.utils.data import DataLoader
from src.model import NeuroplasticLFM


@torch.no_grad()
def perplexity(
    model: NeuroplasticLFM,
    dataloader: DataLoader,
    task_id: Optional[str] = None,
) -> float:
    """Compute per-token perplexity across the dataloader."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    for batch in dataloader:
        device         = next(model.base.parameters()).device
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["labels"].to(device)

        logits = model(input_ids, task_id=task_id, attention_mask=attention_mask)
        loss = F.cross_entropy(
            logits[:, :-1].contiguous().view(-1, logits.size(-1)),
            labels[:, 1:].contiguous().view(-1),
            ignore_index=-100,
            reduction="sum",
        )
        total_loss += loss.item()
        total_tokens += (labels[:, 1:] != -100).sum().item()

    return math.exp(total_loss / max(total_tokens, 1))


def forgetting_score(ppl_before: float, ppl_after: float) -> float:
    """
    Fractional change in perplexity after a new task is trained.
    Positive = forgetting (ppl went up), negative = improvement.
    """
    return (ppl_after - ppl_before) / ppl_before


def tau_stats(cluster) -> Dict[str, float]:
    """Return current tau statistics from a CfCCluster."""
    from src.train import extract_tau_stats

    return extract_tau_stats(cluster)
