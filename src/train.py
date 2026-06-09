import torch
import torch.nn.functional as F
from itertools import cycle
from torch.utils.data import DataLoader
from typing import Dict, List
from tqdm import tqdm
from src.model import NeuroplasticLFM


def extract_tau_stats(cluster) -> Dict[str, float]:
    """
    Extract adaptive time-constant statistics from all CfCCell layers.

    WiredCfCCell (used by AutoNCP) contains layer_0..layer_N, each a CfCCell
    with time_a and time_b projections that compute the per-token τ:
        t_interp = sigmoid(time_a(x) * ts + time_b(x))
    Watching these evolve during training is the functional neuroplasticity signal.
    """
    try:
        cell = cluster.cfc.rnn_cell
        stats: Dict[str, float] = {}
        for i in range(cell.num_layers):
            layer = getattr(cell, f"layer_{i}")
            stats[f"time_a_l{i}_mean"] = layer.time_a.weight.data.mean().item()
            stats[f"time_b_l{i}_mean"] = layer.time_b.weight.data.mean().item()
            stats[f"time_a_l{i}_std"]  = layer.time_a.weight.data.std().item()
        # Aggregate across layers for convenience (used in log line).
        # time_a_std is the key diagnostic: if it grows during training, individual
        # neurons are specialising to different time constants — liquid property active.
        ta_means = [stats[f"time_a_l{i}_mean"] for i in range(cell.num_layers)]
        ta_stds  = [stats[f"time_a_l{i}_std"]  for i in range(cell.num_layers)]
        stats["time_a_mean"] = sum(ta_means) / len(ta_means)
        stats["time_a_std"]  = sum(ta_stds)  / len(ta_stds)
        return stats
    except AttributeError:
        return {}


def train_cluster(
    model: NeuroplasticLFM,
    task_id: str,
    dataloader: DataLoader,
    max_steps: int = 500,
    lr: float = 3e-4,
    log_every: int = 50,
) -> List[Dict]:
    cluster = model.registry.get(task_id)
    # Maturity gate gets 5× higher lr so it can open faster than the adapter weights.
    maturity_params = [cluster.maturity]
    other_params    = [p for n, p in cluster.named_parameters()
                       if p.requires_grad and n != "maturity"]
    optimizer = torch.optim.AdamW(
        [
            {"params": other_params,    "lr": lr},
            {"params": maturity_params, "lr": lr * 5},
        ],
        weight_decay=0.01,
    )

    cluster.train()
    model.base.eval()

    history = []
    device = next(cluster.parameters()).device

    for step, batch in enumerate(tqdm(cycle(dataloader), total=max_steps, desc=f"cluster[{task_id}]")):
        if step >= max_steps:
            break

        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["labels"].to(device)

        logits = model(input_ids, task_id=task_id, attention_mask=attention_mask)
        loss = F.cross_entropy(
            logits[:, :-1].contiguous().view(-1, logits.size(-1)),
            labels[:, 1:].contiguous().view(-1),
            ignore_index=-100,
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(cluster.parameters(), 1.0)
        optimizer.step()

        if step % log_every == 0:
            gate = torch.sigmoid(cluster.maturity).item()
            τ_info = extract_tau_stats(cluster)
            record = {"step": step, "loss": loss.item(), "gate": gate, **τ_info}
            history.append(record)
            # τ_a_std is the key signal: if it grows, neurons are specialising to
            # different time constants — the liquid property is activating.
            # τ_a_mean near 0 is expected; watch τ_a_std diverge from its init value.
            if τ_info:
                τ_str = (f"  τ_a_mean={τ_info.get('time_a_mean', 0):.4f}"
                         f"  τ_a_std={τ_info.get('time_a_std', 0):.4f}")
            else:
                τ_str = ""
            print(f"  step={step:4d}  loss={loss.item():.4f}  gate={gate:.4f}{τ_str}")

    return history
