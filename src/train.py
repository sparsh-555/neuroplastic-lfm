import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Dict, List, Optional
from tqdm import tqdm
from src.model import NeuroplasticLFM


def extract_tau_stats(cluster) -> Dict[str, float]:
    """
    Extract adaptive time-constant statistics from CfC cell.
    time_a and time_b are learned projections that compute τ per token:
        t_interp = sigmoid(time_a(x) * ts + time_b(x))
    Watching these evolve during training is the functional neuroplasticity signal.
    """
    try:
        cell = cluster.cfc.rnn_cell
        return {
            "time_a_mean": cell.time_a.weight.data.mean().item(),
            "time_b_mean": cell.time_b.weight.data.mean().item(),
            "time_a_std": cell.time_a.weight.data.std().item(),
        }
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
    optimizer = torch.optim.AdamW(
        [p for p in cluster.parameters() if p.requires_grad],
        lr=lr,
        weight_decay=0.01,
    )

    cluster.train()
    model.base.eval()

    history = []
    step = 0

    for batch in tqdm(dataloader, desc=f"cluster[{task_id}]"):
        if step >= max_steps:
            break

        input_ids = batch["input_ids"].to(next(cluster.parameters()).device)
        labels = batch["labels"].to(input_ids.device)

        logits = model(input_ids, task_id=task_id)  # (B, L, V)
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
            τ_str = f"  τ_a={τ_info.get('time_a_mean', 0):.4f}" if τ_info else ""
            print(f"  step={step:4d}  loss={loss.item():.4f}  gate={gate:.4f}{τ_str}")

        step += 1

    return history
