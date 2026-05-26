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
        # Aggregate across layers for convenience (used in log line)
        ta_means = [stats[f"time_a_l{i}_mean"] for i in range(cell.num_layers)]
        stats["time_a_mean"] = sum(ta_means) / len(ta_means)
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
    # Maturity gate gets 5× higher lr to break the bootstrapping trap:
    # sigmoid'(-3) ≈ 0.045 suppresses the gate gradient at every step.
    # 5× is enough to open the gate to ~0.3-0.5 without overshooting.
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
        # Hard ceiling: sigmoid(0) = 0.5 → cluster contributes at most 50%
        # of its output. Prevents the gate from replacing the base model.
        cluster.maturity.data.clamp_(max=0.0)

        if step % log_every == 0:
            gate = torch.sigmoid(cluster.maturity).item()
            τ_info = extract_tau_stats(cluster)
            record = {"step": step, "loss": loss.item(), "gate": gate, **τ_info}
            history.append(record)
            τ_str = f"  τ_a={τ_info.get('time_a_mean', 0):.4f}" if τ_info else ""
            print(f"  step={step:4d}  loss={loss.item():.4f}  gate={gate:.4f}{τ_str}")

    return history
