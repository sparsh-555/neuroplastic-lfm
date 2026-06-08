"""
Autonomous spawn trigger for NeuroplasticLFM.

Monitors per-batch loss against a calibrated baseline. When the rolling average
exceeds baseline * threshold (distribution shift detected), spawns a new cluster
and trains it on the buffered high-loss examples — no task labels required.

After each spawn the baseline is re-calibrated with the new cluster active.
This means the threshold adapts to the model's improved capability: subsequent
spawns only fire if the distribution is genuinely novel relative to everything
the model can currently handle, not relative to the original frozen baseline.
"""
from collections import deque
from typing import Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from src.model import NeuroplasticLFM
from src.train import train_cluster


class LossMonitor:
    """Rolling window loss tracker with spawn-trigger logic."""

    def __init__(self, window: int = 20, threshold: float = 1.15):
        self.window = window
        self.threshold = threshold
        self._buffer: deque = deque(maxlen=window)
        self.baseline: Optional[float] = None

    def update(self, loss: float) -> None:
        self._buffer.append(loss)

    def set_baseline(self, loss: float) -> None:
        self.baseline = loss

    def rolling_loss(self) -> float:
        if not self._buffer:
            return float("inf")
        return sum(self._buffer) / len(self._buffer)

    def should_spawn(self) -> bool:
        """True when rolling loss exceeds baseline by threshold ratio."""
        if self.baseline is None or len(self._buffer) < self.window // 2:
            return False
        return self.rolling_loss() > self.baseline * self.threshold


class ExampleBuffer:
    """Accumulates (input_ids, attention_mask, labels) for cluster training."""

    def __init__(self, capacity: int = 200):
        self.capacity = capacity
        self._ids: list = []
        self._masks: list = []
        self._labels: list = []

    def add(self, batch: dict) -> None:
        self._ids.append(batch["input_ids"].cpu())
        self._masks.append(batch["attention_mask"].cpu())
        self._labels.append(batch["labels"].cpu())
        if len(self._ids) > self.capacity:
            self._ids.pop(0)
            self._masks.pop(0)
            self._labels.pop(0)

    def __len__(self) -> int:
        return len(self._ids)

    def as_dataloader(self, batch_size: int = 4) -> DataLoader:
        ids = torch.cat(self._ids, dim=0)
        masks = torch.cat(self._masks, dim=0)
        labels = torch.cat(self._labels, dim=0)

        class _DictDS(torch.utils.data.Dataset):
            def __init__(self, ids, masks, labels):
                self.ids = ids
                self.masks = masks
                self.labels = labels

            def __len__(self):
                return len(self.ids)

            def __getitem__(self, i):
                return {
                    "input_ids": self.ids[i],
                    "attention_mask": self.masks[i],
                    "labels": self.labels[i],
                }

        return DataLoader(
            _DictDS(ids, masks, labels), batch_size=batch_size, shuffle=True
        )


class AdaptiveSpawnController:
    """
    Wraps NeuroplasticLFM with autonomous cluster spawning.

    Usage:
        controller = AdaptiveSpawnController(model)
        controller.calibrate(calibration_dataloader)   # sets baseline loss
        for batch in stream:
            cluster_id = controller.process(batch)     # spawns if needed
    """

    def __init__(
        self,
        model: NeuroplasticLFM,
        window: int = 20,
        threshold: float = 1.15,
        buffer_size: int = 200,
        train_steps: int = 500,
        min_buffer_to_spawn: int = 50,
        cooldown_steps: int = 20,
    ):
        self.model = model
        self.monitor = LossMonitor(window=window, threshold=threshold)
        self.buffer = ExampleBuffer(capacity=buffer_size)
        self.train_steps = train_steps
        self.min_buffer_to_spawn = min_buffer_to_spawn
        self.cooldown_steps = cooldown_steps
        self._spawn_count = 0
        self._active_cluster: Optional[str] = None
        self._cooldown_remaining: int = 0

    @torch.no_grad()
    def _batch_loss(self, batch: dict, task_id: Optional[str] = None) -> float:
        device = next(self.model.parameters()).device
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        logits = self.model(input_ids, task_id=task_id, attention_mask=attention_mask)
        return F.cross_entropy(
            logits[:, :-1].contiguous().view(-1, logits.size(-1)),
            labels[:, 1:].contiguous().view(-1),
            ignore_index=-100,
        ).item()

    @torch.no_grad()
    def _recalibrate(self, n_batches: int = 20) -> float:
        """
        Re-measure baseline with the current active cluster on buffered examples.

        Called after each spawn so the threshold adapts to the improved model
        capability rather than staying fixed at the original frozen-baseline level.
        A second spawn only fires if the data is genuinely novel relative to what
        all existing clusters together can already handle.
        """
        dl = self.buffer.as_dataloader()
        losses = []
        for i, batch in enumerate(dl):
            if i >= n_batches:
                break
            losses.append(self._batch_loss(batch, task_id=self._active_cluster))
        new_baseline = sum(losses) / len(losses) if losses else self.monitor.baseline
        self.monitor.set_baseline(new_baseline)
        print(f"[SpawnTrigger] Baseline recalibrated → {new_baseline:.4f} (with cluster '{self._active_cluster}' active)")
        return new_baseline

    def calibrate(self, dataloader: DataLoader, n_batches: int = 20) -> float:
        """Compute baseline loss on a representative sample. Call before streaming."""
        losses = []
        for i, batch in enumerate(dataloader):
            if i >= n_batches:
                break
            losses.append(self._batch_loss(batch))
        baseline = sum(losses) / len(losses)
        self.monitor.set_baseline(baseline)
        print(f"[SpawnTrigger] Baseline loss calibrated: {baseline:.4f}")
        return baseline

    def process(self, batch: dict) -> Optional[str]:
        """
        Feed one batch through the controller.

        Returns the cluster_id used for this batch (None = base only).
        Spawns and trains a new cluster if the loss monitor fires.
        """
        loss = self._batch_loss(batch, task_id=self._active_cluster)
        self.monitor.update(loss)
        self.buffer.add(batch)

        if self._cooldown_remaining > 0:
            self._cooldown_remaining -= 1

        can_spawn = (
            self._cooldown_remaining == 0
            and self.monitor.should_spawn()
            and len(self.buffer) >= self.min_buffer_to_spawn
        )
        if can_spawn:
            cluster_id = f"auto_{self._spawn_count}"
            print(
                f"\n[SpawnTrigger] Rolling loss {self.monitor.rolling_loss():.4f} > "
                f"{self.monitor.baseline * self.monitor.threshold:.4f} "
                f"— spawning cluster '{cluster_id}'"
            )
            self.model.spawn_cluster(cluster_id)
            dl = self.buffer.as_dataloader()
            train_cluster(
                self.model, cluster_id, dl,
                max_steps=self.train_steps, log_every=100,
            )
            self._spawn_count += 1
            self._active_cluster = cluster_id

            # Re-calibrate baseline with the new cluster active so the threshold
            # reflects current capability, not the original frozen baseline.
            self._recalibrate(n_batches=20)

            self.monitor._buffer.clear()
            self.buffer = ExampleBuffer(capacity=self.buffer.capacity)
            self._cooldown_remaining = self.cooldown_steps  # short: just for buffer refill
            print(f"[SpawnTrigger] Cluster '{cluster_id}' active. Cooldown: {self.cooldown_steps} steps.")

        return self._active_cluster
