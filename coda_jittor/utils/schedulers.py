"""Learning rate schedulers adapted for Jittor optimizers."""
from __future__ import annotations

import math
from typing import List


class _BaseScheduler:
    def __init__(self, optimizer):
        self.optimizer = optimizer
        self.base_lr = float(optimizer.lr)
        self.last_epoch = -1

    def step(self, epoch: int):
        self.last_epoch = epoch
        new_lr = self.get_lr()
        self.optimizer.lr = new_lr

    def get_lr(self) -> float:
        raise NotImplementedError


class CosineScheduler(_BaseScheduler):
    def __init__(self, optimizer, total_epochs: int):
        super().__init__(optimizer)
        self.total_epochs = max(1, total_epochs)

    def get_lr(self) -> float:
        if self.total_epochs <= 1:
            return self.base_lr
        progress = min(self.last_epoch, self.total_epochs)
        return self.base_lr * math.cos((99 * math.pi * progress) / (200 * (self.total_epochs - 1)))


class MultiStepScheduler(_BaseScheduler):
    def __init__(self, optimizer, milestones: List[int], gamma: float = 0.1):
        super().__init__(optimizer)
        self.milestones = set(milestones)
        self.gamma = gamma
        self.decay_count = 0

    def step(self, epoch: int):
        if epoch in self.milestones:
            self.decay_count += 1
            self.optimizer.lr = self.base_lr * (self.gamma ** self.decay_count)
        self.last_epoch = epoch

    def get_lr(self) -> float:
        return self.optimizer.lr


def build_scheduler(optimizer, schedule_type: str, schedule):
    if schedule_type == 'cosine' and schedule:
        return CosineScheduler(optimizer, schedule[-1])
    if schedule_type == 'decay' and schedule:
        return MultiStepScheduler(optimizer, schedule)
    return None
