"""Metric helpers for the Jittor CODA-Prompt port."""
from __future__ import annotations

import time
from typing import Iterable, Sequence, Tuple

import jittor as jt
import numpy as np


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / max(1, self.count)


class Timer:
    def __init__(self):
        self.start = None

    def tic(self):
        self.start = time.time()

    def toc(self):
        if self.start is None:
            return 0.0
        return time.time() - self.start


def accuracy(output, target, topk: Sequence[int] = (1,)):
    if isinstance(output, jt.Var):
        logits = output.numpy()
    else:
        logits = np.asarray(output)
    target = np.asarray(target)
    maxk = max(topk)
    top_indices = np.argsort(-logits, axis=1)[:, :maxk]
    correct = top_indices == target[:, None]
    res = []
    for k in topk:
        correct_k = correct[:, :k].sum()
        res.append(float(correct_k) * 100.0 / max(1, len(target)))
    return res[0] if len(res) == 1 else res


def accumulate_acc(output, target, task, meter: AverageMeter, topk: Sequence[int]):
    acc = accuracy(output, target, topk=topk)
    meter.update(acc, len(target))
    return meter
