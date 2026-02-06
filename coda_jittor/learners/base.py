"""Jittor reimplementation of the baseline learner."""
from __future__ import annotations

import copy
import os
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple

import jittor as jt
import jittor.nn as nn
import numpy as np

from .. import models
from ..utils.metric import AverageMeter, Timer, accumulate_acc
from ..utils.schedulers import build_scheduler


class NormalNN:
    """Jittor analogue of the PyTorch NormalNN baseline."""

    def __init__(self, learner_config: Dict[str, Any]):
        self.log = print
        self.config = copy.deepcopy(learner_config)
        self.out_dim = learner_config['out_dim']
        self.model = self.create_model()
        self.reset_optimizer = True
        self.overwrite = learner_config['overwrite']
        self.batch_size = learner_config['batch_size']
        self.tasks = learner_config['tasks']
        self.top_k = learner_config.get('top_k', 1)
        self.memory_size = self.config['memory']
        self.task_count = 0
        self.dw = bool(self.config['DW']) and self.memory_size > 0
        self.use_cuda = jt.flags.use_cuda != 0
        self.last_valid_out_dim = 0
        self.valid_out_dim = 0
        self.schedule_type = self.config['schedule_type']
        self.schedule = self.config['schedule']
        self.init_optimizer()

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    def learn_batch(self, train_loader: Iterable, train_dataset, model_save_dir: str, val_loader: Optional[Iterable] = None):
        need_train = True
        if not self.overwrite:
            try:
                self.load_model(model_save_dir)
                need_train = False
            except FileNotFoundError:
                need_train = True

        if self.reset_optimizer:
            self.log('Optimizer is reset!')
            self.init_optimizer()

        batch_time = None
        if need_train:
            self.data_weighting(train_dataset)
            losses = AverageMeter()
            acc = AverageMeter()
            batch_time = AverageMeter()
            batch_timer = Timer()
            total_epochs = self.config['schedule'][-1]
            for epoch in range(total_epochs):
                self.epoch = epoch
                if epoch > 0 and self.scheduler is not None:
                    self.scheduler.step(epoch)
                self.log('LR:', float(self.optimizer.lr))
                batch_timer.tic()
                for i, (x, y, task) in enumerate(train_loader):
                    x_var = jt.array(x)
                    y_var = jt.array(y).astype(jt.int32)
                    loss, output = self.update_model(x_var, y_var)
                    batch_time.update(batch_timer.toc())
                    batch_timer.tic()
                    y_np = np.asarray(y)
                    accumulate_acc(output, y_np, task, acc, topk=(self.top_k,))
                    losses.update(float(loss), y_np.shape[0])
                    batch_timer.tic()
                self.log(f'Epoch:{self.epoch + 1:.0f}/{total_epochs:.0f}')
                self.log(f' * Loss {losses.avg:.3f} | Train Acc {acc.avg:.3f}')
                losses.reset()
                acc.reset()

        self.model.eval()
        self.last_valid_out_dim = self.valid_out_dim
        self.first_task = False
        self.task_count += 1
        if self.memory_size > 0:
            train_dataset.update_coreset(self.memory_size, np.arange(self.last_valid_out_dim))

        return None if batch_time is None else batch_time.avg

    def criterion(self, logits: jt.Var, targets: jt.Var, data_weights: jt.Var) -> jt.Var:
        per_example = self._cross_entropy(logits, targets)
        return (per_example * data_weights).mean()

    def update_model(self, inputs: jt.Var, targets: jt.Var, target_scores=None, dw_force=None, kd_index=None):
        weight = jt.ones_like(targets).float32()
        if self.dw_k is not None:
            weight *= self.dw_k[-1]
        logits = self.forward(inputs)
        total_loss = self.criterion(logits, targets, weight)
        self.optimizer.step(total_loss)
        return total_loss.item(), logits

    def validation(self, dataloader: Iterable, model: Optional[Any] = None, task_in=None, task_metric: str = 'acc', verbal: bool = True, task_global: bool = False):
        model = model or self.model
        batch_timer = Timer()
        acc = AverageMeter()
        batch_timer.tic()
        orig_training = getattr(model, 'is_training', lambda: getattr(model, 'training', True))()
        if hasattr(model, 'eval'):
            model.eval()
        for _, (inp, target, task) in enumerate(dataloader):
            inp_var = jt.array(inp)
            if task_in is None:
                output = self._extract_logits(model(inp_var))[:, :self.valid_out_dim]
                accumulate_acc(output, np.asarray(target), task, acc, topk=(self.top_k,))
            else:
                mask = (target >= task_in[0]) & (target < task_in[-1])
                if mask.sum() == 0:
                    continue
                sel_inp = jt.array(inp[mask])
                sel_target = target[mask]
                if task_global:
                    output = self._extract_logits(model(sel_inp))[:, :self.valid_out_dim]
                    accumulate_acc(output, np.asarray(sel_target), task, acc, topk=(self.top_k,))
                else:
                    output = self._extract_logits(model(sel_inp))[:, task_in]
                    accumulate_acc(output, np.asarray(sel_target - task_in[0]), task, acc, topk=(self.top_k,))
        if orig_training and hasattr(model, 'train'):
            model.train()
        elif not orig_training and hasattr(model, 'eval'):
            model.eval()
        if verbal:
            self.log(f' * Val Acc {acc.avg:.3f}, Total time {batch_timer.toc():.2f}')
        return acc.avg

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    def data_weighting(self, dataset, num_seen=None):
        self.dw_k = jt.ones((self.valid_out_dim + 1,)).float32()

    def save_model(self, filename: str):
        path = os.path.join(filename, 'class.pkl')
        self.log('=> Saving Jittor model to:', path)
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
        jt.save(self.model.state_dict(), path)
        self.log('=> Save Done')

    def load_model(self, filename: str):
        path = os.path.join(filename, 'class.pkl')
        state = jt.load(path)
        self.model.load_state_dict(state)
        self.model.eval()
        self.log('=> Load Done')

    def init_optimizer(self):
        opt_name = self.config['optimizer'].lower()
        params = [{'params': self.model.parameters()}]
        lr = self.config['lr']
        weight_decay = self.config['weight_decay']
        momentum = self.config['momentum']
        if opt_name == 'sgd':
            self.optimizer = jt.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
        elif opt_name == 'rmsprop':
            self.optimizer = jt.optim.RMSprop(params, lr=lr, alpha=0.9, eps=1e-8, momentum=momentum, weight_decay=weight_decay)
        elif opt_name == 'adam':
            beta1 = self.config.get('momentum', 0.9)
            self.optimizer = jt.optim.Adam(params, lr=lr, betas=(beta1, 0.999), weight_decay=weight_decay)
        elif opt_name == 'adamw':
            self.optimizer = jt.optim.AdamW(params, lr=lr, betas=(0.9, 0.999), weight_decay=weight_decay)
        else:
            raise ValueError(f'Unsupported optimizer {self.config["optimizer"]} for Jittor port')
        self.scheduler = build_scheduler(self.optimizer, self.schedule_type, self.schedule)

    def create_model(self):
        cfg = self.config
        model_builder = models.get_model(cfg['model_type'], cfg['model_name'])
        model_kwargs = cfg.get('model_kwargs') or {}
        return model_builder(out_dim=self.out_dim, **model_kwargs)

    @staticmethod
    def _extract_logits(output: Any):
        if isinstance(output, (tuple, list)):
            return output[0]
        return output

    def forward(self, x: jt.Var) -> jt.Var:
        logits = self._extract_logits(self.model(x))
        return logits[:, :self.valid_out_dim]

    def add_valid_output_dim(self, dim: int = 0):
        self.log('Incremental class: Old valid output dimension:', self.valid_out_dim)
        self.valid_out_dim += dim
        self.log('Incremental class: New Valid output dimension:', self.valid_out_dim)
        return self.valid_out_dim

    def set_task_id(self, task_id: int) -> None:
        if hasattr(self.model, 'task_id'):
            self.model.task_id = task_id

    def process_task_count(self) -> None:
        if hasattr(self.model, 'process_task_count'):
            self.model.process_task_count()

    def pre_steps(self):
        pass

    def count_parameter(self) -> int:
        return sum(p.numel() for p in self.model.parameters())

    def count_memory(self, dataset_size: Sequence[int]):
        return self.count_parameter() + self.memory_size * np.prod(dataset_size)

    def _cross_entropy(self, logits: jt.Var, targets: jt.Var) -> jt.Var:
        log_probs = nn.log_softmax(logits, dim=1)
        batch_indices = jt.arange(targets.shape[0]).int32()
        gathered = log_probs[batch_indices, targets]
        return -gathered
