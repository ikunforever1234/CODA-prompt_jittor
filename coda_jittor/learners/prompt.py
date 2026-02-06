"""Prompt-based learners implemented with Jittor."""
from __future__ import annotations

from typing import Any, Dict, Iterable, Optional, Sequence

import jittor as jt

from .base import NormalNN
from .. import models
from ..utils.schedulers import build_scheduler


class PromptLearner(NormalNN):
    def __init__(self, learner_config: Dict[str, Any]):
        self.prompt_param = learner_config['prompt_param']
        super().__init__(learner_config)

    def update_model(self, inputs: jt.Var, targets: jt.Var):
        logits, prompt_loss = self.model(inputs, train=True)
        logits = logits[:, :self.valid_out_dim]
        if self.last_valid_out_dim > 0:
            prev = logits[:, :self.last_valid_out_dim] - 1e9
            rest = logits[:, self.last_valid_out_dim:]
            logits = jt.concat([prev, rest], dim=1)
        weight = jt.ones_like(targets).float32()
        if self.dw_k is not None:
            weight *= self.dw_k[-1]
        ce_loss = self.criterion(logits, targets, weight)
        total_loss = ce_loss + prompt_loss.sum()
        self.optimizer.step(total_loss)
        return total_loss.item(), logits

    def init_optimizer(self):
        cfg = self.config
        lr = cfg['lr']
        weight_decay = cfg['weight_decay']
        momentum = cfg['momentum']
        opt_name = cfg['optimizer'].lower()
        prompt_params = []
        if hasattr(self.model, 'prompt') and self.model.prompt is not None:
            prompt_params.extend(self.model.prompt.parameters())
        if hasattr(self.model, 'last'):
            prompt_params.extend(self.model.last.parameters())
        if not prompt_params:
            prompt_params = self.model.parameters()
        param_groups = [{'params': prompt_params}]
        if opt_name == 'sgd':
            self.optimizer = jt.optim.SGD(param_groups, lr=lr, momentum=momentum, weight_decay=weight_decay)
        elif opt_name == 'rmsprop':
            self.optimizer = jt.optim.RMSprop(param_groups, lr=lr, alpha=0.9, eps=1e-8, momentum=momentum, weight_decay=weight_decay)
        elif opt_name == 'adam':
            beta1 = cfg.get('momentum', 0.9)
            self.optimizer = jt.optim.Adam(param_groups, lr=lr, betas=(beta1, 0.999), weight_decay=weight_decay)
        elif opt_name == 'adamw':
            self.optimizer = jt.optim.AdamW(param_groups, lr=lr, betas=(0.9, 0.999), weight_decay=weight_decay)
        else:
            raise ValueError(f'Unsupported optimizer {cfg["optimizer"]} in prompt learner')
        self.scheduler = build_scheduler(self.optimizer, self.config['schedule_type'], self.config['schedule'])

    def create_model(self):
        cfg = self.config
        builder = models.get_model(cfg['model_type'], cfg['model_name'])
        model_kwargs = cfg.get('model_kwargs') or {}
        return builder(out_dim=self.out_dim, prompt_flag='coda', prompt_param=self.prompt_param, **model_kwargs)


class CODAPrompt(PromptLearner):
    pass


class DualPrompt(PromptLearner):
    def create_model(self):
        cfg = self.config
        builder = models.get_model(cfg['model_type'], cfg['model_name'])
        model_kwargs = cfg.get('model_kwargs') or {}
        return builder(out_dim=self.out_dim, prompt_flag='dual', prompt_param=self.prompt_param, **model_kwargs)


class L2P(PromptLearner):
    def create_model(self):
        cfg = self.config
        builder = models.get_model(cfg['model_type'], cfg['model_name'])
        model_kwargs = cfg.get('model_kwargs') or {}
        return builder(out_dim=self.out_dim, prompt_flag='l2p', prompt_param=self.prompt_param, **model_kwargs)
