"""Model zoo for the Jittor CODA-Prompt reproduction."""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import jittor as jt
import jittor.nn as nn

from .vit import VisionTransformer


def ortho_penalty(t: jt.Var) -> jt.Var:
    identity = jt.array(np.eye(t.shape[0], dtype=np.float32)).stop_grad()
    diff = jt.matmul(t, jt.transpose(t, (1, 0))) - identity
    return jt.mean(diff ** 2)


def l2_normalize(x: jt.Var, dim: int, eps: float = 1e-6) -> jt.Var:
    norm = jt.norm(x, dim=dim, keepdims=True) + eps
    return x / norm


def _task_slice(total: int, tasks: int, task_idx: int) -> Tuple[int, int]:
    tasks = max(1, tasks)
    base = total // tasks
    rem = total % tasks
    start = task_idx * base + min(task_idx, rem)
    length = base + (1 if task_idx < rem else 0)
    end = min(total, start + length)
    return start, end


def _orthogonalize_block(tensor: jt.Var, start: int, end: int) -> jt.Var:
    data = tensor.numpy()
    flat = data.reshape(data.shape[0], -1)
    for idx in range(start, end):
        vec = flat[idx]
        for j in range(idx):
            denom = np.dot(flat[j], flat[j]) + 1e-8
            vec = vec - np.dot(vec, flat[j]) / denom * flat[j]
        norm = np.linalg.norm(vec)
        if norm < 1e-6:
            vec = np.random.randn(*vec.shape)
            norm = np.linalg.norm(vec)
        flat[idx] = vec / norm
    return jt.array(data)


def _reinit_slice(param: nn.Parameter, start: int, end: int) -> None:
    data = param.numpy()
    flat = data.reshape(data.shape[0], -1)
    width = flat.shape[1]
    for idx in range(start, end):
        vec = np.random.randn(width)
        for j in range(idx):
            denom = np.dot(flat[j], flat[j]) + 1e-8
            vec = vec - np.dot(vec, flat[j]) / denom * flat[j]
        norm = np.linalg.norm(vec)
        if norm < 1e-6:
            vec = np.random.randn(width)
            norm = np.linalg.norm(vec)
        flat[idx] = vec / norm
    param.assign(jt.array(data))


class CodaPrompt(nn.Module):
    def __init__(self, emb_dim: int, n_tasks: int, prompt_param, key_dim: int = 768):
        super().__init__()
        self.emb_dim = emb_dim
        self.key_dim = key_dim
        self.n_tasks = max(1, int(n_tasks))
        self.e_pool_size = int(prompt_param[0])
        self.e_p_length = int(prompt_param[1])
        self.ortho_mu = float(prompt_param[2])
        self.e_layers = [0, 1, 2, 3, 4]
        self.task_count = 0
        self._initialize_parameters()

    def _initialize_parameters(self):
        for layer in self.e_layers:
            e_p = jt.randn((self.e_pool_size, self.e_p_length, self.emb_dim))
            e_k = jt.randn((self.e_pool_size, self.key_dim))
            e_a = jt.randn((self.e_pool_size, self.key_dim))
            e_p = _orthogonalize_block(e_p, 0, self.e_pool_size)
            e_k = _orthogonalize_block(e_k, 0, self.e_pool_size)
            e_a = _orthogonalize_block(e_a, 0, self.e_pool_size)
            setattr(self, f'e_p_{layer}', nn.Parameter(e_p))
            setattr(self, f'e_k_{layer}', nn.Parameter(e_k))
            setattr(self, f'e_a_{layer}', nn.Parameter(e_a))

    def process_task_count(self):
        if self.task_count + 1 >= self.n_tasks:
            return
        self.task_count += 1
        start, end = _task_slice(self.e_pool_size, self.n_tasks, self.task_count)
        for layer in self.e_layers:
            for suffix in ('p', 'k', 'a'):
                param = getattr(self, f'e_{suffix}_{layer}')
                _reinit_slice(param, start, end)

    def forward(self, q_vec: jt.Var, layer_idx: int, x_block: jt.Var, train: bool = False, task_id: Optional[int] = None):
        if layer_idx not in self.e_layers:
            return None, jt.zeros((1,), dtype=jt.float32), x_block

        start, end = _task_slice(self.e_pool_size, self.n_tasks, self.task_count)
        K = getattr(self, f'e_k_{layer_idx}')
        A = getattr(self, f'e_a_{layer_idx}')
        P = getattr(self, f'e_p_{layer_idx}')

        if train:
            head = K[:start].stop_grad() if start > 0 else None
            body = K[start:end]
            K_use = jt.concat([head, body], dim=0) if head is not None else body
            head = A[:start].stop_grad() if start > 0 else None
            body = A[start:end]
            A_use = jt.concat([head, body], dim=0) if head is not None else body
            head = P[:start].stop_grad() if start > 0 else None
            body = P[start:end]
            P_use = jt.concat([head, body], dim=0) if head is not None else body
        else:
            K_use = K[:end]
            A_use = A[:end]
            P_use = P[:end]

        a_query = jt.einsum('bd,kd->bkd', q_vec, A_use)
        n_K = l2_normalize(K_use, dim=1)
        q_norm = l2_normalize(a_query, dim=2)
        cos_sim = jt.einsum('bkd,kd->bk', q_norm, n_K)
        prompts = jt.einsum('bk,kld->bld', cos_sim, P_use)
        split = P_use.shape[1] // 2
        Ek = prompts[:, :split, :]
        Ev = prompts[:, split:, :]
        min_tokens = min(Ek.shape[1], Ev.shape[1])
        if min_tokens > 0:
            Ek = Ek[:, :min_tokens, :]
            Ev = Ev[:, :min_tokens, :]
            prompt_tokens = [Ek, Ev]
        else:
            prompt_tokens = None

        if train and self.ortho_mu > 0:
            loss = (ortho_penalty(K_use) + ortho_penalty(A_use) + ortho_penalty(P_use.reshape((P_use.shape[0], -1)))) * self.ortho_mu
        else:
            loss = jt.zeros((1,), dtype=jt.float32)

        return prompt_tokens, loss, x_block


class ViTZoo(nn.Module):
    def __init__(self, num_classes: int, prompt_flag: str = 'coda', prompt_param=None,
                 img_size: int = 224, patch_size: int = 16, embed_dim: int = 768,
                 depth: int = 12, num_heads: int = 12):
        super().__init__()
        self.prompt_flag = prompt_flag
        self.task_id = None
        self.feat = VisionTransformer(img_size=img_size, patch_size=patch_size, embed_dim=embed_dim, depth=depth, num_heads=num_heads)
        self.last = nn.Linear(embed_dim, num_classes)
        if prompt_flag == 'coda' and prompt_param is not None:
            n_tasks = prompt_param[0] if isinstance(prompt_param[0], int) else int(prompt_param[0])
            self.prompt = CodaPrompt(embed_dim, n_tasks, prompt_param[1], key_dim=embed_dim)
        else:
            self.prompt = None

    def process_task_count(self):
        if self.prompt is not None:
            self.prompt.process_task_count()

    def execute(self, x: jt.Var, pen: bool = False, train: bool = False):
        if self.prompt is not None:
            with jt.no_grad():
                q_tokens, _ = self.feat(x)
                q = q_tokens[:, 0, :]
            tokens, prompt_loss = self.feat(x, prompt=self.prompt, q=q, train=train, task_id=self.task_id)
            feats = tokens[:, 0, :]
        else:
            tokens, prompt_loss = self.feat(x)
            feats = tokens[:, 0, :]
        logits = self.last(feats)
        if pen:
            return feats
        if self.prompt is not None and train:
            return logits, prompt_loss
        return logits


def vit_pt_imnet(out_dim, block_division=None, prompt_flag='coda', prompt_param=None, **kwargs):
    return ViTZoo(num_classes=out_dim, prompt_flag=prompt_flag, prompt_param=prompt_param, **kwargs)


MODEL_FACTORY = {
    'vit_pt_imnet': vit_pt_imnet,
}
