"""Vision Transformer components implemented in Jittor."""
from __future__ import annotations

from typing import Optional

import jittor as jt
import jittor.nn as nn


class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = (img_size, img_size) if isinstance(img_size, int) else img_size
        patch_size = (patch_size, patch_size) if isinstance(patch_size, int) else patch_size
        self.num_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])
        self.proj = nn.Conv(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def execute(self, x: jt.Var) -> jt.Var:
        x = self.proj(x)
        x = x.reshape((x.shape[0], x.shape[1], -1))
        x = x.permute(0, 2, 1)
        return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def execute(self, x: jt.Var) -> jt.Var:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=True, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def execute(self, x: jt.Var, prompt=None) -> jt.Var:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        if prompt is not None:
            pk, pv = prompt
            pk = pk.reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
            pv = pv.reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
            k = jt.concat([pk, k], dim=2)
            v = jt.concat([pv, v], dim=2)
        attn = jt.matmul(q, jt.transpose(k, (0, 1, 3, 2))) * self.scale
        attn = jt.nn.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)
        x = jt.transpose(jt.matmul(attn, v), (0, 2, 1, 3)).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, drop=0.0, attn_drop=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads=num_heads, attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, int(dim * mlp_ratio), drop=drop)

    def execute(self, x: jt.Var, prompt=None) -> jt.Var:
        x = x + self.attn(self.norm1(x), prompt=prompt)
        x = x + self.mlp(self.norm2(x))
        return x


class VisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4.0, drop_rate=0.0, attn_drop_rate=0.0):
        super().__init__()
        self.num_features = embed_dim
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        self.cls_token = nn.Parameter(jt.randn((1, 1, embed_dim)) * 0.02)
        self.pos_embed = nn.Parameter(jt.randn((1, self.patch_embed.num_patches + 1, embed_dim)) * 0.02)
        self.pos_drop = nn.Dropout(drop_rate)
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, drop=drop_rate, attn_drop=attn_drop_rate)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

    def execute(self, x: jt.Var, prompt=None, q=None, train: bool = False, task_id: Optional[int] = None):
        B = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.broadcast((B, 1, self.cls_token.shape[-1]))
        x = jt.concat([cls_tokens, x], dim=1)
        pos_embed = self.pos_embed[:, :x.shape[1], :].broadcast((B, x.shape[1], self.pos_embed.shape[-1]))
        x = x + pos_embed
        x = self.pos_drop(x)
        prompt_loss = jt.zeros((1,), dtype=jt.float32)
        for layer_id, blk in enumerate(self.blocks):
            if prompt is not None:
                p_list, loss, x = prompt.forward(q, layer_id, x, train=train, task_id=task_id)
                if isinstance(loss, jt.Var):
                    prompt_loss += loss
            else:
                p_list = None
            x = blk(x, prompt=p_list)
        x = self.norm(x)
        return x, prompt_loss
