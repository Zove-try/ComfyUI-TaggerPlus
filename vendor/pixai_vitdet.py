# -*- coding: utf-8 -*-
"""
PixAI Tagger v1.0 的模型架构与预处理（vendored）

来源与署名
----------
- 模型：https://huggingface.co/pixai-labs/pixai-tagger-v1.0  (Apache-2.0)
- 本文件的 PyTorch 实现取自 ComfyUI-Tagger
  https://github.com/sln77/ComfyUI-Tagger  (MIT License, Copyright (c) 2026 sln77)
  为便于本插件独立分发，此处原样保留（未改动算法），仅做整理。

为什么 vendor 而不是 import：
  新插件要能独立安装/开源，不依赖用户是否装了 ComfyUI-Tagger。
  保留同一份实现可以保证输出与原节点逐标签一致。
"""

import os
import json
import math
from functools import partial
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import torchvision.transforms.functional as TF

from safetensors.torch import load_file as load_safetensors
from transformers import PretrainedConfig, PreTrainedModel

try:
    from timm.layers import DropPath, Mlp, trunc_normal_
except ModuleNotFoundError:
    from timm.models.layers import DropPath, Mlp, trunc_normal_

try:
    import comfy.model_management as mm
except ImportError:
    mm = None


class LayerScale(nn.Module):
    def __init__(self, dim: int, init_values: Union[float, Tensor] = 1e-5, inplace: bool = False) -> None:
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


def init_t_xy(end_x: int, end_y: int, scale: float = 1.0, offset: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
    t = torch.arange(end_x * end_y, dtype=torch.float32)
    t_x = (t % end_x).float()
    t_y = torch.div(t, end_x, rounding_mode="floor").float()
    return t_x * scale + offset, t_y * scale + offset


def compute_axial_cis(dim: int, end_x: int, end_y: int, theta: float = 10000.0, scale_pos: float = 1.0, offset: int = 0) -> torch.Tensor:
    freqs_x = 1.0 / (theta ** (torch.arange(0, dim, 4)[: (dim // 4)].float() / dim))
    freqs_y = 1.0 / (theta ** (torch.arange(0, dim, 4)[: (dim // 4)].float() / dim))
    t_x, t_y = init_t_xy(end_x, end_y, scale_pos, offset)
    freqs_cis_x = torch.polar(torch.ones_like(torch.outer(t_x, freqs_x)), torch.outer(t_x, freqs_x))
    freqs_cis_y = torch.polar(torch.ones_like(torch.outer(t_y, freqs_y)), torch.outer(t_y, freqs_y))
    return torch.cat([freqs_cis_x, freqs_cis_y], dim=-1)


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    ndim = x.ndim
    shape = [d if i >= ndim - 2 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_enc(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2)) if xk.shape[-2] != 0 else None
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    if xk_ is None:
        return xq_out.type_as(xq).to(xq.device), xk
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq).to(xq.device), xk_out.type_as(xk).to(xk.device)


def window_partition(x: Tensor, window_size: int) -> Tuple[Tensor, Tuple[int, int]]:
    B, H, W, C = x.shape
    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
    Hp, Wp = H + pad_h, W + pad_w
    x = x.view(B, Hp // window_size, window_size, Wp // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).reshape(-1, window_size, window_size, C)
    return windows, (Hp, Wp)


def window_unpartition(windows: Tensor, window_size: int, pad_hw: Tuple[int, int], hw: Tuple[int, int]) -> Tensor:
    Hp, Wp = pad_hw
    H, W = hw
    B = windows.shape[0] // (Hp * Wp // window_size // window_size)
    x = windows.reshape(B, Hp // window_size, Wp // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).reshape(B, Hp, Wp, -1)
    if Hp > H or Wp > W:
        x = x[:, :H, :W, :]
    return x


def get_rel_pos(q_size: int, k_size: int, rel_pos: Tensor) -> Tensor:
    max_rel_dist = int(2 * max(q_size, k_size) - 1)
    if rel_pos.shape[0] != max_rel_dist:
        rel_pos_resized = F.interpolate(
            rel_pos.reshape(1, rel_pos.shape[0], -1).permute(0, 2, 1),
            size=max_rel_dist,
            mode="linear",
            align_corners=False,
        ).reshape(-1, max_rel_dist).permute(1, 0)
    else:
        rel_pos_resized = rel_pos
    q_coords = torch.arange(q_size)[:, None] * max(k_size / q_size, 1.0)
    k_coords = torch.arange(k_size)[None, :] * max(q_size / k_size, 1.0)
    relative_coords = (q_coords - k_coords) + (k_size - 1) * max(q_size / k_size, 1.0)
    return rel_pos_resized[relative_coords.long()]


def get_abs_pos(abs_pos: Tensor, has_cls_token: bool, hw: Tuple[int, int], retain_cls_token: bool = False, tiling: bool = False) -> Tensor:
    h, w = hw
    if has_cls_token:
        cls_pos = abs_pos[:, :1]
        abs_pos = abs_pos[:, 1:]
    size = int(math.sqrt(abs_pos.shape[1]))
    if size != h or size != w:
        new_abs_pos = abs_pos.reshape(1, size, size, -1).permute(0, 3, 1, 2)
        if tiling:
            new_abs_pos = new_abs_pos.tile([1, 1] + [x // y + 1 for x, y in zip((h, w), new_abs_pos.shape[2:])])[:, :, :h, :w]
        else:
            new_abs_pos = F.interpolate(new_abs_pos, size=(h, w), mode="bicubic", align_corners=False)
        return new_abs_pos.permute(0, 2, 3, 1) if not retain_cls_token else torch.cat([cls_pos, new_abs_pos.permute(0, 2, 3, 1).reshape(1, h * w, -1)], dim=1)
    else:
        return abs_pos.reshape(1, h, w, -1) if not retain_cls_token else torch.cat([cls_pos, abs_pos], dim=1)


def concat_rel_pos(q: Tensor, k: Tensor, q_hw: Tuple[int, int], k_hw: Tuple[int, int], rel_pos_h: Tensor, rel_pos_w: Tensor, rescale: bool = False, relative_coords: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
    q_h, q_w = q_hw
    k_h, k_w = k_hw
    if relative_coords is not None:
        Rh, Rw = rel_pos_h[relative_coords], rel_pos_w[relative_coords]
    else:
        Rh, Rw = get_rel_pos(q_h, k_h, rel_pos_h), get_rel_pos(q_w, k_w, rel_pos_w)
    B, _, dim = q.shape
    r_q = q.reshape(B, q_h, q_w, dim)
    old_scale = dim**0.5
    new_scale = (dim + k_h + k_w) ** 0.5 if rescale else old_scale
    scale_ratio = new_scale / old_scale
    rel_h = torch.einsum("bhwc,hkc->bhwk", r_q, Rh) * new_scale
    rel_w = torch.einsum("bhwc,wkc->bhwk", r_q, Rw) * new_scale
    eye_h = torch.eye(k_h, dtype=q.dtype, device=q.device).view(1, k_h, 1, k_h).expand([B, k_h, k_w, k_h])
    eye_w = torch.eye(k_w, dtype=q.dtype, device=q.device).view(1, 1, k_w, k_w).expand([B, k_h, k_w, k_w])
    q = torch.cat([r_q * scale_ratio, rel_h, rel_w], dim=-1).view(B, q_h * q_w, -1)
    k = torch.cat([k.view(B, k_h, k_w, -1), eye_h, eye_w], dim=-1).view(B, k_h * k_w, -1)
    return q, k


class PatchEmbed(nn.Module):
    def __init__(self, kernel_size=(16, 16), stride=(16, 16), padding=(0, 0), in_chans=3, embed_dim=768, bias=True):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)

    def forward(self, x: Tensor) -> Tensor:
        return self.proj(x).permute(0, 2, 3, 1)


class Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = True, use_rel_pos: bool = False, rel_pos_zero_init: bool = True, input_size: Optional[Tuple[int, int]] = None, cls_token: bool = False, use_rope: bool = False, rope_theta: float = 10000.0, rope_pt_size: Optional[Tuple[int, int]] = None, rope_interp: bool = False):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.cls_token = cls_token
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.use_rel_pos = use_rel_pos
        self.input_size = input_size
        self.use_rope = use_rope
        self.rope_theta = rope_theta
        self.rope_pt_size = rope_pt_size
        self.rope_interp = rope_interp

        if self.use_rel_pos:
            self.rel_pos_h = nn.Parameter(torch.zeros(2 * input_size[0] - 1, self.head_dim))
            self.rel_pos_w = nn.Parameter(torch.zeros(2 * input_size[1] - 1, self.head_dim))
            if not rel_pos_zero_init:
                trunc_normal_(self.rel_pos_h, std=0.02)
                trunc_normal_(self.rel_pos_w, std=0.02)
            H, W = input_size
            relative_coords = (torch.arange(H)[:, None] - torch.arange(W)[None, :]) + (H - 1)
            self.register_buffer("relative_coords", relative_coords.long())
        else:
            self.rel_pos_h = self.rel_pos_w = None

        if self.use_rope:
            scale_pos = (self.rope_pt_size[0] / self.input_size[0]) if (self.rope_interp and self.rope_pt_size) else 1.0
            freqs_cis = compute_axial_cis(self.head_dim, input_size[0], input_size[1], theta=rope_theta, scale_pos=scale_pos)
            if self.cls_token:
                t = torch.zeros(self.head_dim // 2, dtype=torch.float32)
                freqs_cis = torch.cat([torch.polar(torch.ones_like(t), t)[None, :], freqs_cis], dim=0)
            self.register_buffer("freqs_cis", freqs_cis, persistent=False)
        else:
            self.freqs_cis = None

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim == 4:
            B, H, W, _ = x.shape
            L = H * W
            ndim = 4
        else:
            B, L, _ = x.shape
            ndim = 3
            H = W = int(math.sqrt(L))
        qkv = self.qkv(x).reshape(B, L, 3, self.num_heads, -1)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        if self.use_rope and self.freqs_cis is not None:
            q, k = apply_rotary_enc(q, k, freqs_cis=self.freqs_cis)
        if self.use_rel_pos:
            q, k = concat_rel_pos(q.flatten(0, 1), k.flatten(0, 1), (H, W), x.shape[1:3], self.rel_pos_h, self.rel_pos_w, rescale=True, relative_coords=self.relative_coords)
            q = q.reshape(B, self.num_heads, H * W, -1)
            k = k.reshape(B, self.num_heads, H * W, -1)
        x = F.scaled_dot_product_attention(q, k, v)
        if ndim == 4:
            x = x.view(B, self.num_heads, H, W, -1).permute(0, 2, 3, 1, 4).reshape(B, H, W, -1)
        else:
            x = x.view(B, self.num_heads, L, -1).permute(0, 2, 1, 3).reshape(B, L, -1)
        return self.proj(x)


class Block(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0, qkv_bias: bool = True, drop_path: float = 0.0, norm_layer: Callable = nn.LayerNorm, act_layer: Callable = nn.GELU, use_rel_pos: bool = False, rel_pos_zero_init: bool = True, window_size: int = 0, input_size: Optional[Tuple[int, int]] = None, use_rope: bool = False, rope_pt_size: Optional[Tuple[int, int]] = None, rope_interp: bool = False, cls_token: bool = False, dropout: float = 0.0, init_values: Optional[float] = None):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, use_rel_pos=use_rel_pos, rel_pos_zero_init=rel_pos_zero_init, input_size=input_size if window_size == 0 else (window_size, window_size), use_rope=use_rope, rope_pt_size=rope_pt_size, rope_interp=rope_interp, cls_token=cls_token)
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=(dropout, 0.0))
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.dropout = nn.Dropout(dropout)
        self.window_size = window_size

    def forward(self, x: Tensor) -> Tensor:
        shortcut = x
        x = self.norm1(x)
        if self.window_size > 0:
            H, W = x.shape[1], x.shape[2]
            x, pad_hw = window_partition(x, self.window_size)
        x = self.ls1(self.attn(x))
        if self.window_size > 0:
            x = window_unpartition(x, self.window_size, pad_hw, (H, W))
        x = shortcut + self.dropout(self.drop_path(x))
        x = x + self.dropout(self.drop_path(self.ls2(self.mlp(self.norm2(x)))))
        return x


class MHAttnPool(nn.Module):
    def __init__(self, dim, num_heads=8, num_patches=196, use_pos=True):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.use_pos = use_pos
        self.norm = nn.LayerNorm(dim)
        self.q = nn.Parameter(torch.empty(dim))
        trunc_normal_(self.q)
        self.kv = nn.Linear(dim, dim * 2)
        if use_pos:
            self.pos_embed = nn.Parameter(torch.empty(num_patches, dim))
            trunc_normal_(self.pos_embed, std=0.02)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, D = x.shape
        x = self.norm(x)
        if self.use_pos:
            x = x + self.pos_embed[None, ...]
        q = self.q[None, None, ...].expand(B, -1, -1)
        k, v = self.kv(x).chunk(2, dim=-1)
        q = q.view(B, 1, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(1, 2).contiguous().view(B, 1, D)
        return self.proj(out).squeeze(1)


class ViTDetClsConfig(PretrainedConfig):
    model_type = "cls_vitdet"

    def __init__(self, num_classes=30877, img_size=1024, patch_size=16, in_chans=3, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0, qkv_bias=True, drop_path_rate=0.0, norm_layer="LayerNorm", act_layer="GELU", use_abs_pos=True, tile_abs_pos=True, rel_pos_blocks=(2, 5, 8, 11), rel_pos_zero_init=True, window_size=14, global_att_blocks=(2, 5, 8, 11), use_rope=False, rope_pt_size=None, use_interp_rope=False, pretrain_img_size=224, pretrain_use_cls_token=True, retain_cls_token=True, dropout=0.0, return_interm_layers=False, init_values=None, ln_pre=False, ln_post=False, bias_patch_embed=True, use_act_checkpoint=False, tags=None, tags_split=None, tags_best_threshold=None, category_best_threshold=None, **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.qkv_bias = qkv_bias
        self.drop_path_rate = drop_path_rate
        self.norm_layer = norm_layer
        self.act_layer = act_layer
        self.use_abs_pos = use_abs_pos
        self.tile_abs_pos = tile_abs_pos
        self.rel_pos_blocks = rel_pos_blocks
        self.rel_pos_zero_init = rel_pos_zero_init
        self.window_size = window_size
        self.global_att_blocks = global_att_blocks
        self.use_rope = use_rope
        self.rope_pt_size = rope_pt_size
        self.use_interp_rope = use_interp_rope
        self.pretrain_img_size = pretrain_img_size
        self.pretrain_use_cls_token = pretrain_use_cls_token
        self.retain_cls_token = retain_cls_token
        self.dropout = dropout
        self.return_interm_layers = return_interm_layers
        self.init_values = init_values
        self.ln_pre = ln_pre
        self.ln_post = ln_post
        self.bias_patch_embed = bias_patch_embed
        self.use_act_checkpoint = use_act_checkpoint
        self.tags = tags or []
        self.tags_split = tags_split or []
        self.tags_best_threshold = tags_best_threshold
        self.category_best_threshold = category_best_threshold


class ViTDetCls(PreTrainedModel):
    config_class = ViTDetClsConfig

    def __init__(self, config: ViTDetClsConfig):
        super().__init__(config)
        act_layer = getattr(nn, config.act_layer)
        norm_layer = partial(getattr(nn, config.norm_layer), eps=1e-5)
        window_block_indexes = [i for i in range(config.depth) if i not in config.global_att_blocks]
        self.full_attn_ids = list(config.global_att_blocks)
        self.rel_pos_blocks = [i in config.rel_pos_blocks for i in range(config.depth)] if not isinstance(config.rel_pos_blocks, bool) else [config.rel_pos_blocks] * config.depth

        self.retain_cls_token = getattr(config, "retain_cls_token", True)
        self.pretrain_use_cls_token = getattr(config, "pretrain_use_cls_token", True)
        self.return_interm_layers = getattr(config, "return_interm_layers", False)

        self.img_size = config.img_size
        self.patch_size = config.patch_size
        self.embed_dim = config.embed_dim
        self.num_heads = config.num_heads

        if self.retain_cls_token:
            scale = config.embed_dim**-0.5
            self.class_embedding = nn.Parameter(scale * torch.randn(1, 1, config.embed_dim))

        self.patch_embed = PatchEmbed(kernel_size=(config.patch_size, config.patch_size), stride=(config.patch_size, config.patch_size), in_chans=config.in_chans, embed_dim=config.embed_dim, bias=config.bias_patch_embed)
        self.tile_abs_pos = config.tile_abs_pos
        self.use_abs_pos = config.use_abs_pos
        if self.use_abs_pos:
            num_patches = (config.pretrain_img_size // config.patch_size) ** 2
            num_positions = (num_patches + 1) if self.pretrain_use_cls_token else num_patches
            self.pos_embed = nn.Parameter(torch.zeros(1, num_positions, config.embed_dim))
        else:
            self.pos_embed = None

        dpr = [x.item() for x in torch.linspace(0, config.drop_path_rate, config.depth)]
        self.blocks = nn.ModuleList([
            Block(
                dim=config.embed_dim, num_heads=config.num_heads, mlp_ratio=config.mlp_ratio, qkv_bias=config.qkv_bias, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer, use_rel_pos=self.rel_pos_blocks[i], rel_pos_zero_init=config.rel_pos_zero_init,
                window_size=config.window_size if i in window_block_indexes else 0,
                input_size=(config.img_size // config.patch_size, config.img_size // config.patch_size),
                use_rope=config.use_rope,
                rope_pt_size=((config.window_size, config.window_size) if config.rope_pt_size is None else (config.rope_pt_size, config.rope_pt_size)),
                rope_interp=config.use_interp_rope, cls_token=self.retain_cls_token, dropout=config.dropout, init_values=config.init_values
            )
            for i in range(config.depth)
        ])

        self.ln_pre = norm_layer(config.embed_dim) if config.ln_pre else nn.Identity()
        self.ln_post = norm_layer(config.embed_dim) if config.ln_post else nn.Identity()
        self.head_pool = MHAttnPool(config.embed_dim, config.num_heads, num_patches=(config.img_size // config.patch_size) ** 2, use_pos=True)
        self.head = nn.Linear(config.embed_dim, config.num_classes)

    def forward_feature(self, x: torch.Tensor) -> List[torch.Tensor]:
            x = self.patch_embed(x)
            h, w = x.shape[1], x.shape[2]
            s = 0
            if self.retain_cls_token:
                x = torch.cat([self.class_embedding, x.flatten(1, 2)], dim=1)
                s = 1
            if self.pos_embed is not None:
                x = x + get_abs_pos(self.pos_embed, self.pretrain_use_cls_token, (h, w), self.retain_cls_token, tiling=self.tile_abs_pos)
            x = self.ln_pre(x)
            outputs = []
            for i, blk in enumerate(self.blocks):
                x = blk(x)
                if (i == self.full_attn_ids[-1]) or (self.return_interm_layers and i in self.full_attn_ids):
                    if i == self.full_attn_ids[-1]:
                        x = self.ln_post(x)
                    feats = x[:, s:]
                    if feats.ndim == 4:
                        feats = feats.permute(0, 3, 1, 2)
                    else:
                        h_f = w_f = int(math.sqrt(feats.shape[1]))
                        feats = feats.reshape(feats.shape[0], h_f, w_f, feats.shape[-1]).permute(0, 3, 1, 2)
                    outputs.append(feats)
            return outputs

    def forward(self, x):
        x = self.forward_feature(x)[-1]
        x = x.view(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
        x = self.head_pool(x)
        return self.head(x)



def rescale_pad(image_tensor, output_size=1024):
    h, w = image_tensor.shape[-2:]
    if h != output_size or w != output_size:
        r = min(output_size / h, output_size / w)
        new_h, new_w = int(h * r), int(w * r)
        ph = output_size - new_h
        pw = output_size - new_w
        left = pw // 2
        right = pw - left
        top = ph // 2
        bottom = ph - top
        image_tensor = TF.resize(image_tensor, [new_h, new_w], interpolation=TF.InterpolationMode.BILINEAR)
        image_tensor = TF.pad(image_tensor, [left, top, right, bottom], fill=0)
    return image_tensor
