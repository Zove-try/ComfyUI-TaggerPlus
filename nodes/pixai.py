# -*- coding: utf-8 -*-
"""
PixAI Tagger Plus —— 把"手动输入模型路径"换成"从 ComfyUI models 目录下拉选择"。

模型放置（二选一都行）：
  A) 把 HF 仓库整个文件夹拷进  ComfyUI/models/pixai_tagger/pixai-tagger-v1.0/
     （里面要有 model.safetensors 和 config.json）→ 下拉里出现 "pixai-tagger-v1.0"
  B) 只放两个文件到  ComfyUI/models/pixai_tagger/
     model.safetensors + config.json  → 下拉里出现 "model.safetensors"

想继续用现在的目录也行：在插件根目录建 taggerplus_dirs.json，写
  {"pixai_tagger": ["E:/pixai-tagger"]}
"""
import glob
import json
import os
import threading
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms.functional as TF
from safetensors.torch import load_file as load_safetensors
from transformers import PretrainedConfig, PreTrainedModel

from . import fetch
from ..vendor.pixai_vitdet import (          # noqa: F401  (架构 + rescale_pad)
    LayerScale, PatchEmbed, Attention, Block, MHAttnPool,
    ViTDetClsConfig, ViTDetCls, rescale_pad,
)

_HERE = os.path.dirname(os.path.realpath(__file__))
_PACK = os.path.dirname(_HERE)

try:
    import folder_paths
    _MODELS_ROOT = folder_paths.models_dir
    _COMFY_ROOT = folder_paths.base_path
except Exception:
    _MODELS_ROOT = os.path.join(_PACK, "models")
    _COMFY_ROOT = os.path.dirname(os.path.dirname(_PACK))


def search_dirs():
    dirs = [
        os.path.join(_MODELS_ROOT, "pixai_tagger"),
        os.path.join(_PACK, "models", "pixai_tagger"),
        os.path.join(_PACK, "models"),
    ]
    cfg = os.path.join(_PACK, "taggerplus_dirs.json")
    if os.path.exists(cfg):
        try:
            for d in json.load(open(cfg, encoding="utf-8")).get("pixai_tagger", []):
                dirs.append(os.path.expanduser(d))
        except Exception as e:
            _log(f"[TaggerPlus] 读取 taggerplus_dirs.json 失败: {e}")
    seen, out = set(), []
    for d in dirs:
        d = os.path.abspath(d)
        if d not in seen and os.path.isdir(d):
            seen.add(d); out.append(d)
    return out


def list_models():
    """→ {下拉显示名: (权重路径, 配置路径)}"""
    out = {}
    for d in search_dirs():
        for cfg in glob.glob(os.path.join(d, "**", "config.json"), recursive=True):
            folder = os.path.dirname(cfg)
            for w in ("model.safetensors", "diffusion_pytorch_model.safetensors"):
                p = os.path.join(folder, w)
                if os.path.exists(p):
                    rel = os.path.relpath(folder, d).replace("\\", "/")
                    key = rel if rel not in (".", "") else w
                    out.setdefault(key, (p, cfg))
        for p in glob.glob(os.path.join(d, "*.safetensors")):
            cfg = os.path.splitext(p)[0] + ".json"
            if not os.path.exists(cfg):
                cfg = os.path.join(os.path.dirname(p), "config.json")
            if os.path.exists(cfg):
                out.setdefault(os.path.basename(p), (p, cfg))
    return out


def list_configs():
    out = []
    for d in search_dirs():
        for cfg in glob.glob(os.path.join(d, "**", "config.json"), recursive=True):
            rel = os.path.relpath(cfg, d).replace("\\", "/")
            out.append(rel)
    return sorted(set(out))


def models_root():
    """自动下载的落地目录（每个模型一个子文件夹）"""
    d = os.path.join(_MODELS_ROOT, "pixai_tagger")
    os.makedirs(d, exist_ok=True)
    return d


_CACHE = {}
_LOCK = threading.Lock()

from . import tp_cache


def _release_caches():
    """清空本模块的模型缓存（供 tp_cache 调用）"""
    with _LOCK:
        n = len(_CACHE)
        for obj in _CACHE.values():
            try:
                obj.model.to("cpu")            # 先搬离显存，引用一断就能立刻回收
            except Exception:
                pass
        _CACHE.clear()
    return n


tp_cache.register_releaser(_release_caches)


class _Loaded:
    __slots__ = ("model", "config", "tags", "tags_split", "best_thr", "device")

    def __init__(self, model, config, tags, tags_split, best_thr, device):
        self.model, self.config = model, config
        self.tags, self.tags_split, self.best_thr = tags, tags_split, best_thr
        self.device = device


def _torch_device(pref):
    if pref == "cpu":
        return torch.device("cpu")
    try:
        import comfy.model_management as mm
        d = mm.get_torch_device()
        if pref == "cuda" and d.type != "cuda":
            raise RuntimeError("请求了 cuda 但 ComfyUI 没有可用 CUDA 设备")
        return d
    except Exception:
        return torch.device("cuda" if (pref != "cpu" and torch.cuda.is_available()) else "cpu")


def get_loaded(model_path, config_path, device_pref):
    key = (model_path, config_path, device_pref)
    with _LOCK:
        if key in _CACHE:
            return _CACHE[key]

    device = _torch_device(device_pref)
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = ViTDetClsConfig(**json.load(f))
    tags, split = cfg.tags, cfg.tags_split
    if getattr(cfg, "tags_best_threshold", None) is not None:
        best = torch.tensor(cfg.tags_best_threshold, dtype=torch.float32, device=device)
    elif getattr(cfg, "category_best_threshold", None) is not None:
        ct = cfg.category_best_threshold
        best = torch.cat([torch.full([c], ct.get(cat, 0.2), dtype=torch.float32, device=device)
                          for cat, c in split])
    else:
        best = torch.full([len(tags)], 0.2, dtype=torch.float32, device=device)

    _log(f"[TaggerPlus/PixAI] 正在加载 {os.path.basename(model_path)}（约 1.9 GB）→ {device}"
         f"，首次加载需要十几秒，之后常驻显存/内存")
    model = ViTDetCls(cfg)
    sd = (load_safetensors(model_path, device="cpu") if model_path.endswith(".safetensors")
          else torch.load(model_path, map_location="cpu", weights_only=True))
    model.load_state_dict(sd)
    model.to(device)
    model.eval()
    _log(f"[TaggerPlus/PixAI] ready: {len(tags)} tags / {len(split)} categories on {device}")
    obj = _Loaded(model, cfg, tags, split, best, device)
    with _LOCK:
        _CACHE[key] = obj
    return obj



def _log(msg, ascii_fallback=None):
    """安全打印：Windows GBK 控制台打印中文/⚠ 会抛 UnicodeEncodeError，
    这里兜底成 ASCII，保证日志永远不会让节点崩掉。"""
    try:
        print(msg, flush=True)
    except UnicodeEncodeError:
        try:
            print(ascii_fallback if ascii_fallback is not None
                  else msg.encode("ascii", "replace").decode("ascii"), flush=True)
        except Exception:
            pass

class PixAITaggerPlus:
    @classmethod
    def INPUT_TYPES(cls):
        models = fetch.combo_entries("pixai", list(list_models().keys()))
        dirs = search_dirs()
        return {
            "required": {
                "image": ("IMAGE",),
                "model": (models or ["（没有可用模型）"], {"tooltip":
                    "已装模型直接选；带 ⬇ 的条目会在首次使用时**自动下载**（优先官方，失败切国内镜像）\n"
                    "扫描目录：\n" + "\n".join(dirs)}),
                "threshold_mode": (["custom", "optimal_calibrated"], {"default": "custom",
                    "tooltip": "custom = 用下面 6 个阈值；optimal_calibrated = 模型自带的逐标签校准阈值（会忽略下面 6 个）"}),
                "general_threshold": ("FLOAT", {"default": 0.17, "min": 0.0, "max": 1.0, "step": 0.01}),
                "character_threshold": ("FLOAT", {"default": 0.27, "min": 0.0, "max": 1.0, "step": 0.01}),
                "copyright_threshold": ("FLOAT", {"default": 0.24, "min": 0.0, "max": 1.0, "step": 0.01}),
                "style_threshold": ("FLOAT", {"default": 0.15, "min": 0.0, "max": 1.0, "step": 0.01}),
                "meta_threshold": ("FLOAT", {"default": 0.17, "min": 0.0, "max": 1.0, "step": 0.01}),
                "rating_threshold": ("FLOAT", {"default": 0.41, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
            "optional": {
                "device": (["auto", "cuda", "cpu"], {"default": "auto"}),
                "config_override": (["auto"] + list_configs(), {"default": "auto",
                    "tooltip": "一般不用改；只有当同名目录里有多份 config.json 时才需要指定"}),
                "sort_by_confidence": ("BOOLEAN", {"default": True}),
                "character_first": ("BOOLEAN", {"default": False}),
                "include_style": ("BOOLEAN", {"default": True}),
                "include_meta": ("BOOLEAN", {"default": False}),
                "include_rating": ("BOOLEAN", {"default": False}),
                "replace_underscore": ("BOOLEAN", {"default": False}),
                "exclude_tags": ("STRING", {"default": "", "multiline": False}),
                "unload_after_run": ("BOOLEAN", {"default": False,
                    "label_on": "跑完即释放显存", "label_off": "保留缓存（更快）"}),
            },
        }

    RETURN_TYPES = ("STRING",) * 8
    RETURN_NAMES = ("tags_string", "character_tags", "copyright_tags", "general_tags",
                    "artist-style_tags", "meta_tags", "rating_tags", "device")
    FUNCTION = "tag"
    CATEGORY = "TaggerPlus"
    DESCRIPTION = "PixAI Tagger v1.0：模型从 ComfyUI models 目录下拉选择，无需手输路径"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")          # 图变了就必须重算；会话缓存在本节点内部

    def tag(self, image, model, threshold_mode, general_threshold, character_threshold,
            copyright_threshold, style_threshold, meta_threshold, rating_threshold,
            device="auto", config_override="auto", sort_by_confidence=True, character_first=False,
            include_style=True, include_meta=False, include_rating=False,
            replace_underscore=False, exclude_tags=""):

        name, need_dl = fetch.parse_selection(model, "pixai")
        if need_dl:
            if not fetch.fetch("pixai", name, os.path.join(models_root(), name)):
                raise RuntimeError(
                    f"自动下载 {name} 失败（约 1.9 GB）。可手动下载后放进 "
                    f"{os.path.join(models_root(), name)}，"
                    "或设置 TAGGERPLUS_HF_ENDPOINT=https://hf-mirror.com")
        table = list_models()
        if name not in table:
            raise ValueError(
                f"找不到模型 {name!r}。已扫描目录：\n  " + "\n  ".join(search_dirs()) +
                "\n请把 model.safetensors + config.json 放进 models/pixai_tagger/")
        model_path, config_path = table[name]
        if config_override != "auto":
            for d in search_dirs():
                p = os.path.join(d, config_override)
                if os.path.exists(p):
                    config_path = p
                    break

        L = get_loaded(model_path, config_path, device)
        dev_label = f"GPU · {torch.cuda.get_device_name(0)}" if L.device.type == "cuda" else "CPU"

        if threshold_mode == "optimal_calibrated":
            thr = L.best_thr.clone()
        else:
            th_map = {"general": general_threshold, "character": character_threshold,
                      "copyright": copyright_threshold, "style": style_threshold,
                      "meta": meta_threshold, "rating": rating_threshold}
            thr = torch.zeros(len(L.tags), dtype=torch.float32, device=L.device)
            st = 0
            for cat, cnt in L.tags_split:
                thr[st:st + cnt] = th_map.get(cat, 0.2)
                st += cnt

        excl = set()
        for x in (s.strip() for s in exclude_tags.split(",")):
            if x:
                excl |= {x, x.replace("_", " "), x.replace(" ", "_")}

        def fmt(tags):
            out = []
            for t in tags:
                f = t.replace("_", " ") if replace_underscore else t
                if t not in excl and f not in excl:
                    out.append(f)
            return out

        img_size = getattr(L.config, "img_size", 1008)
        res = {c: [] for c in ("general", "character", "copyright", "style", "meta", "rating")}
        for i in range(image.shape[0]):
            arr = (image[i].cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
            pil = Image.fromarray(arr)
            if pil.mode != "RGB":
                pil = pil.convert("RGB")
            t = TF.normalize(rescale_pad(TF.to_tensor(pil), output_size=img_size),
                             mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            with torch.inference_mode():
                probs = L.model(t.unsqueeze(0).to(L.device)).sigmoid()[0]

            st = 0
            for cat, cnt in L.tags_split:
                pc, tc = probs[st:st + cnt], thr[st:st + cnt]
                idx = torch.nonzero(pc > tc, as_tuple=False).flatten().cpu().tolist()
                d = {L.tags[st + j]: float(pc[j]) for j in idx}
                res[cat] = (sorted(d, key=lambda k: -d[k]) if sort_by_confidence else list(d))
                st += cnt

        fg, fc, fp = fmt(res["general"]), fmt(res["character"]), fmt(res["copyright"])
        fs, fm, fr = fmt(res["style"]), fmt(res["meta"]), fmt(res["rating"])
        combined = (fc + fp + fg) if character_first else (fg + fc + fp)
        if include_style:
            combined += fs
        if include_meta:
            combined += fm
        if include_rating:
            combined += fr
        _log(f"[TaggerPlus/PixAI] {os.path.basename(model_path)} | {dev_label} | "
             f"character {len(fc)} copyright {len(fp)} general {len(fg)} style {len(fs)}")
        return (", ".join(combined), ", ".join(fc), ", ".join(fp), ", ".join(fg),
                ", ".join(fs), ", ".join(fm), ", ".join(fr), dev_label)


# ---------------------------------------------------------------------------
# 跑完即释放显存的开关（unload_after_run）
# 包一层而不是改 tag() 签名：所有 return 路径都会被覆盖
# ---------------------------------------------------------------------------
def _install_unload_switch(cls):
    _orig = cls.tag

    def _tag(self, *a, **kw):
        # 先把开关取出来，绝不能把它转发给 tag()（签名里没有这个参数）
        unload = kw.pop("unload_after_run", False)
        try:
            out = _orig(self, *a, **kw)
            tp_cache.mark_used()          # 记下使用时间，避免被常规内存回收清掉
            return out
        finally:
            if unload:
                freed = tp_cache.release_all()
                if freed:                     # 没东西可释放时保持安静
                    detail = ", ".join(f"{k}:{v}" for k, v in freed)
                    print(f"[TaggerPlus] unload_after_run → 已释放缓存（{detail}）")

    _tag.__name__ = "tag"
    _tag.__doc__ = _orig.__doc__
    _tag._taggerplus_orig = _orig        # 暴露原函数，便于调试/内省
    cls.tag = _tag
    return cls


_install_unload_switch(PixAITaggerPlus)
