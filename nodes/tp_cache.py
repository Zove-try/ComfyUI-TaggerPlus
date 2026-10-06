# -*- coding: utf-8 -*-
"""TaggerPlus 缓存回收

节点把 ONNX 会话 / torch 模型缓存在模块级，避免每次执行重建（这是速度的关键）。
但这类对象没有注册进 ComfyUI 的模型管理，ComfyUI 的「卸载模型 / 释放显存」不会碰它们，
用户就会看到显存被占住且无法释放。

这里做两件事：
1. 各节点把「清空自己缓存」的函数注册进来，统一出口释放；
2. 包一层 comfy.model_management.unload_all_models / free_memory，
   让 ComfyUI 自己的释放动作（点卸载、显存紧张时）顺带回收我们的缓存。
"""
import gc
import sys
import threading

_RELEASERS = []
_LOCK = threading.Lock()
_HOOKED = False


def register_releaser(fn):
    """注册一个「清空缓存」的回调（幂等）"""
    with _LOCK:
        if fn not in _RELEASERS:
            _RELEASERS.append(fn)


def release_all(quiet=False):
    """释放所有已注册的缓存，返回释放条数"""
    freed = []
    with _LOCK:
        items = list(_RELEASERS)
    for fn in items:
        try:
            n = fn()
            if n:
                freed.append((getattr(fn, "__name__", "cache"), n))
        except Exception as e:                      # noqa: BLE001
            if not quiet:
                print(f"[TaggerPlus] release failed: {type(e).__name__}: {e}", file=sys.stderr)
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            try:
                torch.cuda.ipc_collect()
            except Exception:                       # noqa: BLE001
                pass
    except Exception:                               # noqa: BLE001
        pass
    return freed


def install_hooks():
    """把释放动作挂到 ComfyUI 的模型管理上（只挂一次）"""
    global _HOOKED
    if _HOOKED:
        return
    try:
        import comfy.model_management as mm
    except Exception:                               # noqa: BLE001
        return

    def wrap(name, after=True):
        orig = getattr(mm, name, None)
        if orig is None or getattr(orig, "_taggerplus_wrapped", False):
            return
        def patched(*a, **kw):
            out = orig(*a, **kw)
            try:
                release_all(quiet=True)
            except Exception:                       # noqa: BLE001
                pass
            return out
        patched._taggerplus_wrapped = True
        patched._taggerplus_orig = orig
        setattr(mm, name, patched)

    wrap("unload_all_models")     # 「卸载模型」按钮 / POST /free
    wrap("free_memory")           # 显存紧张时的自动释放
    _HOOKED = True
    print("[TaggerPlus] 已挂接 ComfyUI 显存释放钩子（卸载模型时同步回收反推缓存）")


class TaggerPlusUnload:
    """把上游内容原样传给下游，同时释放 TaggerPlus 占用的显存。

    接在反推节点后面即可：跑完反推 → 立即释放模型缓存，把显存让给后面的采样器。
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}, "optional": {"anything": ("*",)}}

    RETURN_TYPES = ("*",)
    RETURN_NAMES = ("anything",)
    FUNCTION = "run"
    CATEGORY = "TaggerPlus"
    DESCRIPTION = ("释放 TaggerPlus 反推模型占用的显存（ONNX 会话与 timm/PixAI 模型缓存），"
                   "输入原样透传，可接在反推节点后面")

    def run(self, anything=None):
        freed = release_all()
        if freed:
            detail = ", ".join(f"{k}:{v}" for k, v in freed)
            print(f"[TaggerPlus] 已释放缓存 -> {detail}")
        else:
            print("[TaggerPlus] 无可释放的缓存")
        return (anything,)
