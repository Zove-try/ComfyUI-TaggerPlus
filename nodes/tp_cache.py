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
import time

_RELEASERS = []
_LOCK = threading.Lock()
_HOOKED = False
_LAST_USED = 0.0
IDLE_SECONDS = 30.0      # 距上次反推多久才允许被 ComfyUI 的卸载动作回收



def register_releaser(fn):
    """注册一个「清空缓存」的回调（幂等）"""
    with _LOCK:
        if fn not in _RELEASERS:
            _RELEASERS.append(fn)


def mark_used():
    """节点每次跑完调用：记录使用时间，避免被 ComfyUI 的常规内存回收顺手清掉"""
    global _LAST_USED
    _LAST_USED = time.time()


def release_if_idle(idle_seconds=None, quiet=True):
    """只在「距上次反推超过 idle_seconds」时才释放。

    ComfyUI 在执行流程里会频繁调用 unload_all_models()/free_memory()，
    如果每次都释放，反推模型就会被反复卸载 → 每次都要重新加载（实测 5~7 秒）。
    """
    global _LAST_USED
    idle = IDLE_SECONDS if idle_seconds is None else idle_seconds
    if _LAST_USED and (time.time() - _LAST_USED) < idle:
        return None
    _LAST_USED = 0.0
    return release_all(quiet=quiet)


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

    def wrap(name, idle_guard=False):
        orig = getattr(mm, name, None)
        if orig is None or getattr(orig, "_taggerplus_wrapped", False):
            return
        def patched(*a, **kw):
            out = orig(*a, **kw)
            try:
                if idle_guard:
                    release_if_idle()
                else:
                    release_all(quiet=True)
            except Exception:                       # noqa: BLE001
                pass
            return out
        patched._taggerplus_wrapped = True
        patched._taggerplus_orig = orig
        setattr(mm, name, patched)

    # 只挂 unload_all_models。「卸载模型」按钮和 POST /free 走它；
    # 不能挂 free_memory —— ComfyUI 加载任何模型都会调它，会导致反推模型被反复卸载。
    # 而且 unload_all_models 在执行流程里也会被调用，所以再加一层空闲判定。
    wrap("unload_all_models", idle_guard=True)
    _HOOKED = True
    print("[TaggerPlus] 已挂接 ComfyUI 显存释放钩子（卸载模型时同步回收反推缓存）")
