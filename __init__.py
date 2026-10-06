# -*- coding: utf-8 -*-
"""
ComfyUI-TaggerPlus
==================
两个"纠正版"反推节点：

- **WD14 Tagger Plus** —— 原版 ComfyUI-WD14-Tagger 每次执行都重建 1.2GB 的 ONNX 会话，
  且 CUDA provider 加载失败时会静默退回 CPU（实测 16.5 秒/张，其中 ~9 秒是 CPU 推理、
  ~3-6 秒是重复建会话）。本插件缓存会话与词表，并把**实际使用的设备**作为输出显示出来。
- **PixAI Tagger Plus** —— 原版要求手输模型绝对路径；本插件从 ComfyUI 的
  `models/pixai_tagger/` 目录自动扫描，下拉选择。

开源：MIT。PixAI 的模型架构实现 vendor 自 ComfyUI-Tagger (MIT, sln77)，
详见 vendor/LICENSE.ComfyUI-Tagger.txt。
"""
from .nodes import tp_cache  # noqa: E402
import os
import sys

# Windows 中文环境控制台默认是 GBK，节点里打印 ⚠ / 中文会抛 UnicodeEncodeError。
# 这里只把「无法编码的字符」替换掉，不改编码本身（尽量少侵入全局状态）。
try:
    sys.stdout.reconfigure(errors="replace")
    sys.stderr.reconfigure(errors="replace")
except Exception:
    pass

from .nodes.wd14 import WD14TaggerPlus
from .nodes.pixai import PixAITaggerPlus

tp_cache.install_hooks()

__version__ = "0.1.6"

NODE_CLASS_MAPPINGS = {
    "WD14TaggerPlus": WD14TaggerPlus,
    "PixAITaggerPlus": PixAITaggerPlus,
    "TaggerPlusUnload": tp_cache.TaggerPlusUnload,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WD14TaggerPlus": "WD14 Tagger Plus ⚡",
    "PixAITaggerPlus": "PixAI Tagger Plus ⚡",
    "TaggerPlusUnload": "Tagger Plus 卸载模型 (Unload)",
}

# 把模型目录注册进 ComfyUI（方便用户在标准位置放模型）
try:
    import folder_paths
    for key, sub in (("wd14_tagger", "wd14_tagger"), ("pixai_tagger", "pixai_tagger")):
        p = os.path.join(folder_paths.models_dir, sub)
        os.makedirs(p, exist_ok=True)
        if key not in folder_paths.folder_names_and_paths:
            folder_paths.add_model_folder_path(key, p)
except Exception as e:                                     # pragma: no cover
    print(f"[TaggerPlus] 注册模型目录失败（不影响使用）: {e}")

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "__version__"]
