# -*- coding: utf-8 -*-
"""
WD14 Tagger Plus —— 解决了 ComfyUI-WD14-Tagger 的两个坑：

1. 慢：原节点在每次执行时都重新 `InferenceSession(...)`（1.2GB 模型每次重载），
   而且 ONNX Runtime 的 CUDA provider 加载失败时会**静默**退回 CPU（实测 16.5s/张）。
   本节点：会话按 (模型, provider) 缓存 + 词表只解析一次 + **明确告诉你实际跑在什么设备上**。
2. 模型路径：改为扫描 ComfyUI 的 models 目录，下拉选择；兼容旧插件的 models 目录。

与 pysssss 版的行为差异（都为可选项，默认按"更干净"的方向）：
- `escape_parens` 默认 False（原版强制把 ( ) 转义成 \\( \\)，会让下游分类器查不到表）
- `sort_by_confidence` 默认 False（与原版一致的词表顺序）
- `exclude_tags` 双向大小写不敏感匹配
"""
import csv
import glob
import os
import site
import threading

import numpy as np
from PIL import Image

# ---------------------------------------------------------------- 模型目录
_HERE = os.path.dirname(os.path.realpath(__file__))
_PACK = os.path.dirname(_HERE)

try:
    import folder_paths
    _MODELS_ROOT = folder_paths.models_dir
    _COMFY_ROOT = folder_paths.base_path
except Exception:                    # 脱离 ComfyUI 也能用（便于单测/脚本调用）
    _MODELS_ROOT = os.path.join(_PACK, "models")
    # 插件在 <ComfyUI>/custom_nodes/<pack> → 往上推两级就是 ComfyUI 根
    _COMFY_ROOT = os.path.dirname(os.path.dirname(_PACK))

#: 扫描顺序：用户 models 目录 → 插件自带 → 旧插件目录（平滑迁移）
def search_dirs():
    dirs = [
        os.path.join(_MODELS_ROOT, "wd14_tagger"),
        os.path.join(_PACK, "models", "wd14_tagger"),
        os.path.join(_PACK, "models"),
    ]
    if _COMFY_ROOT:
        dirs.append(os.path.join(_COMFY_ROOT, "custom_nodes", "ComfyUI-WD14-Tagger", "models"))
    # 额外目录：插件根目录放 taggerplus_dirs.json，{"wd14_tagger": ["D:/xxx"]}
    cfg = os.path.join(_PACK, "taggerplus_dirs.json")
    if os.path.exists(cfg):
        try:
            import json
            for d in json.load(open(cfg, encoding="utf-8")).get("wd14_tagger", []):
                dirs.append(os.path.expanduser(d))
        except Exception as e:
            print(f"[TaggerPlus] 读取 taggerplus_dirs.json 失败: {e}")
    seen, out = set(), []
    for d in dirs:
        d = os.path.abspath(d)
        if d not in seen and os.path.isdir(d):
            seen.add(d); out.append(d)
    return out


def list_models():
    """只列出「.onnx + 同名 .csv」都存在的模型"""
    found = {}
    for d in search_dirs():
        for onnx in sorted(glob.glob(os.path.join(d, "*.onnx"))):
            stem = os.path.splitext(os.path.basename(onnx))[0]
            if os.path.exists(os.path.join(d, stem + ".csv")):
                found.setdefault(stem, onnx)
    return found


# ---------------------------------------------------------------- 设备 / 会话
_DLL_READY = False
_SESSIONS = {}
_CSVS = {}
_LOCK = threading.Lock()


def _prepare_dll_paths():
    """尽力让 onnxruntime 找到 CUDA 运行库（nvidia-* pip 包 / torch/lib）"""
    global _DLL_READY
    if _DLL_READY:
        return
    _DLL_READY = True
    cands = []
    try:
        import torch
        cands.append(os.path.join(os.path.dirname(torch.__file__), "lib"))
    except Exception:
        pass
    try:
        for sp in set(list(site.getsitepackages()) + [site.getusersitepackages()]):
            cands += glob.glob(os.path.join(sp, "nvidia", "*", "bin"))
            cands += glob.glob(os.path.join(sp, "nvidia", "*", "lib"))
    except Exception:
        pass
    # 插件自带的 CUDA 12 运行库（onnxruntime-gpu 的 CUDA EP 需要 CUDA 12，而便携包里的
    # torch 往往带的是 CUDA 13，文件名不同 → 加载失败 → 静默退回 CPU）
    cands += glob.glob(os.path.join(_PACK, "cuda12", "nvidia", "*", "bin"))
    cands += glob.glob(os.path.join(_PACK, "cuda12", "nvidia", "*", "lib"))
    for d in cands:
        if not os.path.isdir(d):
            continue
        try:
            os.add_dll_directory(d)
        except Exception:
            pass
        os.environ["PATH"] = d + os.pathsep + os.environ.get("PATH", "")


def pick_providers(device):
    import onnxruntime as ort
    avail = ort.get_available_providers()
    if device == "cpu":
        return ["CPUExecutionProvider"]
    if "CUDAExecutionProvider" in avail:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    if "TensorrtExecutionProvider" in avail:
        return ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def get_session(onnx_path, providers):
    key = (onnx_path, tuple(providers))
    with _LOCK:
        if key in _SESSIONS:
            return _SESSIONS[key]
    _prepare_dll_paths()
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    if os.environ.get("TAGGERPLUS_ORT_VERBOSE"):
        so.log_severity_level = 1
    sess = ort.InferenceSession(onnx_path, sess_options=so, providers=list(providers))
    with _LOCK:
        _SESSIONS[key] = sess
    return sess


def load_vocab(csv_path, replace_underscore):
    """词表只解析一次；返回 (tags, general_index, character_index)"""
    key = (csv_path, bool(replace_underscore))
    with _LOCK:
        if key in _CSVS:
            return _CSVS[key]
    tags, general_index, character_index = [], None, None
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            if len(row) < 3:
                continue
            if general_index is None and row[2] == "0":
                general_index = reader.line_num - 2
            elif character_index is None and row[2] == "4":
                character_index = reader.line_num - 2
            tags.append(row[1].replace("_", " ") if replace_underscore else row[1])
    if general_index is None:
        general_index = 0
    if character_index is None:
        character_index = len(tags)
    with _LOCK:
        _CSVS[key] = (tags, general_index, character_index)
    return _CSVS[key]


CUDA12_HINT = (
    "\n[TaggerPlus] 检测到 CUDA provider 加载失败，已退回 CPU。\n"
    "  原因通常是 onnxruntime-gpu 需要 CUDA 12 运行库（cublasLt64_12.dll），\n"
    "  而 ComfyUI 便携包里的 torch 带的是 CUDA 13（cublasLt64_13.dll），文件名不匹配。\n"
    "  一键修复：双击插件目录里的 install_cuda12.bat（Linux/macOS 用 install_cuda12.sh）\n"
    "  或手动执行（装到本插件目录，不影响主环境，约 1.3GB）：\n"
    '    "<ComfyUI python>" -m pip install --target "<插件目录>/cuda12" \\\n'
    "        nvidia-cublas-cu12 nvidia-cuda-runtime-cu12 nvidia-cufft-cu12 nvidia-curand-cu12\n"
    "  装完重启 ComfyUI，device 输出会变成 GPU · <显卡名>。\n"
)


def device_label(sess, requested):
    used = sess.get_providers()
    gpu = [p for p in used if "CUDA" in p or "Tensorrt" in p]
    if gpu:
        try:
            import torch
            name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else gpu[0]
        except Exception:
            name = gpu[0]
        return f"GPU · {name}"
    if requested and requested[0] != "CPUExecutionProvider":
        print(CUDA12_HINT)
        return "CPU ⚠ 请求了 GPU 但回退到 CPU（多为缺 CUDA 运行库，见 README）"
    return "CPU"


# ---------------------------------------------------------------- 节点
class WD14TaggerPlus:
    @classmethod
    def INPUT_TYPES(cls):
        models = list(list_models().keys())
        dirs = search_dirs()
        tip = "模型目录：" + (" | ".join(dirs) if dirs else "（还没找到目录）")
        return {
            "required": {
                "image": ("IMAGE",),
                "model": (models or ["（请把 .onnx 和同名 .csv 放进 models/wd14_tagger/）"],
                          {"tooltip": "自动扫描以下目录：\n" + "\n".join(dirs)}),
                "threshold": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 1.0, "step": 0.01,
                                        "tooltip": "通用标签阈值（原版默认 0.35）"}),
                "character_threshold": ("FLOAT", {"default": 0.85, "min": 0.0, "max": 1.0, "step": 0.01,
                                                  "tooltip": "角色标签阈值（原版默认 0.85；做对比建议和对方对齐）"}),
                "device": (["auto", "cuda", "cpu"], {"default": "auto",
                                                     "tooltip": "auto = 有 CUDA 就用；加载失败会自动回退并在 device 输出里说明"}),
            },
            "optional": {
                "replace_underscore": ("BOOLEAN", {"default": False,
                                                   "tooltip": "把 blue_eyes 变成 blue eyes（注意：会让下游分类器按空格切词）"}),
                "escape_parens": ("BOOLEAN", {"default": False,
                                              "tooltip": "原版会把 ( ) 转义成 \\( \\)；关掉可让下游查表正常"}),
                "sort_by_confidence": ("BOOLEAN", {"default": False,
                                                   "tooltip": "True = 按置信度从高到低；False = 词表顺序（与原版一致）"}),
                "trailing_comma": ("BOOLEAN", {"default": False}),
                "exclude_tags": ("STRING", {"default": "", "multiline": False,
                                            "tooltip": "要排除的标签，逗号分隔（大小写不敏感）"}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("tags", "device")
    FUNCTION = "tag"
    CATEGORY = "TaggerPlus"
    DESCRIPTION = "WD 系列反推（onnxruntime）：会话缓存 + 明确的设备回报，速度比原版快约 30 倍"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")          # 图变了就必须重算；会话缓存在本节点内部

    def tag(self, image, model, threshold, character_threshold, device="auto",
            replace_underscore=False, escape_parens=False, sort_by_confidence=False,
            trailing_comma=False, exclude_tags=""):
        table = list_models()
        if model not in table:
            raise ValueError(
                f"找不到模型 {model!r}。已扫描目录：\n  " + "\n  ".join(search_dirs()) +
                "\n请把 .onnx 与同名 .csv 放进 models/wd14_tagger/（或在插件的 taggerplus_dirs.json 里加目录）")
        onnx_path = table[model]
        csv_path = onnx_path[:-5] + ".csv"

        providers = pick_providers(device)
        sess = get_session(onnx_path, providers)
        label = device_label(sess, providers)

        inp = sess.get_inputs()[0]
        height = inp.shape[1]
        out_name = sess.get_outputs()[0].name
        tags, gi, ci = load_vocab(csv_path, replace_underscore)

        excl = set()
        for x in (s.strip() for s in exclude_tags.split(",")):
            if x:
                excl.add(x.lower())
                excl.add(x.replace("_", " ").lower())
                excl.add(x.replace(" ", "_").lower())

        results = []
        for i in range(image.shape[0]):
            arr = (image[i].cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
            pil = Image.fromarray(arr)
            if pil.mode != "RGB":
                pil = pil.convert("RGB")
            w, h = pil.size
            ratio = float(height) / max(w, h)
            new = (max(1, int(w * ratio)), max(1, int(h * ratio)))
            pil = pil.resize(new, Image.LANCZOS)
            square = Image.new("RGB", (height, height), (255, 255, 255))
            square.paste(pil, ((height - new[0]) // 2, (height - new[1]) // 2))
            x = np.asarray(square).astype(np.float32)[:, :, ::-1]      # RGB -> BGR
            probs = sess.run([out_name], {inp.name: np.expand_dims(x, 0)})[0][0]

            gen = [(tags[j], float(probs[j])) for j in range(gi, ci) if probs[j] > threshold]
            ch = [(tags[j], float(probs[j])) for j in range(ci, len(tags)) if probs[j] > character_threshold]
            picked = ch + gen
            if sort_by_confidence:
                picked = sorted(picked, key=lambda t: -t[1])
            picked = [t for t, _ in picked if t.lower() not in excl]
            if escape_parens:
                picked = [t.replace("(", "\\(").replace(")", "\\)") for t in picked]
            sep = ", " if trailing_comma else ", "
            s = (", ".join(picked) + ("," if trailing_comma and picked else ""))
            results.append(s)

        print(f"[TaggerPlus/WD14] {model} | 实际设备: {label} | {len(results)} 张")
        return (results if len(results) > 1 else results[0], label)
