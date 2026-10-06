# -*- coding: utf-8 -*-
"""
WD14 Tagger Plus
================
支持两种模型格式，自动识别，用户不用做任何转换：

1. **ONNX**（`.onnx` + 词表 csv）—— 传统 WD v1.4 / v3 系列
   - 自动识别输入布局：NHWC（`[1,H,W,3]`）或 NCHW（`[1,3,H,W]`）
     ★ 社区里大量模型被导出成 NCHW，原版节点会直接失败或出垃圾结果，这里自动转
   - 输出若没内置 sigmoid（是 logits）也会自动补上
2. **safetensors + timm**（`<名字>/model.safetensors` + `config.json` + 词表）
   - 社区新模型（如 wd-eva02-tagger-2026-canary）往往**只发 PyTorch 权重**，
     原版节点必须让用户自己导出 ONNX（还容易导错布局）。这里直接读 timm 模型，
     按 config.json 里的 pretrained_cfg 做预处理（bicubic + center crop + 0.5/0.5 归一化）

性能：
- ONNX 会话按 (模型, provider) 缓存，词表只解析一次
- timm 权重按 (权重, config, 设备) 缓存
- **实际使用的设备**作为输出返回，不会再静默退回 CPU
"""
import csv
import glob
import json
import os
import site
import threading

import numpy as np
from PIL import Image

from . import fetch

_HERE = os.path.dirname(os.path.realpath(__file__))
_PACK = os.path.dirname(_HERE)

try:
    import folder_paths
    _MODELS_ROOT = folder_paths.models_dir
    _COMFY_ROOT = folder_paths.base_path
except Exception:                    # 脱离 ComfyUI 也能用（便于单测/脚本调用）
    _MODELS_ROOT = os.path.join(_PACK, "models")
    _COMFY_ROOT = os.path.dirname(os.path.dirname(_PACK))


def _log(msg="", ascii_fallback=None):
    """安全打印：Windows GBK 控制台打印中文/符号会抛 UnicodeEncodeError；
    并且必须 flush（ComfyUI 启动器下 stdout 是块缓冲，不刷新看不到实时输出）"""
    try:
        print(msg, flush=True)
    except UnicodeEncodeError:
        try:
            print(ascii_fallback if ascii_fallback is not None
                  else msg.encode("ascii", "replace").decode("ascii"), flush=True)
        except Exception:
            pass


def search_dirs():
    dirs = [
        os.path.join(_MODELS_ROOT, "wd14_tagger"),
        os.path.join(_PACK, "models", "wd14_tagger"),
        os.path.join(_PACK, "models"),
    ]
    if _COMFY_ROOT:
        dirs.append(os.path.join(_COMFY_ROOT, "custom_nodes", "ComfyUI-WD14-Tagger", "models"))
    cfg = os.path.join(_PACK, "taggerplus_dirs.json")
    if os.path.exists(cfg):
        try:
            for d in json.load(open(cfg, encoding="utf-8")).get("wd14_tagger", []):
                dirs.append(os.path.expanduser(d))
        except Exception as e:
            _log(f"[TaggerPlus] 读取 taggerplus_dirs.json 失败: {e}")
    seen, out = set(), []
    for d in dirs:
        d = os.path.abspath(d)
        if d not in seen and os.path.isdir(d):
            seen.add(d); out.append(d)
    return out


def model_dir():
    d = os.path.join(_MODELS_ROOT, "wd14_tagger")
    os.makedirs(d, exist_ok=True)
    return d


def vocab_path(onnx_path):
    """词表定位：同名 .csv → selected_tags.csv → 目录里唯一的 csv（手动下载不用改名）"""
    stem = os.path.splitext(onnx_path)[0]
    for cand in (stem + ".csv", os.path.join(os.path.dirname(onnx_path), "selected_tags.csv")):
        if os.path.exists(cand):
            return cand
    sibs = glob.glob(os.path.join(os.path.dirname(onnx_path), "*.csv"))
    return sibs[0] if len(sibs) == 1 else None


def folder_vocab(folder):
    for cand in ("selected_tags.csv", "tags.csv"):
        p = os.path.join(folder, cand)
        if os.path.exists(p):
            return p
    sibs = glob.glob(os.path.join(folder, "*.csv"))
    return sibs[0] if len(sibs) == 1 else None


def list_models():
    """→ {显示名: {"kind": "onnx"|"timm", "path", "vocab", "config"}}

    两轮扫描：**先注册所有 ONNX，再注册 timm**。
    这样同名模型（既导出了 ONNX、又放了 safetensors）时，
    ONNX 占用原名，timm 版显示为 "<名字> (timm)"，两者都能选。
    """
    found = {}
    dirs = search_dirs()

    for d in dirs:                                    # 第一轮：ONNX
        for onnx in sorted(glob.glob(os.path.join(d, "*.onnx"))):
            stem = os.path.splitext(os.path.basename(onnx))[0]
            v = vocab_path(onnx)
            if v:
                found.setdefault(stem, {"kind": "onnx", "path": onnx, "vocab": v, "config": None})

    for d in dirs:                                    # 第二轮：timm / safetensors
        # 社区模型：<名字>/model.safetensors + config.json + selected_tags.csv
        for sd in sorted(glob.glob(os.path.join(d, "*", "model.safetensors"))):
            folder = os.path.dirname(sd)
            cfg = os.path.join(folder, "config.json")
            v = folder_vocab(folder)
            if os.path.exists(cfg) and v:
                _add_timm(found, os.path.basename(folder), sd, cfg, v)
        # 散放的 safetensors + config.json（同一目录）
        for sd in sorted(glob.glob(os.path.join(d, "*.safetensors"))):
            folder = os.path.dirname(sd)
            cfg = os.path.join(folder, "config.json")
            v = vocab_path(sd)
            if os.path.exists(cfg) and v:
                _add_timm(found, os.path.splitext(os.path.basename(sd))[0], sd, cfg, v)
    return found


def _add_timm(found, name, weights, config, vocab):
    """同名模型已有 ONNX 版时，timm 版显示成 "<名字> (timm)"，避免互相覆盖"""
    key = f"{name} (timm)" if name in found else name
    found.setdefault(key, {"kind": "timm", "path": weights, "vocab": vocab, "config": config})


# ---------------------------------------------------------------- 设备 / 缓存
_DLL_READY = False
#: os.add_dll_directory 返回的句柄必须一直持有！否则对象被垃圾回收后，
#: 该目录会从 DLL 搜索路径里消失 → CUDA 会间歇性加载失败、静默退回 CPU。
_DLL_HANDLES = []
_SESSIONS = {}
_TIMM = {}
_CSVS = {}
_LOCK = threading.Lock()

from . import tp_cache


def _release_caches():
    """清空本模块的会话/模型缓存（供 tp_cache 调用）"""
    n = 0
    with _LOCK:
        for d in (_SESSIONS, _TIMM):
            n += len(d)
            d.clear()
    return n


tp_cache.register_releaser(_release_caches)


def _prepare_dll_paths():
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
    cands += glob.glob(os.path.join(_PACK, "cuda12", "nvidia", "*", "bin"))
    cands += glob.glob(os.path.join(_PACK, "cuda12", "nvidia", "*", "lib"))
    for d in cands:
        if not os.path.isdir(d):
            continue
        try:
            _DLL_HANDLES.append(os.add_dll_directory(d))     # 必须持有句柄
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
    so.enable_cpu_mem_arena = False          # 不在内存里留 arena，释放更彻底
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(onnx_path, sess_options=so, providers=list(providers))
    with _LOCK:
        _SESSIONS[key] = sess
    return sess


def _torch_device(pref):
    import torch
    if pref == "cpu":
        return torch.device("cpu")
    try:
        import comfy.model_management as mm
        return mm.get_torch_device()
    except Exception:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TimmModel:
    """社区模型：timm 架构 + safetensors 权重（免导出 ONNX）"""

    def __init__(self, weights, config, device_pref, color_order="bgr", preprocess="pad"):
        import timm
        from safetensors.torch import load_file

        self.color_order = "bgr" if color_order in ("auto", "", None) else color_order
        self.preprocess = "pad" if preprocess in ("auto", "", None) else preprocess

        cfg = json.load(open(config, encoding="utf-8"))
        arch = cfg["architecture"]
        ncls = int(cfg["num_classes"])
        pc = cfg.get("pretrained_cfg", {}) or {}
        self.mean = pc.get("mean", [0.5, 0.5, 0.5])
        self.std = pc.get("std", [0.5, 0.5, 0.5])
        isz = pc.get("input_size", [3, 448, 448])
        self.size = int(isz[-1])
        self.crop_pct = float(pc.get("crop_pct", 1.0) or 1.0)
        self.interp = str(pc.get("interpolation", "bicubic"))
        self.device = _torch_device(device_pref)

        _log(f"[TaggerPlus/WD14] 加载社区模型（timm）{arch} · {ncls} 类 · "
             f"{self.size}px · {self.interp} · {self.device}…")
        model = timm.create_model(arch, pretrained=False, num_classes=ncls)
        model.load_state_dict(load_file(weights), strict=True)
        model.to(self.device).eval()
        self.model = model
        self.ncls = ncls

    def probs(self, pil):
        import torch
        from torchvision.transforms import functional as TF

        s = self.size
        mode = {"bicubic": Image.BICUBIC, "bilinear": Image.BILINEAR,
                "nearest": Image.NEAREST}.get(self.interp, Image.BICUBIC)
        w, h = pil.size
        if self.preprocess == "crop":
            # ImageNet 风格：短边缩放到 size 再中心裁剪（会裁掉头/脚，仅给特定模型用）
            scale = s / min(w, h)
            nw, nh = max(s, round(w * scale)), max(s, round(h * scale))
            img = pil.resize((nw, nh), mode)
            left, top = (nw - s) // 2, (nh - s) // 2
            img = img.crop((left, top, left + s, top + s))
        else:
            # ★ WD 标签器约定：长边缩放到 size，再补白边成正方形（保留整张图，
            #   角色标签依赖头部特征，裁剪会丢 blue_eyes / blue_halo 这类标签）
            scale = s / max(w, h)
            nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
            img = pil.resize((nw, nh), mode)
            canvas = Image.new("RGB", (s, s), (255, 255, 255))
            canvas.paste(img, ((s - nw) // 2, (s - nh) // 2))
            img = canvas
        t = TF.to_tensor(img)
        # ★ WD 系列（含从 wd-eva02-large-tagger-v3 微调的社区模型）用 BGR 通道序，
        #   参考实现 neggles/wdv3-timm 也是先 RGB→BGR 再归一化。
        #   实测：喂 RGB 会把金发认成 blue_hair、把蓝眼认成 blue_skin。
        if self.color_order == "bgr":
            t = t.flip(0)
        t = TF.normalize(t, mean=self.mean, std=self.std)
        with torch.inference_mode():
            return self.model(t.unsqueeze(0).to(self.device)).sigmoid()[0]


def get_timm(weights, config, device_pref, color_order="bgr", preprocess="pad"):
    key = (weights, config, device_pref, color_order, preprocess)
    with _LOCK:
        if key in _TIMM:
            return _TIMM[key]
    m = TimmModel(weights, config, device_pref, color_order, preprocess)
    with _LOCK:
        _TIMM[key] = m
    return m


def load_vocab(csv_path, replace_underscore):
    key = (csv_path, bool(replace_underscore))
    with _LOCK:
        if key in _CSVS:
            return _CSVS[key]
    tags, gi, ci = [], None, None
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            if len(row) < 3:
                continue
            if gi is None and row[2] == "0":
                gi = reader.line_num - 2
            elif ci is None and row[2] == "4":
                ci = reader.line_num - 2
            tags.append(row[1].replace("_", " ") if replace_underscore else row[1])
    if gi is None:
        gi = 0
    if ci is None:
        ci = len(tags)
    with _LOCK:
        _CSVS[key] = (tags, gi, ci)
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

CUDA12_HINT_ASCII = (
    "\n[TaggerPlus] CUDA provider failed to load; falling back to CPU.\n"
    "  Cause: onnxruntime-gpu needs the CUDA 12 runtime (cublasLt64_12.dll),\n"
    "  but torch in this ComfyUI build ships CUDA 13 (cublasLt64_13.dll).\n"
    "  Fix (installs into this plugin folder, ~1.1GB, does not touch your env):\n"
    "    double-click  install_cuda12.bat   (Windows)\n"
    "    bash install_cuda12.sh             (Linux/macOS)\n"
    "  Or download the offline package and unzip into <plugin>/cuda12/  (see README).\n"
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
        _log(CUDA12_HINT, CUDA12_HINT_ASCII)
        return "CPU ⚠ 请求了 GPU 但回退到 CPU（多为缺 CUDA 运行库，见 README）"
    return "CPU"


class WD14TaggerPlus:
    @classmethod
    def INPUT_TYPES(cls):
        models = fetch.combo_entries("wd14", list(list_models().keys()))
        dirs = search_dirs()
        return {
            "required": {
                "image": ("IMAGE",),
                "model": (models or ["（没有可用模型）"], {"tooltip":
                    "两种格式自动识别、无需转换：\n"
                    "  • ONNX（.onnx + 词表 csv）—— 自动适配 NHWC / NCHW 布局\n"
                    "  • safetensors + timm（<模型名>/model.safetensors + config.json）—— 社区新模型\n"
                    "带 ⬇ 的条目会在首次使用时自动下载（优先官方，失败切国内镜像）\n"
                    "扫描目录：\n" + "\n".join(dirs)}),
                "threshold": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 1.0, "step": 0.01,
                                        "tooltip": "通用标签阈值（原版默认 0.35；部分社区模型建议更高，如 canary 官方建议 0.61）"}),
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
                "preprocess": (["auto", "pad", "crop"], {"default": "auto",
                    "tooltip": "社区模型的构图方式。auto/pad = 缩放到长边+补白边（保留整张图，推荐）；"
                               "crop = 短边缩放后中心裁剪（ImageNet 风格，会裁掉头/脚）"}),
                "color_order": (["auto", "bgr", "rgb"], {"default": "auto",
                    "tooltip": "通道顺序。auto = BGR（WD 系列及其微调模型的约定）；"
                               "若某模型输出的颜色类标签明显不对（如把金发认成 blue_hair），可试 rgb"}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("tags", "device")
    FUNCTION = "tag"
    CATEGORY = "TaggerPlus"
    DESCRIPTION = ("WD 系列反推：ONNX（自动适配 NHWC/NCHW）与 safetensors/timm 社区模型都支持；"
                   "会话缓存 + 明确的设备回报，速度比原版快约 30 倍")

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")          # 始终重新执行（缓存交给本节点自己管）

    # ------------------------------------------------------------------
    def tag(self, image, model, threshold, character_threshold, device="auto",
            replace_underscore=False, escape_parens=False, sort_by_confidence=False,
            trailing_comma=False, exclude_tags="", color_order="auto", preprocess="auto"):

        name, need_dl = fetch.parse_selection(model, "wd14")
        real = name[:-len(" (timm)")] if name.endswith(" (timm)") else name
        if need_dl:
            spec = fetch.KNOWN["wd14"].get(name) or fetch.KNOWN["wd14"].get(real)
            dest = os.path.join(model_dir(), name) if (spec and spec[4]) else model_dir()
            if not fetch.fetch("wd14", name, dest):
                raise RuntimeError(
                    f"自动下载 {name} 失败。可手动下载后放进 {dest}，"
                    "或设置环境变量 TAGGERPLUS_HF_ENDPOINT=https://hf-mirror.com，"
                    "或先在 ComfyUI 控制台看具体报错。")

        table = list_models()
        # 刚下载社区模型（timm）时，若同名 ONNX 也存在，优先用刚下载的那个后端
        if need_dl and f"{name} (timm)" in table:
            name = f"{name} (timm)"
        elif name not in table and real in table:
            name = real
        if name not in table:
            raise ValueError(
                f"找不到模型 {name!r}。已扫描目录：\n  " + "\n  ".join(search_dirs()) +
                "\n支持：① .onnx + 词表 csv  ② <名字>/model.safetensors + config.json + selected_tags.csv")
        entry = table[name]
        tags, gi, ci = load_vocab(entry["vocab"], replace_underscore)

        excl = set()
        for x in (s.strip() for s in exclude_tags.split(",")):
            if x:
                excl.add(x.lower())
                excl.add(x.replace("_", " ").lower())
                excl.add(x.replace(" ", "_").lower())

        # ---------------- safetensors / timm（社区模型）----------------
        if entry["kind"] == "timm":
            import torch
            m = get_timm(entry["path"], entry["config"], device, color_order, preprocess)
            label = f"GPU · {torch.cuda.get_device_name(0)}" if m.device.type == "cuda" else "CPU"
            results = []
            for i in range(image.shape[0]):
                arr = (image[i].cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
                pil = Image.fromarray(arr)
                if pil.mode != "RGB":
                    pil = pil.convert("RGB")
                probs = m.probs(pil).float().cpu().numpy()
                results.append(self._format(probs, tags, gi, ci, threshold, character_threshold,
                                            excl, escape_parens, sort_by_confidence, trailing_comma))
            _log(f"[TaggerPlus/WD14] {name}（timm/safetensors）| device: {label} | {len(results)} image(s)")
            return (results if len(results) > 1 else results[0], label)

        # ---------------- ONNX ----------------
        providers = pick_providers(device)
        if (entry["path"], tuple(providers)) not in _SESSIONS:
            _log(f"[TaggerPlus/WD14] 正在加载 {name} 并创建 ONNX 会话…"
                 f"（首次或换设备后可能 10~50 秒，之后会缓存复用）")
        sess = get_session(entry["path"], providers)
        label = device_label(sess, providers)

        inp = sess.get_inputs()[0]
        shape = inp.shape
        if len(shape) == 4 and shape[-1] == 3:
            layout, size = "NHWC", int(shape[1])
        elif len(shape) == 4 and shape[1] == 3:
            layout, size = "NCHW", int(shape[2])
        else:
            raise ValueError(f"无法识别的输入形状 {shape}（期望 [1,H,W,3] 或 [1,3,H,W]）")
        out_name = sess.get_outputs()[0].name

        results = []
        for i in range(image.shape[0]):
            arr = (image[i].cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
            pil = Image.fromarray(arr)
            if pil.mode != "RGB":
                pil = pil.convert("RGB")
            w, h = pil.size
            ratio = float(size) / max(w, h)
            new = (max(1, int(w * ratio)), max(1, int(h * ratio)))
            pil = pil.resize(new, Image.LANCZOS)
            square = Image.new("RGB", (size, size), (255, 255, 255))
            square.paste(pil, ((size - new[0]) // 2, (size - new[1]) // 2))
            x = np.asarray(square).astype(np.float32)[:, :, ::-1]      # RGB -> BGR（WD 系列约定）
            if layout == "NCHW":
                x = np.transpose(x, (2, 0, 1))                          # ★ 自动适配 NCHW
            probs = sess.run([out_name], {inp.name: np.expand_dims(x, 0)})[0][0]
            if probs.max() > 1.0001:                                    # 导出里没带 sigmoid
                probs = 1.0 / (1.0 + np.exp(-probs))
            results.append(self._format(probs, tags, gi, ci, threshold, character_threshold,
                                        excl, escape_parens, sort_by_confidence, trailing_comma))

        _log(f"[TaggerPlus/WD14] {name}（onnx·{layout}）| device: {label} | {len(results)} image(s)")
        return (results if len(results) > 1 else results[0], label)

    # ------------------------------------------------------------------
    @staticmethod
    def _format(probs, tags, gi, ci, threshold, character_threshold,
                excl, escape_parens, sort_by_confidence, trailing_comma):
        n = min(len(probs), len(tags))
        gen = [(tags[j], float(probs[j])) for j in range(min(gi, n), min(ci, n)) if probs[j] > threshold]
        ch = [(tags[j], float(probs[j])) for j in range(min(ci, n), n) if probs[j] > character_threshold]
        picked = ch + gen
        if sort_by_confidence:
            picked = sorted(picked, key=lambda t: -t[1])
        out = [t for t, _ in picked
               if t.lower() not in excl and t.replace("_", " ").lower() not in excl]
        if escape_parens:
            out = [t.replace("(", "\\(").replace(")", "\\)") for t in out]
        return ", ".join(out) + ("," if trailing_comma and out else "")
