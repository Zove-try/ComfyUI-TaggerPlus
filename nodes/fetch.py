# -*- coding: utf-8 -*-
"""
模型自动下载（WD14 / PixAI）
=============================

设计目标：新用户装完插件，下拉里直接选"⬇ 需要下载"的条目就能跑起来，
不需要知道 HuggingFace、不需要命令行、不需要手动改名。

- 默认先试 HuggingFace 官方，失败自动切 hf-mirror.com 国内镜像
- 可用环境变量 TAGGERPLUS_HF_ENDPOINT 强制指定镜像（如 https://hf-mirror.com）
- 可用插件目录下的 taggerplus_dirs.json 里的 "hf_endpoint" 字段指定
- 下载走 .part 临时文件 + 原子改名，中断不会留下坏文件
- 全部输出 ASCII，避免 Windows GBK 控制台崩溃
"""
import json
import os
import shutil
import sys
import time
import urllib.error
import urllib.request

_HERE = os.path.dirname(os.path.realpath(__file__))
_PACK = os.path.dirname(_HERE)

#: 已知模型：显示名 -> (HF 仓库, [(远端文件名, 保存名)], 体积说明)
KNOWN = {
    "wd14": {
        "wd-eva02-large-tagger-v3": ("SmilingWolf/wd-eva02-large-tagger-v3",
                                     [("model.onnx", "{name}.onnx"),
                                      ("selected_tags.csv", "{name}.csv")], "1.2 GB", "最准（推荐）"),
        "wd-swinv2-tagger-v3": ("SmilingWolf/wd-swinv2-tagger-v3",
                                [("model.onnx", "{name}.onnx"),
                                 ("selected_tags.csv", "{name}.csv")], "446 MB", "较快"),
        "wd-convnext-tagger-v3": ("SmilingWolf/wd-convnext-tagger-v3",
                                  [("model.onnx", "{name}.onnx"),
                                   ("selected_tags.csv", "{name}.csv")], "377 MB", "较快"),
        "wd-vit-tagger-v3": ("SmilingWolf/wd-vit-tagger-v3",
                             [("model.onnx", "{name}.onnx"),
                              ("selected_tags.csv", "{name}.csv")], "361 MB", "最快"),
        "wd-v1-4-moat-tagger-v2": ("SmilingWolf/wd-v1-4-moat-tagger-v2",
                                   [("model.onnx", "{name}.onnx"),
                                    ("selected_tags.csv", "{name}.csv")], "311 MB", "旧版 v1.4，可作对照"),
    },
    "pixai": {
        "pixai-tagger-v1.0": ("pixai-labs/pixai-tagger-v1.0",
                              [("model.safetensors", "model.safetensors"),
                               ("config.json", "config.json")], "1.9 GB", "官方 v1.0"),
    },
}

MARK = "\u2b07 "          # ⬇
SUFFIX = "  (需下载)"


def endpoints():
    """返回要依次尝试的下载源"""
    envs = [os.environ.get("TAGGERPLUS_HF_ENDPOINT")]
    cfg = os.path.join(_PACK, "taggerplus_dirs.json")
    if os.path.exists(cfg):
        try:
            envs.append(json.load(open(cfg, encoding="utf-8")).get("hf_endpoint"))
        except Exception:
            pass
    out = [e.rstrip("/") for e in envs if e]
    for e in ("https://huggingface.co", "https://hf-mirror.com"):
        if e not in out:
            out.append(e)
    return out


def _log(msg):
    try:
        print(msg)
    except UnicodeEncodeError:
        print(msg.encode("ascii", "replace").decode("ascii"))


def _human(n):
    return f"{n/1024/1024:.1f} MB" if n >= 1024 * 1024 else f"{n/1024:.0f} KB"


def download_file(url, dest, timeout=30, max_bytes=None):
    """流式下载 + 进度输出 + .part 原子改名"""
    part = dest + ".part"
    req = urllib.request.Request(url, headers={"User-Agent": "ComfyUI-TaggerPlus"})
    t0 = time.time()
    got = 0
    last = -1
    with urllib.request.urlopen(req, timeout=timeout) as r:
        total = int(r.headers.get("Content-Length") or 0)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        with open(part, "wb") as f:
            while True:
                chunk = r.read(4 << 20)
                if not chunk:
                    break
                f.write(chunk)
                got += len(chunk)
                if max_bytes and got >= max_bytes:
                    _log(f"    [test] stopped after {_human(got)}")
                    break
                if total:
                    pct = int(got * 100 / total)
                    if pct >= last + 10:
                        last = pct
                        sp = got / max(time.time() - t0, 1e-6)
                        _log(f"    {pct:3d}%  {_human(got)}/{_human(total)}  {sp/1024/1024:.1f} MB/s")
    if max_bytes and got >= max_bytes:
        os.remove(part)
        return got
    os.replace(part, dest)
    sp = got / max(time.time() - t0, 1e-6)
    _log(f"    100%  {_human(got)}  avg {sp/1024/1024:.1f} MB/s  -> {os.path.basename(dest)}")
    return got


def fetch(kind, name, dest_dir, max_bytes=None):
    """下载一个已知模型到 dest_dir；返回 True/False"""
    spec = KNOWN.get(kind, {}).get(name)
    if not spec:
        _log(f"[TaggerPlus] unknown model: {name}")
        return False
    repo, files, size, _note = spec
    os.makedirs(dest_dir, exist_ok=True)
    errors = []
    for ep in endpoints():
        try:
            _log(f"[TaggerPlus] downloading {name} ({size}) from {ep} ...")
            for remote, tpl in files:
                url = f"{ep}/{repo}/resolve/main/{remote}"
                download_file(url, os.path.join(dest_dir, tpl.format(name=name)), max_bytes=max_bytes)
            _log(f"[TaggerPlus] done: {name} -> {dest_dir}")
            return True
        except Exception as e:
            msg = f"{type(e).__name__}: {e}"
            errors.append(f"{ep} -> {msg}")
            _log(f"[TaggerPlus] failed from {ep}: {msg}")
    _log("[TaggerPlus] all endpoints failed:\n  " + "\n  ".join(errors))
    _log("[TaggerPlus] tip: set TAGGERPLUS_HF_ENDPOINT=https://hf-mirror.com "
         "or add a proxy (HTTPS_PROXY=http://127.0.0.1:port)")
    return False


def combo_entries(kind, installed):
    """下拉列表 = 已装模型 + 未装但可下载的模型"""
    out = list(installed)
    for name, (repo, files, size, note) in KNOWN.get(kind, {}).items():
        if name not in installed:
            out.append(f"{MARK}{name}{SUFFIX} · {size} · {note}")
    return out


def parse_selection(sel):
    """从下拉选择里解析出 (真实模型名, 是否需要下载)"""
    if sel.startswith(MARK):
        return sel[len(MARK):].split(SUFFIX)[0].strip(), True
    return sel, False
