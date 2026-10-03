# -*- coding: utf-8 -*-
"""
模型自动下载（WD14 / PixAI）
=============================

设计目标：新用户装完插件，下拉里直接选"⬇ 需要下载"的条目就能跑起来，
不需要知道 HuggingFace、不需要命令行、不需要手动改名。

进度可见性（重点）：
- 用 ComfyUI 官方的 `comfy.utils.ProgressBar` → **网页 UI 的节点上会显示进度条**
- 所有日志 flush=True（ComfyUI 启动器下 stdout 是块缓冲，不 flush 就看不到）
- 每 3 秒至少输出一次（哪怕网速很慢也有动静），并显示 MB/s 与剩余时间估算
- 下载前后都有明确的开始/结束提示，避免"看起来卡住"
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

MARK = "\u2b07 "
SUFFIX = "  (需下载)"
TAG = "[TaggerPlus]"
_print_lock = None


def endpoints():
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


def _log(msg=""):
    """安全 + 立即刷新（ComfyUI 启动器下 stdout 是块缓冲，不 flush 控制台看不到）"""
    try:
        print(msg, flush=True)
    except UnicodeEncodeError:
        try:
            print(msg.encode("ascii", "replace").decode("ascii"), flush=True)
        except Exception:
            pass


def _human(n):
    if n >= 1024 ** 3:
        return f"{n/1024**3:.2f} GB"
    if n >= 1024 * 1024:
        return f"{n/1024/1024:.1f} MB"
    return f"{n/1024:.0f} KB"


class _NullBar:
    def update(self, *a, **k):
        pass

    def update_absolute(self, *a, **k):
        pass


def _make_bar(total_mb):
    """ComfyUI 官方进度条 → 网页 UI 节点上会出现进度条；不在 ComfyUI 里则静默降级"""
    try:
        import comfy.utils
        return comfy.utils.ProgressBar(max(1, int(total_mb)))
    except Exception:
        return _NullBar()


def download_file(url, dest, timeout=30, max_bytes=None, label=None):
    """流式下载：进度条 + 每 3 秒输出一次 + .part 原子改名"""
    part = dest + ".part"
    req = urllib.request.Request(url, headers={"User-Agent": "ComfyUI-TaggerPlus"})
    label = label or os.path.basename(dest)
    t0 = time.time()
    got = 0
    with urllib.request.urlopen(req, timeout=timeout) as r:
        total = int(r.headers.get("Content-Length") or 0)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        bar = _make_bar(total / 1024 / 1024) if total else _NullBar()
        _log(f"{TAG} 开始下载 {label}"
             + (f"（{_human(total)}）" if total else "（大小未知）") + f" → {os.path.basename(dest)}")
        last_t, last_mb = t0, 0
        win = [(t0, 0)]                                  # 滑动窗口，用于算瞬时速度
        with open(part, "wb") as f:
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
                got += len(chunk)
                mb = got >> 20
                if mb > last_mb:                       # 每 1 MB 推进一次进度条
                    try:
                        bar.update(mb - last_mb)
                    except Exception:
                        pass
                    last_mb = mb
                now = time.time()
                if now - last_t >= 3.0:                # 至少每 3 秒说一句话
                    win.append((now, got))
                    while len(win) > 2 and now - win[0][0] > 8.0:
                        win.pop(0)
                    dt_win = now - win[0][0]
                    sp = (got - win[0][1]) / dt_win if dt_win > 0.5 else got / max(now - t0, 1e-6)
                    if total:
                        pct = got * 100 / total
                        eta = (total - got) / max(sp, 1e-6)
                        eta_s = f"{int(eta//60)}分{int(eta%60):02d}秒" if eta < 3600 else f"{eta/60:.0f}分"
                        _log(f"{TAG}   下载中 {pct:5.1f}%  {_human(got)} / {_human(total)}"
                             f"  {sp/1024/1024:.1f} MB/s  剩余约 {eta_s}")
                    else:
                        _log(f"{TAG}   下载中 {_human(got)}  {sp/1024/1024:.1f} MB/s")
                    last_t = now
                if max_bytes and got >= max_bytes:
                    _log(f"{TAG}   [测试] 已到上限，停止在 {_human(got)}")
                    break
    if max_bytes and got >= max_bytes:
        os.remove(part)
        return got
    os.replace(part, dest)
    sp = got / max(time.time() - t0, 1e-6)
    _log(f"{TAG} 下载完成 {os.path.basename(dest)}  {_human(got)}  用时 {time.time()-t0:.1f}s"
         f"  平均 {sp/1024/1024:.1f} MB/s")
    return got


def fetch(kind, name, dest_dir, max_bytes=None):
    """下载一个已知模型；返回 True/False"""
    spec = KNOWN.get(kind, {}).get(name)
    if not spec:
        _log(f"{TAG} 未知模型：{name}")
        return False
    repo, files, size, _note = spec
    os.makedirs(dest_dir, exist_ok=True)
    eps = endpoints()
    _log(f"{TAG} 需要下载模型 {name}（{size}），共 {len(files)} 个文件，"
         f"依次尝试 {len(eps)} 个下载源")
    errors = []
    for ep in eps:
        try:
            _log(f"{TAG} 使用下载源：{ep}")
            for i, (remote, tpl) in enumerate(files, 1):
                _log(f"{TAG} 文件 {i}/{len(files)}：{remote}")
                download_file(f"{ep}/{repo}/resolve/main/{remote}",
                              os.path.join(dest_dir, tpl.format(name=name)),
                              max_bytes=max_bytes, label=f"{name} · {remote}")
            _log(f"{TAG} ✓ 模型就绪：{name} → {dest_dir}")
            return True
        except Exception as e:
            msg = f"{type(e).__name__}: {e}"
            errors.append(f"{ep} → {msg}")
            _log(f"{TAG} ✗ 该源失败：{msg}")
            _log(f"{TAG} 换下一个源重试…")
    _log(f"{TAG} ✗ 所有下载源都失败了：")
    for e in errors:
        _log(f"{TAG}     {e}")
    _log(f"{TAG} 建议：① 设置环境变量 TAGGERPLUS_HF_ENDPOINT=https://hf-mirror.com "
         f"② 或设置 HTTPS_PROXY=http://127.0.0.1:端口 走代理 ③ 或按 README 手动下载")
    return False


def combo_entries(kind, installed):
    out = list(installed)
    for name, (repo, files, size, note) in KNOWN.get(kind, {}).items():
        if name not in installed:
            out.append(f"{MARK}{name}{SUFFIX} · {size} · {note}")
    return out


def parse_selection(sel):
    if sel.startswith(MARK):
        return sel[len(MARK):].split(SUFFIX)[0].strip(), True
    return sel, False
