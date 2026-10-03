# -*- coding: utf-8 -*-
"""
模型自动下载（WD14 / PixAI）
=============================

模型由下拉列表直接触发下载：选中带 `⬇` 的条目即可，无需命令行，也无需手动改名。

进度可见性：
- 使用 ComfyUI 官方 `comfy.utils.ProgressBar`，在网页 UI 的节点上显示进度条
- 所有日志 flush=True（ComfyUI 启动器下 stdout 为块缓冲，不刷新则不可见）
- 无论网速快慢，至少每 3 秒输出一次，包含 MB/s 与剩余时间估算
- 下载开始与结束均输出明确提示
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

#: 结构：显示名 -> (HF 仓库, [(远端文件, 保存名)], 体积说明, 备注, 是否解压到子目录)
KNOWN = {
    "wd14": {
        "wd-eva02-large-tagger-v3": ("SmilingWolf/wd-eva02-large-tagger-v3",
                                     [("model.onnx", "{name}.onnx"),
                                      ("selected_tags.csv", "{name}.csv")], "1.2 GB", "最准（推荐）", False),
        "wd-swinv2-tagger-v3": ("SmilingWolf/wd-swinv2-tagger-v3",
                                [("model.onnx", "{name}.onnx"),
                                 ("selected_tags.csv", "{name}.csv")], "446 MB", "较快", False),
        "wd-convnext-tagger-v3": ("SmilingWolf/wd-convnext-tagger-v3",
                                  [("model.onnx", "{name}.onnx"),
                                   ("selected_tags.csv", "{name}.csv")], "377 MB", "较快", False),
        "wd-vit-tagger-v3": ("SmilingWolf/wd-vit-tagger-v3",
                             [("model.onnx", "{name}.onnx"),
                              ("selected_tags.csv", "{name}.csv")], "361 MB", "最快", False),
        "wd-v1-4-moat-tagger-v2": ("SmilingWolf/wd-v1-4-moat-tagger-v2",
                                   [("model.onnx", "{name}.onnx"),
                                    ("selected_tags.csv", "{name}.csv")], "311 MB", "旧版 v1.4，可作对照", False),
        # ★ 社区新模型：官方只发 PyTorch 权重（timm），没有 ONNX。
        #   原版节点要求用户自己导出 ONNX（还容易导成 NCHW 用不了），
        #   本插件直接读 safetensors + timm，免转换。
        "wd-eva02-tagger-2026-canary": (
            "ashen-sensored/wd-eva02-tagger-2026-canary",
            [("model.safetensors", "model.safetensors"),
             ("config.json", "config.json"),
             ("selected_tags.csv", "selected_tags.csv")],
            "1.2 GB", "社区模型 · 截止 2026-05（比 v3 新两年）· 16473 标签", True),
    },
    "pixai": {
        "pixai-tagger-v1.0": ("pixai-labs/pixai-tagger-v1.0",
                              [("model.safetensors", "model.safetensors"),
                               ("config.json", "config.json")], "1.9 GB", "官方 v1.0", False),
    },
}

TAG = "[TaggerPlus]"
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
    repo, files, size, _note, _sub = spec
    os.makedirs(dest_dir, exist_ok=True)

    # 快速路径：文件都在本地就直接用，不联网（选中已下载条目时走这里）
    targets = [os.path.join(dest_dir, tpl.format(name=name)) for _remote, tpl in files]
    if not max_bytes and all(os.path.exists(x) for x in targets):
        _log(f"{TAG} {name} 已在本地，直接使用 → {dest_dir}")
        return True

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
    """下拉条目 = 已安装的模型 + 尚未安装的已知模型，**统一用裸模型名**。

    值的稳定性靠"名字本身不变"保证：下载前后都是同一个字符串，
    因此已保存的工作流不会失效，也不需要任何图标或体积后缀做标记。
    尚未下载的模型在首次运行时自动获取（fetch 内部判断文件是否存在）。
    """
    out = list(installed)
    for name in KNOWN.get(kind, {}):
        if name not in installed:
            out.append(name)
    return out


def parse_selection(sel, kind=None):
    """解析下拉值 -> (模型名, 是否需要确保本地存在)

    值是裸模型名。若它是本插件管理的已知模型，返回 need=True，
    由 fetch() 判断本地是否已有（有则直接使用，无则下载）。
    兼容早期带图标的旧值。
    """
    name = sel.strip().lstrip("\u2b07").strip()
    if "\u00b7" in name:                      # 兼容旧格式 "名字 · 体积"
        name = name.split("\u00b7")[0].strip()
    known = name in KNOWN.get(kind, {}) if kind else False
    return name, known
