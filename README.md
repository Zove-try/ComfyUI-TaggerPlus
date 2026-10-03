# ComfyUI-TaggerPlus

两个「纠正版」反推节点，修掉现有 WD14 / PixAI 节点的实际痛点。

![category](https://img.shields.io/badge/ComfyUI-custom--node-blue) ![license](https://img.shields.io/badge/license-MIT-green)

---

## 为什么做这个

| 痛点 | 现状 | 本插件 |
|---|---|---|
| **WD14 慢** | `ComfyUI-WD14-Tagger` 每次执行都重新 `InferenceSession(...)`（1.2 GB 模型重载），且 CUDA provider 加载失败时**静默**退回 CPU | 会话 + 词表**只加载一次**；并把**实际使用的设备**作为输出显示出来 |
| **WD14 静默降级** | 节点日志只打印「请求的 provider」，不打印「实际生效的 provider」，你永远不知道自己在跑 CPU | `device` 输出直接写：`GPU · RTX 5060 Ti` 或 `CPU ⚠ 请求了 GPU 但回退到 CPU` |
| **PixAI 要手输路径** | `model_file` / `config_file` 是文本框，要填绝对路径 | 从 `ComfyUI/models/pixai_tagger/` **自动扫描、下拉选择** |

## 安装

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/<you>/ComfyUI-TaggerPlus
# 依赖（ComfyUI 里通常已装 torch；onnxruntime-gpu 需要）
pip install -r ComfyUI-TaggerPlus/requirements.txt
```

重启 ComfyUI，节点出现在 **TaggerPlus** 分类下。

## 节点

### 1. WD14 Tagger Plus ⚡

- 输入：`image`、`model`（下拉）、`threshold`、`character_threshold`、`device`
- 输出：`tags`(STRING)、`device`(STRING)
- 模型放置：`ComfyUI/models/wd14_tagger/` 下放 **`.onnx` 与同名 `.csv`**
  （默认也会扫描 `custom_nodes/ComfyUI-WD14-Tagger/models/`，所以现有模型不用搬）
- 相对原版的差异（都是可选项，默认更"干净"）：
  - `escape_parens` 默认 **False**（原版强制把 `(` `)` 转成 `\(` `\)`，会让下游按标签查表的节点失效）
  - `sort_by_confidence` 默认 **False**（与原版一致的词表顺序）
  - `exclude_tags` 大小写不敏感、下划线/空格两种写法都匹配

### 2. PixAI Tagger Plus ⚡

- 输入：`image`、`model`（下拉）、6 个类别阈值、`threshold_mode`、`device` 等
- 输出：与原版**完全一致**的 7 个 STRING + 新增 `device`(STRING)
- 模型放置（二选一）：
  - **A**：整个 HF 仓库文件夹拷进 `ComfyUI/models/pixai_tagger/pixai-tagger-v1.0/`
    （含 `model.safetensors` + `config.json`）→ 下拉里出现 `pixai-tagger-v1.0`
  - **B**：只放两个文件到 `ComfyUI/models/pixai_tagger/` → 下拉里出现 `model.safetensors`
- 已经在别处的模型（例如 `E:/pixai-tagger`）不用搬，在插件根目录建 `taggerplus_dirs.json`：

```json
{
  "pixai_tagger": ["E:/pixai-tagger"],
  "wd14_tagger": ["D:/models/wd"]
}
```

## 关于「WD14 慢」的完整结论

在本机（RTX 5060 Ti + `onnxruntime-gpu 1.23.2` + `torch 2.9.1+cu130`）实测：

```
原版 WD14 节点：16.50s / 16.47s / 16.21s（三张不同图，耗时完全一样）
PixAI 节点：    5.92s（首张含加载） / 0.62s / 0.61s
```

拆开后：

| 环节 | 耗时 | 归属 |
|---|---|---|
| 磁盘顺序读 1.2 GB | 0.55 s | 无关（2,199 MB/s） |
| 解析 10,861 行词表 | 0.01 s | 无关 |
| 创建 ONNX 会话 | 3 – 6 s | **原节点每次执行都重建** |
| 推断（448²） | CPU 约 1.6 s / GPU 约 0.2 s | 取决于 provider |

**最坑的一点**：`onnxruntime-gpu` 的 CUDA provider 需要 **CUDA 12** 运行库（`cublasLt64_12.dll`），
而 ComfyUI 便携包里的 torch 往往带的是 **CUDA 13**（`cublasLt64_13.dll`），文件名不同 → 加载失败 →
**静默退回 CPU**。ORT 只在 stderr 打一行警告，节点日志完全看不到。

### 让 CUDA 真正生效（实测有效）

不用改动 ComfyUI 主环境 —— 把 CUDA 12 运行库装到**插件自己的目录**里，本插件会自动把它
加入 DLL 搜索路径：

```bash
"<ComfyUI>/python/python.exe" -m pip install --target "<ComfyUI>/custom_nodes/ComfyUI-TaggerPlus/cuda12" \
    nvidia-cublas-cu12 nvidia-cuda-runtime-cu12 nvidia-cufft-cu12 nvidia-curand-cu12
```

> 约 1.35 GB。之所以需要它：`onnxruntime-gpu` 的 CUDA EP 是针对 **CUDA 12** 编译的，而
> ComfyUI 便携包里的 `torch` 通常带 **CUDA 13**（`cublasLt64_13.dll`），**文件名不同 → 找不到**。
> cuDNN 9 一般已随 torch 提供，无需另装。

### 修好之后（本机 RTX 5060 Ti 实测）

| | 原版节点 | TaggerPlus |
|---|---|---|
| WD14 首张 | 16.50 s | 49 s（一次性：加载 1.35 GB CUDA 12 运行库 + 建 CUDA 会话） |
| WD14 后续每张 | 16.47 s | **0.11 – 0.19 s** |
| PixAI 首张 | 5.92 s | 3.7 s |
| PixAI 后续每张 | 0.61 s | **0.55 s** |

**93 张图只出标签：25 分钟 → 约 1 分钟。**
修好后 `device` 输出会从 `CPU ⚠ 请求了 GPU 但回退到 CPU` 变成 `GPU · NVIDIA GeForce RTX 5060 Ti`，
并且控制台会打印具体的修复命令（如果没修的话）。

## 兼容性 / 一致性

新节点与原节点在**同一张图**上做过逐标签比对（`escape_parens=True` 对齐原版行为）：

| 节点 | 结果 |
|---|---|
| WD14 Tagger Plus vs `WD14Tagger|pysssss` | ✅ 输出字符串完全相同 |
| PixAI Tagger Plus vs `PixAITagger` | ✅ 输出字符串完全相同 |

## 许可与署名

- 本插件：**MIT**
- `vendor/pixai_vitdet.py`（PixAI 的模型架构与预处理）取自
  [sln77/ComfyUI-Tagger](https://github.com/sln77/ComfyUI-Tagger)（MIT, Copyright (c) 2026 sln77），
  原文见 `vendor/LICENSE.ComfyUI-Tagger.txt`
- PixAI Tagger v1.0 权重：[pixai-labs/pixai-tagger-v1.0](https://huggingface.co/pixai-labs/pixai-tagger-v1.0)（Apache-2.0）
- WD 系列权重：[SmilingWolf](https://huggingface.co/SmilingWolf)（各自仓库许可）

---

## English

Two "fixed" tagger nodes for ComfyUI:

- **WD14 Tagger Plus** — the upstream `ComfyUI-WD14-Tagger` rebuilds the ONNX `InferenceSession`
  on *every* execution (1.2 GB reload) and **silently falls back to CPU** when the CUDA provider
  fails to load. This node caches the session and vocabulary, and **reports the device actually
  used** as an output.
- **PixAI Tagger Plus** — upstream requires typing absolute model paths. This node auto-scans
  `ComfyUI/models/pixai_tagger/` and gives you a dropdown.

Outputs are byte-identical to the upstream nodes (verified), so you can swap them in safely.
MIT licensed; the PixAI architecture code is vendored from
[sln77/ComfyUI-Tagger](https://github.com/sln77/ComfyUI-Tagger) with attribution.
