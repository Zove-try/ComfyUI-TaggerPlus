# ComfyUI-TaggerPlus

![cover](assets/cover.png)

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

插件本体**只有约 60 KB**（纯代码），正常安装即可：

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/Zove-try/ComfyUI-TaggerPlus
```

或用 ComfyUI-Manager →「Install via Git URL」填仓库地址。重启 ComfyUI，
节点出现在 **TaggerPlus** 分类下。

### CUDA 运行库（可选，只有需要时才装）

**先直接跑一次。** 打开 WD14 Tagger Plus，看它的 `device` 输出：

| 输出 | 含义 | 要做什么 |
|---|---|---|
| `GPU · NVIDIA GeForce RTX xxxx` | 已经在用显卡 | **什么都不用装** |
| `CPU ⚠ 请求了 GPU 但回退到 CPU` | 缺 CUDA 12 运行库 | 选下面任一种方式装 |

插件在创建 ONNX 会话前会**自动依次扫描**三处，任意一处命中就直接用 GPU：

1. `site-packages/nvidia/*/bin` —— 装过 `nvidia-*-cu12` 的话
2. `torch/lib` —— **如果你的 ComfyUI 便携包用的是 CUDA 12 版 torch
   （cu121 / cu124 / cu126，绝大多数便携包都是），`cublasLt64_12.dll` 这里就有，零下载直接可用**
3. `<插件目录>/cuda12/` —— 下面两种安装方式的落地位置

只有三处都找不到时才会退回 CPU（典型情况：**torch 是 CUDA 13 版本**）。

---

#### 方式 A：网盘下载（推荐，不用命令行、不用联网 pip）

> **下载地址（夸克网盘）：https://pan.quark.cn/s/77632a836e96**
> 提取方式：打开链接直接下载，无需提取码
> 文件名：`ComfyUI-TaggerPlus_CUDA12运行库_Windows.zip`
> 大小：**787 MB**（解压后约 1.1 GB）
> SHA256：`287643aa255738c49ead59e2d3dc7562879d4bebb6a0ef1a7d221936de607df5`
> 适用：**仅 Windows**（Linux / macOS 请用方式 B）

下载后**把压缩包里的 `cuda12` 文件夹整体解压到插件目录**，与 `nodes`、`vendor` 同级：

```
ComfyUI/
└── custom_nodes/
    └── ComfyUI-TaggerPlus/          ← 插件目录
        ├── nodes/
        ├── vendor/
        ├── __init__.py
        └── cuda12/                  ← 解压到这里
            └── nvidia/
                ├── cublas/bin/cublasLt64_12.dll
                ├── cuda_runtime/bin/cudart64_12.dll
                ├── cufft/bin/cufft64_11.dll
                └── curand/bin/curand64_10.dll
```

解压后最终路径必须是
`ComfyUI-TaggerPlus/cuda12/nvidia/cublas/bin/cublasLt64_12.dll`。
如果变成了 `cuda12/cuda12/nvidia/...`（多了一层），手动把里面那层提上来即可。

重启 ComfyUI，`device` 显示 `GPU · ...` 就成了。

#### 方式 B：一键脚本（自动 pip，跨平台）

| 平台 | 操作 |
|---|---|
| Windows | 双击插件目录里的 **`install_cuda12.bat`** |
| Linux / macOS | `bash install_cuda12.sh` |

脚本会自动找到 ComfyUI 的 python（便携包路径优先），把 4 个 NVIDIA 运行库装进
`<插件目录>/cuda12/`，并校验结果。失败时会提示换国内镜像。

> 为什么仓库里不直接附带这些 DLL：解压后 1.1 GB，GitHub 单文件上限 100 MB，仓库也没法 clone；
> 而且 NVIDIA 运行时库应通过官方渠道分发。所以仓库保持 **约 60 KB 纯代码**，运行库按需获取。
> 不需要了直接删掉 `cuda12/` 文件夹即可，不影响任何其他东西。

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

**Windows users who need the CUDA 12 runtime** can grab the offline package (787 MB) from
[Quark Drive](https://pan.quark.cn/s/77632a836e96) and unzip its `cuda12/` folder into the plugin directory —
no pip, no command line. Linux/macOS use `install_cuda12.sh`.

Outputs are byte-identical to the upstream nodes (verified), so you can swap them in safely.
MIT licensed; the PixAI architecture code is vendored from
[sln77/ComfyUI-Tagger](https://github.com/sln77/ComfyUI-Tagger) with attribution.

## FAQ

<details>
<summary><b>Q：不下载 CUDA 运行库、也不放进目录，会怎样？</b></summary>

**完全不影响使用，只是慢。** 实测（同一台机、同一批图）：

| | 第 1 张 | 第 2 张 | 第 3 张 | 标签结果 |
|---|---|---|---|---|
| 有 CUDA 运行库（GPU） | 2.33 s | 0.09 s | 0.17 s | — |
| 没有（自动退回 CPU） | 4.01 s | 1.64 s | 1.89 s | **与 GPU 逐字节相同** |

也就是说：

- ✅ 功能、参数、输出**一模一样**，不会有任何报错或缺失
- ✅ 只是单张从 **0.1 秒变成 1.6 秒左右**（约 10–20 倍）
- ✅ `device` 输出会明确写 `CPU ⚠ 请求了 GPU 但回退到 CPU`，控制台也会打印修复方法
- ⚠️ 唯一的代价是 CPU 占用（会和 ComfyUI 里的其他 CPU 任务抢资源）

而且即使不装运行库，本插件相对原版**仍然是大幅提速**的
（原版 16.5 s/张 → 本插件 CPU 约 1.6 s/张，因为省掉了每张重建 ONNX 会话的开销）。

</details>


<details>
<summary><b>Q：装了插件但还是 CPU，device 显示 <code>CPU ⚠ …</code></b></summary>

按顺序检查：

1. `cuda12/nvidia/cublas/bin/cublasLt64_12.dll` 这个路径存在吗？（多一层 `cuda12/` 是常见错误）
2. 重启的是 **跑着 8188 端口的那个 ComfyUI 进程**吗？（不是关掉网页就行）
3. 控制台（不是节点界面）里有没有 ORT 的报错？把报错发到 issue 里
4. 如果用的是 CUDA 12 版 torch，其实不用装 —— 检查一下 `torch/lib/cublasLt64_12.dll` 是否存在
</details>

<details>
<summary><b>Q：第一次跑要等几十秒，正常吗？</b></summary>

正常。首次创建 CUDA 会话要从磁盘读入约 1 GB 的 CUDA 运行库并初始化，
冷启动可能 10–50 秒；之后会话被缓存，**每张图只要 0.07–0.2 秒**。
热缓存下建会话约 1.8 秒。
</details>

<details>
<summary><b>Q：和原版节点的输出会不一样吗？</b></summary>

不会。与原版做过逐标签比对（把 `escape_parens` 设为 `True` 对齐原版行为时），
输出字符串**完全相同**。默认关闭括号转义，是为了让下游按标签查表的节点能正常工作
（原版会把 `(` `)` 转成 `\(` `\)`，导致 `xxx_(series)` 这类标签查不到）。
</details>

<details>
<summary><b>Q：模型一定要搬到 models 目录吗？</b></summary>

不用。插件默认也会扫描 `custom_nodes/ComfyUI-WD14-Tagger/models/`（WD14），
另外可以在插件根目录建 `taggerplus_dirs.json` 指定任意目录：

```json
{
  "pixai_tagger": ["E:/pixai-tagger"],
  "wd14_tagger": []
}
```
</details>

<details>
<summary><b>Q：为什么我的 WD14 有 16 秒那么慢？</b></summary>

两个原因叠加：① 原版节点每次执行都重建 ONNX 会话；② CUDA provider 加载失败后
**静默**退回 CPU（CPU 单张要 1.6 秒以上）。本插件把两者都解决了，
并且会把实际设备显示出来，不会再让你蒙在鼓里。
</details>

