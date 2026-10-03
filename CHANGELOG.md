# Changelog

## v0.1.3 — 2026-10-03

### 新增

- **社区模型支持：直接读 safetensors + timm，免导 ONNX**
  - 社区新模型（如 `wd-eva02-tagger-2026-canary`）通常只发 PyTorch 权重，
    原版节点要求用户自己导出 ONNX，且极易导成 NCHW 布局导致不可用
  - 现在把 `<名字>/model.safetensors` + `config.json` + 词表放进 `models/wd14_tagger/` 即可
- **ONNX 布局自动适配**：自动识别 NHWC / NCHW 并转置，导出成 NCHW 的社区 ONNX 不再需要手动转换
- ONNX 输出若不是概率（导出时没带 sigmoid）会自动补 sigmoid
- 自动下载表新增 `wd-eva02-tagger-2026-canary`（Apache-2.0 / 16,473 标签 / 训练截止 2026-05-18）
- 同名模型同时有 ONNX 与 timm 版时，timm 版显示为 `<名字> (timm)`，两个都能选
- 新增可选项 `color_order`(auto/bgr/rgb) 与 `preprocess`(auto/pad/crop)

### 修复（开发中发现的两个预处理陷阱，实测数据）

- **通道顺序**：社区模型沿用 WD 的 **BGR** 约定。喂 RGB 会把金发识别成 `blue_hair`、
  蓝眼识别成 `blue_skin` —— 与 ONNX 版一致率只有 43.8%
- **构图**：标签器必须"长边缩放 + 白边补方"保留整张图；用 ImageNet 的中心裁剪会把头/脚裁掉，
  丢掉 `blue_eyes` / `blue_halo` / `blue_ribbon` 等标签，还误报 `head_out_of_frame` —— 一致率仅 50%

两项修正后，同一份权重的 safetensors 与 ONNX 路径**一致率 98.5%**（实测同一张图 65 vs 64 个标签）。


## v0.1.2 — 2026-10-03

### 新增

- **下载进度可见**：用 ComfyUI 官方 `comfy.utils.ProgressBar`，网页 UI 的节点上会显示进度条；
  控制台至少每 3 秒输出一行（百分比 / 已下载 / 实时速度 / 剩余时间估算）
- 下载每个文件、切换下载源、加载模型与创建 ONNX 会话都有明确状态输出

### 修复

- **CUDA 间歇性回退 CPU**：`os.add_dll_directory()` 的返回值之前没有持有，
  对象被垃圾回收后该目录会从 DLL 搜索路径中消失，导致 CUDA provider 时而可用时而不可用。
  现在句柄保存在模块级列表中常驻。
- 所有日志改为 `flush=True`：ComfyUI 启动器下 stdout 是块缓冲，不刷新则控制台看不到实时输出


## v0.1.1 — 2026-10-03

### 新增

- **模型自动下载**：下拉列表里带 `⬇` 的条目，选中即自动下载
  （先试 HuggingFace 官方，失败自动切 hf-mirror.com 国内镜像；`.part` 临时文件 + 原子改名，
  中断不留坏文件）
- 支持 `TAGGERPLUS_HF_ENDPOINT` 环境变量 / `taggerplus_dirs.json` 的 `hf_endpoint` 指定镜像
- WD14 词表匹配放宽：现在同时认 `<模型名>.csv` 与 HF 原始的 `selected_tags.csv`，
  手动下载的用户**不需要改名**
- README 增加「模型下载」章节（自动 / 镜像 / 手动三条路径 + 体积表）


## v0.1.0 — 2026-10-03

首个版本。

### 新增

- **WD14 Tagger Plus**
  - ONNX 会话与词表缓存（原版每次执行都重建 1.2 GB 会话）
  - `device` 输出：明确报告**实际使用**的设备，修掉原版"静默退回 CPU"的问题
  - 模型从 `models/wd14_tagger/` 下拉选择，并兼容原插件的 models 目录
  - 自动依次扫描 `site-packages/nvidia/*/bin`、`torch/lib`、`<插件>/cuda12/` 加载 CUDA 12 运行库
  - 可选 `escape_parens`（默认关闭）、`sort_by_confidence`（默认关闭，与原版一致）
- **PixAI Tagger Plus**
  - 模型与配置从 `models/pixai_tagger/` 自动扫描，下拉选择（原版要手输绝对路径）
  - 输出与原版完全一致的 7 个 STRING，另加 `device`
  - 支持 `taggerplus_dirs.json` 指定额外模型目录
- `install_cuda12.bat` / `install_cuda12.sh`：一键安装 CUDA 12 运行库到插件目录
- README：包含完整的性能实测、安装说明（网盘 / 脚本两条路径）与 FAQ

### 修复

- **Windows GBK 控制台崩溃**：CPU 回退路径打印 `⚠` / 中文会抛 `UnicodeEncodeError`，
  导致"没装 CUDA 运行库"的用户直接报错。现在所有日志走安全包装，并配 ASCII 备用文本。

### 实测数据（RTX 5060 Ti）

| | 原版节点 | TaggerPlus |
|---|---|---|
| WD14 首张 | 16.50 s | 2.3 s（热缓存建会话）/ 最多数十秒（冷启动读 1 GB DLL） |
| WD14 后续每张 | 16.47 s | **0.09–0.17 s**（GPU）/ 1.6 s（无运行库时 CPU） |
| PixAI 后续每张 | 0.61 s | 0.55 s |

与原版逐标签比对：**输出字符串完全相同**。
