# Changelog

## v0.1.4 — 2026-10-03
### 修复

- **下载完成后工作流校验失败**：模型下载完毕后，下拉中的 `⬇ <模型名>` 条目会消失，
  而已保存的工作流按字符串记录该值，再次运行时报
  `Value not in list: model: '⬇ …' not in [...]`，必须手动重选模型。
  现在该条目始终保留，选中时直接使用本地文件（不重复下载），历史工作流也随之恢复正常。
- 选中已下载的社区模型（timm）时，优先使用刚下载的那个后端，避免同名 ONNX 抢先匹配。


## v0.1.3 — 2026-10-03
### 新增

- 支持社区模型：直接读取 safetensors + timm，无需导出 ONNX
  - 社区模型多只发布 PyTorch 权重，原版节点要求用户自行导出 ONNX，且容易导出为 NCHW 布局
  - 现在将 `<名字>/model.safetensors` + `config.json` + 词表放入 `models/wd14_tagger/` 即可
- ONNX 输入布局自动适配：识别 NHWC / NCHW 并转置；输出若非概率则自动应用 sigmoid
- 自动下载列表新增 `wd-eva02-tagger-2026-canary`（Apache-2.0，16,473 标签，训练数据截止 2026-05-18）
- 同名模型同时存在 ONNX 与 safetensors 版本时，后者显示为 `<名字> (timm)`
- 新增可选参数 `color_order`（auto/bgr/rgb）与 `preprocess`（auto/pad/crop）

### 修复

- 通道顺序：社区模型沿用 WD 的 BGR 约定。输入 RGB 时金发会被识别为 `blue_hair`、蓝眼被识别为
  `blue_skin`，与 ONNX 路径一致率仅 43.8%
- 构图：标签器需「长边缩放 + 白边补方」。改用 ImageNet 式中心裁剪会裁掉头/脚，
  丢失 `blue_eyes` / `blue_halo` / `blue_ribbon` 并误报 `head_out_of_frame`，一致率 50%
- 两项修正后，同一份权重的 safetensors 与 ONNX 路径一致率达 98.5%

## v0.1.2 — 2026-10-03
### 新增

- 下载进度可视化：使用 `comfy.utils.ProgressBar` 在节点上显示进度条；
  控制台至少每 3 秒输出一行（百分比 / 已下载 / 实时速度 / 剩余时间）
- 下载文件、切换下载源、加载模型与创建 ONNX 会话均输出状态信息

### 修复

- CUDA 间歇性退回 CPU：`os.add_dll_directory()` 的返回值未被持有，对象被回收后该目录
  会从 DLL 搜索路径中移除，导致 CUDA provider 时可用时不可用。现在句柄常驻于模块级列表
- 日志改为 `flush=True`：ComfyUI 启动器下 stdout 为块缓冲，未刷新时控制台看不到实时输出

## v0.1.1 — 2026-10-03
### 新增

- 模型自动下载：下拉列表中带 `⬇` 的条目选中即自动下载，先试 HuggingFace 官方，
  失败切换到 hf-mirror.com 镜像；使用 `.part` 临时文件与原子改名
- 支持通过 `TAGGERPLUS_HF_ENDPOINT` 环境变量或 `taggerplus_dirs.json` 的 `hf_endpoint` 指定镜像
- 词表匹配放宽：同时识别 `<模型名>.csv` 与 `selected_tags.csv`，手动下载无需改名

## v0.1.0 — 2026-10-03
### 新增

- **WD14 Tagger Plus**
  - ONNX 会话与词表缓存
  - `device` 输出：报告实际使用的设备
  - 模型从 `models/wd14_tagger/` 下拉选择，并兼容原插件的 models 目录
  - 依次扫描 `site-packages/nvidia/*/bin`、`torch/lib`、`<插件>/cuda12/` 加载 CUDA 12 运行库
  - 可选 `escape_parens`（默认关闭）、`sort_by_confidence`（默认关闭）
- **PixAI Tagger Plus**
  - 模型与配置从 `models/pixai_tagger/` 自动扫描、下拉选择
  - 输出与原版一致的 7 个 STRING，另加 `device`
  - 支持 `taggerplus_dirs.json` 指定额外模型目录
- `install_cuda12.bat` / `install_cuda12.sh`：将 CUDA 12 运行库安装到插件目录
- README：性能实测、安装说明与 FAQ

### 修复

- Windows GBK 控制台崩溃：CPU 回退路径打印 `⚠` 与中文时抛出 `UnicodeEncodeError`。
  现在日志经安全包装，并配有 ASCII 备用文本

### 实测（RTX 5060 Ti）

| | 原版节点 | TaggerPlus |
|---|---|---|
| WD14 首次 | 16.50 s | 冷启动 10–50 s（每进程一次） |
| WD14 之后每张 | 16.47 s | 0.09 – 0.17 s（GPU）/ 约 1.6 s（CPU） |
| PixAI 之后每张 | 0.61 s | 0.55 s |

与原版逐标签比对：输出字符串完全相同。
