# Changelog

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
