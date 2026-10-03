#!/usr/bin/env bash
# ComfyUI-TaggerPlus · 安装 CUDA 12 运行库（Linux / macOS）
#
# 为什么需要：onnxruntime-gpu 的 CUDA provider 是针对 CUDA 12 编译的。
# 如果你的 torch 是 CUDA 13 版本，ORT 会因为找不到 libcublasLt.so.12 而静默退回 CPU。
# 本脚本把运行库装到插件目录里（不影响主环境）。
set -euo pipefail

PACK="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET="$PACK/cuda12"

echo "================================================================"
echo "  ComfyUI-TaggerPlus · CUDA 12 运行库安装"
echo "================================================================"
echo "  装到: $TARGET"
echo

# 找 ComfyUI 的 python：便携包 <ComfyUI>/python/bin/python3，否则用 PATH 里的
PY="$PACK/../../python/bin/python3"
if [ -x "$PY" ]; then
  echo "  [1/3] 使用便携包 python: $PY"
else
  PY="${PYTHON:-python3}"
  echo "  [1/3] 使用 PATH 里的: $PY"
fi
"$PY" -c "import sys; print('       版本:', sys.version.split()[0])"

echo
echo "  [2/3] 下载安装（约 1.3 GB）..."
PKGS=(nvidia-cublas-cu12 nvidia-cuda-runtime-cu12 nvidia-cufft-cu12 nvidia-curand-cu12)
if ! "$PY" -m pip install --target "$TARGET" --no-warn-script-location "${PKGS[@]}"; then
  echo
  echo "  [X] 失败。可换镜像重试（中国大陆）："
  echo "      $PY -m pip install --target \"$TARGET\" -i https://pypi.tuna.tsinghua.edu.cn/simple ${PKGS[*]}"
  exit 1
fi

echo
echo "  [3/3] 校验..."
miss=0
for f in "$TARGET"/nvidia/*/lib/libcublasLt.so.12 "$TARGET"/nvidia/*/lib/libcublas.so.12; do
  if ls $f >/dev/null 2>&1; then echo "      OK  $(basename "$f")"; else echo "      缺失 $(basename "$f")"; miss=1; fi
done

echo
if [ "$miss" = "0" ]; then
  echo "  完成！重启 ComfyUI，WD14 Tagger Plus 的 device 输出应从 CPU 变成 GPU。"
else
  echo "  有文件缺失，请把上面的输出发出来。"
fi
