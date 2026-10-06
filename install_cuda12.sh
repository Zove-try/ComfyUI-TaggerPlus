#!/usr/bin/env bash
# 只使用整合包自带的 python，不会用系统 PATH 里的 python。
# 自动找不到时，会让用户手动输入目录。
set -u
PACK="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET="$PACK/cuda12"
DETECT_ONLY=""
GIVEN=""
USESYS=0
for a in "$@"; do
    case "$a" in
        --detect-only) DETECT_ONLY="--detect-only" ;;
        --use-system-python) USESYS=1 ;;
    esac
done
_prev=""
for a in "$@"; do
    if [ "$_prev" = "--python" ]; then GIVEN="$a"; fi
    case "$a" in
        --python=*) GIVEN="${a#--python=}" ;;
    esac
    _prev="$a"
done
PY=""
FOUND_BY=""

echo "================================================================"
echo "  ComfyUI-TaggerPlus · CUDA 12 运行库安装"
echo "================================================================"
echo "  插件目录 : $PACK"
echo "  装到哪里 : $TARGET"
echo

if [ -n "${TAGGERPLUS_PYTHON:-}" ] && [ -x "$TAGGERPLUS_PYTHON" ]; then
    PY="$TAGGERPLUS_PYTHON"; FOUND_BY="环境变量 TAGGERPLUS_PYTHON"
fi

if [ -z "$PY" ] && [ -f "$PACK/.taggerplus_runtime.txt" ]; then
    cand="$(head -n1 "$PACK/.taggerplus_runtime.txt" | tr -d '\r')"
    if [ -n "$cand" ] && [ -x "$cand" ]; then PY="$cand"; FOUND_BY="插件记录的 ComfyUI 运行时路径"; fi
fi

if [ -z "$PY" ] && [ -n "$GIVEN" ]; then
    for c in "$GIVEN/python/bin/python3" "$GIVEN/python/bin/python" "$GIVEN/bin/python3" "$GIVEN/bin/python" "$GIVEN"; do
        if [ -x "$c" ]; then PY="$c"; FOUND_BY="命令行 --python"; break; fi
    done
    if [ -z "$PY" ]; then echo "  命令行给的路径里没有可执行的 python：$GIVEN"; fi
fi

if [ -z "$PY" ]; then
    DIR="$PACK"
    for _ in 1 2 3 4 5 6; do
        for sub in python/bin/python3 python/bin/python python_embeded/python.exe bin/python3 venv/bin/python .venv/bin/python; do
            if [ -x "$DIR/$sub" ]; then PY="$DIR/$sub"; FOUND_BY="自动查找（$DIR）"; break 2; fi
        done
        DIR="$(dirname "$DIR")"
    done
fi

if [ -z "$PY" ] && [ "$USESYS" = "1" ]; then
    for c in "$(command -v python3 2>/dev/null || true)" "$(command -v python 2>/dev/null || true)"; do
        if [ -n "$c" ] && [ -x "$c" ]; then PY="$c"; FOUND_BY="系统 PATH（--use-system-python 显式指定）"; break; fi
    done
    if [ -n "$PY" ]; then echo "  注意：正在使用系统 python，它未必和你的 ComfyUI 环境一致。"; fi
fi

if [ -z "$PY" ]; then
    echo "  没有自动找到整合包自带的 python。"
    echo
    echo "  请把下面任意一个粘贴进来（直接回车 = 取消）："
    echo "     · python 可执行文件的完整路径，例如 /opt/ComfyUI/venv/bin/python"
    echo "     · 或者整合包根目录，例如           /opt/ComfyUI"
    echo "     （也可用 --python <路径> 或 --use-system-python 直接指定）"
    echo
    while true; do
        printf "  路径: "
        read -r ANS || exit 0
        [ -z "$ANS" ] && exit 0
        ANS="${ANS%\"}"; ANS="${ANS#\"}"
        for c in "$ANS/python/bin/python3" "$ANS/python/bin/python" "$ANS/bin/python3" "$ANS/bin/python" "$ANS/venv/bin/python" "$ANS"; do
            if [ -x "$c" ] && [ -f "$c" ]; then PY="$c"; FOUND_BY="手动输入"; break; fi
        done
        [ -n "$PY" ] && break
        echo "  这个路径下没找到可执行的 python，请再试一次（回车取消）。"
        echo
    done
fi

if [ -z "$PY" ]; then
    echo
    echo "  没有指定 python，已取消。随时可以重新运行本脚本。"
    exit 0
fi

echo
echo "  [1/3] 使用 python : $PY"
echo "        来源       : $FOUND_BY"
"$PY" -c "import sys; print('         版本       :', sys.version.split()[0]); print('         环境前缀   :', sys.prefix)" || {
    echo "  这个 python 跑不起来，请重新运行本脚本并输入正确的路径。"; exit 0; }

if [ "$DETECT_ONLY" = "--detect-only" ]; then
    echo
    echo "  [--detect-only] 只检测不安装，结束。"
    exit 0
fi

echo
echo "  [2/3] 开始下载安装（约 1.3 GB，视网速可能几分钟）..."
"$PY" -m pip install --target "$TARGET" --no-warn-script-location \
    nvidia-cublas-cu12 nvidia-cuda-runtime-cu12 nvidia-cufft-cu12 nvidia-curand-cu12 || {
    echo
    echo "  安装失败。国内网络可加镜像重试："
    echo "      \"$PY\" -m pip install --target \"$TARGET\" -i https://pypi.tuna.tsinghua.edu.cn/simple \\"
    echo "          nvidia-cublas-cu12 nvidia-cuda-runtime-cu12 nvidia-cufft-cu12 nvidia-curand-cu12"
    exit 1; }

echo
echo "  [3/3] 校验关键文件..."
OK=1
for F in cublasLt64_12.dll cublas64_12.dll cudart64_12.dll; do
    if [ -f "$TARGET/nvidia/cublas/bin/$F" ] || [ -f "$TARGET/nvidia/cuda_runtime/bin/$F" ]; then
        echo "      OK  $F"
    else
        echo "      缺失 $F"; OK=0
    fi
done
echo
if [ "$OK" = "1" ]; then
    echo "  完成！请重启 ComfyUI，然后看反推节点的 device 输出，"
    echo "  应该从 \"CPU ...\" 变成 \"GPU · 你的显卡\"。"
else
    echo "  有些文件没装上，请把上面的输出发出来。"
fi
