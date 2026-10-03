@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion
title ComfyUI-TaggerPlus - 安装 CUDA 12 运行库

set "PACK=%~dp0"
set "TARGET=%PACK%cuda12"

echo ================================================================
echo   ComfyUI-TaggerPlus · CUDA 12 运行库安装
echo ================================================================
echo.
echo   装到哪里 : %TARGET%
echo   为什么   : onnxruntime-gpu 的 CUDA provider 是针对 CUDA 12 编译的，
echo              如果你的 torch 是 CUDA 13 版本（cublasLt64_13.dll），
echo              ORT 会因为找不到 cublasLt64_12.dll 而静默退回 CPU。
echo              本脚本把这几个运行库装到插件目录里，不影响你的主环境。
echo.

rem ---- 找 ComfyUI 的 python ----
rem 便携包：<ComfyUI>\python\python.exe；插件在 <ComfyUI>\custom_nodes\<插件>
set "PY=%PACK%..\..\python\python.exe"
if exist "%PY%" (
    echo   [1/3] 找到便携包 python: %PY%
) else (
    set "PY=python"
    echo   [1/3] 没找到便携包 python，改用 PATH 里的 python
)

"%PY%" -c "import sys; print('       版本:', sys.version.split()[0])" 2>nul
if errorlevel 1 (
    echo.
    echo   [X] python 不可用。请手动执行：
    echo       ^<ComfyUI^>\python\python.exe -m pip install --target "%TARGET%" ^
    echo           nvidia-cublas-cu12 nvidia-cuda-runtime-cu12 nvidia-cufft-cu12 nvidia-curand-cu12
    echo.
    pause
    exit /b 1
)

echo.
echo   [2/3] 开始下载安装（约 1.3 GB，视网速可能几分钟）...
echo.
"%PY%" -m pip install --target "%TARGET%" --no-warn-script-location ^
    nvidia-cublas-cu12 nvidia-cuda-runtime-cu12 nvidia-cufft-cu12 nvidia-curand-cu12
if errorlevel 1 (
    echo.
    echo   [X] 安装失败。国内网络可加镜像重试：
    echo       "%PY%" -m pip install --target "%TARGET%" -i https://pypi.tuna.tsinghua.edu.cn/simple ^
    echo           nvidia-cublas-cu12 nvidia-cuda-runtime-cu12 nvidia-cufft-cu12 nvidia-curand-cu12
    echo.
    pause
    exit /b 1
)

echo.
echo   [3/3] 校验关键文件...
set "OK=1"
for %%F in (cublasLt64_12.dll cublas64_12.dll cudart64_12.dll) do (
    if exist "%TARGET%\nvidia\cublas\bin\%%F" (
        echo       OK  %%F
    ) else if exist "%TARGET%\nvidia\cuda_runtime\bin\%%F" (
        echo       OK  %%F
    ) else (
        echo       缺失 %%F
        set "OK=0"
    )
)

echo.
if "%OK%"=="1" (
    echo   完成！请重启 ComfyUI，然后看 WD14 Tagger Plus 的 device 输出，
    echo   应该从 "CPU ..." 变成 "GPU · <你的显卡>"。
) else (
    echo   有些文件没装上，请把上面的输出发出来。
)
echo.
pause
