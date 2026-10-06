@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion
title ComfyUI-TaggerPlus - CUDA 12 运行库安装

set "PACK=%~dp0"
set "TARGET=%PACK%cuda12"
set "DETECT=0"
set "PY="
set "BY="

rem 参数：--detect-only 只检测；--python <路径> 手动指定
if /i "%~1"=="--detect-only" set "DETECT=1"
if /i "%~2"=="--detect-only" set "DETECT=1"
if /i "%~3"=="--detect-only" set "DETECT=1"
if /i "%~1"=="--python" set "GIVEN=%~2"
if /i "%~2"=="--python" set "GIVEN=%~3"
if /i "%~1"=="--use-system-python" set "USESYS=1"
if /i "%~2"=="--use-system-python" set "USESYS=1"
if /i "%~3"=="--use-system-python" set "USESYS=1"

echo ================================================================
echo   ComfyUI-TaggerPlus · CUDA 12 运行库安装
echo ================================================================
echo   插件目录 : %PACK%
echo   装到哪里 : %TARGET%
echo.

rem ---------- 1) 命令行 --python ----------
if defined GIVEN (
    set "C=%GIVEN:"=%"
    if exist "!C!\python\python.exe"          ( set "PY=!C!\python\python.exe" & set "BY=命令行 --python" )
    if not defined PY if exist "!C!\python_embeded\python.exe" ( set "PY=!C!\python_embeded\python.exe" & set "BY=命令行 --python" )
    if not defined PY if exist "!C!\python.exe" ( set "PY=!C!\python.exe" & set "BY=命令行 --python" )
    if not defined PY if exist "!C!"            ( set "PY=!C!" & set "BY=命令行 --python" )
    if not defined PY echo   命令行给的路径里没有 python：!C!
)

rem ---------- 2) 环境变量 ----------
if not defined PY if defined TAGGERPLUS_PYTHON (
    if exist "%TAGGERPLUS_PYTHON%" ( set "PY=%TAGGERPLUS_PYTHON%" & set "BY=环境变量 TAGGERPLUS_PYTHON" )
)

rem ---------- 3) 插件记录的 ComfyUI 运行时路径 ----------
if not defined PY if exist "%PACK%.taggerplus_runtime.txt" (
    set /p PY=<"%PACK%.taggerplus_runtime.txt"
    if defined PY if not exist "!PY!" set "PY="
    if defined PY set "BY=插件记录的 ComfyUI 运行时路径"
)

rem ---------- 4) 从插件目录逐级向上找 ----------
if not defined PY call :probe "%PACK%"
if not defined PY call :probe "%PACK%..\"
if not defined PY call :probe "%PACK%..\..\"
if not defined PY call :probe "%PACK%..\..\..\"
if not defined PY call :probe "%PACK%..\..\..\..\"
if not defined PY call :probe "%PACK%..\..\..\..\..\"
if not defined PY call :probe "%PACK%..\..\..\..\..\..\"

rem ---------- 4.5) 显式允许使用系统 python（桌面端等场景）----------
if not defined PY if "%USESYS%"=="1" (
    for /f "delims=" %%P in ('where python 2^>nul') do (
        if not defined PY set "PY=%%P" & set "BY=系统 PATH（--use-system-python 显式指定）"
    )
    if defined PY echo   注意：正在使用系统 python，它未必和你的 ComfyUI 环境一致。
)

rem ---------- 5) 还找不到：请用户输入 ----------
if not defined PY (
    echo   没有自动找到整合包自带的 python。
    echo.
    echo   请把下面任意一个粘贴进来（直接回车 = 取消）：
    echo       · python.exe 的完整路径，例如  D:\ComfyUI_windows_portable\python_embeded\python.exe
    echo       · 或者整合包根目录，例如        D:\ComfyUI_windows_portable
    echo.
    echo   如果你用的是 ComfyUI 桌面端，或确实想用系统 python，可以重新运行：
    echo       install_cuda12.bat --use-system-python
    echo.
    set /p "ANS=  路径: "
    if defined ANS (
        set "ANS=!ANS:"=!"
        if exist "!ANS!\python\python.exe"          ( set "PY=!ANS!\python\python.exe" & set "BY=手动输入" )
        if not defined PY if exist "!ANS!\python_embeded\python.exe" ( set "PY=!ANS!\python_embeded\python.exe" & set "BY=手动输入" )
        if not defined PY if exist "!ANS!\python.exe" ( set "PY=!ANS!\python.exe" & set "BY=手动输入" )
        if not defined PY if exist "!ANS!"            ( set "PY=!ANS!" & set "BY=手动输入" )
        if not defined PY echo   这个路径下没找到 python.exe。
    )
)

if not defined PY (
    echo.
    echo   没有指定 python，已取消。随时可以重新运行本脚本。
    echo.
    pause
    exit /b 0
)

echo.
echo   [1/3] 使用 python : %PY%
echo         来源       : %BY%
"%PY%" -c "import sys; print('         版本       :', sys.version.split()[0]); print('         环境前缀   :', sys.prefix)"
if errorlevel 1 (
    echo.
    echo   这个 python 跑不起来，请重新运行本脚本并输入正确的路径。
    pause
    exit /b 0
)
if "%DETECT%"=="1" (
    echo.
    echo   [--detect-only] 只检测不安装，结束。
    exit /b 0
)

echo.
echo   [2/3] 开始下载安装（约 1.3 GB，视网速可能几分钟）...
"%PY%" -m pip install --target "%TARGET%" --no-warn-script-location ^
    nvidia-cublas-cu12 nvidia-cuda-runtime-cu12 nvidia-cufft-cu12 nvidia-curand-cu12
if errorlevel 1 (
    echo.
    echo   安装失败。国内网络可加镜像重试：
    echo       "%PY%" -m pip install --target "%TARGET%" -i https://pypi.tuna.tsinghua.edu.cn/simple ^
    echo           nvidia-cublas-cu12 nvidia-cuda-runtime-cu12 nvidia-cufft-cu12 nvidia-curand-cu12
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
    echo   完成！请重启 ComfyUI，然后看反推节点的 device 输出，
    echo   应该从 "CPU ..." 变成 "GPU · 你的显卡"。
) else (
    echo   有些文件没装上，请把上面的输出发出来。
)
echo.
pause
exit /b 0

:probe
if defined PY exit /b
if exist "%~1python\python.exe"          set "PY=%~1python\python.exe" & set "BY=自动查找" & exit /b
if exist "%~1python_embeded\python.exe"  set "PY=%~1python_embeded\python.exe" & set "BY=自动查找" & exit /b
if exist "%~1python\Scripts\python.exe"  set "PY=%~1python\Scripts\python.exe" & set "BY=自动查找" & exit /b
if exist "%~1Scripts\python.exe"         set "PY=%~1Scripts\python.exe" & set "BY=自动查找" & exit /b
if exist "%~1bin\python.exe"             set "PY=%~1bin\python.exe" & set "BY=自动查找" & exit /b
exit /b
