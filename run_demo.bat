@echo off
REM SAM3 Gradio演示系统启动脚本 (Windows)

echo SAM3 Gradio演示系统启动中...

REM 检查Python环境
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo 错误: 未找到Python，请确保已安装Python 3.8+
    pause
    exit /b 1
)

REM 检查模型文件
if not exist "models\sam3.pt" (
    echo 警告: 未找到模型文件 models\sam3.pt
    echo 请从SAM3官方仓库下载模型文件并放入models目录
)

if not exist "assets\bpe_simple_vocab_16e6.txt.gz" (
    echo 警告: 未找到BPE词汇表文件 assets\bpe_simple_vocab_16e6.txt.gz
    echo 请从SAM3官方仓库下载BPE文件并放入assets目录
)

REM 创建models目录（如果不存在）
if not exist "models" mkdir models

REM 启动演示系统
echo 启动Gradio演示系统...
python sam3_gradio_demo.py

pause