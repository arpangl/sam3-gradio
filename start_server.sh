#!/bin/bash

# SAM3 Gradio 服务器部署脚本

echo "SAM3 Gradio 服务器部署启动中..."

# 检查Python环境
if ! command -v python &> /dev/null; then
    echo "错误: 未找到Python，请确保已安装Python 3.8+"
    exit 1
fi

# 检查模型文件
if [ ! -f "models/sam3.pt" ]; then
    echo "警告: 未找到模型文件 models/sam3.pt"
    echo "请从SAM3官方仓库下载模型文件并放入models目录"
fi

if [ ! -f "assets/bpe_simple_vocab_16e6.txt.gz" ]; then
    echo "警告: 未找到BPE词汇表文件 assets/bpe_simple_vocab_16e6.txt.gz"
    echo "请从SAM3官方仓库下载BPE文件并放入assets目录"
fi

# 创建models目录（如果不存在）
mkdir -p models

# 启动演示系统
echo "启动Gradio演示系统..."
echo "应用将在 http://0.0.0.0:7860 上运行"
echo "您也可以通过Gradio提供的公共链接访问"
echo "按 Ctrl+C 停止服务器"
python sam3_gradio_demo.py