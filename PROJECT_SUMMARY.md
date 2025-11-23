# SAM3 Gradio 演示系统

## 项目概述

本项目提供了一个基于 Gradio 的 SAM3 (Segment Anything Model 3) 演示系统，支持图像分割和视频目标跟踪功能。系统使用相对路径，可在不同环境下运行。

## 项目结构

```
sam3/
├── sam3_gradio_demo.py          # 主程序文件
├── download_models.py           # 模型下载脚本
├── requirements.txt             # 依赖列表
├── run_demo.sh                  # Linux/Mac 启动脚本
├── run_demo.bat                 # Windows 启动脚本
├── Gradio_Demo_README.md        # 详细使用说明
├── models/                      # 模型文件目录
│   └── README.md               # 模型下载说明
└── sam3/                        # SAM3 核心代码
    ├── model_builder.py         # 模型构建器
    └── model/                   # 模型实现
```

## 主要功能

1. **图像分割**
   - 支持文本提示分割
   - 支持点提示分割
   - 支持框提示分割
   - 实时预览和结果可视化

2. **视频目标跟踪**
   - 基于文本提示的目标跟踪
   - 自动传播和更新
   - 结果可视化

## 技术特点

1. **跨环境兼容性**
   - 使用相对路径
   - 自动模型下载
   - 多平台启动脚本

2. **用户友好性**
   - 直观的 Gradio 界面
   - 详细的提示和说明
   - 错误处理和反馈

3. **模块化设计**
   - 清晰的代码结构
   - 分离的功能模块
   - 易于扩展和维护

## 使用方法

1. **环境准备**
   ```bash
   pip install -r requirements.txt
   ```

2. **模型下载**
   ```bash
   python download_models.py
   ```

3. **启动系统**
   - Linux/Mac: `bash run_demo.sh`
   - Windows: `run_demo.bat`
   - 直接运行: `python sam3_gradio_demo.py`

4. **访问界面**
   - 打开浏览器访问 http://localhost:7860

## 文件说明

- `sam3_gradio_demo.py`: 主程序，包含 Gradio 界面和核心功能
- `download_models.py`: 自动下载所需模型文件
- `requirements.txt`: 项目依赖列表
- `run_demo.sh/bat`: 便捷启动脚本
- `Gradio_Demo_README.md`: 详细使用说明和提示格式

## 注意事项

1. 首次运行需要下载模型文件，确保网络连接正常
2. 模型文件较大（约2.5GB），确保有足够的存储空间
3. 建议使用 GPU 运行以获得更好的性能

## 故障排除

1. **模型文件缺失**: 运行 `python download_models.py` 下载模型
2. **导入错误**: 检查依赖是否正确安装
3. **GPU 不可用**: 系统会自动切换到 CPU 模式，但速度较慢

## 扩展功能

系统设计为可扩展的，可以轻松添加以下功能：
- 更多提示类型支持
- 批量处理功能
- 自定义模型加载
- 结果导出功能