#!/usr/bin/env python3
"""
SAM3 Interactive Vision Studio
基于 SAM3 的交互式图像分割系统
"""

import os
import sys
import time
import io
import numpy as np
import torch
import gradio as gr
from PIL import Image
import cv2
from pathlib import Path
import json

# 添加当前目录到Python路径，以便导入sam3模块
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

# 导入SAM3相关模块
try:
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
    from sam3.model.data_misc import FindStage
    from sam3.visualization_utils import plot_results
    from sam3.model import box_ops
except ImportError as e:
    print(f"导入SAM3模块失败: {e}")
    print("请确保已正确安装SAM3依赖")
    sys.exit(1)

# 全局变量
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"使用设备: {DEVICE}")

# 初始化模型
def initialize_models():
    """初始化SAM3图像预测器"""
    try:
        # 检查模型文件是否存在
        model_dir = current_dir / "models"
        checkpoint_path = model_dir / "sam3.pt"
        bpe_path = current_dir / "assets" / "bpe_simple_vocab_16e6.txt.gz"
        
        if not checkpoint_path.exists():
            print(f"模型文件不存在: {checkpoint_path}")
            print("请下载SAM3模型文件到目录")
            return None
            
        if not bpe_path.exists():
            print(f"BPE文件不存在: {bpe_path}")
            return None
            
        # 初始化图像模型
        image_model = build_sam3_image_model(
            checkpoint_path=str(checkpoint_path),
            bpe_path=str(bpe_path),
            device=DEVICE
        )
        
        # 创建图像处理器
        image_predictor = Sam3Processor(image_model, device=DEVICE)

        print("模型初始化成功")
        return image_predictor
        
    except Exception as e:
        print(f"模型初始化失败: {e}")
        return None

# 全局预测器实例
image_predictor = initialize_models()

def handle_image_click(img, original_img, evt: gr.SelectData, mode, current_points, current_boxes, click_state):
    """处理图像点击事件，提供实时视觉反馈"""
    if img is None:
        return img, current_points, current_boxes, click_state, "请先上传图像"
    
    # 如果没有原始图像，就使用当前图像作为原始图像
    if original_img is None:
        original_img = img.copy()
        
    # 基于当前显示的图像进行绘制
    vis_img = img.copy()
    
    x, y = evt.index
    x, y = int(x), int(y)
    
    info_msg = ""
    
    if mode == "📍 点提示 (Point)":
        new_point = f"{x},{y}"
        if current_points:
            current_points += f";{new_point}"
        else:
            current_points = new_point
            
        # 在图像上画红色圆点
        cv2.circle(vis_img, (x, y), 6, (255, 0, 0), -1) # 红色实心圆
        cv2.circle(vis_img, (x, y), 6, (255, 255, 255), 1) # 白色描边
        
        info_msg = f"✅ 已添加点: {new_point}"
        return vis_img, current_points, current_boxes, None, info_msg
        
    elif mode == "🔲 框提示 (Box)":
        if click_state is None:
            # 第一次点击 - 画起点（蓝色）
            click_state = [x, y]
            cv2.circle(vis_img, (x, y), 6, (0, 0, 255), -1) # 蓝色实心圆
            cv2.circle(vis_img, (x, y), 6, (255, 255, 255), 1) # 白色描边
            info_msg = f"🔵 已记录起点: {x},{y}，请点击对角点完成框选"
            return vis_img, current_points, current_boxes, click_state, info_msg
        else:
            # 第二次点击 - 画框（绿色）
            x1, y1 = click_state
            x2, y2 = x, y
            
            xmin = min(x1, x2)
            ymin = min(y1, y2)
            xmax = max(x1, x2)
            ymax = max(y1, y2)
            
            # 确保框有大小
            if xmin == xmax: xmax += 1
            if ymin == ymax: ymax += 1
            
            new_box = f"{xmin},{ymin},{xmax},{ymax}"
            if current_boxes:
                current_boxes += f";{new_box}"
            else:
                current_boxes = new_box
            
            # 画绿色矩形框
            cv2.rectangle(vis_img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 3)
                
            info_msg = f"✅ 已添加框: {new_box}"
            return vis_img, current_points, current_boxes, None, info_msg
    
    return vis_img, current_points, current_boxes, click_state, info_msg

def segment_image(
    input_image,
    text_prompt,
    confidence_threshold,
    point_prompt,
    box_prompt,
    original_image=None,
    progress=gr.Progress()
):
    """图像分割功能"""
    # 优先使用原始图像，如果不存在则使用输入图像
    image_to_process = original_image if original_image is not None else input_image
    
    if image_to_process is None:
        return None, "请上传图像"
        
    if not text_prompt and not point_prompt and not box_prompt:
        return None, "请提供至少一种提示（文本、点或框）"
    
    try:
        if image_predictor is None:
            return None, "模型未初始化，请检查模型文件"
            
        start_time = time.time()
        progress(0.1, desc="正在加载图像...")
        
        # 转换图像格式
        if isinstance(image_to_process, np.ndarray):
            image = Image.fromarray(image_to_process)
        else:
            image = image_to_process
            
        # 设置图像
        state = image_predictor.set_image(image)
        progress(0.3, desc="解析提示信息...")
        
        # 处理文本提示
        if text_prompt:
            state = image_predictor.set_text_prompt(text_prompt, state)
            
        # 处理点提示
        if point_prompt:
            points = []
            for point_str in point_prompt.split(';'):
                if point_str:
                    try:
                        x, y = map(float, point_str.split(','))
                        points.append([x, y])
                    except ValueError:
                        continue
                        
            if points:
                width, height = image.size
                normalized_points = []
                for x, y in points:
                    normalized_points.append([x/width, y/height])
                    
                for point in normalized_points:
                    box_size = min(width, height) * 0.05
                    box_width = box_size / width
                    box_height = box_size / height
                    box = [point[0], point[1], box_width, box_height]
                    state = image_predictor.add_geometric_prompt(box, True, state)
        
        # 处理框提示
        if box_prompt:
            boxes = []
            for box_str in box_prompt.split(';'):
                if box_str:
                    try:
                        x1, y1, x2, y2 = map(float, box_str.split(','))
                        boxes.append([x1, y1, x2, y2])
                    except ValueError:
                        continue
                        
            if boxes:
                width, height = image.size
                normalized_boxes = []
                for x1, y1, x2, y2 in boxes:
                    center_x = (x1 + x2) / 2 / width
                    center_y = (y1 + y2) / 2 / height
                    box_width = (x2 - x1) / width
                    box_height = (y2 - y1) / height
                    normalized_boxes.append([center_x, center_y, box_width, box_height])
                    
                for box in normalized_boxes:
                    state = image_predictor.add_geometric_prompt(box, True, state)
        
        # 设置置信度阈值
        state = image_predictor.set_confidence_threshold(confidence_threshold, state)
        
        progress(0.7, desc="模型推理中...")
        
        # 获取结果
        if "boxes" in state and len(state["boxes"]) > 0:
            # 可视化结果
            import matplotlib.pyplot as plt
            
            # 使用官方的 plot_results 接口进行绘制
            # plot_results 内部会创建 figure 并绘制 masks, boxes, scores
            # 注意：它会打印找到的对象数量，但这不影响 Gradio 显示
            plot_results(image, state)
            
            # 获取当前的 figure (由 plot_results 创建) 并转换为 PIL 图像
            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
            buf.seek(0)
            result_image = Image.open(buf)
            plt.close() # 关闭 figure 释放内存
            
            processing_time = time.time() - start_time
            info = f"✨ 处理完成 | 耗时: {processing_time:.2f}s | 检测到 {len(state['boxes'])} 个目标"
            
            return result_image, info
        else:
            return image, "⚠️ 未检测到任何对象，请尝试调整提示或降低置信度阈值"
            
    except Exception as e:
        return None, f"❌ 处理失败: {str(e)}"

def create_demo():
    """创建美化后的Gradio演示界面"""

    # 自定义CSS
    custom_css = """
    .container { max-width: 1200px; margin: auto; padding-top: 20px; }
    h1 { text-align: center; font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif; color: #2d3748; margin-bottom: 10px; }
    .description { text-align: center; font-size: 1.1em; color: #4a5568; margin-bottom: 30px; }
    .gr-button-primary { background: linear-gradient(90deg, #4b6cb7 0%, #182848 100%); border: none; }
    .gr-box { border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
    #interaction-info { font-weight: bold; color: #2b6cb0; text-align: center; background-color: #ebf8ff; padding: 10px; border-radius: 5px; border: 1px solid #bee3f8; }

    /* 让Radio按钮组水平撑满 */
    .mode-radio .wrap { display: flex; width: 100%; gap: 10px; }
    .mode-radio .wrap label { flex: 1; justify-content: center; text-align: center; }
    """

    theme = gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="slate",
        font=[gr.themes.GoogleFont('Inter'), 'ui-sans-serif', 'system-ui', 'sans-serif']
    )

    with gr.Blocks(theme=theme, css=custom_css, title="SAM3 交互式视觉工作台") as demo:

        with gr.Column(elem_classes="container"):
            gr.Markdown("# 👁️ SAM3 交互式视觉工作台")
            gr.Markdown("专注于 SAM3 的交互式图像分割体验", elem_classes="description")

            with gr.Row():
                # 左侧控制栏
                with gr.Column(scale=1):
                    image_input = gr.Image(type="numpy", label="原始图像 (点击进行交互)", elem_id="input_image")

                    # 存储原始图像状态
                    original_image_state = gr.State(None)
                    click_state = gr.State(None)

                    with gr.Group():
                        gr.Markdown("### 🎮 交互模式")
                        # 第一行：模式选择
                        interaction_mode = gr.Radio(
                            choices=["📍 点提示 (Point)", "🔲 框提示 (Box)"],
                            value="📍 点提示 (Point)",
                            label="选择模式",
                            show_label=False,
                            elem_classes="mode-radio",
                        )
                        # 第二行：清空按钮（全宽）
                        with gr.Row():
                            clear_prompts_btn = gr.Button("🗑️ 清空提示 (Clear Prompts)", size="sm", variant="secondary")

                        interaction_info = gr.Markdown("👆 点击图像开始添加提示...", elem_id="interaction-info")

                    with gr.Accordion("📝 高级提示选项", open=True):
                        text_prompt = gr.Textbox(
                            label="文本提示 (Text Prompt)",
                            placeholder="输入物体描述，例如：'a red car' 或 '一只猫'",
                            lines=1
                        )

                        with gr.Row():
                            gr.Markdown("示例快速填充：")
                            example_text_btn = gr.Button("🐱 猫", size="sm")
                            example_point_btn = gr.Button("📍 示例点", size="sm")

                        with gr.Row(visible=False): # 隐藏原始坐标输入框，保持后端逻辑但减少界面干扰
                            point_prompt = gr.Textbox(label="点坐标")
                            box_prompt = gr.Textbox(label="框坐标")

                    confidence_threshold = gr.Slider(
                        minimum=0.0, maximum=1.0, value=0.4, step=0.05,
                        label="🎯 置信度阈值 (Confidence)"
                    )

                    segment_button = gr.Button("🚀 开始分割 (Segment)", variant="primary", size="lg")

                # 右侧结果栏
                with gr.Column(scale=1):
                    image_output = gr.Image(type="numpy", label="✨ 分割结果")
                    image_info = gr.Textbox(label="📊 分析报告", interactive=False, lines=2)

            # 事件绑定

            # 1. 上传图片时保存原图
            def store_original_image(img): return img, None # Reset click state
            image_input.upload(fn=store_original_image, inputs=[image_input], outputs=[original_image_state, click_state])

            # 2. 点击图片处理
            image_input.select(
                fn=handle_image_click,
                inputs=[image_input, original_image_state, interaction_mode, point_prompt, box_prompt, click_state],
                outputs=[image_input, point_prompt, box_prompt, click_state, interaction_info]
            )

            # 3. 清空提示
            def clear_prompts(orig_img):
                if orig_img is None: return None, "", "", None, "请先上传图像"
                return orig_img, "", "", None, "♻️ 提示已清空，图像已重置"
            clear_prompts_btn.click(
                fn=clear_prompts,
                inputs=[original_image_state],
                outputs=[image_input, point_prompt, box_prompt, click_state, interaction_info]
            )

            # 4. 分割按钮
            segment_button.click(
                fn=segment_image,
                inputs=[image_input, text_prompt, confidence_threshold, point_prompt, box_prompt, original_image_state],
                outputs=[image_output, image_info]
            )

            # 5. 示例按钮
            example_text_btn.click(fn=lambda: "a cat", outputs=[text_prompt])
            example_point_btn.click(fn=lambda: "100,100", outputs=[point_prompt])

        # 页脚
        gr.Markdown("""
        ---
        <div style="text-align: center; color: #718096; font-size: 0.9em;">
            Powered by <strong>SAM3</strong> | 2025 SAM3 Interactive Studio
        </div>
        """)

    return demo

def main():
    """主函数"""
    # 检查模型文件
    model_dir = current_dir / "models"
    if not model_dir.exists():
        print(f"创建模型目录: {model_dir}")
        model_dir.mkdir(exist_ok=True)
        
    checkpoint_path = model_dir / "sam3.pt"
    bpe_path = current_dir / "assets" / "bpe_simple_vocab_16e6.txt.gz"
    
    if not checkpoint_path.exists() or not bpe_path.exists():
        print("⚠️ 模型文件缺失")
        print(f"请确保以下文件存在:\n1. {checkpoint_path}\n2. {bpe_path}")
        
        response = input("是否尝试自动下载模型文件？(y/n): ").lower().strip()
        if response == 'y':
            try:
                import download_models
                download_models.main()
            except Exception as e:
                print(f"自动下载失败: {e}")
                return
        else:
            return
    
    print("🚀 正在启动 SAM3 交互式视觉工作台...")
    demo = create_demo()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7890,
        share=False,
        debug=True,
        allowed_paths=[str(current_dir)]
    )

if __name__ == "__main__":
    main()
