import torch
from PIL import Image
import matplotlib.pyplot as plt
import os
from datetime import datetime
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 加载 SAM3 模型
model = build_sam3_image_model()
processor = Sam3Processor(model)

# 加载图像
image = Image.open("/home/jikangyi/sam3/assets/images/image.png")
inference_state = processor.set_image(image)

# 文本提示
text_prompt = "Steel bar inspection"
output = processor.set_text_prompt(state=inference_state, prompt=text_prompt)

# 获取分割结果
masks = output["masks"]
boxes = output["boxes"]
scores = output["scores"]

# 辅助函数：将CUDA张量转换为CPU NumPy数组
def tensor_to_numpy(tensor):
    if isinstance(tensor, torch.Tensor):
        # 如果是CUDA张量，先移到CPU再转换为NumPy
        return tensor.cpu().numpy()
    return tensor

# 简化的调试信息
print("分割结果摘要:")
print(f"掩码数量: {len(masks) if masks is not None else 'None'}")
print(f"边界框数量: {len(boxes) if boxes is not None else 'None'}")
if masks is not None and len(masks) > 0:
    # 修复布尔类型无法直接求平均的问题
    mask_tensor = masks[0]
    if len(mask_tensor.shape) > 2:
        mask_tensor = mask_tensor[0]
    # 确保在CPU上进行计算
    if isinstance(mask_tensor, torch.Tensor):
        mask_tensor = mask_tensor.cpu()
    # 转换为浮点型后再计算比例
    nonzero_ratio = mask_tensor.float().mean().item()
    print(f"第一个掩码的有效像素比例: {nonzero_ratio:.4f}")

# 简化的结果显示和保存函数
def save_results(image, masks, boxes, scores, text_prompt, save_path):
    # 处理没有分割结果的情况
    if masks is None or len(masks) == 0:
        plt.figure(figsize=(10, 8))
        plt.imshow(image)
        plt.title("原图 - 未找到匹配的分割结果")
        plt.axis('off')
        plt.suptitle(f'文本指令: "{text_prompt}"')
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"未找到分割结果，仅保存原图至: {save_path}")
        plt.close()
        return
    
    # 只显示和保存得分最高的分割结果
    best_idx = 0
    if hasattr(scores, 'argmax'):
        best_idx = scores.argmax().item()
    elif hasattr(scores, '__getitem__'):
        best_idx = scores.index(max(scores))
    
    best_mask = masks[best_idx]
    best_box = boxes[best_idx]
    best_score = scores[best_idx]
    
    # 将所有张量转换为NumPy数组以用于matplotlib
    best_mask = tensor_to_numpy(best_mask)
    best_box = tensor_to_numpy(best_box)
    best_score = float(best_score) if isinstance(best_score, torch.Tensor) else best_score
    
    plt.figure(figsize=(10, 8))
    plt.imshow(image)
    
    # 显示掩码
    if best_mask.ndim > 2:
        best_mask = best_mask[0]
    plt.imshow(best_mask, alpha=0.6, cmap='jet')
    
    # 绘制边界框
    x1, y1, x2, y2 = best_box
    rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                        fill=False, color='yellow', linewidth=2)
    plt.gca().add_patch(rect)
    
    plt.title(f"分割结果 (置信度: {best_score:.3f})")
    plt.suptitle(f'文本指令: "{text_prompt}"')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    print(f"最佳分割结果已保存至: {save_path}")
    
    # 单独保存掩码
    mask_only_path = save_path.replace('.png', '_mask.png')
    plt.figure(figsize=(10, 8))
    plt.imshow(best_mask, cmap='jet')
    plt.title(f"分割掩码")
    plt.axis('off')
    plt.savefig(mask_only_path, bbox_inches='tight', dpi=300)
    print(f"分割掩码已保存至: {mask_only_path}")
    
    plt.close('all')

# 创建输出目录并生成保存路径
output_dir = "/home/jikangyi/sam3/output"
os.makedirs(output_dir, exist_ok=True)

safe_prompt = text_prompt.replace(" ", "_")[:10]
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
save_path = os.path.join(output_dir, f"seg_{safe_prompt}_{timestamp}.png")

# 保存结果
save_results(image, masks, boxes, scores, text_prompt, save_path)