#!/usr/bin/env python3
"""
SAM3模型下载脚本
用于下载SAM3演示系统所需的模型文件
"""

import os
import sys
from pathlib import Path
import requests
from tqdm import tqdm

def download_file(url: str, dest_path: str, description: str = "Downloading"):
    """下载文件并显示进度条"""
    dest_path = Path(dest_path)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    
    if dest_path.exists():
        print(f"文件已存在: {dest_path}")
        return True
    
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024  # 1 KB
        
        with open(dest_path, 'wb') as f, tqdm(
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
            desc=description
        ) as bar:
            for data in response.iter_content(block_size):
                bar.update(len(data))
                f.write(data)
        
        print(f"下载完成: {dest_path}")
        return True
    except Exception as e:
        print(f"下载失败: {e}")
        if dest_path.exists():
            dest_path.unlink()  # 删除部分下载的文件
        return False

def main():
    """主函数"""
    print("SAM3模型文件下载脚本")
    print("=" * 50)
    
    # 模型文件信息
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    # 注意：这些URL是示例，实际使用时需要替换为真实的下载链接
    assets_dir = Path("assets")
    assets_dir.mkdir(exist_ok=True)
    
    model_files = [
        {
            "name": "SAM3模型文件",
            "url": "https://huggingface.co/facebook/sam3/resolve/main/sam3.pt",
            "dest": models_dir / "sam3.pt"
        },
        {
            "name": "BPE词汇表文件",
            "url": "https://huggingface.co/facebook/sam3/resolve/main/bpe_simple_vocab_16e6.txt.gz",
            "dest": assets_dir / "bpe_simple_vocab_16e6.txt.gz"
        }
    ]
    
    # 检查是否已有模型文件
    missing_files = []
    for file_info in model_files:
        if not file_info["dest"].exists():
            missing_files.append(file_info)
    
    if not missing_files:
        print("所有模型文件已存在，无需下载。")
        return
    
    print(f"需要下载 {len(missing_files)} 个文件:")
    for file_info in missing_files:
        print(f"- {file_info['name']}: {file_info['dest']}")
    
    print("\n注意：您可能需要先在Hugging Face上申请访问权限。")
    print("请访问 https://huggingface.co/facebook/sam3 申请访问权限。")
    
    # 询问用户是否继续
    response = input("\n是否继续下载？(y/n): ").lower().strip()
    if response != 'y':
        print("下载已取消。")
        return
    
    # 下载文件
    success_count = 0
    for file_info in missing_files:
        print(f"\n下载 {file_info['name']}...")
        if download_file(file_info["url"], file_info["dest"], file_info['name']):
            success_count += 1
    
    print(f"\n下载完成: {success_count}/{len(missing_files)} 个文件成功下载")
    
    if success_count < len(missing_files):
        print("部分文件下载失败，请检查网络连接或手动下载。")
        print("您也可以从以下链接手动下载:")
        for file_info in missing_files:
            print(f"- {file_info['name']}: {file_info['url']}")
            print(f"  保存到: {file_info['dest']}")
    else:
        print("所有模型文件下载完成！现在可以运行SAM3 Gradio演示系统了。")

if __name__ == "__main__":
    main()