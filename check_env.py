#!/usr/bin/env python3
"""
测试当前环境的包版本，验证requirements.txt的准确性
"""

import sys
import importlib
import subprocess

def get_package_version(package_name):
    """获取包的版本信息"""
    try:
        package = importlib.import_module(package_name)
        return getattr(package, '__version__', 'Unknown')
    except ImportError:
        return 'Not installed'

def get_pip_version(package_name):
    """通过pip获取包版本"""
    try:
        result = subprocess.run([sys.executable, '-m', 'pip', 'show', package_name], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            for line in result.stdout.split('\n'):
                if line.startswith('Version:'):
                    return line.split(':')[1].strip()
        return 'Not found'
    except:
        return 'Error'

def main():
    print("🔍 检查当前Python环境中的包版本")
    print("=" * 50)
    
    # 定义要检查的关键包
    packages_to_check = [
        ('torch', 'torch'),
        ('torchvision', 'torchvision'), 
        ('diffusers', 'diffusers'),
        ('transformers', 'transformers'),
        ('gradio', 'gradio'),
        ('PIL', 'Pillow'),
        ('cv2', 'opencv-python'),
        ('numpy', 'numpy'),
        ('requests', 'requests'),
        ('safetensors', 'safetensors'),
        ('accelerate', 'accelerate'),
        ('scipy', 'scipy')
    ]
    
    print(f"Python版本: {sys.version}")
    print("-" * 50)
    
    for import_name, pip_name in packages_to_check:
        import_version = get_package_version(import_name)
        pip_version = get_pip_version(pip_name)
        
        print(f"{pip_name:20} | Import: {import_version:12} | Pip: {pip_version}")
    
    print("-" * 50)
    print("✅ 检查完成！")
    
    # 检查CUDA可用性
    try:
        import torch
        print(f"\n🚀 CUDA信息:")
        print(f"   CUDA可用: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"   CUDA版本: {torch.version.cuda}")
            print(f"   GPU数量: {torch.cuda.device_count()}")
            print(f"   当前GPU: {torch.cuda.get_device_name(0)}")
    except:
        print("\n❌ 无法获取CUDA信息")

if __name__ == "__main__":
    main()
