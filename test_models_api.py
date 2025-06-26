#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
API模型支持测试脚本
测试新的模型API支持检测功能
"""

import requests
import os
import sys

# 从app.py导入配置
sys.path.append(os.path.dirname(__file__))
from app import API_ENDPOINTS, API_SUPPORTED_MODELS, get_available_models

def test_model_endpoints():
    """测试模型端点是否存在"""
    print("🔍 测试 Hugging Face API 模型端点...")
    print("=" * 60)
    
    for model_id, endpoint in API_ENDPOINTS.items():
        model_name = API_SUPPORTED_MODELS.get(model_id, model_id)
        print(f"\n📦 测试模型: {model_name}")
        print(f"🎯 模型ID: {model_id}")
        print(f"🔗 端点: {endpoint}")
        
        try:
            # 使用HEAD请求检查端点是否存在
            response = requests.head(endpoint, timeout=10)
            
            if response.status_code == 200:
                print(f"✅ 端点存在且可访问")
            elif response.status_code == 503:
                print(f"⚠️ 模型正在加载中 (503)")
            elif response.status_code == 404:
                print(f"❌ 端点不存在 (404)")
            elif response.status_code == 401:
                print(f"🔐 需要认证 (401) - 端点存在")
            else:
                print(f"⚠️ 状态码: {response.status_code}")
        
        except requests.exceptions.Timeout:
            print(f"⏰ 连接超时")
        except requests.exceptions.ConnectionError:
            print(f"❌ 连接失败")
        except Exception as e:
            print(f"❌ 错误: {str(e)}")
    
    print("\n" + "=" * 60)

def test_model_selection():
    """测试模型选择功能"""
    print("🎯 测试模型选择功能...")
    print("=" * 60)
    
    api_models = get_available_models("api")
    local_models = get_available_models("local")
    
    print(f"📊 API模式可用模型: {len(api_models)} 个")
    print("🔸 API支持的模型列表:")
    for i, (model_id, model_name) in enumerate(api_models.items(), 1):
        print(f"   {i:2d}. {model_name}")
    
    print(f"\n📊 本地模式可用模型: {len(local_models)} 个")
    print(f"🔸 额外的本地模型: {len(local_models) - len(api_models)} 个")
    
    print("\n" + "=" * 60)

def recommend_best_models():
    """推荐最佳模型"""
    print("🏆 推荐的最佳API模型...")
    print("=" * 60)
    
    # 按优先级排序的推荐模型
    recommended = [
        ("black-forest-labs/FLUX.1-dev", "🥇 最强大的图像生成模型"),
        ("black-forest-labs/FLUX.1-schnell", "⚡ 快速生成，高质量"),  
        ("stabilityai/stable-diffusion-xl-base-1.0", "🎨 SDXL经典选择"),
        ("stabilityai/stable-diffusion-3.5-large", "🚀 最新SD3.5"),
        ("Kwai-Kolors/Kolors", "📸 逼真图像生成")
    ]
    
    print("基于官方文档和社区反馈的推荐:")
    for i, (model_id, description) in enumerate(recommended, 1):
        if model_id in API_ENDPOINTS:
            model_name = API_SUPPORTED_MODELS.get(model_id, model_id)
            print(f"{i}. {description}")
            print(f"   模型: {model_name}")
            print(f"   ID: {model_id}")
            print()
    
    print("💡 建议:")
    print("• 首次使用推荐 FLUX.1-dev - 质量最高")
    print("• 需要快速生成推荐 FLUX.1-schnell")
    print("• 需要稳定性推荐 SDXL Base 1.0")
    print("• 对于人像生成推荐 Kolors")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    print("🎨 Stable Diffusion API 模型支持测试")
    print("=" * 60)
    print("该脚本将测试新添加的API模型支持功能")
    print()
    
    # 测试模型选择功能
    test_model_selection()
    
    # 推荐最佳模型
    recommend_best_models()
    
    # 询问是否测试API端点
    try:
        test_endpoints = input("是否测试API端点连接? (y/N): ").strip().lower()
        if test_endpoints in ['y', 'yes']:
            test_model_endpoints()
        else:
            print("⏭️ 跳过端点测试")
    except KeyboardInterrupt:
        print("\n\n👋 测试结束")
    
    print("\n✅ 测试完成！")
    print("💡 现在可以在Gradio界面中选择推荐的API模型了")
