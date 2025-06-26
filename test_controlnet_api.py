#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试 ControlNet API 端点可用性
"""

import requests

# 当前使用的 ControlNet 端点
CONTROLNET_ENDPOINTS = {
    "canny": "https://api-inference.huggingface.co/models/lllyasviel/sd-controlnet-canny",
    "scribble": "https://api-inference.huggingface.co/models/lllyasviel/sd-controlnet-scribble", 
    "depth": "https://api-inference.huggingface.co/models/lllyasviel/sd-controlnet-depth"
}

# 新的可能端点
NEW_CONTROLNET_ENDPOINTS = {
    "canny": "https://api-inference.huggingface.co/models/lllyasviel/control_v11p_sd15_canny",
    "scribble": "https://api-inference.huggingface.co/models/lllyasviel/control_v11p_sd15_scribble",
    "depth": "https://api-inference.huggingface.co/models/lllyasviel/control_v11p_sd15_depth",
    "openpose": "https://api-inference.huggingface.co/models/lllyasviel/control_v11p_sd15_openpose",
    "seg": "https://api-inference.huggingface.co/models/lllyasviel/control_v11p_sd15_seg"
}

def test_endpoint(name, url):
    """测试单个端点"""
    try:
        response = requests.head(url, timeout=10)
        status = response.status_code
        
        if status == 200:
            return f"✅ {name}: 可用 (200)"
        elif status == 401:
            return f"🔐 {name}: 需要认证，但端点存在 (401)"
        elif status == 503:
            return f"⚠️ {name}: 模型加载中 (503)"
        elif status == 404:
            return f"❌ {name}: 端点不存在 (404)"
        else:
            return f"⚠️ {name}: 状态码 {status}"
    except requests.exceptions.Timeout:
        return f"⏰ {name}: 连接超时"
    except requests.exceptions.ConnectionError:
        return f"❌ {name}: 连接失败"
    except Exception as e:
        return f"❌ {name}: 错误 - {str(e)}"

def main():
    print("🔍 测试 ControlNet API 端点可用性")
    print("=" * 60)
    
    print("\n📋 测试当前端点:")
    for name, url in CONTROLNET_ENDPOINTS.items():
        result = test_endpoint(name, url)
        print(f"  {result}")
        print(f"     URL: {url}")
    
    print("\n📋 测试新版本端点:")
    for name, url in NEW_CONTROLNET_ENDPOINTS.items():
        result = test_endpoint(name, url)
        print(f"  {result}")
        print(f"     URL: {url}")
    
    print("\n" + "=" * 60)
    print("💡 建议:")
    print("• 如果当前端点不可用，应该更新到新版本端点")
    print("• ControlNet v1.1 系列是更新的版本，应该优先使用")
    print("• 404错误表示端点已经不存在或已迁移")

if __name__ == "__main__":
    main()
