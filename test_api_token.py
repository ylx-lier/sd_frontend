#!/usr/bin/env python3
"""
API Token 测试脚本
用于调试和验证 Hugging Face API Token 的有效性
"""

import requests
import json
import sys

def test_api_token(token):
    """测试API Token的详细信息"""
    print(f"🔍 测试API Token: {token[:10]}...{token[-5:] if len(token) > 15 else token}")
    print("=" * 60)
    
    headers = {"Authorization": f"Bearer {token}"}
    
    # 测试1: whoami API
    print("\n📋 测试1: Whoami API")
    try:
        response = requests.get(
            "https://huggingface.co/api/whoami-v2",
            headers=headers,
            timeout=10
        )
        print(f"状态码: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ 用户信息: {json.dumps(data, indent=2, ensure_ascii=False)}")
        else:
            print(f"❌ 响应内容: {response.text}")
    except Exception as e:
        print(f"❌ 请求失败: {e}")
    
    # 测试2: 模型API调用
    print("\n🤖 测试2: 模型 API 调用")
    try:
        test_endpoint = "https://api-inference.huggingface.co/models/runwayml/stable-diffusion-v1-5"
        test_payload = {
            "inputs": "a beautiful landscape",
            "parameters": {
                "num_inference_steps": 1,
                "guidance_scale": 1.0
            }
        }
        
        response = requests.post(
            test_endpoint,
            headers=headers,
            json=test_payload,
            timeout=15
        )
        
        print(f"状态码: {response.status_code}")
        if response.status_code == 200:
            print(f"✅ 模型API调用成功，返回图像数据 ({len(response.content)} bytes)")
        elif response.status_code == 401:
            print(f"❌ 认证失败: {response.text}")
        elif response.status_code == 403:
            print(f"❌ 权限不足: {response.text}")
        elif response.status_code == 503:
            print(f"⚠️ 模型加载中: {response.text}")
        else:
            print(f"⚠️ 其他状态: {response.text}")
            
    except Exception as e:
        print(f"❌ 请求失败: {e}")
    
    # 测试3: Token信息
    print("\n🔑 测试3: Token 格式检查")
    if token.startswith('hf_'):
        print("✅ Token格式正确 (以 hf_ 开头)")
    else:
        print("❌ Token格式可能有误 (应该以 hf_ 开头)")
    
    if len(token) > 30:
        print("✅ Token长度看起来合理")
    else:
        print("❌ Token长度可能过短")

def main():
    print("🔧 Hugging Face API Token 测试工具")
    print("=" * 60)
    
    if len(sys.argv) > 1:
        token = sys.argv[1]
    else:
        token = input("请输入您的 Hugging Face API Token: ").strip()
    
    if not token:
        print("❌ 未提供Token")
        return
    
    test_api_token(token)
    
    print("\n" + "=" * 60)
    print("💡 如果所有测试都失败，请检查:")
    print("1. Token是否正确复制 (注意前后空格)")
    print("2. Token是否有正确的权限 (至少需要 Read 权限)")
    print("3. 网络连接是否正常")
    print("4. 是否需要使用代理")
    print("\n🔗 获取Token: https://huggingface.co/settings/tokens")

if __name__ == "__main__":
    main()
