#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
img2img API测试工具
测试Hugging Face API的img2img功能支持情况
"""

import os
import sys
import requests
import base64
import io
from PIL import Image, ImageDraw
import json

def create_test_image():
    """创建一个简单的测试图像"""
    # 创建一个512x512的测试图像
    image = Image.new('RGB', (512, 512), color='lightblue')
    draw = ImageDraw.Draw(image)
    
    # 画一个简单的图形
    draw.rectangle([100, 100, 400, 400], fill='white', outline='black', width=3)
    draw.text((200, 250), "TEST", fill='black')
    
    return image

def image_to_base64(image):
    """将PIL图像转换为base64字符串"""
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

def test_img2img_api(api_token, test_image):
    """测试不同的img2img API格式"""
    
    if not api_token or not api_token.strip():
        print("❌ 错误: 需要有效的 Hugging Face API Token")
        print("请设置环境变量: set HF_API_TOKEN=your_token_here")
        return
    
    # 将测试图像转换为base64
    image_b64 = image_to_base64(test_image)
    
    # 测试的模型和端点
    models_to_test = {
        "runwayml/stable-diffusion-v1-5": "https://api-inference.huggingface.co/models/runwayml/stable-diffusion-v1-5",
        "stabilityai/stable-diffusion-2-1": "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-2-1",
        "stabilityai/stable-diffusion-xl-base-1.0": "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-xl-base-1.0",
    }
    
    # 测试不同的payload格式
    test_prompt = "a beautiful landscape, detailed, high quality"
    
    payloads_to_test = [
        {
            "name": "Stability AI 风格",
            "payload": {
                "inputs": test_prompt,
                "parameters": {
                    "init_image": image_b64,
                    "negative_prompt": "blurry, low quality",
                    "strength": 0.7,
                    "num_inference_steps": 20,
                    "guidance_scale": 7.5
                }
            }
        },
        {
            "name": "Diffusers 标准格式",
            "payload": {
                "inputs": {
                    "prompt": test_prompt,
                    "image": image_b64,
                    "negative_prompt": "blurry, low quality",
                    "strength": 0.7
                }
            }
        },
        {
            "name": "简化格式",
            "payload": {
                "inputs": test_prompt,
                "image": image_b64,
                "strength": 0.7
            }
        },
        {
            "name": "兼容性格式",
            "payload": {
                "prompt": test_prompt,
                "init_image": image_b64,
                "strength": 0.7,
                "negative_prompt": "blurry, low quality"
            }
        }
    ]
    
    headers = {
        "Authorization": f"Bearer {api_token}",
        "Content-Type": "application/json"
    }
    
    print("🧪 开始测试img2img API...")
    print("=" * 80)
    
    results = {}
    
    for model_name, endpoint in models_to_test.items():
        print(f"\n🎯 测试模型: {model_name}")
        print("-" * 60)
        
        model_results = {}
        
        for payload_info in payloads_to_test:
            payload_name = payload_info["name"]
            payload = payload_info["payload"]
            
            print(f"🔄 尝试格式: {payload_name}")
            
            try:
                response = requests.post(
                    endpoint,
                    headers=headers,
                    json=payload,
                    timeout=60
                )
                
                print(f"   状态码: {response.status_code}")
                
                if response.status_code == 200:
                    try:
                        # 尝试解析图像
                        image_data = response.content
                        if len(image_data) > 0:
                            test_img = Image.open(io.BytesIO(image_data))
                            print(f"   ✅ 成功! 图像尺寸: {test_img.size}")
                            model_results[payload_name] = "SUCCESS"
                            
                            # 保存测试结果图像
                            filename = f"img2img_test_{model_name.replace('/', '_')}_{payload_name.replace(' ', '_')}.png"
                            test_img.save(filename)
                            print(f"   💾 已保存: {filename}")
                        else:
                            print(f"   ❌ 响应为空")
                            model_results[payload_name] = "EMPTY_RESPONSE"
                    except Exception as e:
                        print(f"   ❌ 图像解析失败: {e}")
                        model_results[payload_name] = f"IMAGE_ERROR: {e}"
                        
                elif response.status_code == 503:
                    print(f"   ⏳ 模型加载中...")
                    model_results[payload_name] = "MODEL_LOADING"
                    
                elif response.status_code == 400:
                    try:
                        error_info = response.json()
                        print(f"   ❌ 请求错误: {error_info}")
                        model_results[payload_name] = f"BAD_REQUEST: {error_info}"
                    except:
                        print(f"   ❌ 请求错误: {response.text[:100]}")
                        model_results[payload_name] = f"BAD_REQUEST: {response.text[:100]}"
                        
                else:
                    print(f"   ❌ 其他错误: {response.status_code}")
                    try:
                        error_text = response.text[:200]
                        print(f"   错误详情: {error_text}")
                        model_results[payload_name] = f"HTTP_{response.status_code}: {error_text}"
                    except:
                        model_results[payload_name] = f"HTTP_{response.status_code}"
                        
            except requests.exceptions.Timeout:
                print(f"   ⏰ 请求超时")
                model_results[payload_name] = "TIMEOUT"
                
            except Exception as e:
                print(f"   ❌ 请求异常: {e}")
                model_results[payload_name] = f"EXCEPTION: {e}"
        
        results[model_name] = model_results
    
    # 输出测试总结
    print("\n" + "=" * 80)
    print("📊 测试结果总结")
    print("=" * 80)
    
    success_count = 0
    total_count = 0
    
    for model_name, model_results in results.items():
        print(f"\n🎯 {model_name}:")
        for payload_name, result in model_results.items():
            total_count += 1
            status_icon = "✅" if result == "SUCCESS" else "❌"
            print(f"   {status_icon} {payload_name}: {result}")
            if result == "SUCCESS":
                success_count += 1
    
    print(f"\n📈 总体成功率: {success_count}/{total_count} ({success_count/total_count*100:.1f}%)")
    
    if success_count == 0:
        print("\n💡 建议:")
        print("• img2img功能在Hugging Face公共API中支持有限")
        print("• 建议使用本地模式进行img2img生成")
        print("• 或者使用ControlNet模式实现类似效果")
        print("• API模式更适合text-to-image生成")
    
    return results

def main():
    """主函数"""
    print("🎨 img2img API测试工具")
    print("=" * 50)
    
    # 获取API Token
    api_token = os.environ.get('HF_API_TOKEN')
    
    if not api_token:
        print("❌ 未找到 HF_API_TOKEN 环境变量")
        print("\n设置方法:")
        print("Windows: set HF_API_TOKEN=your_token_here")
        print("Linux/Mac: export HF_API_TOKEN=your_token_here")
        print("\n获取Token: https://huggingface.co/settings/tokens")
        return
    
    print(f"✅ 使用API Token: {api_token[:10]}...{api_token[-4:]}")
    
    # 创建测试图像
    print("🖼️ 创建测试图像...")
    test_image = create_test_image()
    test_image.save("test_input_image.png")
    print("💾 已保存测试图像: test_input_image.png")
    
    # 执行测试
    results = test_img2img_api(api_token, test_image)
    
    # 保存详细结果
    with open("img2img_test_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print("\n💾 详细结果已保存: img2img_test_results.json")

if __name__ == "__main__":
    main()
