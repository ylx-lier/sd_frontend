import requests
import json
import base64
from PIL import Image
import io

def test_controlnet_api_format():
    """测试ControlNet API的正确格式"""
    
    # 创建一个简单的测试图像
    test_image = Image.new('RGB', (512, 512), color='white')
    buffered = io.BytesIO()
    test_image.save(buffered, format="PNG")
    image_b64 = base64.b64encode(buffered.getvalue()).decode()
    
    endpoint = "https://api-inference.huggingface.co/models/lllyasviel/control_v11p_sd15_canny"
    
    # 测试不同的payload格式
    payloads = [
        # 格式1: 简单格式
        {
            "inputs": "a beautiful landscape",
            "parameters": {
                "image": image_b64
            }
        },
        
        # 格式2: 更详细的参数
        {
            "inputs": "a beautiful landscape",
            "parameters": {
                "image": image_b64,
                "negative_prompt": "blurry",
                "num_inference_steps": 20,
                "guidance_scale": 7.5
            }
        },
        
        # 格式3: 直接图像输入
        {
            "inputs": {
                "prompt": "a beautiful landscape",
                "image": image_b64
            }
        }
    ]
    
    headers = {
        "Authorization": "Bearer hf_fake_token",  # 使用假token来测试格式
        "Content-Type": "application/json"
    }
    
    for i, payload in enumerate(payloads, 1):
        print(f"\n=== 测试格式 {i} ===")
        print(f"Payload: {json.dumps(payload, indent=2)[:200]}...")
        
        try:
            response = requests.post(endpoint, headers=headers, json=payload, timeout=10)
            print(f"Status Code: {response.status_code}")
            print(f"Response: {response.text[:200]}...")
            
            if response.status_code == 422:
                print("❌ 格式错误 (422)")
            elif response.status_code == 401:
                print("✅ 格式正确，但需要有效token (401)")
            elif response.status_code == 503:
                print("✅ 格式正确，模型正在加载 (503)")
            
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    test_controlnet_api_format()
