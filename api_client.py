"""
API客户端模块 - 处理与Hugging Face API的通信
"""

import requests
import io
import base64
from PIL import Image
from config import API_ENDPOINTS, CONTROLNET_API_ENDPOINTS, CONTROLNET_TYPES, PROXY_CONFIG, API_SUPPORTED_MODELS

# 全局变量
HF_API_TOKEN = None

def set_api_token(token):
    """设置API Token"""
    global HF_API_TOKEN
    HF_API_TOKEN = token.strip() if token else None

def validate_api_key(api_token):
    """验证API Key的有效性 - 改进版本"""
    if not api_token.strip():
        return "⚠️ 请输入有效的API Token"
    
    token = api_token.strip()
    
    # 基本格式检查
    if not token.startswith('hf_'):
        return "❌ Token格式错误：应该以 'hf_' 开头"
    
    if len(token) < 30:
        return "❌ Token长度过短：请检查是否完整复制"
    
    try:
        # 构建代理配置
        proxies = None
        if PROXY_CONFIG.get('enabled'):
            proxies = {}
            if PROXY_CONFIG.get('http'):
                proxies['http'] = PROXY_CONFIG['http']
            if PROXY_CONFIG.get('https'):
                proxies['https'] = PROXY_CONFIG['https']
        
        headers = {"Authorization": f"Bearer {token}"}
        
        # 方法1: 尝试访问用户信息API (使用正确的v2端点)
        try:
            response = requests.get(
                "https://huggingface.co/api/whoami-v2",
                headers=headers,
                timeout=15,
                proxies=proxies
            )
            
            if response.status_code == 200:
                try:
                    user_info = response.json()
                    username = user_info.get('name', 'User')
                    return f"✅ Token验证成功 - 用户: {username}"
                except:
                    return f"✅ Token验证成功 - API响应正常"
            elif response.status_code == 401:
                return "❌ Token无效：请检查Token是否正确或已过期"
            elif response.status_code == 403:
                return "⚠️ Token权限受限，但可能可用于基础API调用"
            else:
                # 如果whoami失败，继续尝试其他验证方法
                pass
                 
        except requests.exceptions.RequestException:
            # whoami API失败，尝试其他方法
            pass
        
        # 方法2: 尝试访问模型列表API（更宽松的验证）
        try:
            response = requests.get(
                "https://huggingface.co/api/models",
                headers=headers,
                timeout=15,
                proxies=proxies,
                params={"limit": 1}  # 只请求1个模型，减少流量
            )
            
            if response.status_code == 200:
                return f"✅ Token基本有效 - 可访问模型API"
            elif response.status_code == 401:
                return "❌ Token无效或已过期"
            elif response.status_code == 403:
                return "⚠️ Token权限不足，但格式正确"
            else:
                return f"⚠️ API返回状态 {response.status_code}，请检查Token权限"
                
        except requests.exceptions.RequestException:
            pass
        
        # 方法3: 最后尝试简单的推理API检查（HEAD请求）
        try:
            test_endpoint = "https://api-inference.huggingface.co/models/runwayml/stable-diffusion-v1-5"
            response = requests.head(
                test_endpoint,
                headers=headers,
                timeout=10,
                proxies=proxies
            )
            
            if response.status_code in [200, 503]:  # 503表示模型在加载
                return f"✅ Token可用于推理API"
            elif response.status_code == 401:
                return "❌ Token无效，无法访问推理API"
            elif response.status_code == 403:
                return "❌ Token权限不足，无法访问推理API"
            else:
                return f"⚠️ 推理API返回状态 {response.status_code}，Token可能有效"
                
        except requests.exceptions.Timeout:
            return f"⚠️ 网络超时，Token格式正确但无法验证连接"
        except requests.exceptions.ConnectionError:
            return f"⚠️ 网络连接失败，请检查网络设置或代理配置"
            
    except Exception as e:
        return f"❌ 验证过程出错: {str(e)[:50]}..."
    
    # 如果所有API调用都失败，但Token格式正确
    return f"⚠️ 无法验证Token有效性，但格式正确。可能是网络问题或API服务异常"

def check_model_api_support(model_id, run_mode):
    """检查模型是否支持API模式"""
    if run_mode != "api":
        return f"✅ 本地模式 - 支持所有模型"
    
    if model_id in API_ENDPOINTS:
        from config import MODELS
        return f"✅ API模式支持 - {MODELS.get(model_id, model_id)}"
    else:
        available_models = ", ".join([API_SUPPORTED_MODELS.get(m, m) for m in API_ENDPOINTS.keys()])
        return f"❌ API模式不支持此模型\n💡 支持的模型: {available_models}"

def test_model_api_connection(model_id, api_token):
    """测试模型API连接 - 改进版本"""
    if not api_token.strip():
        return "⚠️ 请先输入有效的API Token"
    
    if model_id not in API_ENDPOINTS:
        return f"❌ 模型 {model_id} 不支持API模式"
    
    try:
        endpoint = API_ENDPOINTS[model_id]
        headers = {"Authorization": f"Bearer {api_token.strip()}"}
        
        # 构建代理配置
        proxies = None
        if PROXY_CONFIG.get('enabled'):
            proxies = {}
            if PROXY_CONFIG.get('http'):
                proxies['http'] = PROXY_CONFIG['http']
            if PROXY_CONFIG.get('https'):
                proxies['https'] = PROXY_CONFIG['https']
        
        # 使用HEAD请求检查API可访问性（不实际生成图片）
        response = requests.head(
            endpoint,
            headers=headers,
            timeout=10,
            proxies=proxies
        )
        
        from config import MODELS
        model_name = MODELS.get(model_id, model_id)
        
        if response.status_code == 200:
            return f"✅ 模型API连接成功 - {model_name} 可用"
        elif response.status_code == 503:
            return f"⚠️ 模型正在加载中 - {model_name} (请稍后重试)"
        elif response.status_code == 401:
            return "❌ API Token无效或无权限访问此模型"
        elif response.status_code == 403:
            return "❌ Token权限不足，无法访问推理API"
        elif response.status_code == 404:
            return f"❌ 模型端点不存在 - {model_name}"
        elif response.status_code == 429:
            return f"⚠️ API调用频率限制 - {model_name} (Token有效)"
        else:
            return f"⚠️ API返回状态码 {response.status_code} - 连接可能有问题"
    
    except requests.exceptions.Timeout:
        return f"❌ 连接超时 - 请检查网络或启用代理"
    except requests.exceptions.ConnectionError:
        return f"❌ 网络连接失败 - 请检查网络设置或代理配置"
    except Exception as e:
        return f"❌ 连接测试失败: {str(e)[:50]}..."

def query_hf_api(endpoint, payload, api_token=None):
    """Call Hugging Face API with proxy support"""
    headers = {"Content-Type": "application/json"}
    if api_token:
        headers["Authorization"] = f"Bearer {api_token}"
    
    # 配置代理
    proxies = {}
    if PROXY_CONFIG["enabled"]:
        if PROXY_CONFIG["http"]:
            proxies["http"] = PROXY_CONFIG["http"]
        if PROXY_CONFIG["https"]:
            proxies["https"] = PROXY_CONFIG["https"]
    
    try:
        # 增加超时时间并使用代理
        response = requests.post(
            endpoint, 
            headers=headers, 
            json=payload, 
            timeout=120,  # 增加到2分钟
            proxies=proxies if proxies else None
        )
        
        if response.status_code == 200:
            return response.content
        elif response.status_code == 503:
            raise Exception("Model is loading, please try again later")
        elif response.status_code == 429:
            raise Exception("API rate limit exceeded, please try again later")
        elif response.status_code == 401:
            raise Exception("Invalid or missing API token")
        elif response.status_code == 404:
            raise Exception("Model endpoint not found")
        else:
            # Ensure error message is ASCII safe
            error_text = "Unknown API error"
            try:
                if response.text:
                    # Try to get ASCII-safe error message
                    error_text = response.text.encode('ascii', 'ignore').decode('ascii')
                    if not error_text.strip():
                        error_text = "API error with non-ASCII response"
            except:
                error_text = "API response encoding error"
            raise Exception(f"API call failed: {response.status_code}, {error_text}")
    except requests.exceptions.Timeout:
        proxy_info = f" (using proxy: {proxies})" if proxies else " (no proxy)"
        raise Exception(f"API call timeout after 120s{proxy_info}, please check network connection or proxy settings")
    except requests.exceptions.ConnectionError as e:
        proxy_info = f" (using proxy: {proxies})" if proxies else " (no proxy)"
        raise Exception(f"Network connection error{proxy_info}, please check network settings or try enabling proxy")
    except Exception as e:
        # Ensure all error messages are ASCII safe
        error_msg = str(e)
        try:
            error_msg.encode('ascii')
        except UnicodeEncodeError:
            error_msg = "API call error with encoding issues"
        raise Exception(error_msg)

def generate_image_api(prompt, negative_prompt="", model_id="runwayml/stable-diffusion-v1-5"):
    """Generate image using API"""
    endpoint = API_ENDPOINTS.get(model_id)
    if not endpoint:
        raise Exception(f"Model {model_id} does not support API mode")
    
    # Ensure prompt and negative_prompt are ASCII safe
    try:
        safe_prompt = prompt.encode('utf-8', 'ignore').decode('utf-8')
        safe_negative_prompt = negative_prompt.encode('utf-8', 'ignore').decode('utf-8') if negative_prompt else ""
    except:
        safe_prompt = "safe prompt"
        safe_negative_prompt = ""
    
    payload = {
        "inputs": safe_prompt,
        "parameters": {
            "negative_prompt": safe_negative_prompt,
            "num_inference_steps": 20,
            "guidance_scale": 7.5,
        }
    }
    
    try:
        image_bytes = query_hf_api(endpoint, payload, HF_API_TOKEN)
        image = Image.open(io.BytesIO(image_bytes))
        return image, "API image generation successful!"
    except Exception as e:
        return None, f"API generation failed: {str(e)}"

def generate_controlnet_image_api(prompt, negative_prompt, control_image, control_type):
    """Generate ControlNet image using API"""
    endpoint = CONTROLNET_API_ENDPOINTS.get(control_type)
    if not endpoint:
        raise Exception(f"ControlNet type {control_type} does not support API mode")
    
    # Convert control image to base64
    buffered = io.BytesIO()
    control_image.save(buffered, format="PNG")
    control_image_b64 = base64.b64encode(buffered.getvalue()).decode()
    
    # Ensure prompt and negative_prompt are safe
    try:
        safe_prompt = prompt.encode('utf-8', 'ignore').decode('utf-8')
        safe_negative_prompt = negative_prompt.encode('utf-8', 'ignore').decode('utf-8') if negative_prompt else ""
    except:
        safe_prompt = "safe prompt"
        safe_negative_prompt = ""
    
    payload = {
        "inputs": {
            "prompt": safe_prompt,
            "image": control_image_b64,
            "negative_prompt": safe_negative_prompt
        }
    }
    
    try:
        image_bytes = query_hf_api(endpoint, payload, HF_API_TOKEN)
        image = Image.open(io.BytesIO(image_bytes))
        control_type_name = CONTROLNET_TYPES[control_type]['name']
        return image, f"API mode {control_type_name} image generation successful!"
    except Exception as e:
        return None, f"ControlNet API generation failed: {str(e)}"

def generate_img2img_api(prompt, negative_prompt, input_image, strength):
    """Generate img2img image using API"""
    # Note: Hugging Face public API has limited img2img support
    # This is a basic implementation that may need adjustment
    endpoint = API_ENDPOINTS.get("runwayml/stable-diffusion-v1-5")  # Use default model
    if not endpoint:
        raise Exception("img2img API mode not supported")
    
    # Convert input image to base64
    buffered = io.BytesIO()
    input_image.save(buffered, format="PNG")
    input_image_b64 = base64.b64encode(buffered.getvalue()).decode()
    
    # Ensure prompt and negative_prompt are safe
    try:
        safe_prompt = prompt.encode('utf-8', 'ignore').decode('utf-8')
        safe_negative_prompt = negative_prompt.encode('utf-8', 'ignore').decode('utf-8') if negative_prompt else ""
    except:
        safe_prompt = "safe prompt"
        safe_negative_prompt = ""
    
    payload = {
        "inputs": {
            "prompt": safe_prompt,
            "image": input_image_b64,
            "negative_prompt": safe_negative_prompt,
            "strength": strength
        }
    }
    
    try:
        image_bytes = query_hf_api(endpoint, payload, HF_API_TOKEN)
        image = Image.open(io.BytesIO(image_bytes))
        return image, "API mode img2img generation successful!"
    except Exception as e:
        return None, f"img2img API generation failed: {str(e)}"
