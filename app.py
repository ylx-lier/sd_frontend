import os
import sys
import signal
import subprocess
import requests
import torch
import numpy as np
import cv2
import gradio as gr
import warnings
import io
import base64
from PIL import Image
from datetime import datetime
from diffusers import StableDiffusionPipeline, StableDiffusionControlNetPipeline, ControlNetModel
from diffusers import StableDiffusionImg2ImgPipeline, DPMSolverMultistepScheduler

warnings.filterwarnings("ignore")

# 全局变量存储Gradio应用实例
gradio_app = None
current_port = None

def cleanup_resources():
    """清理资源和释放端口"""
    global gradio_app, current_port
    
    print("\n🧹 正在清理资源...")
    
    try:
        # 关闭Gradio应用
        if gradio_app is not None:
            print("📱 关闭Gradio应用...")
            try:
                gradio_app.close()
            except:
                pass
            gradio_app = None
        
        # 强制释放端口（Windows）
        if current_port and os.name == 'nt':  # Windows系统
            try:
                print(f"🔌 释放端口 {current_port}...")
                # 查找占用端口的进程
                result = subprocess.run(
                    f'netstat -ano | findstr :{current_port}',
                    shell=True, capture_output=True, text=True
                )
                
                if result.stdout:
                    lines = result.stdout.strip().split('\n')
                    for line in lines:
                        if 'LISTENING' in line:
                            parts = line.split()
                            if len(parts) >= 5:
                                pid = parts[-1]
                                print(f"🎯 终止进程 PID: {pid}")
                                subprocess.run(f'taskkill /f /pid {pid}', shell=True, capture_output=True)
                                
            except Exception as e:
                print(f"⚠️ 端口释放警告: {e}")
        
        print("✅ 资源清理完成")
        
    except Exception as e:
        print(f"❌ 清理过程出错: {e}")

def signal_handler(signum, frame):
    """信号处理器 - 捕获Ctrl+C等中断信号"""
    print(f"\n🛑 收到中断信号 {signum}")
    cleanup_resources()
    print("👋 应用已安全退出")
    sys.exit(0)

def setup_signal_handlers():
    """设置信号处理器"""
    try:
        # 捕获常见的中断信号
        signal.signal(signal.SIGINT, signal_handler)   # Ctrl+C
        if hasattr(signal, 'SIGTERM'):
            signal.signal(signal.SIGTERM, signal_handler) # 终止信号
        if hasattr(signal, 'SIGBREAK'):
            signal.signal(signal.SIGBREAK, signal_handler) # Windows Ctrl+Break
        print("🔧 信号处理器已设置")
    except Exception as e:
        print(f"⚠️ 信号处理器设置失败: {e}")

# 设置自动清理
import atexit
atexit.register(cleanup_resources)

# 全局变量存储管道
pipe = None
controlnet_pipe = None
img2img_pipe = None
device = "cuda" if torch.cuda.is_available() else "cpu"
current_model = "runwayml/stable-diffusion-v1-5"
current_controlnet = None

# 运行模式选择
RUN_MODE = "api"  # "local" 或 "api"
HF_API_TOKEN = None  # 在这里设置您的 Hugging Face API Token

# 代理设置 (用于解决网络连接问题)
PROXY_CONFIG = {
    "enabled": False,
    "http": None,
    "https": None
}

def update_proxy_config(enabled, http_proxy, https_proxy):
    """更新代理配置"""
    global PROXY_CONFIG
    PROXY_CONFIG["enabled"] = enabled
    PROXY_CONFIG["http"] = http_proxy if http_proxy.strip() else None
    PROXY_CONFIG["https"] = https_proxy if https_proxy.strip() else None
    
    if enabled and (PROXY_CONFIG["http"] or PROXY_CONFIG["https"]):
        return f"✅ 代理已启用: HTTP={PROXY_CONFIG['http'] or 'None'}, HTTPS={PROXY_CONFIG['https'] or 'None'}"
    else:
        return "❌ 代理已禁用"

def auto_push_to_github():
    """自动推送到 GitHub"""
    try:
        print("🚀 开始自动推送到 GitHub...")
        
        # 检查是否在 git 仓库中
        result = subprocess.run("git status", shell=True, capture_output=True, text=True)
        if result.returncode != 0:
            return "❌ 当前目录不是 git 仓库或 git 未安装"
        
        # 添加所有更改的文件
        result = subprocess.run("git add .", shell=True, capture_output=True, text=True)
        if result.returncode != 0:
            return f"❌ 添加文件失败: {result.stderr}"
        
        # 检查是否有更改需要提交
        result = subprocess.run("git diff --staged --quiet", shell=True, capture_output=True, text=True)
        if result.returncode == 0:  # 如果命令成功，说明没有更改
            return "✅ 没有新的更改需要提交"
        
        # 生成时间戳
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # 准备提交信息
        commit_message = f"Auto update: {timestamp} - 功能更新和优化"
        
        # 提交更改
        result = subprocess.run(f'git commit -m "{commit_message}"', shell=True, capture_output=True, text=True)
        if result.returncode != 0:
            return f"❌ 提交失败: {result.stderr}"
        
        # 推送到远程仓库
        result = subprocess.run("git push origin main", shell=True, capture_output=True, text=True)
        if result.returncode != 0:
            return f"❌ 推送失败: {result.stderr}\n💡 请检查网络连接或 GitHub 权限"
        
        # 获取仓库 URL
        result = subprocess.run("git remote get-url origin", shell=True, capture_output=True, text=True)
        repo_url = result.stdout.strip() if result.returncode == 0 else "未知"
        
        return f"✅ 成功推送到 GitHub!\n🔗 仓库: {repo_url}\n⏰ 时间: {timestamp}"
        
    except Exception as e:
        return f"❌ 推送过程中发生错误: {str(e)}"

# API模式下的推理端点 - 官方支持的热门模型
API_ENDPOINTS = {
    # 最新推荐模型 (官方文档推荐)
    "black-forest-labs/FLUX.1-dev": "https://api-inference.huggingface.co/models/black-forest-labs/FLUX.1-dev",
    "black-forest-labs/FLUX.1-schnell": "https://api-inference.huggingface.co/models/black-forest-labs/FLUX.1-schnell",
    "stabilityai/stable-diffusion-xl-base-1.0": "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-xl-base-1.0",
    "stabilityai/stable-diffusion-3.5-large": "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-3.5-large",
    "stabilityai/stable-diffusion-3-medium-diffusers": "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-3-medium-diffusers",
    "latent-consistency/lcm-lora-sdxl": "https://api-inference.huggingface.co/models/latent-consistency/lcm-lora-sdxl",
    "Kwai-Kolors/Kolors": "https://api-inference.huggingface.co/models/Kwai-Kolors/Kolors",
    
    # 经典稳定的API模型
    "runwayml/stable-diffusion-v1-5": "https://api-inference.huggingface.co/models/runwayml/stable-diffusion-v1-5",
    "stabilityai/stable-diffusion-2-1": "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-2-1",
    "prompthero/openjourney": "https://api-inference.huggingface.co/models/prompthero/openjourney",
    "dreamlike-art/dreamlike-diffusion-1.0": "https://api-inference.huggingface.co/models/dreamlike-art/dreamlike-diffusion-1.0",
}

# ControlNet API endpoints - 更新为最新版本
CONTROLNET_API_ENDPOINTS = {
    "canny": "https://api-inference.huggingface.co/models/lllyasviel/control_v11p_sd15_canny",
    "scribble": "https://api-inference.huggingface.co/models/lllyasviel/control_v11p_sd15_scribble", 
    "depth": "https://api-inference.huggingface.co/models/lllyasviel/control_v11p_sd15_depth",
    "openpose": "https://api-inference.huggingface.co/models/lllyasviel/control_v11p_sd15_openpose",
    "seg": "https://api-inference.huggingface.co/models/lllyasviel/control_v11p_sd15_seg"
}

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
        return f"✅ API模式支持 - {MODELS.get(model_id, model_id)}"
    else:
        available_models = ", ".join([MODELS.get(m, m) for m in API_ENDPOINTS.keys()])
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

def test_controlnet_api_connection(control_type, api_token):
    """测试ControlNet API连接"""
    if not api_token.strip():
        return "⚠️ 请先输入有效的API Token"
    
    if control_type not in CONTROLNET_API_ENDPOINTS:
        return f"❌ ControlNet类型 {control_type} 不支持API模式"
    
    try:
        endpoint = CONTROLNET_API_ENDPOINTS[control_type]
        headers = {"Authorization": f"Bearer {api_token.strip()}"}
        
        # 构建代理配置
        proxies = None
        if PROXY_CONFIG.get('enabled'):
            proxies = {}
            if PROXY_CONFIG.get('http'):
                proxies['http'] = PROXY_CONFIG['http']
            if PROXY_CONFIG.get('https'):
                proxies['https'] = PROXY_CONFIG['https']
        
        # 使用HEAD请求检查API可访问性
        response = requests.head(
            endpoint,
            headers=headers,
            timeout=10,
            proxies=proxies
        )
        
        control_name = CONTROLNET_TYPES[control_type]['name']
        
        if response.status_code == 200:
            return f"✅ ControlNet API连接成功 - {control_name} 可用"
        elif response.status_code == 503:
            return f"⚠️ ControlNet模型正在加载 - {control_name} (请稍等1-2分钟)"
        elif response.status_code == 401:
            return "❌ API Token无效或无权限访问ControlNet模型"
        elif response.status_code == 403:
            return "❌ Token权限不足，无法访问ControlNet推理API"
        elif response.status_code == 404:
            return f"❌ ControlNet端点不存在 - {control_name}"
        elif response.status_code == 429:
            return f"⚠️ API调用频率限制 - {control_name} (Token有效，请稍后重试)"
        else:
            return f"⚠️ 未知状态 ({response.status_code}) - {control_name}"
    
    except requests.exceptions.Timeout:
        return f"❌ 连接超时 - 请检查网络或启用代理"
    except requests.exceptions.ConnectionError:
        return f"❌ 网络连接失败 - 请检查网络设置或代理配置"
    except Exception as e:
        return f"❌ 连接测试失败: {str(e)[:50]}..."

def test_img2img_api_connection(api_token):
    """img2img API技术说明 - 解释为什么API模式不适合img2img"""
    
    return """🔬 img2img技术原理分析

📋 **技术栈要求**：
✅ VAE编码器 - 图像→潜在空间转换
✅ 噪声调制器 - 根据strength参数调制
✅ UNet采样器 - 潜在空间去噪过程  
✅ VAE解码器 - 潜在空间→图像转换
✅ 调度器 - 控制采样步骤

🌐 **API模式现状**：
❌ 公共API通常只提供text-to-image接口
❌ 缺少独立的VAE编码器访问
❌ 无法进行潜在空间操作
❌ 复杂流程不适合HTTP API封装

💡 **ComfyUI对比**：
ComfyUI通过节点化设计，提供完整的VAE编码器节点，
这正是高质量img2img的关键。每个步骤都可以独立配置。

🎯 **推荐解决方案**：

1. 🏠 **本地模式** (最佳选择)
   • 完整img2img管道支持
   • 包含专用VAE编码器/解码器
   • 可精确控制strength等参数

2. 🖼️ **ControlNet模式** (API兼容)
   • Canny边缘检测 + 文生图
   • 保持原图结构，改变风格  
   • API模式完全支持

3. 📊 **Inpainting模式** (局部编辑)
   • 针对特定区域修改
   • 某些API服务支持

� **技术结论**：
img2img本质上是一个需要完整模型管道的复杂流程，
更适合本地计算而非API调用。"""

# ControlNet 类型选项 - 更新为最新版本
CONTROLNET_TYPES = {
    "canny": {
        "name": "Canny边缘检测",
        "model_id": "lllyasviel/control_v11p_sd15_canny",
        "description": "检测图像边缘轮廓，保持物体形状"
    },
    "scribble": {
        "name": "Scribble涂鸦控制",
        "model_id": "lllyasviel/control_v11p_sd15_scribble", 
        "description": "基于手绘涂鸦或简笔画生成图像"
    },
    "depth": {
        "name": "Depth深度控制",
        "model_id": "lllyasviel/control_v11p_sd15_depth",
        "description": "基于深度图控制空间结构和层次"
    },
    "openpose": {
        "name": "OpenPose姿态控制",
        "model_id": "lllyasviel/control_v11p_sd15_openpose",
        "description": "基于人体姿态骨架控制人物姿势"
    },
    "seg": {
        "name": "Segmentation分割控制",
        "model_id": "lllyasviel/control_v11p_sd15_seg",
        "description": "基于语义分割图控制物体分布"
    }
}

# 预定义模型列表 (分为API支持和仅本地支持)
API_SUPPORTED_MODELS = {
    # 最新推荐模型 (官方文档推荐，性能优异)
    "black-forest-labs/FLUX.1-dev": "FLUX.1 Dev (最强大的图像生成模型，推荐)",
    "black-forest-labs/FLUX.1-schnell": "FLUX.1 Schnell (快速生成，高质量)",
    "stabilityai/stable-diffusion-xl-base-1.0": "SDXL Base 1.0 (高分辨率，经典选择)",
    "stabilityai/stable-diffusion-3.5-large": "SD 3.5 Large (最新版本)",
    "stabilityai/stable-diffusion-3-medium-diffusers": "SD 3 Medium (强大的文生图)",
    "latent-consistency/lcm-lora-sdxl": "LCM-LoRA SDXL (快速且强大)",
    "Kwai-Kolors/Kolors": "Kolors (逼真图像生成)",
    
    # 经典稳定的API模型
    "runwayml/stable-diffusion-v1-5": "Stable Diffusion v1.5 (经典基础模型)",
    "stabilityai/stable-diffusion-2-1": "Stable Diffusion v2.1 (更高质量)",
    "prompthero/openjourney": "OpenJourney (多样化艺术风格)",
    "dreamlike-art/dreamlike-diffusion-1.0": "Dreamlike Diffusion (梦幻艺术风格)",
}

# 仅本地模式支持的模型
LOCAL_ONLY_MODELS = {
    "wavymulder/Analog-Diffusion": "Analog Diffusion (胶片风格)",
    "22h/vintedois-diffusion-v0-1": "VintedoisDiffusion (复古风格)",
    "nitrosocke/Arcane-Diffusion": "Arcane Diffusion (动画风格)",
    "hakurei/waifu-diffusion": "Waifu Diffusion (动漫风格)"
}

# 根据运行模式动态获取可用模型
def get_available_models(run_mode):
    if run_mode == "api":
        return API_SUPPORTED_MODELS
    else:
        # 本地模式支持所有模型
        return {**API_SUPPORTED_MODELS, **LOCAL_ONLY_MODELS}

# 兼容性：保持原有MODELS变量
MODELS = {**API_SUPPORTED_MODELS, **LOCAL_ONLY_MODELS}

# Prompt 辅助词条
PROMPT_CATEGORIES = {
    "质量增强": [
        "masterpiece", "best quality", "ultra detailed", "extremely detailed", 
        "high resolution", "8k", "4k", "highly detailed", "sharp focus",
        "professional photography", "award winning", "cinematic lighting"
    ],
    "艺术风格": [
        "oil painting", "watercolor", "digital art", "concept art", "illustration",
        "anime style", "cartoon style", "realistic", "photorealistic", "hyperrealistic",
        "art nouveau", "baroque", "impressionist", "surreal", "abstract"
    ],
    "光照效果": [
        "soft lighting", "dramatic lighting", "cinematic lighting", "golden hour",
        "studio lighting", "natural lighting", "ambient lighting", "rim lighting",
        "volumetric lighting", "god rays", "neon lighting", "sunset", "sunrise"
    ],
    "构图视角": [
        "close-up", "portrait", "full body", "wide shot", "aerial view",
        "bird's eye view", "low angle", "high angle", "profile view",
        "three-quarter view", "dynamic pose", "action shot"
    ],
    "情绪氛围": [
        "peaceful", "dramatic", "mysterious", "romantic", "epic", "serene",
        "energetic", "melancholic", "cheerful", "dark", "bright", "cozy",
        "majestic", "elegant", "powerful", "gentle"
    ],
    "环境场景": [
        "forest", "mountain", "ocean", "city", "countryside", "desert",
        "fantasy world", "sci-fi", "medieval", "modern", "futuristic",
        "indoor", "outdoor", "studio", "landscape", "urban", "nature"
    ],
    "色彩风格": [
        "vibrant colors", "muted colors", "monochrome", "black and white",
        "warm colors", "cool colors", "pastel colors", "neon colors",
        "earth tones", "jewel tones", "vintage colors", "saturated"
    ]
}

# 负面提示词辅助词条
NEGATIVE_PROMPT_CATEGORIES = {
    "画质问题": [
        "blurry", "low quality", "bad quality", "worst quality", "poor quality",
        "pixelated", "jpeg artifacts", "compression artifacts", "distorted",
        "low resolution", "grainy", "noisy", "oversaturated", "undersaturated"
    ],
    "解剖错误": [
        "bad anatomy", "bad hands", "bad fingers", "extra fingers", "missing fingers",
        "extra limbs", "missing limbs", "deformed", "mutated", "disfigured",
        "malformed", "extra arms", "extra legs", "fused fingers", "too many fingers"
    ],
    "面部问题": [
        "bad face", "ugly face", "distorted face", "asymmetrical face",
        "bad eyes", "cross-eyed", "extra eyes", "missing eyes", "bad mouth",
        "bad teeth", "crooked teeth", "bad nose", "asymmetrical features"
    ],
    "艺术风格": [
        "cartoon", "anime", "manga", "3d render", "painting", "sketch",
        "watercolor", "oil painting", "digital art", "illustration",
        "abstract", "surreal", "unrealistic", "stylized"
    ],
    "技术问题": [
        "watermark", "signature", "text", "logo", "copyright", "username",
        "frame", "border", "cropped", "cut off", "out of frame",
        "duplicate", "error", "glitch", "artifact"
    ],
    "光照问题": [
        "bad lighting", "harsh lighting", "overexposed", "underexposed",
        "too dark", "too bright", "uneven lighting", "poor contrast",
        "washed out", "flat lighting", "artificial lighting"
    ],
    "构图问题": [
        "bad composition", "off-center", "tilted", "crooked", "unbalanced",
        "cluttered", "messy", "chaotic", "poor framing", "bad angle",
        "awkward pose", "stiff pose", "unnatural pose"
    ]
}

def load_models(run_mode, selected_model, controlnet_type="canny", api_token=""):
    """加载模型管道 - 改进版本，支持API模型检测"""
    global pipe, controlnet_pipe, img2img_pipe, current_model, current_controlnet, RUN_MODE, HF_API_TOKEN
    
    if not selected_model:
        return "❌ 请选择一个模型"
    
    # 更新全局配置
    RUN_MODE = run_mode
    current_model = selected_model
    if api_token.strip():
        HF_API_TOKEN = api_token.strip()
    
    # 获取模型信息
    available_models = get_available_models(run_mode)
    model_name = available_models.get(selected_model, selected_model)
    
    if run_mode == "api":
        # API模式 - 检查模型支持
        if selected_model not in API_ENDPOINTS:
            supported_models = list(API_SUPPORTED_MODELS.keys())
            recommended = supported_models[:3]  # 推荐前3个
            
            return f"❌ 模型 {model_name} 不支持API模式\n\n� 推荐支持API的模型:\n" + \
                   "\n".join([f"• {API_SUPPORTED_MODELS[m]}" for m in recommended]) + \
                   f"\n\n💡 共有 {len(supported_models)} 个模型支持API模式，请在下拉菜单中选择"
        
        # 检查Token有效性（如果提供）
        token_status = ""
        if api_token.strip():
            # 可以在这里调用Token验证函数
            token_status = "\n🔑 使用认证Token"
        
        # 模拟加载成功
        pipe = "api_mode"
        img2img_pipe = "api_mode" 
        controlnet_pipe = "api_mode"
        current_controlnet = controlnet_type
        
        # 判断模型类型并给出相应提示
        if selected_model.startswith("black-forest-labs/FLUX"):
            quality_tip = "\n⚡ FLUX系列 - 最新一代模型，图像质量极高"
        elif selected_model.startswith("stabilityai/stable-diffusion-xl"):
            quality_tip = "\n� SDXL系列 - 高分辨率生成，经典选择"
        elif selected_model.startswith("stabilityai/stable-diffusion-3"):
            quality_tip = "\n🚀 SD3系列 - 最新技术，文本理解能力强"
        else:
            quality_tip = "\n📝 经典模型 - 稳定可靠"
        
        return f"✅ API模式配置成功！\n📦 当前模型: {model_name}\n🎯 模型ID: {selected_model}\n🎮 ControlNet: {CONTROLNET_TYPES[controlnet_type]['name']}{quality_tip}{token_status}\n💾 存储空间占用: 0 GB\n\n💡 API模式无需下载模型，生成图片通过云端推理"
    
    else:
        # 本地模式 - 下载模型到本地
        try:
            # 基础文生图管道
            pipe = StableDiffusionPipeline.from_pretrained(
                selected_model,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                safety_checker=None,
                requires_safety_checker=False
            )
            pipe = pipe.to(device)
            pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
            
            # 传统图生图管道
            img2img_pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
                selected_model,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                safety_checker=None,
                requires_safety_checker=False
            )
            img2img_pipe = img2img_pipe.to(device)
            img2img_pipe.scheduler = DPMSolverMultistepScheduler.from_config(img2img_pipe.scheduler.config)
            
            # ControlNet 管道
            try:
                current_controlnet = controlnet_type
                controlnet_info = CONTROLNET_TYPES[controlnet_type]
                
                controlnet = ControlNetModel.from_pretrained(
                    controlnet_info["model_id"],
                    torch_dtype=torch.float16 if device == "cuda" else torch.float32
                )
                controlnet_pipe = StableDiffusionControlNetPipeline.from_pretrained(
                    selected_model,
                    controlnet=controlnet,
                    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                    safety_checker=None,
                    requires_safety_checker=False
                )
                controlnet_pipe = controlnet_pipe.to(device)
                controlnet_pipe.scheduler = DPMSolverMultistepScheduler.from_config(controlnet_pipe.scheduler.config)
                return f"✅ 本地模式所有模型加载成功！\n📦 当前模型: {model_name}\n🎯 模型ID: {selected_model}\n🎮 ControlNet: {controlnet_info['name']}\n💾 预计存储占用: ~6-10 GB"
            except Exception as controlnet_error:
                return f"✅ 本地模式基础模型加载成功！\n📦 当前模型: {model_name}\n🎯 模型ID: {selected_model}\n⚠️ ControlNet加载失败: {str(controlnet_error)}\n💡 文生图和传统图生图功能可正常使用\n💾 预计存储占用: ~4-7 GB"
            
        except Exception as e:
            return f"❌ 本地模式加载失败: {str(e)}\n💡 建议尝试API模式以避免存储空间问题"

def generate_image(prompt, negative_prompt, num_steps, guidance_scale, width, height, seed):
    """基础文生图功能"""
    global pipe, current_model, RUN_MODE
    
    if pipe is None:
        return None, "Please load the model first"
    
    if RUN_MODE == "api":
        # API模式
        try:
            image, status = generate_image_api(prompt, negative_prompt, current_model)
            return image, status
        except Exception as e:
            return None, f"❌ API生成失败: {str(e)}"
    
    else:
        # 本地模式
        try:
            # 设置随机种子
            if seed != -1:
                generator = torch.Generator(device=device).manual_seed(seed)
            else:
                generator = None
                
            # 生成图像
            with torch.autocast(device):
                result = pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt if negative_prompt else None,
                    num_inference_steps=num_steps,
                    guidance_scale=guidance_scale,
                    width=width,
                    height=height,
                    generator=generator
                )
            
            image = result.images[0]
            return image, "✅ 本地图像生成成功！"
            
        except Exception as e:
            return None, f"❌ 本地生成失败: {str(e)}"

def preprocess_canny(image, low_threshold=100, high_threshold=200):
    """预处理图像为Canny边缘"""
    image = np.array(image)
    canny = cv2.Canny(image, low_threshold, high_threshold)
    canny_image = canny[:, :, None]
    canny_image = np.concatenate([canny_image, canny_image, canny_image], axis=2)
    return Image.fromarray(canny_image)

def preprocess_scribble(image):
    """预处理图像为涂鸦风格（简化边缘）"""
    image = np.array(image)
    # 转换为灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # 使用边缘检测但参数更宽松，模拟涂鸦效果
    edges = cv2.Canny(gray, 50, 150)
    # 膨胀操作使线条更粗，更像涂鸦
    kernel = np.ones((3,3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    # 转换回RGB
    scribble_image = edges[:, :, None]
    scribble_image = np.concatenate([scribble_image, scribble_image, scribble_image], axis=2)
    return Image.fromarray(scribble_image)

def preprocess_depth(image):
    """预处理图像为深度图（使用简单的深度估计）"""
    image = np.array(image)
    # 转换为灰度图作为简单的深度估计
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # 应用高斯模糊来模拟深度感
    depth = cv2.GaussianBlur(gray, (5, 5), 0)
    # 增强对比度
    depth = cv2.equalizeHist(depth)
    # 转换回RGB
    depth_image = depth[:, :, None]
    depth_image = np.concatenate([depth_image, depth_image, depth_image], axis=2)
    return Image.fromarray(depth_image)

def preprocess_control_image(image, control_type):
    """根据控制类型预处理图像"""
    if control_type == "canny":
        return preprocess_canny(image)
    elif control_type == "scribble":
        return preprocess_scribble(image)
    elif control_type == "depth":
        return preprocess_depth(image)
    else:
        return preprocess_canny(image)  # 默认使用canny

def generate_controlnet_image(prompt, negative_prompt, control_image, control_type, num_steps, guidance_scale, controlnet_conditioning_scale, width, height, seed):
    """ControlNet图像引导生成"""
    global controlnet_pipe, current_controlnet, RUN_MODE
    
    if controlnet_pipe is None:
        return None, None, "❌ 请先加载模型"
    
    if control_image is None:
        return None, None, "❌ 请上传控制图像"
    
    # 检查当前加载的ControlNet类型是否匹配
    if current_controlnet != control_type:
        return None, None, f"❌ 当前加载的是 {CONTROLNET_TYPES[current_controlnet]['name']}，请重新加载模型选择 {CONTROLNET_TYPES[control_type]['name']}"
    
    # 预处理控制图像
    processed_image = preprocess_control_image(control_image, control_type)
    
    if RUN_MODE == "api":
        # API模式
        try:
            image, status = generate_controlnet_image_api(prompt, negative_prompt, processed_image, control_type)
            return image, processed_image, status
        except Exception as e:
            return None, processed_image, f"❌ API生成失败: {str(e)}"
    
    else:
        # 本地模式
        try:
            # 设置随机种子
            if seed != -1:
                generator = torch.Generator(device=device).manual_seed(seed)
            else:
                generator = None
                
            # 生成图像
            with torch.autocast(device):
                result = controlnet_pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt if negative_prompt else None,
                    image=processed_image,
                    num_inference_steps=num_steps,
                    guidance_scale=guidance_scale,
                    controlnet_conditioning_scale=controlnet_conditioning_scale,
                    width=width,
                    height=height,
                    generator=generator
                )
            
            image = result.images[0]
            control_type_name = CONTROLNET_TYPES[control_type]['name']
            return image, processed_image, f"✅ {control_type_name}图像生成成功！"
            
        except Exception as e:
            return None, processed_image, f"❌ 生成失败: {str(e)}"

def generate_img2img(prompt, negative_prompt, input_image, strength, num_steps, guidance_scale, width, height, seed):
    """传统图生图功能 - 改进版本，包含更好的API处理和用户指导"""
    global img2img_pipe, RUN_MODE
    
    if img2img_pipe is None:
        return None, "❌ 请先加载模型"
    
    if input_image is None:
        return None, "❌ 请上传输入图像"
    
    # 验证参数
    if not prompt or not prompt.strip():
        return None, "❌ 请输入描述提示词"
    
    if strength < 0 or strength > 1:
        return None, "❌ 变换强度应在0-1之间"
    
    # 调整图像大小
    try:
        input_image = input_image.resize((width, height))
    except Exception as e:
        return None, f"❌ 图像处理失败: {str(e)}"
    
    if RUN_MODE == "api":
        # API模式 - 包含详细的错误处理和用户指导
        try:
            print(f"🌐 API模式: 尝试img2img生成...")
            image, status = generate_img2img_api(prompt, negative_prompt, input_image, strength)
            
            if image is not None:
                return image, status
            else:
                # API失败时，提供详细的指导信息
                fallback_message = f"""🔄 img2img API暂不可用，建议尝试以下替代方案:

🎯 最佳替代方案:
1. 🏠 切换到本地模式 - img2img功能完全支持
2. 🖼️ 使用ControlNet模式:
   • 选择Canny边缘检测
   • 上传您的图像作为控制图
   • 输入相同的提示词
   • 获得类似的图像变换效果

📋 操作步骤:
• 点击上方"切换到本地模式"按钮
• 或切换到"ControlNet生成"标签页
• 选择"canny"类型，上传同一张图像

💡 为什么会这样:
{status}

🔧 技术原因:
• Hugging Face公共API主要支持text-to-image
• img2img需要专门的API端点支持
• 大多数模型尚未提供img2img API接口"""
                
                return None, fallback_message
                
        except Exception as e:
            error_message = f"""❌ img2img API调用异常: {str(e)}

🔄 推荐解决方案:
1. 🏠 切换到本地模式 (100%支持img2img)
2. 🖼️ 使用ControlNet模式替代
3. 🎨 使用纯文生图模式

📋 快速操作:
• 点击"切换到本地模式"
• 或使用ControlNet的Canny功能"""
            
            return None, error_message
    
    else:
        # 本地模式 - 完全支持img2img
        try:
            print(f"🏠 本地模式: 开始img2img生成...")
            print(f"   输入图像尺寸: {input_image.size}")
            print(f"   变换强度: {strength}")
            print(f"   生成步数: {num_steps}")
            
            # 设置随机种子
            if seed != -1:
                generator = torch.Generator(device=device).manual_seed(seed)
                print(f"   使用种子: {seed}")
            else:
                generator = None
                print(f"   随机种子")
                
            # 生成图像
            with torch.autocast(device):
                result = img2img_pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt if negative_prompt else None,
                    image=input_image,
                    strength=strength,
                    num_inference_steps=num_steps,
                    guidance_scale=guidance_scale,
                    generator=generator
                )
            
            image = result.images[0]
            
            success_message = f"""✅ 本地img2img生成成功！
🖼️ 输出尺寸: {image.size}
🎯 变换强度: {strength}
📝 生成步数: {num_steps}
⚡ 引导强度: {guidance_scale}"""
            
            return image, success_message
            
        except Exception as e:
            error_message = f"""❌ 本地img2img生成失败: {str(e)}

🔧 可能的解决方案:
1. 📉 降低图像分辨率 (512x512)
2. 📊 减少生成步数 (15-25)
3. 🎚️ 调整变换强度 (0.5-0.8)
4. 💾 检查显存是否充足
5. 🔄 重新加载模型

⚠️ 如果问题持续，建议使用ControlNet模式"""
            
            return None, error_message

def add_prompt_tags(current_prompt, selected_tags):
    """添加选中的标签到prompt中"""
    if not selected_tags:
        return current_prompt
    
    # 将选中的标签合并
    new_tags = ", ".join(selected_tags)
    
    if current_prompt:
        # 如果已有prompt，则添加到末尾
        return f"{current_prompt}, {new_tags}"
    else:
        # 如果没有prompt，直接使用标签
        return new_tags

def get_current_model_info():
    """获取当前模型信息"""
    global current_model
    if current_model:
        model_name = MODELS.get(current_model, current_model)
        return f"📦 当前模型: {model_name}"
    else:
        return "❌ 未加载模型"

# 创建Gradio界面
def create_interface():
    with gr.Blocks(title="🎨 AI 图像生成器", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # 🎨 AI 图像生成器 Pro
        
        支持多种模型和三种生成模式：
        - **📝 文生图模式**：纯文本描述生成图像
        - **🔄 传统图生图**：图像风格转换
        - **🖼️ ControlNet模式**：精确结构控制的图像生成
        
        > 基于 Stable Diffusion 系列模型 + ControlNet
        """)
        
        # 模式说明信息面板
        gr.Markdown("""
        ### 🚀 运行模式说明
        - **🌐 API模式 (推荐)**: 通过 Hugging Face API 在线生成，**无需下载任何模型**，节省 4-10GB 存储空间！
        - **💻 本地模式**: 下载模型到本地运行，需要 4-10GB 存储空间，但运行速度更快，支持更多自定义参数
        - **🔑 API Token**: API模式需要 [Hugging Face Token](https://huggingface.co/settings/tokens) (免费账户即可)
        
        ### 🎯 推荐API模型（按性能排序）
        1. **FLUX.1 Dev** - 🥇 最强大的图像生成模型，质量极高
        2. **FLUX.1 Schnell** - ⚡ 快速生成，高质量输出
        3. **SDXL Base 1.0** - 🎨 高分辨率生成，经典选择
        4. **SD 3.5 Large** - 🚀 最新版本，文本理解能力强
        5. **Kolors** - 📸 逼真图像生成，人像效果佳
        """)
        
        # 模型选择和加载区域
        with gr.Row():
            with gr.Column(scale=3):
                # 运行模式选择
                run_mode_radio = gr.Radio(
                    choices=[
                        ("🌐 API模式 (推荐) - 无需下载，节省存储空间", "api"),
                        ("💻 本地模式 - 下载到本地，需要大量存储空间", "local")
                    ],
                    value="api",
                    label="⚙️ 运行模式",
                    info="API模式通过网络调用，本地模式需要下载4-10GB模型文件"
                )
                
                model_dropdown = gr.Dropdown(
                    choices=[(name, model_id) for model_id, name in API_SUPPORTED_MODELS.items()],
                    value="black-forest-labs/FLUX.1-dev",
                    label="🤖 选择基础模型 (仅API支持的模型)",
                    info="✅ API模式 - 这些模型支持云端推理，无需下载"
                )
                controlnet_dropdown = gr.Dropdown(
                    choices=[(f"{info['name']} - {info['description']}", key) for key, info in CONTROLNET_TYPES.items()],
                    value="canny",
                    label="🎮 选择ControlNet类型",
                    info="选择不同的控制方式"
                )
                
                # API Token 设置
                with gr.Accordion("🔑 API设置 (API模式必看)", open=True):
                    gr.Markdown("""
                    **🎯 获取免费API Token：**
                    1. 访问 [Hugging Face](https://huggingface.co/settings/tokens)
                    2. 创建新Token (Read权限即可)
                    3. 复制并粘贴到下方输入框
                    
                    **💡 提示：** 免费账户每月有一定调用限制，付费账户无限制且速度更快
                    """)
                    api_token_input = gr.Textbox(
                        label="🔑 Hugging Face API Token",
                        placeholder="hf_xxxxxxxxxxxxxxxxxxxx (建议设置，否则可能遇到限流)",
                        type="password",
                        info="点击上方链接获取免费Token，提升API调用稳定性"
                    )
                    
                    # API Token 验证状态
                    token_status = gr.Textbox(
                        label="Token验证状态",
                        value="⚠️ 请输入API Token进行验证",
                        interactive=False,
                        lines=1
                    )
                    
                    # 模型API支持状态
                    model_api_status = gr.Textbox(
                        label="模型API支持状态",
                        value="✅ 当前模型支持API模式",
                        interactive=False,
                        lines=1
                    )
                    
                    # API连接测试按钮
                    test_api_btn = gr.Button("🔗 测试基础模型API连接", variant="secondary")
                    
                    # ControlNet API 测试
                    with gr.Row():
                        controlnet_test_dropdown = gr.Dropdown(
                            choices=list(CONTROLNET_TYPES.keys()),
                            value="canny",
                            label="选择ControlNet类型进行测试",
                            scale=2
                        )
                        test_controlnet_api_btn = gr.Button("🎮 测试ControlNet API", variant="secondary", scale=1)
                    
                    controlnet_api_status = gr.Textbox(
                        label="ControlNet API测试状态",
                        value="点击测试按钮检查ControlNet API连接",
                        interactive=False,
                        lines=1
                    )
                
                # 代理设置
                with gr.Accordion("🌐 网络代理设置 (解决连接超时问题)", open=False):
                    gr.Markdown("""
                    **🚨 如果遇到 "API call timeout" 错误，请启用代理：**
                    
                    **Clash 代理设置：**
                    - HTTP代理端口通常是：`http://127.0.0.1:7890`
                    - HTTPS代理端口通常是：`http://127.0.0.1:7890`
                    - 如果端口不同，请查看 Clash 的端口设置
                    
                    **其他代理软件：**
                    - V2Ray: `http://127.0.0.1:10809`
                    - Shadowsocks: `http://127.0.0.1:1080`
                    - 请根据您的代理软件实际端口填写
                    """)
                    
                    proxy_enabled = gr.Checkbox(
                        label="启用代理",
                        value=False,
                        info="如果网络连接超时，请启用此选项"
                    )
                    
                    with gr.Row():
                        http_proxy_input = gr.Textbox(
                            label="HTTP 代理",
                            placeholder="http://127.0.0.1:7890",
                            info="填写 HTTP 代理地址和端口"
                        )
                        https_proxy_input = gr.Textbox(
                            label="HTTPS 代理", 
                            placeholder="http://127.0.0.1:7890",
                            info="填写 HTTPS 代理地址和端口"
                        )
                    
                    proxy_status = gr.Textbox(
                        label="代理状态",
                        value="❌ 代理已禁用",
                        interactive=False
                    )
                    
                    test_proxy_btn = gr.Button("🔗 测试代理连接", variant="secondary")
                    
                    def test_proxy_connection(enabled, http_proxy, https_proxy):
                        """测试代理连接"""
                        if not enabled:
                            return "❌ 代理未启用，无法测试"
                        
                        if not (http_proxy or https_proxy):
                            return "❌ 请填写代理地址"
                        
                        proxies = {}
                        if http_proxy:
                            proxies["http"] = http_proxy
                        if https_proxy:
                            proxies["https"] = https_proxy
                        
                        try:
                            # 测试连接到 Hugging Face
                            response = requests.get(
                                "https://huggingface.co", 
                                proxies=proxies, 
                                timeout=10
                            )
                            if response.status_code == 200:
                                return "✅ 代理连接测试成功！"
                            else:
                                return f"⚠️ 代理连接测试失败，状态码: {response.status_code}"
                        except Exception as e:
                            return f"❌ 代理连接测试失败: {str(e)}"
                    
                    test_proxy_btn.click(
                        test_proxy_connection,
                        inputs=[proxy_enabled, http_proxy_input, https_proxy_input],
                        outputs=[proxy_status]
                    )
                    
                load_btn = gr.Button("🚀 加载选中模型", variant="primary", size="lg")
            with gr.Column(scale=2):
                current_model_display = gr.Textbox(
                    label="当前模型状态", 
                    value="📦 默认模型: Stable Diffusion v1.5\n🎮 默认ControlNet: Canny边缘检测\n⚙️ 默认模式: API模式",
                    interactive=False,
                    lines=3
                )
        
        load_status = gr.Textbox(label="加载状态", value="选择模型后点击加载开始使用", lines=3)
        
        # GitHub 自动推送区域
        with gr.Accordion("🚀 GitHub 自动推送", open=False):
            gr.Markdown("""
            **📦 代码同步功能：**
            - 自动将当前所有更改推送到 GitHub 仓库
            - 包含代码更新、新增文件、配置修改等
            - 适合开发过程中的版本备份和同步
            
            **⚠️ 注意事项：**
            - 确保已配置 GitHub 访问权限
            - 建议在重要功能完成后使用
            - 推送前会自动添加所有更改文件
            """)
            
            with gr.Row():
                push_to_github_btn = gr.Button("🚀 推送到 GitHub", variant="primary", size="lg")
                github_status = gr.Textbox(
                    label="推送状态",
                    value="点击按钮将代码推送到 GitHub 仓库",
                    interactive=False,
                    lines=2
                )
        
        
        # Prompt 辅助选择器
        with gr.Accordion("🎯 Prompt 辅助选择器", open=False):
            gr.Markdown("### 💡 选择词条快速构建高质量提示词")
            
            # 正面词条
            gr.Markdown("#### ✨ 正面提示词 (Positive Prompt)")
            with gr.Row():
                with gr.Column():
                    quality_tags = gr.CheckboxGroup(
                        choices=PROMPT_CATEGORIES["质量增强"],
                        label="🌟 质量增强",
                        info="提升图像质量和细节"
                    )
                    style_tags = gr.CheckboxGroup(
                        choices=PROMPT_CATEGORIES["艺术风格"],
                        label="🎨 艺术风格",
                        info="选择艺术表现形式"
                    )
                with gr.Column():
                    lighting_tags = gr.CheckboxGroup(
                        choices=PROMPT_CATEGORIES["光照效果"],
                        label="💡 光照效果",
                        info="设置光照和氛围"
                    )
                    composition_tags = gr.CheckboxGroup(
                        choices=PROMPT_CATEGORIES["构图视角"],
                        label="📐 构图视角",
                        info="选择拍摄角度和构图"
                    )
                with gr.Column():
                    mood_tags = gr.CheckboxGroup(
                        choices=PROMPT_CATEGORIES["情绪氛围"],
                        label="😊 情绪氛围",
                        info="设定画面情感色调"
                    )
                    scene_tags = gr.CheckboxGroup(
                        choices=PROMPT_CATEGORIES["环境场景"],
                        label="🌍 环境场景",
                        info="选择背景和环境"
                    )
                with gr.Column():
                    color_tags = gr.CheckboxGroup(
                        choices=PROMPT_CATEGORIES["色彩风格"],
                        label="🎨 色彩风格",
                        info="设定色彩主调"
                    )
            
            # 负面词条
            gr.Markdown("#### 🚫 负面提示词 (Negative Prompt)")
            with gr.Row():
                with gr.Column():
                    neg_quality_tags = gr.CheckboxGroup(
                        choices=NEGATIVE_PROMPT_CATEGORIES["画质问题"],
                        label="🚫 画质问题",
                        info="避免画质相关问题"
                    )
                    neg_anatomy_tags = gr.CheckboxGroup(
                        choices=NEGATIVE_PROMPT_CATEGORIES["解剖错误"],
                        label="🚫 解剖错误",
                        info="避免身体结构错误"
                    )
                with gr.Column():
                    neg_face_tags = gr.CheckboxGroup(
                        choices=NEGATIVE_PROMPT_CATEGORIES["面部问题"],
                        label="🚫 面部问题",
                        info="避免面部相关问题"
                    )
                    neg_style_tags = gr.CheckboxGroup(
                        choices=NEGATIVE_PROMPT_CATEGORIES["艺术风格"],
                        label="🚫 避免风格",
                        info="排除不想要的艺术风格"
                    )
                with gr.Column():
                    neg_tech_tags = gr.CheckboxGroup(
                        choices=NEGATIVE_PROMPT_CATEGORIES["技术问题"],
                        label="🚫 技术问题",
                        info="避免水印、裁剪等技术问题"
                    )
                    neg_lighting_tags = gr.CheckboxGroup(
                        choices=NEGATIVE_PROMPT_CATEGORIES["光照问题"],
                        label="🚫 光照问题",
                        info="避免光照相关问题"
                    )
                with gr.Column():
                    neg_composition_tags = gr.CheckboxGroup(
                        choices=NEGATIVE_PROMPT_CATEGORIES["构图问题"],
                        label="🚫 构图问题",
                        info="避免构图相关问题"
                    )
            
            with gr.Row():
                clear_tags_btn = gr.Button("🗑️ 清空所有选择", variant="secondary")
                apply_positive_tags_btn = gr.Button("✨ 应用正面词条到当前标签页", variant="primary")
                apply_negative_tags_btn = gr.Button("🚫 应用负面词条到当前标签页", variant="secondary")
        
        with gr.Tabs() as tabs:
            # Tab 1: 基础文生图
            with gr.TabItem("📝 文生图模式"):
                with gr.Row():
                    with gr.Column(scale=1):
                        with gr.Row():
                            prompt1 = gr.Textbox(
                                label="提示词 (Prompt)",
                                placeholder="描述你想要的图像，例如：a beautiful landscape with mountains and lakes, highly detailed, 4k",
                                lines=3,
                                scale=4
                            )
                            with gr.Column(scale=1):
                                apply_positive_to_prompt1 = gr.Button("➕ 正面词条", variant="secondary", size="sm")
                                apply_negative_to_prompt1 = gr.Button("➖ 负面词条", variant="secondary", size="sm")
                        
                        negative_prompt1 = gr.Textbox(
                            label="负面提示词 (Negative Prompt)",
                            placeholder="不想要的元素，例如：blurry, low quality, distorted",
                            lines=2
                        )
                        
                        with gr.Row():
                            num_steps1 = gr.Slider(10, 50, value=20, step=1, label="采样步数")
                            guidance_scale1 = gr.Slider(1, 20, value=7.5, step=0.5, label="引导强度")
                        
                        with gr.Row():
                            width1 = gr.Slider(256, 1024, value=512, step=64, label="宽度")
                            height1 = gr.Slider(256, 1024, value=512, step=64, label="高度")
                        
                        seed1 = gr.Number(label="随机种子 (-1为随机)", value=-1)
                        generate_btn1 = gr.Button("🎨 生成图像", variant="primary")
                    
                    with gr.Column(scale=1):
                        output_image1 = gr.Image(label="生成的图像", type="pil")
                        output_status1 = gr.Textbox(label="生成状态")
            
            # Tab 2: 传统图生图
            with gr.TabItem("🔄 传统图生图"):
                # 添加技术说明
                with gr.Accordion("🔬 技术原理说明", open=False):
                    gr.Markdown("""
### 📚 img2img技术原理

**img2img** 是一个复杂的图像处理流程，需要完整的AI模型管道：

🔄 **完整流程**：
1. **VAE编码器** → 将输入图像编码到潜在空间 (latent space)
2. **噪声添加** → 根据strength参数添加不同程度的噪声
3. **UNet采样** → 在潜在空间进行去噪过程，结合文本提示
4. **VAE解码器** → 将潜在表示解码回图像空间

🏠 **本地模式** vs 🌐 **API模式**：
- **本地模式**：✅ 完整支持，包含VAE编码器/解码器
- **API模式**：❌ 限制较大，公共API通常只提供简化接口

💡 **ComfyUI对比**：
ComfyUI使用完整的工作流节点，包含独立的VAE编码器节点，
这正是实现高质量img2img的关键组件。

🎯 **最佳替代方案**：
如果您在API模式下需要类似效果，推荐使用 **ControlNet + Canny边缘检测**
                    """)
                
                with gr.Row():
                    with gr.Column(scale=1):
                        input_image = gr.Image(label="上传输入图像", type="pil")
                        
                        with gr.Row():
                            prompt_img2img = gr.Textbox(
                                label="提示词 (Prompt)",
                                placeholder="描述想要的风格变化，例如：oil painting style, vibrant colors",
                                lines=3,
                                scale=4
                            )
                            with gr.Column(scale=1):
                                apply_positive_to_img2img = gr.Button("➕ 正面词条", variant="secondary", size="sm")
                                apply_negative_to_img2img = gr.Button("➖ 负面词条", variant="secondary", size="sm")
                        
                        negative_prompt_img2img = gr.Textbox(
                            label="负面提示词 (Negative Prompt)",
                            placeholder="不想要的元素",
                            lines=2
                        )
                        
                        strength = gr.Slider(0.1, 1.0, value=0.7, step=0.1, label="变化强度 (越高变化越大)")
                        
                        with gr.Row():
                            num_steps_img2img = gr.Slider(10, 50, value=20, step=1, label="采样步数")
                            guidance_scale_img2img = gr.Slider(1, 20, value=7.5, step=0.5, label="引导强度")
                        
                        with gr.Row():
                            width_img2img = gr.Slider(256, 1024, value=512, step=64, label="宽度")
                            height_img2img = gr.Slider(256, 1024, value=512, step=64, label="高度")
                        
                        seed_img2img = gr.Number(label="随机种子 (-1为随机)", value=-1)
                        
                        with gr.Row():
                            generate_btn_img2img = gr.Button("🔄 传统图生图", variant="secondary", scale=3)
                            test_img2img_api_btn = gr.Button("🧪 测试API", variant="secondary", scale=1, 
                                                           visible=True, size="sm")
                    
                    with gr.Column(scale=1):
                        output_image_img2img = gr.Image(label="生成的图像", type="pil")
                        output_status_img2img = gr.Textbox(label="生成状态")
            
            # Tab 3: ControlNet图像引导
            with gr.TabItem("🖼️ ControlNet模式"):
                with gr.Row():
                    with gr.Column(scale=1):
                        control_image = gr.Image(label="上传控制图像", type="pil")
                        
                        control_type_radio = gr.Radio(
                            choices=[(f"{info['name']} - {info['description']}", key) for key, info in CONTROLNET_TYPES.items()],
                            value="canny",
                            label="🎮 控制类型",
                            info="选择控制方式（需要与加载的ControlNet类型一致）"
                        )
                        
                        with gr.Row():
                            prompt2 = gr.Textbox(
                                label="提示词 (Prompt)",
                                placeholder="基于上传图像的结构，描述想要的风格，例如：oil painting style, sunset colors",
                                lines=3,
                                scale=4
                            )
                            with gr.Column(scale=1):
                                apply_positive_to_prompt2 = gr.Button("➕ 正面词条", variant="secondary", size="sm")
                                apply_negative_to_prompt2 = gr.Button("➖ 负面词条", variant="secondary", size="sm")
                        
                        negative_prompt2 = gr.Textbox(
                            label="负面提示词 (Negative Prompt)",
                            placeholder="不想要的元素",
                            lines=2
                        )
                        
                        with gr.Row():
                            num_steps2 = gr.Slider(10, 50, value=20, step=1, label="采样步数")
                            guidance_scale2 = gr.Slider(1, 20, value=7.5, step=0.5, label="引导强度")
                        
                        controlnet_scale = gr.Slider(0.0, 2.0, value=1.0, step=0.1, label="ControlNet强度")
                        
                        with gr.Row():
                            width2 = gr.Slider(256, 1024, value=512, step=64, label="宽度")
                            height2 = gr.Slider(256, 1024, value=512, step=64, label="高度")
                        
                        seed2 = gr.Number(label="随机种子 (-1为随机)", value=-1)
                        generate_btn2 = gr.Button("🎨 ControlNet生成", variant="primary")
                    
                    with gr.Column(scale=1):
                        with gr.Row():
                            control_preview = gr.Image(label="控制图像预览", type="pil")
                            output_image2 = gr.Image(label="生成的图像", type="pil")
                        output_status2 = gr.Textbox(label="生成状态")
        
        # 示例和对比说明
        gr.Markdown("""
        ## 💡 三种模式对比与使用指南
        
        ### � **运行模式详细对比**
        
        | 运行模式 | 存储空间 | 初始化时间 | 生成速度 | 网络要求 | 成本 | 推荐指数 |
        |----------|----------|------------|----------|----------|------|----------|
        | **🌐 API模式** | **0 GB** | 即时 | 中等 | 需要网络 | 免费额度+付费 | ⭐⭐⭐⭐⭐ |
        | **💻 本地模式** | **4-10 GB** | 5-15分钟 | 快速 | 仅下载时需要 | 硬件成本 | ⭐⭐⭐ |
        
        **🎯 模型存储空间详情 (本地模式)：**
        - 基础模型 (SD v1.5): ~4GB
        - ControlNet 模型: ~1.5GB 每个
        - 高级模型 (SD v2.1): ~5-6GB
        - 完整配置总计: **6-10GB**
        
        **💡 建议：**
        - 🟢 **存储空间紧张** → 选择API模式
        - 🟢 **需要频繁生成** → 选择本地模式  
        - 🟢 **初次体验** → 建议API模式
        
        ### �🔍 **生成模式对比表**
        
        | 模式 | 输入 | 控制方式 | 优势 | 适用场景 |
        |------|------|----------|------|----------|
        | **📝 文生图** | 仅文本 | 提示词 | 完全创新，无限可能 | 原创作品，概念设计 |
        | **🔄 传统图生图** | 图片+文本 | strength参数 | 快速风格转换 | 简单风格化，快速修改 |
        | **🖼️ ControlNet** | 图片+文本 | 精确结构控制 | 保持结构，精确控制 | 建筑重设计，姿态保持 |
        
        ### 📝 **文生图模式示例：**
        - `a majestic dragon flying over a medieval castle, fantasy art, highly detailed`
        - `portrait of a young woman, oil painting style, soft lighting, renaissance art`
        
        ### 🎯 **Prompt 辅助器使用说明：**
        - **✨ 正面词条**：描述你想要的效果、风格、质量等
        - **🚫 负面词条**：描述你不想要的问题、风格、瑕疵等
        - **📝 应用方式**：点击 "➕正面词条" 或 "➖负面词条" 按钮直接替换当前内容
        - **💡 使用技巧**：先选择词条，再点击应用到对应的提示词框中
        
        ### 🔄 **传统图生图 vs 🖼️ ControlNet 详细对比：**
        
        **🔬 传统图生图技术原理：**
        - 🔧 **VAE编码器**：将输入图像编码到潜在空间
        - 🎲 **噪声调制**：根据strength添加不同程度噪声
        - 🔄 **UNet采样**：在潜在空间结合文本进行去噪
        - 🖼️ **VAE解码器**：将潜在表示解码回图像空间
        
        **传统图生图的技术限制：**
        - 🔸 **API模式受限**：需要完整VAE编码器，公共API通常不提供
        - 🔸 **结构不稳定**：噪声调制可能破坏重要结构信息
        - 🔸 **参数敏感**：strength参数难以精确控制变化程度
        - 🔸 **ComfyUI优势**：通过独立VAE节点实现精确控制
        
        **🌐 API模式 vs 🏠 本地模式：**
        - **API模式**：❌ 缺少VAE编码器访问，无法进行潜在空间操作
        - **本地模式**：✅ 完整img2img管道，包含独立VAE编码器/解码器
        
        **ControlNet的技术优势：**
        - ✅ **结构保持**：通过边缘、深度等控制信号保持结构
        - ✅ **API兼容**：可通过预处理图像+文生图实现
        - ✅ **可预测性**：相同控制信号产生一致结果
        - ✅ **高保真度**：保持原图关键特征的同时进行风格转换
        
        ### 🛠️ **参数调节建议：**
        - **采样步数**：20-30 (质量与速度平衡)
        - **引导强度**：7-12 (文本描述影响力)
        - **变化强度**(传统图生图)：0.6-0.8 (保留原图程度)
        - **ControlNet强度**：0.8-1.2 (结构控制强度)
        """)
        
        # 绑定事件
        
        # API Token 设置事件
        def update_api_token(token):
            global HF_API_TOKEN
            HF_API_TOKEN = token.strip() if token else None
            return f"🔑 API Token {'已设置' if token else '未设置'}"
        
        # 运行模式切换事件 - 更新模型选择器和显示
        def update_run_mode_and_models(mode):
            global RUN_MODE
            RUN_MODE = mode
            mode_text = "🌐 API模式" if mode == "api" else "💻 本地模式"
            storage_text = "存储占用: 0 GB" if mode == "api" else "存储占用: 4-10 GB"
            status_text = f"⚙️ {mode_text}\n💾 {storage_text}"
            
            # 同时更新模型选择器
            model_update = update_model_choices(mode)
            return status_text, model_update
        
        run_mode_radio.change(
            update_run_mode_and_models,
            inputs=[run_mode_radio],
            outputs=[current_model_display, model_dropdown]
        )
        
        # 代理设置事件
        def update_proxy_settings(enabled, http_proxy, https_proxy):
            status = update_proxy_config(enabled, http_proxy, https_proxy)
            return status
        
        proxy_enabled.change(
            update_proxy_settings,
            inputs=[proxy_enabled, http_proxy_input, https_proxy_input],
            outputs=[proxy_status]
        )
        
        http_proxy_input.change(
            update_proxy_settings,
            inputs=[proxy_enabled, http_proxy_input, https_proxy_input],
            outputs=[proxy_status]
        )
        
        https_proxy_input.change(
            update_proxy_settings,
            inputs=[proxy_enabled, http_proxy_input, https_proxy_input],
            outputs=[proxy_status]
        )
        
        # GitHub 推送事件
        push_to_github_btn.click(
            auto_push_to_github,
            inputs=[],
            outputs=[github_status]
        )
        
        # API Token 实时验证
        api_token_input.change(
            validate_api_key,
            inputs=[api_token_input],
            outputs=[token_status]
        )
        
        # 模型API支持检测
        def update_model_api_status(model_id, run_mode):
            return check_model_api_support(model_id, run_mode)
        
        model_dropdown.change(
            update_model_api_status,
            inputs=[model_dropdown, run_mode_radio],
            outputs=[model_api_status]
        )
        
        run_mode_radio.change(
            update_model_api_status,
            inputs=[model_dropdown, run_mode_radio],
            outputs=[model_api_status]
        )
        
        # API连接测试
        test_api_btn.click(
            test_model_api_connection,
            inputs=[model_dropdown, api_token_input],
            outputs=[model_api_status]
        )
        
        # ControlNet API连接测试
        test_controlnet_api_btn.click(
            test_controlnet_api_connection,
            inputs=[controlnet_test_dropdown, api_token_input],
            outputs=[controlnet_api_status]
        )
        
        # 模型加载事件
        load_btn.click(
            load_models, 
            inputs=[run_mode_radio, model_dropdown, controlnet_dropdown, api_token_input], 
            outputs=[load_status]
        )
        
        # 更新当前模型显示
        model_dropdown.change(
            lambda x: f"📦 选中模型: {MODELS.get(x, x)}",
            inputs=[model_dropdown],
            outputs=[current_model_display]
        )
        
        # Prompt 辅助器事件
        def get_selected_positive_tags(*tag_groups):
            """获取所有选中的正面标签"""
            selected_tags = []
            for tags in tag_groups:
                if tags:
                    selected_tags.extend(tags)
            return ", ".join(selected_tags) if selected_tags else ""
        
        def get_selected_negative_tags(*tag_groups):
            """获取所有选中的负面标签"""
            selected_tags = []
            for tags in tag_groups:
                if tags:
                    selected_tags.extend(tags)
            return ", ".join(selected_tags) if selected_tags else ""
        
        def clear_all_tags():
            return [[] for _ in range(14)]  # 7个正面tag组 + 7个负面tag组
        
        # 正面词条应用到各个prompt框的事件
        apply_positive_to_prompt1.click(
            get_selected_positive_tags,
            inputs=[quality_tags, style_tags, lighting_tags, composition_tags, mood_tags, scene_tags, color_tags],
            outputs=[prompt1]
        )
        
        apply_positive_to_img2img.click(
            get_selected_positive_tags,
            inputs=[quality_tags, style_tags, lighting_tags, composition_tags, mood_tags, scene_tags, color_tags],
            outputs=[prompt_img2img]
        )
        
        apply_positive_to_prompt2.click(
            get_selected_positive_tags,
            inputs=[quality_tags, style_tags, lighting_tags, composition_tags, mood_tags, scene_tags, color_tags],
            outputs=[prompt2]
        )
        
        # 负面词条应用到各个negative prompt框的事件
        apply_negative_to_prompt1.click(
            get_selected_negative_tags,
            inputs=[neg_quality_tags, neg_anatomy_tags, neg_face_tags, neg_style_tags, neg_tech_tags, neg_lighting_tags, neg_composition_tags],
            outputs=[negative_prompt1]
        )
        
        apply_negative_to_img2img.click(
            get_selected_negative_tags,
            inputs=[neg_quality_tags, neg_anatomy_tags, neg_face_tags, neg_style_tags, neg_tech_tags, neg_lighting_tags, neg_composition_tags],
            outputs=[negative_prompt_img2img]
        )
        
        apply_negative_to_prompt2.click(
            get_selected_negative_tags,
            inputs=[neg_quality_tags, neg_anatomy_tags, neg_face_tags, neg_style_tags, neg_tech_tags, neg_lighting_tags, neg_composition_tags],
            outputs=[negative_prompt2]
        )
        
        # 全局应用按钮事件（兼容性保留）
        apply_positive_tags_btn.click(
            get_selected_positive_tags,
            inputs=[quality_tags, style_tags, lighting_tags, composition_tags, mood_tags, scene_tags, color_tags],
            outputs=[]
        )
        
        apply_negative_tags_btn.click(
            get_selected_negative_tags,
            inputs=[neg_quality_tags, neg_anatomy_tags, neg_face_tags, neg_style_tags, neg_tech_tags, neg_lighting_tags, neg_composition_tags],
            outputs=[]
        )
        
        clear_tags_btn.click(
            clear_all_tags,
            outputs=[quality_tags, style_tags, lighting_tags, composition_tags, mood_tags, scene_tags, color_tags,
                    neg_quality_tags, neg_anatomy_tags, neg_face_tags, neg_style_tags, neg_tech_tags, neg_lighting_tags, neg_composition_tags]
        )
        
        # 原有的生成事件
        generate_btn1.click(
            generate_image,
            inputs=[prompt1, negative_prompt1, num_steps1, guidance_scale1, width1, height1, seed1],
            outputs=[output_image1, output_status1]
        )
        
        generate_btn_img2img.click(
            generate_img2img,
            inputs=[prompt_img2img, negative_prompt_img2img, input_image, strength, num_steps_img2img, guidance_scale_img2img, width_img2img, height_img2img, seed_img2img],
            outputs=[output_image_img2img, output_status_img2img]
        )
        
        # img2img API测试按钮
        test_img2img_api_btn.click(
            test_img2img_api_connection,
            inputs=[api_token_input],
            outputs=[output_status_img2img]
        )
        
        generate_btn2.click(
            generate_controlnet_image,
            inputs=[prompt2, negative_prompt2, control_image, control_type_radio, num_steps2, guidance_scale2, controlnet_scale, width2, height2, seed2],
            outputs=[output_image2, control_preview, output_status2]
        )
        
        # 更新模型选择器
        run_mode_radio.change(
            update_model_choices,
            inputs=[run_mode_radio],
            outputs=[model_dropdown]
        )
        
        return demo

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
    """Generate ControlNet image using API - 修复版本"""
    endpoint = CONTROLNET_API_ENDPOINTS.get(control_type)
    if not endpoint:
        raise Exception(f"ControlNet type {control_type} does not support API mode")
    
    # 检查API Token
    if not HF_API_TOKEN or not HF_API_TOKEN.strip():
        raise Exception("ControlNet API requires a valid Hugging Face API Token. Please set your token in the API settings.")
    
    # Convert control image to base64
    import base64
    import io
    
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
    
    # 使用经过测试验证的 ControlNet API 格式
    # 根据测试结果，使用简单的inputs格式
    payload = {
        "inputs": safe_prompt,
        "parameters": {
            "image": control_image_b64,
            "negative_prompt": safe_negative_prompt,
            "num_inference_steps": 20,
            "guidance_scale": 7.5
        }
    }
    
    try:
        image_bytes = query_hf_api(endpoint, payload, HF_API_TOKEN)
        image = Image.open(io.BytesIO(image_bytes))
        control_type_name = CONTROLNET_TYPES[control_type]['name']
        return image, f"✅ API模式 {control_type_name} 图像生成成功！"
    except Exception as e:
        error_msg = str(e)
        if "Model endpoint not found" in error_msg or "404" in error_msg:
            return None, f"❌ ControlNet模型 {control_type} 端点不可用。建议：1) 检查网络连接 2) 尝试其他控制类型 3) 使用本地模式"
        elif "401" in error_msg or "Invalid" in error_msg or "credentials" in error_msg.lower():
            return None, f"❌ API Token无效或未设置。请在API设置中输入有效的 Hugging Face Token"
        elif "503" in error_msg or "loading" in error_msg.lower():
            return None, f"⏳ ControlNet模型正在加载，请稍等1-2分钟后重试"
        elif "timeout" in error_msg.lower():
            return None, f"⏰ 连接超时，请检查网络连接或启用代理设置"
        else:
            return None, f"❌ ControlNet API调用失败: {error_msg}"

def generate_img2img_api(prompt, negative_prompt, input_image, strength):
    """Generate img2img image using API - 技术说明版本"""
    
    # 技术原理说明：img2img需要完整的VAE编码/解码流程
    # 1. VAE编码器将图像编码到潜在空间
    # 2. 根据strength添加噪声
    # 3. UNet进行去噪采样
    # 4. VAE解码器解码回图像
    # 这个复杂流程不适合简化的API调用
    
    
    # 检查API Token
    if not HF_API_TOKEN or not HF_API_TOKEN.strip():
        return None, "❌ img2img API需要有效的 Hugging Face API Token"
    
    # 技术限制说明
    return None, f"""❌ img2img API功能受限

🔬 技术原理说明:
img2img需要完整的VAE编码/解码流程：
1. VAE编码器：将输入图像编码到潜在空间
2. 噪声调制：根据strength参数添加噪声
3. UNet采样：在潜在空间进行去噪
4. VAE解码器：将结果解码回图像空间

🚫 API限制:
• 公共API通常只提供简化的text-to-image端点
• img2img需要完整的模型管道（VAE+UNet+调度器）
• 复杂的潜在空间操作不适合API封装

💡 推荐解决方案:
1. 🏠 本地模式 - 完整支持img2img流程
2. 🖼️ ControlNet模式 - 可实现类似的图像引导效果：
   • Canny边缘检测 + 文生图
   • 保持原图结构，改变风格
   • API模式完全支持
3. � Inpainting模式 - 局部修改（如果支持）

🎯 ComfyUI对比:
您提到的ComfyUI确实是完整的本地工作流，
包含完整的VAE编码器，这正是API模式缺少的部分。"""

def update_model_choices(run_mode):
    """根据运行模式动态更新模型选择器"""
    available_models = get_available_models(run_mode)
    
    if run_mode == "api":
        # API模式：只显示支持API的模型，按推荐程度排序
        recommended_order = [
            "black-forest-labs/FLUX.1-dev",
            "black-forest-labs/FLUX.1-schnell", 
            "stabilityai/stable-diffusion-xl-base-1.0",
            "stabilityai/stable-diffusion-3.5-large",
            "stabilityai/stable-diffusion-3-medium-diffusers",
            "latent-consistency/lcm-lora-sdxl",
            "Kwai-Kolors/Kolors",
            "runwayml/stable-diffusion-v1-5",
            "stabilityai/stable-diffusion-2-1",
            "prompthero/openjourney",
            "dreamlike-art/dreamlike-diffusion-1.0"
        ]
        
        choices = []
        for model_id in recommended_order:
            if model_id in available_models:
                model_name = available_models[model_id]
                choices.append((model_name, model_id))
        
        # 默认选择第一个推荐模型
        default_value = recommended_order[0] if recommended_order else "black-forest-labs/FLUX.1-dev"
        
        return gr.Dropdown.update(
            choices=choices,
            value=default_value,
            label="🤖 选择基础模型 (仅API支持的模型)",
            info="✅ API模式 - 这些模型支持云端推理，无需下载"
        )
    else:
        # 本地模式：显示所有模型
        choices = [(name, model_id) for model_id, name in available_models.items()]
        return gr.Dropdown.update(
            choices=choices,
            value="runwayml/stable-diffusion-v1-5",
            label="🤖 选择基础模型 (支持所有模型)",
            info="💾 本地模式 - 首次使用需要下载模型文件（4-10GB）"
        )

# 主函数：启动Gradio应用 - 带自动清理功能
if __name__ == "__main__":
    print("🎨 启动 AI 图像生成器...")
    print("=" * 60)
    
    # 设置信号处理器
    setup_signal_handlers()
    
    print("🚀 正在初始化界面...")
    
    # 创建界面
    demo = create_interface()
    gradio_app = demo  # 保存到全局变量
    
    print("✅ 界面初始化完成！")
    print("🌐 正在启动服务器...")
    print("=" * 60)
    
    # 智能端口分配
    ports_to_try = [7860, 7861, 7862, 7863, 7864]
    
    for port in ports_to_try:
        try:
            print(f"🔄 尝试启动在端口 {port}...")
            current_port = port  # 保存当前端口到全局变量
            
            demo.launch(
                server_name="0.0.0.0",
                server_port=port,
                share=False,
                inbrowser=True,
                show_error=True,
                debug=False
            )
            break
            
        except (OSError, Exception) as e:
            if "Address already in use" in str(e) or "10048" in str(e) or "Cannot find empty port" in str(e):
                print(f"⚠️ 端口 {port} 被占用，尝试下一个端口...")
                continue
            else:
                print(f"❌ 启动失败: {e}")
                cleanup_resources()
                sys.exit(1)
        except KeyboardInterrupt:
            print("\n🛑 用户中断启动")
            cleanup_resources()
            sys.exit(0)
    
    # 程序结束时自动清理
    cleanup_resources()