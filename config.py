"""
配置模块 - 存储应用的配置信息和常量
"""

import torch

# 设备配置
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 默认运行模式
DEFAULT_RUN_MODE = "api"  # "local" 或 "api"

# 代理设置
PROXY_CONFIG = {
    "enabled": False,
    "http": None,
    "https": None
}

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

# ControlNet API endpoints
CONTROLNET_API_ENDPOINTS = {
    "canny": "https://api-inference.huggingface.co/models/lllyasviel/sd-controlnet-canny",
    "scribble": "https://api-inference.huggingface.co/models/lllyasviel/sd-controlnet-scribble", 
    "depth": "https://api-inference.huggingface.co/models/lllyasviel/sd-controlnet-depth"
}

# ControlNet 类型配置
CONTROLNET_TYPES = {
    "canny": {
        "name": "Canny边缘检测",
        "model_id": "lllyasviel/sd-controlnet-canny",
        "description": "检测图像边缘轮廓，保持物体形状"
    },
    "scribble": {
        "name": "Scribble涂鸦控制",
        "model_id": "lllyasviel/sd-controlnet-scribble", 
        "description": "基于手绘涂鸦或简笔画生成图像"
    },
    "depth": {
        "name": "Depth深度控制",
        "model_id": "lllyasviel/sd-controlnet-depth",
        "description": "基于深度图控制空间结构和层次"
    }
}

# API支持的模型
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

# 根据运行模式动态获取可用模型
def get_available_models(run_mode):
    """根据运行模式获取可用模型列表"""
    if run_mode == "api":
        return API_SUPPORTED_MODELS
    else:
        # 本地模式支持所有模型
        return {**API_SUPPORTED_MODELS, **LOCAL_ONLY_MODELS}

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
