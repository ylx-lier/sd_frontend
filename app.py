import gradio as gr
import torch
from diffusers import StableDiffusionPipeline, StableDiffusionControlNetPipeline, ControlNetModel
from diffusers import StableDiffusionImg2ImgPipeline, DPMSolverMultistepScheduler
import cv2
import numpy as np
from PIL import Image
import warnings
import requests
import io
import base64
import subprocess
import os
from datetime import datetime

warnings.filterwarnings("ignore")

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

# API模式下的推理端点
API_ENDPOINTS = {
    "runwayml/stable-diffusion-v1-5": "https://api-inference.huggingface.co/models/runwayml/stable-diffusion-v1-5",
    "stabilityai/stable-diffusion-2-1": "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-2-1",
    "dreamlike-art/dreamlike-diffusion-1.0": "https://api-inference.huggingface.co/models/dreamlike-art/dreamlike-diffusion-1.0",
    "prompthero/openjourney": "https://api-inference.huggingface.co/models/prompthero/openjourney",
}

# ControlNet API endpoints
CONTROLNET_API_ENDPOINTS = {
    "canny": "https://api-inference.huggingface.co/models/lllyasviel/sd-controlnet-canny",
    "scribble": "https://api-inference.huggingface.co/models/lllyasviel/sd-controlnet-scribble", 
    "depth": "https://api-inference.huggingface.co/models/lllyasviel/sd-controlnet-depth"
}

# ControlNet 类型选项
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

# 预定义模型列表
MODELS = {
    "runwayml/stable-diffusion-v1-5": "Stable Diffusion v1.5 (标准模型)",
    "stabilityai/stable-diffusion-2-1": "Stable Diffusion v2.1 (更高质量)",
    "dreamlike-art/dreamlike-diffusion-1.0": "Dreamlike Diffusion (艺术风格)",
    "prompthero/openjourney": "OpenJourney (多样化风格)",
    "wavymulder/Analog-Diffusion": "Analog Diffusion (胶片风格)",
    "22h/vintedois-diffusion-v0-1": "VintedoisDiffusion (复古风格)",
    "nitrosocke/Arcane-Diffusion": "Arcane Diffusion (动画风格)",
    "hakurei/waifu-diffusion": "Waifu Diffusion (动漫风格)"
}

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
    """加载模型管道"""
    global pipe, controlnet_pipe, img2img_pipe, current_model, current_controlnet, RUN_MODE, HF_API_TOKEN
    
    if not selected_model:
        return "❌ 请选择一个模型"
    
    # 更新全局配置
    RUN_MODE = run_mode
    current_model = selected_model
    if api_token.strip():
        HF_API_TOKEN = api_token.strip()
    model_name = MODELS.get(selected_model, selected_model)
    
    if run_mode == "api":
        # API模式 - 不下载模型
        if selected_model not in API_ENDPOINTS:
            return f"❌ 模型 {model_name} 不支持API模式\n💡 请选择支持API的模型或切换到本地模式"
        
        # 模拟加载成功
        pipe = "api_mode"
        img2img_pipe = "api_mode" 
        controlnet_pipe = "api_mode"
        current_controlnet = controlnet_type
        
        api_status = "🌐 API模式 - 无需下载模型" if not HF_API_TOKEN else "🌐 API模式 - 使用认证Token"
        return f"✅ API模式配置成功！\n📦 当前模型: {model_name}\n🎯 模型ID: {selected_model}\n🎮 ControlNet: {CONTROLNET_TYPES[controlnet_type]['name']}\n{api_status}\n💾 存储空间占用: 0 GB"
    
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
    """传统图生图功能"""
    global img2img_pipe, RUN_MODE
    
    if img2img_pipe is None:
        return None, "❌ 请先加载模型"
    
    if input_image is None:
        return None, "❌ 请上传输入图像"
    
    # 调整图像大小
    input_image = input_image.resize((width, height))
    
    if RUN_MODE == "api":
        # API模式
        try:
            image, status = generate_img2img_api(prompt, negative_prompt, input_image, strength)
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
            return image, "✅ 传统图生图成功！"
            
        except Exception as e:
            return None, f"❌ 生成失败: {str(e)}"

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
                    choices=list(MODELS.keys()),
                    value="runwayml/stable-diffusion-v1-5",
                    label="🤖 选择基础模型",
                    info="选择不同的预训练模型以获得不同的艺术风格"
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
                        generate_btn_img2img = gr.Button("🔄 传统图生图", variant="secondary")
                    
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
        
        **传统图生图的问题：**
        - 🔸 结构不稳定：同样参数可能产生完全不同结果
        - 🔸 strength难调：太高丢失原图，太低改变不够
        - 🔸 细节丢失：容易失去重要的结构信息
        
        **ControlNet的优势：**
        - ✅ 精确控制：保留边缘、深度、姿态等结构信息
        - ✅ 可预测性：相同输入产生一致结果
        - ✅ 高保真度：保持原图关键特征的同时进行风格转换
        
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
        
        # 运行模式切换事件  
        def update_run_mode(mode):
            global RUN_MODE
            RUN_MODE = mode
            mode_text = "🌐 API模式" if mode == "api" else "💻 本地模式"
            storage_text = "存储占用: 0 GB" if mode == "api" else "存储占用: 4-10 GB"
            return f"⚙️ {mode_text}\n💾 {storage_text}"
        
        api_token_input.change(
            update_api_token,
            inputs=[api_token_input],
            outputs=[]
        )
        
        run_mode_radio.change(
            update_run_mode,
            inputs=[run_mode_radio],
            outputs=[]
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
        
        generate_btn2.click(
            generate_controlnet_image,
            inputs=[prompt2, negative_prompt2, control_image, control_type_radio, num_steps2, guidance_scale2, controlnet_scale, width2, height2, seed2],
            outputs=[output_image2, control_preview, output_status2]
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
    """Generate ControlNet image using API"""
    endpoint = CONTROLNET_API_ENDPOINTS.get(control_type)
    if not endpoint:
        raise Exception(f"ControlNet type {control_type} does not support API mode")
    
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
    import base64
    import io
    
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
    
    # Note: This is a simplified implementation, real img2img API may need different payload format
    payload = {
        "inputs": {
            "prompt": safe_prompt,
            "image": input_image_b64,
            "negative_prompt": safe_negative_prompt,
            "strength": strength
        }
    }
    
    try:
        # Note: Since Hugging Face public API has limited img2img support, this may fail
        # Users are recommended to use text-to-image function in API mode
        image_bytes = query_hf_api(endpoint, payload, HF_API_TOKEN)
        image = Image.open(io.BytesIO(image_bytes))
        return image, "API mode img2img image generation successful!"
    except Exception as e:
        return None, f"img2img API not supported, recommend using local mode or text-to-image function: {str(e)}"
    
    # 将输入图像转换为base64
    import base64
    import io
    
    buffered = io.BytesIO()
    input_image.save(buffered, format="PNG")
    input_image_b64 = base64.b64encode(buffered.getvalue()).decode()
    
    # 注意：这是一个简化的实现，真实的img2img API可能需要不同的payload格式
    payload = {
        "inputs": {
            "prompt": prompt,
            "image": input_image_b64,
            "negative_prompt": negative_prompt if negative_prompt else "",
            "strength": strength
        }
    }
    
    try:
        # 注意：由于Hugging Face公共API对img2img支持有限，这里可能会失败
        # 建议用户在API模式下优先使用文生图功能
        image_bytes = query_hf_api(endpoint, payload, HF_API_TOKEN)
        image = Image.open(io.BytesIO(image_bytes))
        return image, "✅ API模式 img2img 图像生成成功！"
    except Exception as e:
        return None, f"❌ img2img API暂不支持，建议使用本地模式或文生图功能: {str(e)}"

if __name__ == "__main__":
    demo = create_interface()
    demo.launch(debug=True)