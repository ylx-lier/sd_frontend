"""
模型管理模块 - 处理本地模型的加载和管理
"""

import torch
from diffusers import StableDiffusionPipeline, StableDiffusionControlNetPipeline, ControlNetModel
from diffusers import StableDiffusionImg2ImgPipeline, DPMSolverMultistepScheduler
from config import DEVICE, CONTROLNET_TYPES, get_available_models, API_SUPPORTED_MODELS, API_ENDPOINTS

# 全局变量存储管道
pipe = None
controlnet_pipe = None
img2img_pipe = None
current_model = "runwayml/stable-diffusion-v1-5"
current_controlnet = None
RUN_MODE = "api"

def get_current_model_info():
    """获取当前模型信息"""
    global current_model
    if current_model:
        from config import MODELS
        model_name = MODELS.get(current_model, current_model)
        return f"📦 当前模型: {model_name}"
    else:
        return "❌ 未加载模型"

def load_models(run_mode, selected_model, controlnet_type="canny", api_token=""):
    """加载模型管道 - 改进版本，支持API模型检测"""
    global pipe, controlnet_pipe, img2img_pipe, current_model, current_controlnet, RUN_MODE
    
    if not selected_model:
        return "❌ 请选择一个模型"
    
    # 更新全局配置
    RUN_MODE = run_mode
    current_model = selected_model
    
    # 设置API Token
    if api_token.strip():
        from api_client import set_api_token
        set_api_token(api_token.strip())
    
    # 获取模型信息
    available_models = get_available_models(run_mode)
    model_name = available_models.get(selected_model, selected_model)
    
    if run_mode == "api":
        # API模式 - 检查模型支持
        if selected_model not in API_ENDPOINTS:
            supported_models = list(API_SUPPORTED_MODELS.keys())
            recommended = supported_models[:3]  # 推荐前3个
            
            return f"❌ 模型 {model_name} 不支持API模式\n\n🌟 推荐支持API的模型:\n" + \
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
            quality_tip = "\n🎨 SDXL系列 - 高分辨率生成，经典选择"
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
                torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
                safety_checker=None,
                requires_safety_checker=False
            )
            pipe = pipe.to(DEVICE)
            pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
            
            # 传统图生图管道
            img2img_pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
                selected_model,
                torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
                safety_checker=None,
                requires_safety_checker=False
            )
            img2img_pipe = img2img_pipe.to(DEVICE)
            img2img_pipe.scheduler = DPMSolverMultistepScheduler.from_config(img2img_pipe.scheduler.config)
            
            # ControlNet 管道
            try:
                current_controlnet = controlnet_type
                controlnet_info = CONTROLNET_TYPES[controlnet_type]
                
                controlnet = ControlNetModel.from_pretrained(
                    controlnet_info["model_id"],
                    torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32
                )
                controlnet_pipe = StableDiffusionControlNetPipeline.from_pretrained(
                    selected_model,
                    controlnet=controlnet,
                    torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
                    safety_checker=None,
                    requires_safety_checker=False
                )
                controlnet_pipe = controlnet_pipe.to(DEVICE)
                controlnet_pipe.scheduler = DPMSolverMultistepScheduler.from_config(controlnet_pipe.scheduler.config)
                return f"✅ 本地模式所有模型加载成功！\n📦 当前模型: {model_name}\n🎯 模型ID: {selected_model}\n🎮 ControlNet: {controlnet_info['name']}\n💾 预计存储占用: ~6-10 GB"
            except Exception as controlnet_error:
                return f"✅ 本地模式基础模型加载成功！\n📦 当前模型: {model_name}\n🎯 模型ID: {selected_model}\n⚠️ ControlNet加载失败: {str(controlnet_error)}\n💡 文生图和传统图生图功能可正常使用\n💾 预计存储占用: ~4-7 GB"
            
        except Exception as e:
            return f"❌ 本地模式加载失败: {str(e)}\n💡 建议尝试API模式以避免存储空间问题"

def is_model_loaded():
    """检查模型是否已加载"""
    return pipe is not None

def get_model_mode():
    """获取当前模型运行模式"""
    return RUN_MODE

def get_current_model():
    """获取当前模型ID"""
    return current_model

def get_current_controlnet():
    """获取当前ControlNet类型"""
    return current_controlnet
