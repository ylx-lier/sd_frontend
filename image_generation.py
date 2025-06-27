"""
图像生成模块 - 处理各种图像生成功能
"""

import torch
import cv2
import numpy as np
from PIL import Image
from models import pipe, controlnet_pipe, img2img_pipe, current_model, current_controlnet, RUN_MODE
from config import DEVICE, CONTROLNET_TYPES
from api_client import generate_image_api, generate_controlnet_image_api, generate_img2img_api

def generate_image(prompt, negative_prompt, num_steps, guidance_scale, width, height, seed):
    """基础文生图功能"""
    from models import pipe, current_model, RUN_MODE
    
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
                generator = torch.Generator(device=DEVICE).manual_seed(seed)
            else:
                generator = None
                
            # 生成图像
            with torch.autocast(DEVICE):
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
    from models import controlnet_pipe, current_controlnet, RUN_MODE
    
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
                generator = torch.Generator(device=DEVICE).manual_seed(seed)
            else:
                generator = None
                
            # 生成图像
            with torch.autocast(DEVICE):
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
    from models import img2img_pipe, RUN_MODE
    
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
                generator = torch.Generator(device=DEVICE).manual_seed(seed)
            else:
                generator = None
                
            # 生成图像
            with torch.autocast(DEVICE):
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
