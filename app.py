"""
主应用文件 - AI图像生成器界面
重构后的模块化版本
"""

import gradio as gr
import warnings

# 导入自定义模块
from config import CONTROLNET_TYPES, PROMPT_CATEGORIES, NEGATIVE_PROMPT_CATEGORIES, API_SUPPORTED_MODELS, MODELS, update_proxy_config
from models import load_models, get_current_model_info
from image_generation import generate_image, generate_controlnet_image, generate_img2img, add_prompt_tags
from api_client import validate_api_key, check_model_api_support, test_model_api_connection, set_api_token
from utils import auto_push_to_github, test_proxy_connection, update_model_choices, setup_cleanup_handlers, find_free_port
import utils  # 导入utils模块以便访问全局变量

warnings.filterwarnings("ignore")

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
                    choices=list(API_SUPPORTED_MODELS.keys()),
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
                    test_api_btn = gr.Button("🔗 测试API连接", variant="secondary")
                
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
        
        ### 🌐 **运行模式详细对比**
        
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
        
        ### 🔍 **生成模式对比表**
        
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
            set_api_token(token)
            return f"🔑 API Token {'已设置' if token else '未设置'}"
        
        # 运行模式切换事件 - 更新模型选择器和显示
        def update_run_mode_and_models(mode):
            mode_text = "🌐 API模式" if mode == "api" else "💻 本地模式"
            storage_text = "存储占用: 0 GB" if mode == "api" else "存储占用: 4-10 GB"
            status_text = f"⚙️ {mode_text}\n💾 {storage_text}"
            
            # 同时更新模型选择器
            model_choices_info = update_model_choices(mode)
            return status_text, gr.Dropdown.update(**model_choices_info)
        
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
        
        # 清空标签
        clear_tags_btn.click(
            clear_all_tags,
            outputs=[quality_tags, style_tags, lighting_tags, composition_tags, mood_tags, scene_tags, color_tags,
                    neg_quality_tags, neg_anatomy_tags, neg_face_tags, neg_style_tags, neg_tech_tags, neg_lighting_tags, neg_composition_tags]
        )
        
        # 图像生成事件
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

# 主函数：启动Gradio应用
if __name__ == "__main__":
    print("🎨 启动 AI 图像生成器...")
    print("=" * 60)
    print("🚀 正在初始化界面...")
    
    # 设置自动端口释放机制
    print("🛡️ 设置自动端口释放机制...")
    setup_cleanup_handlers()
    
    # 寻找可用端口
    available_port = find_free_port(7861)
    
    # 创建并启动界面
    demo = create_interface()
    
    # 设置全局变量，用于清理函数
    utils.demo_instance = demo
    utils.server_port = available_port
    
    print("✅ 界面初始化完成！")
    print(f"🌐 正在启动服务器，端口: {available_port}")
    print("💡 程序退出时将自动释放端口")
    print("=" * 60)
    
    try:
        # 启动Gradio应用
        demo.launch(
            server_name="0.0.0.0",        # 允许外部访问
            server_port=available_port,    # 使用找到的可用端口
            share=False,                   # 不使用公共链接
            inbrowser=True,                # 自动打开浏览器
            show_error=True,               # 显示错误信息
            debug=False                    # 生产模式
        )
    except KeyboardInterrupt:
        print("\n🛑 收到键盘中断信号...")
        utils.cleanup_on_exit()
    except Exception as e:
        print(f"\n❌ 启动失败: {e}")
        utils.cleanup_on_exit()
    finally:
        print("\n🔄 程序结束，确保资源清理...")
        utils.cleanup_on_exit()
