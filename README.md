# 🎨 AI 图像生成器 Pro

一个基于 Stable Diffusion 和 ControlNet 的智能图像生成应用，**支持零存储空间的API模式**和本地部署两种方式。

## ✨ 核心特性

### 🌐 **双模式运行**
- **API模式 (推荐)**: **完全零存储占用**，通过云端API生成图像
- **💻 本地模式**: 下载模型到本地，需要6-10GB存储空间

### 🖼️ **三种生成模式**
- **📝 文生图模式**：纯文本描述生成图像  
- **🔄 传统图生图**：图像风格转换
- **🖼️ ControlNet模式**：精确结构控制的图像生成

### 🎯 **智能功能**
- **多模型选择**：支持8种不同风格的预训练模型
- **ControlNet控制**：Canny边缘/Scribble涂鸦/Depth深度三种控制方式
- **Prompt辅助器**：7大类别的提示词快速选择
- **参数优化**：采样步数、引导强度、分辨率等精细控制

## 💾 **存储空间对比**

| 运行模式 | 存储占用 | 模型下载 | 网络要求 | 推荐指数 |
|----------|----------|----------|----------|----------|
| **🌐 API模式** | **0 GB** | 无需下载 | 需要网络连接 | ⭐⭐⭐⭐⭐ |
| **💻 本地模式** | **6-10 GB** | 需要下载 | 仅下载时需要 | ⭐⭐⭐ |

**💡 API模式优势：**
- ✅ **零存储占用** - 完全无需下载模型文件
- ✅ **即开即用** - 无需等待模型下载和加载
- ✅ **自动更新** - 始终使用最新版本的模型
- ✅ **多设备友好** - 任何设备都可以运行
- ✅ **免费使用** - Hugging Face提供免费API额度

## 🛠️ 技术栈

- **模型**: Stable Diffusion v1.5 + ControlNet (Canny)
- **框架**: Diffusers + Gradio
- **部署**: Hugging Face Spaces

## 🚀 快速开始

### 🌐 API模式（推荐 - 零存储占用）

```bash
# 克隆项目
git clone <your-repo-url>
cd diffusion-app

# 安装依赖
pip install -r requirements.txt

# 直接运行 - 无需下载任何模型！
python app.py
```

**🔑 获取免费API Token（可选但推荐）：**
1. 访问 [Hugging Face Tokens](https://huggingface.co/settings/tokens)
2. 创建新Token（Read权限即可）
3. 在界面中输入Token以获得更稳定的服务

### 💻 本地模式（需要6-10GB存储空间）

### 💻 本地模式（需要6-10GB存储空间）

```bash
# 克隆项目
git clone <your-repo-url>
cd diffusion-app

# 安装依赖
pip install -r requirements.txt

# 运行应用
python app.py
```

## 💡 **API模式 vs 本地模式详细对比**

### 🌐 **API模式（零存储占用）**

**存储空间：**
- ✅ **0 GB** - 完全无需下载模型
- ✅ 无需清理缓存文件
- ✅ 适合存储空间有限的设备

**数据传输：**
- 📤 **仅上传**：文本prompt、控制图像（几KB-几MB）
- 📥 **仅下载**：生成的图像（1-5MB）
- 🌐 **总流量**：每次生成约5-10MB

**性能特点：**
- ⚡ 启动速度：即时（无需加载模型）
- 🔄 生成速度：取决于网络和API队列
- 💰 成本：免费额度 + 可选付费提升

### 💻 **本地模式（大存储占用）**

**存储空间：**
- 📦 **6-10 GB** - 需要下载完整模型
- 📂 存储位置：`C:\Users\{用户名}\.cache\huggingface\hub`
- 🗂️ 包含：基础模型(4GB) + ControlNet模型(1.5GB×3种类型)

**性能特点：**
- 🐌 首次启动：需要5-15分钟下载模型
- ⚡ 后续启动：需要2-5分钟加载模型到显存
- 🚀 生成速度：本地GPU加速，速度快
- 💰 成本：仅硬件成本，无API费用

### Hugging Face Spaces 部署

1. 登录 [Hugging Face](https://huggingface.co/spaces)
2. 点击 "Create new Space"
3. 选择 **Gradio** 类型
4. 上传项目文件
5. 等待构建完成

## 📋 使用说明

### 🌐 API模式使用步骤
1. **选择运行模式** - 界面默认选择"🌐 API模式"
2. **输入API Token**（可选）- 提升调用稳定性
3. **点击"🚀 加载模型"** - 无需等待，即时配置完成
4. **开始生成图像** - 享受零存储占用的AI图像生成

### 💻 本地模式使用步骤
1. **选择运行模式** - 切换到"💻 本地模式"
2. **点击"🚀 加载模型"** - 首次需要下载6-10GB模型文件
3. **等待加载完成** - 显示"✅ 模型加载成功"
4. **开始生成图像** - 使用本地GPU加速生成

### 📝 文生图模式
1. 输入提示词，例如：`a beautiful landscape with mountains`
2. 可选设置负面提示词、采样步数等参数
3. 点击"🎨 生成图像"

### 🖼️ ControlNet模式
1. 上传一张参考图片
2. 选择控制类型（Canny边缘/Scribble涂鸦/Depth深度）
3. 输入风格描述，例如：`oil painting style`
4. 调整 ControlNet 强度
5. 生成结果会保持原图结构，改变风格

## 🎯 示例提示词

**风景类：**
```
a serene mountain lake at sunset, golden hour lighting, highly detailed, 4k
```

**人物类：**
```
portrait of a wise old wizard, fantasy art, detailed facial features, soft lighting
```

**建筑类：**
```
futuristic cityscape with flying cars, cyberpunk style, neon lights, highly detailed
```

**负面提示词：**
```
blurry, low quality, distorted, ugly, bad anatomy, extra limbs
```

## ⚙️ 参数说明

| 参数 | 范围 | 推荐值 | 说明 |
|------|------|--------|------|
| 采样步数 | 10-50 | 20-30 | 越高质量越好但更慢 |
| 引导强度 | 1-20 | 7-12 | 越高越符合提示词 |
| ControlNet强度 | 0-2 | 0.8-1.2 | 控制结构影响程度 |
| 分辨率 | 256-1024 | 512x512 | 更高分辨率需要更多资源 |

## ❓ 常见问题

### 关于存储空间

**Q: API模式真的完全不占用本地存储空间吗？**
A: 是的！API模式下：
- ✅ 不下载任何模型文件（0GB模型存储）
- ✅ 不缓存生成的图像（除非用户主动保存）
- ✅ 仅程序本身占用几MB空间
- ✅ 所有AI计算都在Hugging Face云端完成

**Q: 我之前下载的模型文件在哪里？**
A: 本地模式下载的模型存储在：
- Windows: `C:\Users\{用户名}\.cache\huggingface\hub`
- Linux/Mac: `~/.cache/huggingface/hub`
- 总大小约6-10GB，可通过删除该目录释放空间

**Q: 切换到API模式后，之前的模型文件会自动删除吗？**
A: 不会自动删除。如需释放空间，可手动删除缓存目录：
```bash
# Windows PowerShell
Remove-Item "C:\Users\{用户名}\.cache\huggingface\hub" -Recurse -Force

# Linux/Mac
rm -rf ~/.cache/huggingface/hub
```

**Q: API模式有使用限制吗？**
A: 
- 免费用户：有一定的每月调用次数限制
- 付费用户：更高的调用限制和优先级
- 可通过设置API Token获得更稳定的服务

**Q: 网络断开时API模式还能使用吗？**
A: 不能。API模式需要网络连接到Hugging Face服务器。如需离线使用，请选择本地模式。

## 🔧 自定义扩展

### 添加新的 ControlNet 模型

```python
# 在 app.py 中修改模型ID
controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/sd-controlnet-depth",  # 深度控制
    # "lllyasviel/sd-controlnet-pose",   # 姿态控制
    # "lllyasviel/sd-controlnet-seg",    # 分割控制
)
```

### 更换基础模型

```python
# 替换为其他 Stable Diffusion 模型
model_id = "stabilityai/stable-diffusion-2-1"
# model_id = "CompVis/stable-diffusion-v1-4"
# model_id = "runwayml/stable-diffusion-v1-5"
```

## 📦 项目结构

```
diffusion-app/
├── app.py              # 主应用程序
├── requirements.txt    # Python依赖
├── README.md          # 项目说明
└── .gitignore         # Git忽略文件
```

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

## 📄 许可证

MIT License

## 🙏 致谢

- [Hugging Face Diffusers](https://github.com/huggingface/diffusers)
- [Gradio](https://gradio.app/)
- [ControlNet](https://github.com/lllyasviel/ControlNet)