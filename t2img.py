from diffusers import StableDiffusionPipeline
import torch
import requests

# 加载预训练模型
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16
)
pipe = pipe.to("cuda")

# 生成图像
prompt = "一只可爱的橙色小猫在花园里玩耍"
image = pipe(prompt).images[0]
image.save("generated_cat.png")