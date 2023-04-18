import numpy as np
import torch
from diffusers import StableDiffusionImg2ImgPipeline
from PIL import Image

model = StableDiffusionImg2ImgPipeline.from_pretrained(
            "stabilityai/stable-diffusion-2-base", torch_dtype=torch.float16)
model.to('cuda')


img_path = 'temp/inpaint-2023-04-16_230739/frame_0100_0174.png'
image = Image.open(img_path).resize((512, 512))
image.save(f"test.png")

prompt = "Real estate photo"
inpaint_image = model(
    prompt=prompt, 
    image=image, 
    strength=0.6,
    guidance_scale=7.5,  # default value
).images[0]
inpaint_image.save(f"test_inpaint.png")