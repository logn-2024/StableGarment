import torch
from torchvision import transforms
from PIL import Image

from diffusers import UniPCMultistepScheduler,DDIMScheduler
from diffusers import AutoencoderKL
from diffusers.loaders import LoraLoaderMixin

from stablegarment.models.garment_encoder import GarmentEncoderModel
from stablegarment.pipelines.pipeline_text2img_attn import StableGarmentPipeline

import os

device = "cuda:0"
torch_dtype = torch.float16 
height = 512
width = 384
seed = None
if seed is not None:
    generator = torch.Generator(device=device).manual_seed(seed)
else:
    generator = None

garment_image_path = "./assets/images/garment/00126_00.jpg"

pretrained_garment_encoder_path,fusion_blocks = "loooooong/StableGarment_text2img", "midup"
# pretrained_garment_encoder_path,fusion_blocks = "loooooong/StableGarment_tryon","full"
vae_model = "stabilityai/sd-vae-ft-mse"
base_model = "SG161222/Realistic_Vision_V4.0_noVAE" # change base model for different style

vae = AutoencoderKL.from_pretrained(vae_model).to(dtype=torch_dtype,device=device)
pipeline = StableGarmentPipeline.from_pretrained(base_model, vae=vae, torch_dtype=torch_dtype, variant="fp16",).to(device=device)
pipeline.scheduler = DDIMScheduler.from_pretrained(base_model,subfolder="scheduler")

garment_encoder = GarmentEncoderModel.from_pretrained(
    pretrained_garment_encoder_path,subfolder="garment_encoder",
    torch_dtype=torch_dtype,ignore_mismatched_sizes=True,low_cpu_mem_usage=False
).to(device=device,dtype=torch_dtype)
garment_image = Image.open(garment_image_path).resize((width,height))
garment_image = transforms.CenterCrop((height,width))(transforms.Resize(max(height, width))(garment_image))

garment_images = [garment_image,]
prompt = ["a photo of a woman, full body, best quality, high quality",]
garment_prompt = ["bad quality, worst quality",]

# tune down style_fidelity(0-1) to reduce white edge but may cause garment distortion
images = pipeline(prompt,garment_prompt=garment_prompt,height=height,width=width,generator=generator,
                  num_inference_steps=30,guidance_scale=4.0,num_images_per_prompt=1,style_fidelity=1.,
                  garment_encoder=garment_encoder,garment_image=garment_images,fusion_blocks=fusion_blocks).images
os.makedirs("results",exist_ok=True)
image_row = []
import numpy as np
for i,image in enumerate(images):
    image_row.append(np.array(image))
Image.fromarray(np.concatenate(image_row,axis=1)).save("results/sample.jpg")