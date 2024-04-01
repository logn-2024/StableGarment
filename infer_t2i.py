import torch
from torchvision import transforms
from PIL import Image

from diffusers import UniPCMultistepScheduler
from diffusers import AutoencoderKL

from stablegarment.models.garment_encoder import GarmentEncoderModel
from stablegarment.piplines.pipeline_attn_text import StableGarmentPipeline

import os

device = "cuda"
torch_dtype = torch.float16
height = 512
width = 384

garment_image_path = "./assets/images/garment/00126_00.jpg"
pretrained_garment_encoder_path = "loooooong/StableGarment_text2img"
base_model = "SG161222/Realistic_Vision_V4.0_noVAE" # change base model for different style

vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(dtype=torch_dtype,device=device)
pipeline = StableGarmentPipeline.from_pretrained(base_model, vae=vae, torch_dtype=torch_dtype, variant="fp16").to(device=device)
pipeline.scheduler = UniPCMultistepScheduler.from_pretrained("runwayml/stable-diffusion-v1-5",subfolder="scheduler")

garment_encoder = GarmentEncoderModel.from_pretrained(pretrained_garment_encoder_path,torch_dtype=torch_dtype,subfolder="garment_encoder")
garment_encoder = garment_encoder.to(device=device,dtype=torch_dtype)
garment_image = Image.open(garment_image_path).resize((width,height))
garment_image = transforms.CenterCrop((height,width))(transforms.Resize(max(height, width))(garment_image))

garment_images = [garment_image,]
prompt = ["a photo of a woman",]
cloth_prompt = ["",]

# tune down style_fidelity(0-1) to reduce white edge but may cause garment distortion
images = pipeline(prompt,cloth_prompt=cloth_prompt,height=height,width=width,
                  num_inference_steps=30,guidance_scale=4,num_images_per_prompt=1,style_fidelity=1.,
                  garment_encoder=garment_encoder,garment_image=garment_images,).images
os.makedirs("results",exist_ok=True)
images[0].save("results/sample.jpg")