import torch

from PIL import Image

from diffusers import UniPCMultistepScheduler
from diffusers import AutoencoderKL

from stablegarment.models.appearance_encoder import AppearanceEncoderModel
from stablegarment.piplines.pipeline_attn_text import StableGarmentPipeline

device = "cuda:3"
torch_dtype = torch.float16
height = 512
width = 384

pretrained_garment_encoder_path = "stablegarment_text2img"
vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(dtype=torch_dtype,device=device)
pipeline = StableGarmentPipeline.from_pretrained("SG161222/Realistic_Vision_V4.0_noVAE", vae=vae, torch_dtype=torch_dtype, variant="fp16").to(device=device)
pipeline.scheduler = UniPCMultistepScheduler.from_pretrained("runwayml/stable-diffusion-v1-5",subfolder="scheduler")

garment_encoder = AppearanceEncoderModel.from_pretrained(pretrained_garment_encoder_path,torch_dtype=torch_dtype,subfolder="garment_encoder")
garment_encoder = garment_encoder.to(device=device,dtype=torch_dtype)
garment_image = Image.open("./assets/images/garment/00126_00.jpg").resize((width,height))

garment_images = [garment_image,]
prompt = ["a photo of a woman",]
cloth_prompt = ["",]

images = pipeline(prompt,cloth_prompt=cloth_prompt,
                  height=height,width=width,
                  num_inference_steps=30,guidance_scale=4,num_images_per_prompt=1,
                  garment_encoder=garment_encoder,garment_image=garment_images,).images
images[0].save("results/sample.jpg")