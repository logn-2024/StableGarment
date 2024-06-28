import os
import torch

from PIL import Image
import numpy as np

from transformers import CLIPTextModel, CLIPTokenizer

from diffusers.schedulers import DDIMScheduler
from diffusers.models import AutoencoderKL,UNet2DConditionModel

from stablegarment.models.garment_encoder import GarmentEncoderModel
from stablegarment.models.controlnet import ControlNetModel
from stablegarment.pipelines.pipeline_controlnet_tryon_attn import StableGarmentControlNetTryonPipeline

def prepare_controlnet_inputs(agn_mask_list,densepose_list):
    for i,agn_mask_img in enumerate(agn_mask_list):
        agn_mask_img = np.array(agn_mask_img.convert("L"))
        agn_mask_img = np.expand_dims(agn_mask_img, axis=-1)
        agn_mask_img = (agn_mask_img >= 128).astype(np.float32)  # 0 or 1
        agn_mask_list[i] = 1. - agn_mask_img
    densepose_list = [np.array(img)/255. for img in densepose_list]
    controlnet_inputs = []
    for mask,pose in zip(agn_mask_list,densepose_list):
        controlnet_inputs.append(torch.tensor(np.concatenate([mask, pose], axis=-1)).permute(2,0,1))
    controlnet_inputs = torch.stack(controlnet_inputs)
    return controlnet_inputs

device = "cuda:0" # "cpu"
torch_dtype = torch.float32 if device=="cpu" else torch.float16
height = 512 # between 512 and 1024 
width = 384 # between 384 and 768
seed = 42
if seed is not None:
    generator = torch.Generator(device=device).manual_seed(seed)
else:
    generator = None

pretrained_model_path = "loooooong/StableGarment_tryon"
base_model_path = "SG161222/Realistic_Vision_V4.0_noVAE" # runwayml/stable-diffusion-v1-5
vae_path = "stabilityai/sd-vae-ft-mse"
vae = AutoencoderKL.from_pretrained(vae_path)
unet = UNet2DConditionModel.from_pretrained(base_model_path, subfolder='unet')
garment_encoder = GarmentEncoderModel.from_pretrained(pretrained_model_path, subfolder="garment_encoder")
controlnet = ControlNetModel.from_pretrained(pretrained_model_path,subfolder="controlnet",ignore_mismatched_sizes=True,low_cpu_mem_usage=False)
scheduler = DDIMScheduler.from_pretrained(base_model_path, subfolder="scheduler")
pipeline = StableGarmentControlNetTryonPipeline.from_pretrained(
    base_model_path,
    vae=vae,
    unet=unet,
    controlnet=controlnet,
    scheduler=scheduler,
    torch_dtype=torch_dtype,
).to(device=device,dtype=torch_dtype)
garment_encoder = garment_encoder.to(device=device,dtype=torch_dtype)

garment_image = Image.open("./assets/images/garment/00126_00.jpg").resize((width,height))
densepose_image = Image.open("./assets/images/image_parse/13987_00_densepose.png").resize((width,height))
image_agn_mask = Image.open("./assets/images/image_parse/13987_00_mask.png").resize((width,height))
image_agn = Image.open("./assets/images/image_parse/13987_00_agn.jpg").resize((width,height))

prompts = ["a photo of a woman, full body", ]
garment_prompt = ["",]

garment_images = [garment_image]
densepose_images = [densepose_image]
image_agn_masks = [image_agn_mask]
image_agns = [image_agn]

# image_agns, image_agn_masks = None, [Image.new('L', (width, height), 255) for _ in image_agn_masks] # generate without person and background control
# densepose_images = [Image.new('RGB', (width, height), (0,0,0)) for _ in densepose_images] # generate without densepose control
controlnet_condition = prepare_controlnet_inputs(image_agn_masks,densepose_images)
images = pipeline(prompts, negative_prompt=[""]*len(garment_images),garment_prompt=garment_prompt,
    control_image = [Image.new('RGB', (width, height), (0,0,0))]*len(garment_images),
    height=height,width=width,num_inference_steps=25,guidance_scale=2.5,eta=0.0,
    controlnet_condition=controlnet_condition,garment_image=garment_images,controlnet_conditioning_scale=1.0,
    garment_encoder=garment_encoder,condition_extra=image_agns,num_images_per_prompt=1,
    generator=generator,fusion_blocks="full",
).images
os.makedirs("results",exist_ok=True)
images[0].save("./results/sample.jpg")