import torch

from PIL import Image
import numpy as np

from transformers import CLIPTextModel, CLIPTokenizer

from diffusers.schedulers import DDIMScheduler
from diffusers.models import AutoencoderKL,UNet2DConditionModel

from stablegarment.models.appearance_encoder import AppearanceEncoderModel
from stablegarment.models.controlnet import ControlNetModel
from stablegarment.piplines.pipeline_densepose_attn_text import StableGarmentControlNetPipeline

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

device = "cuda:3"
torch_dtype = torch.float16
height = 512
width = 384
seed = None
if seed is not None:
    generator = torch.Generator(device=device).manual_seed(seed)
else:
    generator = None

pretrained_model_path = "part_module_controlnet_imp2"
base_model_path = "runwayml/stable-diffusion-v1-5"
vae_path = "stabilityai/sd-vae-ft-mse"
vae = AutoencoderKL.from_pretrained(vae_path)
controlnet = ControlNetModel.from_pretrained(pretrained_model_path,subfolder="controlnet")
text_encoder = CLIPTextModel.from_pretrained(base_model_path, subfolder='text_encoder')
tokenizer = CLIPTokenizer.from_pretrained(base_model_path, subfolder='tokenizer')
unet = UNet2DConditionModel.from_pretrained(base_model_path, subfolder='unet')
garment_encoder = AppearanceEncoderModel.from_pretrained(pretrained_model_path, subfolder="garment_encoder")
scheduler = DDIMScheduler.from_pretrained(base_model_path, subfolder="scheduler")
pipeline = StableGarmentControlNetPipeline(
    vae,
    text_encoder, 
    tokenizer,
    unet,
    controlnet,
    scheduler,
).to(device=device,dtype=torch_dtype)
garment_encoder = garment_encoder.to(device=device,dtype=torch_dtype)

garment_image = Image.open("./assets/images/garment/00126_00.jpg").resize((width,height))
densepose_image = Image.open("./assets/images/image_parse/13987_00_densepose.jpg").resize((width,height))
image_agn_mask = Image.open("./assets/images/image_parse/13987_00_mask.png").resize((width,height))
image_agn = Image.open("./assets/images/image_parse/13987_00_agn.jpg").resize((width,height))

prompts = ["a photo of a woman", ]
cloth_prompt = ["",]

garment_images = [garment_image]
densepose_image = [densepose_image]
image_agn_mask = [image_agn_mask]
image_agn = [image_agn]
controlnet_condition = prepare_controlnet_inputs(image_agn_mask,densepose_image)

images = pipeline(prompts, negative_prompt="",cloth_prompt=cloth_prompt, # negative_cloth_prompt = n_prompt,
                  height=height,width=width,num_inference_steps=25,guidance_scale=1.5,eta=0.0,
                  controlnet_condition=controlnet_condition,reference_image=garment_images, 
                  garment_encoder=garment_encoder,condition_extra=image_agn,
                  generator=generator,).images
images[0].save("results/sample.jpg")