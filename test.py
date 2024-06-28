import torch
import torch.nn.functional as F

from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL,UNet2DConditionModel
from diffusers import DDPMScheduler, DDIMScheduler


from stablegarment.models.garment_encoder import GarmentEncoderModel
from stablegarment.models.controlnet import ControlNetModel

from stablegarment.data.vitonhd import VITONHDDataset
from stablegarment.data.dresscode import DressCodeDataset
from stablegarment.models.self_attention_modules import ReferenceAttentionControl

import os
from os.path import join as opj
from tqdm import tqdm
import random

from PIL import Image
import numpy as np
import cv2

seed = 42
# seed all
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

device = 'cuda:0'
weight_dtype = torch.float16

pretrained_vae_model_path = "stabilityai/sd-vae-ft-mse"
pretrained_model_name_or_path = "SG161222/Realistic_Vision_V4.0_noVAE" #"runwayml/stable-diffusion-v1-5"
pretrained_controlnet_path = "loooooong/StableGarment_tryon/controlnet"
pretrained_garment_encoder_path = "loooooong/StableGarment_tryon/garment_encoder"


vton_root_dir = "/tiamat-vePFS/share_data/hailong/data/zalando-hd-resized"
drcd_root_dir = "data/DressCode"
target_dir =  "./results/vthd_tryon"
os.makedirs(target_dir, exist_ok=True)
os.makedirs(opj(target_dir, "samples"), exist_ok=True)
os.makedirs(opj(target_dir, "compare"), exist_ok=True)

vae = AutoencoderKL.from_pretrained(pretrained_vae_model_path, subfolder="vae")
vae.enable_slicing()
controlnet = ControlNetModel.from_pretrained(pretrained_controlnet_path)
unet = UNet2DConditionModel.from_pretrained(pretrained_model_name_or_path, subfolder="unet", use_safetensors=True, torch_dtype=weight_dtype)
garment_encoder = GarmentEncoderModel.from_pretrained(pretrained_garment_encoder_path, subfolder="garment_encoder")

tokenizer = CLIPTokenizer.from_pretrained(pretrained_model_name_or_path, subfolder="tokenizer")
text_encoder = CLIPTextModel.from_pretrained(pretrained_model_name_or_path, subfolder="text_encoder")

noise_scheduler = DDPMScheduler.from_pretrained(pretrained_model_name_or_path, subfolder="scheduler")
scheduler = DDIMScheduler.from_pretrained(pretrained_model_name_or_path, subfolder="scheduler")

vae = vae.to(device, dtype=weight_dtype).eval()
unet = unet.to(device, dtype=weight_dtype).eval()
text_encoder = text_encoder.to(device, dtype=weight_dtype).eval()
controlnet = controlnet.to(device, dtype=weight_dtype).eval()
garment_encoder = garment_encoder.to(device, dtype=weight_dtype).eval()

img_H = 1024
img_W = 768
is_pair = False
is_test = True
is_sorted = True
do_classifier_free_guidance = True
num_inference_steps = 25
guidance_scale = 2.5

# inference_data = DressCodeDataset(
#     data_root_dir = drcd_root_dir,
#     img_H = img_H,
#     img_W = img_W,
#     tokenizer = tokenizer,
#     is_paired = is_pair,
#     is_test = is_test,
#     is_sorted = is_sorted,
#     category = "upper_body", # "dresses", # "lower_body", # 
#     p_flip=0., p_crop=0., p_rmask=0., 
# )
inference_data = VITONHDDataset(
    data_root_dir = vton_root_dir,
    img_H = img_H,
    img_W = img_W,
    tokenizer = tokenizer,
    is_paired = is_pair,
    is_test = is_test,
    is_sorted = is_sorted,
    p_flip=0., p_crop=0., p_rmask=0., 
)

inference_data_loader = torch.utils.data.DataLoader(
    inference_data,
    batch_size=64,
    shuffle=False,
    num_workers=0,
    drop_last=False,
)

reference_control_writer = ReferenceAttentionControl(garment_encoder, mode='write', fusion_blocks='full',do_classifier_free_guidance=False)
reference_control_reader = ReferenceAttentionControl(unet, mode='read', fusion_blocks='full',do_classifier_free_guidance=True)
reference_control_reader.share_bank(reference_control_writer)

with torch.no_grad():
    for idx, batch in enumerate(inference_data_loader):
        ref_latents = vae.encode(batch["garment"].to(device=device, dtype=weight_dtype)).latent_dist.mean
        ref_latents = ref_latents * vae.config.scaling_factor
        agn_img_latents = vae.encode(batch["agn"].to(device=device, dtype=weight_dtype)).latent_dist.sample()
        agn_img_latents = agn_img_latents * vae.config.scaling_factor

        latents = torch.randn_like(ref_latents) * scheduler.init_noise_sigma
        noise = latents.clone()

        # text_input_cnt = "a photo of woman"
        # text_input_ids = tokenizer(text_input_cnt,padding="max_length", max_length=77, truncation=True, return_tensors="pt").input_ids.to(device)
        # text_embeddings = text_encoder(text_input_ids)[0]
        text_input_ids = batch["img_text_token_ids"].to(device)
        text_embeddings = text_encoder(text_input_ids)[0]

        un_text_input_ids = batch["null_token_id"].to(device)
        un_text_embeddings = text_encoder(un_text_input_ids)[0]
        text_embeddings = torch.cat([un_text_embeddings, text_embeddings], dim=0)

        garment_text_input_ids = batch["garment_text_token_ids"].to(device)
        garment_text_embeddings = text_encoder(garment_text_input_ids)[0]
        un_garment_text_input_ids = batch["null_token_id"].to(device)
        un_garment_text_embeddings = text_encoder(un_garment_text_input_ids)[0]
        garment_text_embeddings = torch.cat([un_garment_text_embeddings, garment_text_embeddings], dim=0)

        image_latents = vae.encode(batch["image"].to(device=device, dtype=weight_dtype)).latent_dist.mean * vae.config.scaling_factor

        scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = scheduler.timesteps

        image_densepose = (batch['image_densepose'].to(device, dtype=weight_dtype) + 1) / 2
        image_agn = (batch['agn'].to(device, dtype=weight_dtype) + 1) / 2
        image_agn_mask = batch['agn_mask'].to(device, dtype=weight_dtype)
        image_cond = torch.cat([image_agn_mask, image_densepose], dim=1)
        image_cond = torch.cat([image_cond] * 2) if do_classifier_free_guidance else image_cond
        ref_latents = torch.cat([ref_latents] * 2) if do_classifier_free_guidance else ref_latents

        for i, t in tqdm(enumerate(timesteps), total=len(timesteps)):
            # only use the condition tensor
            garment_encoder(
                ref_latents[ref_latents.shape[0]//2:],
                t,
                encoder_hidden_states=text_embeddings[text_embeddings.shape[0]//2:],
                return_dict=False,
            )

            latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            agn_img_latents_input = torch.cat([agn_img_latents] * 2) if do_classifier_free_guidance else agn_img_latents
            down_block_res_samples, mid_block_res_sample = controlnet(
                latent_model_input,
                agn_img_latents_input,
                t,
                encoder_hidden_states=text_embeddings,
                controlnet_cond = image_cond,
                return_dict=False,
            )

            noise_pred = unet(
                latent_model_input, 
                t, 
                encoder_hidden_states=text_embeddings,
                down_block_additional_residuals=down_block_res_samples,
                mid_block_additional_residual=mid_block_res_sample,
                return_dict=False,
            )[0]

            reference_control_reader.clear()
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            extra_step_kwargs = {}
            extra_step_kwargs["eta"] = 0.
            latents = scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample
            reference_control_writer.clear() 

        dec_latents = 1 / 0.18215 * latents
        gen_img = vae.decode(dec_latents).sample
        gen_img = (gen_img / 2 + 0.5).clamp(0, 1).permute(0, 2, 3, 1)

        samples = gen_img.cpu().float().numpy()

        bz = samples.shape[0]
        for bz_idx in range(bz):
            sample = samples[bz_idx]
            sample = (sample * 255).astype(np.uint8)
            
            src_img = torch.clamp((batch['image'] + 1) / 2, 0, 1)
            src_img = src_img.permute(0,2,3,1).cpu().numpy()[bz_idx]
            src_img = (src_img * 255).astype(np.uint8)

            garment_img = torch.clamp((batch['garment'] + 1) / 2, 0, 1)
            garment_img = garment_img.permute(0,2,3,1).cpu().numpy()[bz_idx]
            garment_img = (garment_img * 255).astype(np.uint8)

            densepose_img = torch.clamp((batch['image_densepose'] + 1) / 2, 0, 1)
            densepose_img = densepose_img.permute(0,2,3,1).cpu().numpy()[bz_idx]
            densepose_img = (densepose_img * 255).astype(np.uint8)

            agn_img = torch.clamp((batch['agn'] + 1) / 2, 0, 1)
            agn_img = agn_img.permute(0,2,3,1).cpu().numpy()[bz_idx]
            agn_img = (agn_img * 255).astype(np.uint8)
            
            result = np.concatenate([src_img, garment_img, densepose_img, agn_img, sample], axis=1)
            result = Image.fromarray(result)
            src_basename,ref_basename = os.path.basename(batch['img_fn'][bz_idx]),os.path.basename(batch['garment_fn'][bz_idx])
            src_id,ref_id = os.path.splitext(src_basename)[0],os.path.splitext(ref_basename)[0]
            basename = f"{src_id}.png"
            result.save(opj(target_dir,"compare",f"{src_id}-{ref_id}.png"))
            Image.fromarray(np.concatenate([src_img, garment_img], axis=1)).save(opj(target_dir,"compare",f"{src_id}-{ref_id}-cond1.png"))
            Image.fromarray(np.concatenate([agn_img, densepose_img], axis=1)).save(opj(target_dir,"compare",f"{src_id}-{ref_id}-cond2.png"))
            cv2.imwrite(opj(target_dir,"samples",basename),sample[:,:,::-1])