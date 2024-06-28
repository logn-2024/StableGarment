# from stablegarment.models.controlnet import ControlNetModel
from stablegarment.models.controlnet_lora import ControlLoRAModel

from diffusers.models.unet_2d_condition import UNet2DConditionModel

# model_path = "runwayml/stable-diffusion-v1-5"
model_path = "stabilityai/stable-diffusion-xl-base-1.0"
unet = UNet2DConditionModel.from_pretrained(model_path,subfolder="unet", use_safetensors=True)
print(dir(unet))
# unet.save_config("./configs/controlnet_config.json")

# cn = ControlNetModel.from_unet(unet)
cn = ControlLoRAModel.from_unet(unet,lora_linear_rank=8,lora_conv2d_rank=8)
cn.save_config("./configs")