# Adapted from https://github.com/magic-research/magic-animate

import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np

import inspect
from typing import Callable, List, Optional, Union
from packaging import version
from transformers import CLIPTextModel, CLIPTokenizer

from diffusers.configuration_utils import FrozenDict
from diffusers.models import AutoencoderKL,UNet2DConditionModel
from diffusers import DiffusionPipeline
from diffusers.schedulers import (
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
)
from diffusers.utils import deprecate, logging
from diffusers.utils import is_accelerate_available
from diffusers.pipelines.stable_diffusion.pipeline_output import StableDiffusionPipelineOutput
from diffusers.image_processor import VaeImageProcessor

from einops import rearrange

from ..models.controlnet import ControlNetModel
from ..models.self_attention_modules import ReferenceAttentionControl

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

class StableGarmentControlNetPipeline(DiffusionPipeline):
    _optional_components = []

    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet2DConditionModel,
        controlnet: ControlNetModel,
        scheduler: Union[
            DDIMScheduler,
            PNDMScheduler,
            LMSDiscreteScheduler,
            EulerDiscreteScheduler,
            EulerAncestralDiscreteScheduler,
            DPMSolverMultistepScheduler,
        ],
    ):
        super().__init__()

        if hasattr(scheduler.config, "steps_offset") and scheduler.config.steps_offset != 1:
            deprecation_message = (
                f"The configuration file of this scheduler: {scheduler} is outdated. `steps_offset`"
                f" should be set to 1 instead of {scheduler.config.steps_offset}. Please make sure "
                "to update the config accordingly as leaving `steps_offset` might led to incorrect results"
                " in future versions. If you have downloaded this checkpoint from the Hugging Face Hub,"
                " it would be very nice if you could open a Pull request for the `scheduler/scheduler_config.json`"
                " file"
            )
            deprecate("steps_offset!=1", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(scheduler.config)
            new_config["steps_offset"] = 1
            scheduler._internal_dict = FrozenDict(new_config)

        if hasattr(scheduler.config, "clip_sample") and scheduler.config.clip_sample is True:
            deprecation_message = (
                f"The configuration file of this scheduler: {scheduler} has not set the configuration `clip_sample`."
                " `clip_sample` should be set to False in the configuration file. Please make sure to update the"
                " config accordingly as not setting `clip_sample` in the config might lead to incorrect results in"
                " future versions. If you have downloaded this checkpoint from the Hugging Face Hub, it would be very"
                " nice if you could open a Pull request for the `scheduler/scheduler_config.json` file"
            )
            deprecate("clip_sample not set", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(scheduler.config)
            new_config["clip_sample"] = False
            scheduler._internal_dict = FrozenDict(new_config)

        is_unet_version_less_0_9_0 = hasattr(unet.config, "_diffusers_version") and version.parse(
            version.parse(unet.config._diffusers_version).base_version
        ) < version.parse("0.9.0.dev0")
        is_unet_sample_size_less_64 = hasattr(unet.config, "sample_size") and unet.config.sample_size < 64
        if is_unet_version_less_0_9_0 and is_unet_sample_size_less_64:
            deprecation_message = (
                "The configuration file of the unet has set the default `sample_size` to smaller than"
                " 64 which seems highly unlikely. If your checkpoint is a fine-tuned version of any of the"
                " following: \n- CompVis/stable-diffusion-v1-4 \n- CompVis/stable-diffusion-v1-3 \n-"
                " CompVis/stable-diffusion-v1-2 \n- CompVis/stable-diffusion-v1-1 \n- runwayml/stable-diffusion-v1-5"
                " \n- runwayml/stable-diffusion-inpainting \n you should change 'sample_size' to 64 in the"
                " configuration file. Please make sure to update the config accordingly as leaving `sample_size=32`"
                " in the config might lead to incorrect results in future versions. If you have downloaded this"
                " checkpoint from the Hugging Face Hub, it would be very nice if you could open a Pull request for"
                " the `unet/config.json` file"
            )
            deprecate("sample_size<64", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(unet.config)
            new_config["sample_size"] = 64
            unet._internal_dict = FrozenDict(new_config)

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            controlnet=controlnet,
            scheduler=scheduler,
        )
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)

    def enable_vae_slicing(self):
        self.vae.enable_slicing()

    def disable_vae_slicing(self):
        self.vae.disable_slicing()

    def enable_sequential_cpu_offload(self, gpu_id=0):
        if is_accelerate_available():
            from accelerate import cpu_offload
        else:
            raise ImportError("Please install accelerate via `pip install accelerate`")

        device = torch.device(f"cuda:{gpu_id}")

        for cpu_offloaded_model in [self.unet, self.text_encoder, self.vae]:
            if cpu_offloaded_model is not None:
                cpu_offload(cpu_offloaded_model, device)

    @property
    def _execution_device(self):
        if self.device != torch.device("meta") or not hasattr(self.unet, "_hf_hook"):
            return self.device
        for module in self.unet.modules():
            if (
                hasattr(module, "_hf_hook")
                and hasattr(module._hf_hook, "execution_device")
                and module._hf_hook.execution_device is not None
            ):
                return torch.device(module._hf_hook.execution_device)
        return self.device

    def _encode_prompt(self, prompt, device, do_classifier_free_guidance, negative_prompt):
        if isinstance(prompt, torch.Tensor):
            batch_size = prompt.shape[0]
            text_input_ids = prompt
        else:
            batch_size = 1 if isinstance(prompt, str) else len(prompt)
            text_inputs = self.tokenizer(
                prompt,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
                )
            text_input_ids = text_inputs.input_ids
            untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

            if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
                removed_text = self.tokenizer.batch_decode(untruncated_ids[:, self.tokenizer.model_max_length - 1 : -1])
                logger.warning(
                    "The following part of your input was truncated because CLIP can only handle sequences up to"
                    f" {self.tokenizer.model_max_length} tokens: {removed_text}"
                )

        if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
            attention_mask = text_inputs.attention_mask.to(device)
        else:
            attention_mask = None

        text_embeddings = self.text_encoder(
            text_input_ids.to(device),
            attention_mask=attention_mask,
        )
        text_embeddings = text_embeddings[0]

        # duplicate text embeddings for each generation per prompt, using mps friendly method
        bs_embed, seq_len, _ = text_embeddings.shape

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance:
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                uncond_tokens = negative_prompt

            max_length = text_input_ids.shape[-1]
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )

            if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
                attention_mask = uncond_input.attention_mask.to(device)
            else:
                attention_mask = None

            uncond_embeddings = self.text_encoder(
                uncond_input.input_ids.to(device),
                attention_mask=attention_mask,
            )
            uncond_embeddings = uncond_embeddings[0]

            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = uncond_embeddings.shape[1]

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

        return text_embeddings

    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    def check_inputs(self, prompt, height, width, callback_steps):
        if not isinstance(prompt, str) and not isinstance(prompt, list) and not isinstance(prompt, torch.Tensor):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        if (callback_steps is None) or (
            callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0)
        ):
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                f" {type(callback_steps)}."
            )

    def _encode_vae_image(self, image: torch.Tensor, generator: torch.Generator):
        if isinstance(generator, list):
            image_latents = [
                self.vae.encode(image[i : i + 1]).latent_dist.sample(generator=generator[i])
                for i in range(image.shape[0])
            ]
            image_latents = torch.cat(image_latents, dim=0)
        else:
            image_latents = self.vae.encode(image).latent_dist.sample(generator=generator)

        image_latents = self.vae.config.scaling_factor * image_latents

        return image_latents

    def prepare_latents(self, batch_size, num_channels_latents, height, width, dtype, device, generator, latents=None, clip_length=16):
        shape = (batch_size, num_channels_latents, height // self.vae_scale_factor, width // self.vae_scale_factor)
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )
        if latents is None:
            rand_device = "cpu" if device.type == "mps" else device

            if isinstance(generator, list):
                latents = [
                    torch.randn(shape, generator=generator[i], device=rand_device, dtype=dtype)
                    for i in range(batch_size)
                ]
                latents = torch.cat(latents, dim=0).to(device)
            else:
                latents = torch.randn(shape, generator=generator, device=rand_device, dtype=dtype).to(device)
                
        else:
            if latents.shape != shape:
                raise ValueError(f"Unexpected latents shape, got {latents.shape}, expected {shape}")
            latents = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        noise = latents.clone()
        latents = latents * self.scheduler.init_noise_sigma
        return latents, noise

    def prepare_condition(self, condition, device, dtype, do_classifier_free_guidance):
        if do_classifier_free_guidance:
            condition = torch.cat([condition] * 2)
        return condition.to(device,dtype=dtype)

    @torch.no_grad()
    def images2latents(self, images, dtype):
        """
        Convert RGB image to VAE latents
        """
        device = self._execution_device
        if isinstance(images, torch.Tensor):
            # suppose input is [-1, 1]
            images = images.to(dtype)
            if images.ndim == 3:
                images = images.unsqueeze(0)
        elif isinstance(images, np.ndarray):
            # suppose input is [0, 255]
            images = torch.from_numpy(images).float().to(dtype) / 127.5 - 1
            images = rearrange(images, "h w c -> c h w").to(device)[None, :]
        latents = self.vae.encode(images)['latent_dist'].mean * self.vae.config.scaling_factor
        return latents

    @torch.no_grad()
    def encode_single_image_latents(self, images, mask, dtype):
        device = self._execution_device
        images = torch.from_numpy(images).float().to(dtype) / 127.5 - 1
        images = rearrange(images, "h w c -> c h w").to(device)
        latents = self.vae.encode(images[None, :])['latent_dist'].mean * 0.18215

        images = images.unsqueeze(0)


        mask = torch.from_numpy(mask).float().to(dtype).to(device) / 255.0
        if mask.ndim == 2:
            mask = mask[None, None, :]
        elif mask.ndim == 3:
            mask = mask[:, None, :, :]
        
        mask = F.interpolate(mask, size=latents.shape[-2:], mode='nearest')
        return latents, images,  mask
    
    def prepare_garment_embeds(self,garment_image,device,dtype,normalize=False):
        if not isinstance(garment_image, list):
            garment_image = [garment_image]
        garment_embeds = []
        for single_garment_image in garment_image:
            single_garment_image = np.array(single_garment_image)/127.5-1
            single_garment_embed = self.vae.encode(torch.tensor(single_garment_image.transpose(2,0,1)[None]).to(device=device,dtype=dtype)).latent_dist.mean
            single_garment_embed = single_garment_embed * self.vae.config.scaling_factor
            garment_embeds.append(single_garment_embed)
        return torch.cat(garment_embeds,dim=0)
    
    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]],
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        cloth_prompt: Optional[Union[str, List[str]]] = None,
        # negative_cloth_prompt: Optional[Union[str, List[str]]] = None,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: Optional[int] = 1,
        controlnet_condition: list = None,
        controlnet_conditioning_scale: float = 1.0,
        num_actual_inference_steps: Optional[int] = None,
        garment_encoder = None, 
        reference_image: str = None,
        condition_extra = None,
        **kwargs,
    ):
        controlnet = self.controlnet

        # Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        # Check inputs. Raise error if not correct
        self.check_inputs(prompt, height, width, callback_steps)

        # Define call parameters
        # batch_size = 1 if isinstance(prompt, str) else len(prompt)
        batch_size =1
        if latents is not None:
            batch_size = latents.shape[0]
        if isinstance(prompt, list):
            batch_size = len(prompt)

        device = self._execution_device
        do_classifier_free_guidance = guidance_scale > 1.0

        # Encode input prompt
        if not isinstance(prompt, torch.Tensor):
            prompt = prompt if isinstance(prompt, list) else [prompt] * batch_size
        if negative_prompt is not None:
            negative_prompt = negative_prompt if isinstance(negative_prompt, list) else [negative_prompt] * batch_size 
        text_embeddings = self._encode_prompt(
            prompt, device, do_classifier_free_guidance, negative_prompt
        )
        text_embeddings = torch.cat([text_embeddings])
        # encode cloth prompt
        if not isinstance(cloth_prompt, torch.Tensor):
            cloth_prompt = cloth_prompt if isinstance(cloth_prompt, list) else [cloth_prompt] * batch_size
        negative_cloth_prompt = ""
        if negative_cloth_prompt is not None:
            negative_cloth_prompt = negative_cloth_prompt if isinstance(negative_cloth_prompt, list) else [negative_cloth_prompt] * batch_size
        cloth_text_embeddings = self._encode_prompt(
            cloth_prompt, device, do_classifier_free_guidance, negative_cloth_prompt
        )
        cloth_text_embeddings = torch.cat([cloth_text_embeddings])
        if garment_encoder is not None:
            reference_control_writer = ReferenceAttentionControl(garment_encoder, do_classifier_free_guidance=False, mode='write', fusion_blocks='midup')
            reference_control_reader = ReferenceAttentionControl(self.unet, do_classifier_free_guidance=do_classifier_free_guidance, mode='read', fusion_blocks='midup')

        # Prepare control for controlnet
        control = self.prepare_condition(
                condition=controlnet_condition,
                device=device,
                dtype=controlnet.dtype,
                do_classifier_free_guidance=do_classifier_free_guidance,
            )

        # Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # Prepare latent variables
        num_channels_latents = self.unet.in_channels
        latents = self.prepare_latents(
            batch_size,
            num_channels_latents,
            height,
            width,
            text_embeddings.dtype,
            device,
            generator,
            latents,
        )
        if isinstance(latents, tuple):
            latents, noise = latents

        # Prepare extra step kwargs.
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # For img2img setting
        if num_actual_inference_steps is None:
            num_actual_inference_steps = num_inference_steps
        
        # convert garment_image to latent for garment encoder
        if reference_image is not None:
            # for now we just use the conditional part
            cloth_text_embeddings = cloth_text_embeddings[cloth_text_embeddings.shape[0]//2:]
            ref_image_latents = self.prepare_garment_embeds(reference_image,device,text_embeddings.dtype)
            assert batch_size==cloth_text_embeddings.shape[0]==ref_image_latents.shape[0]

        # another special input for controlnet
        if condition_extra is not None:
            condition_extra = self.prepare_garment_embeds(condition_extra,device,text_embeddings.dtype)
            condition_extra = torch.cat([condition_extra] * 2) if do_classifier_free_guidance else condition_extra
        
        # context_scheduler = get_context_scheduler(context_schedule)
        
        # Denoising loop
        for i, t in tqdm(enumerate(timesteps), total=len(timesteps)):
            if num_actual_inference_steps is not None and i < num_inference_steps - num_actual_inference_steps:
                continue
            if garment_encoder is not None:
                garment_encoder(
                    ref_image_latents,
                    t,
                    encoder_hidden_states=cloth_text_embeddings,
                    return_dict=False,
                )
                reference_control_reader.update(reference_control_writer,dtype=ref_image_latents.dtype,num_repeat=1) # num_images_per_prompt

            latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            down_block_res_samples, mid_block_res_sample = self.controlnet(
                latent_model_input,
                condition_extra,
                t,
                encoder_hidden_states=text_embeddings,
                controlnet_cond = control,
                return_dict=False,
            )
            # predict the noise residual
            noise_pred = self.unet(
                latent_model_input, 
                t, 
                encoder_hidden_states=text_embeddings,
                down_block_additional_residuals=down_block_res_samples,
                mid_block_additional_residual=mid_block_res_sample,
                return_dict=False,
            )[0]
              
            # perform guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

            if garment_encoder is not None:
                reference_control_reader.clear()
                reference_control_writer.clear()

        samples = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False, 
                                  generator=generator)[0]

        images = self.image_processor.postprocess(samples, output_type=output_type, do_denormalize=None)
        
        if not return_dict:
            return (images,None)
        
        return StableDiffusionPipelineOutput(images=images, nsfw_content_detected=None)