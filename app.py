# adapted from https://huggingface.co/spaces/HumanAIGC/OutfitAnyone/blob/main/app.py
import os
from os.path import join as opj

import torch
import gradio as gr
from PIL import Image
import numpy as np
from torchvision import transforms

from transformers import CLIPTextModel, CLIPTokenizer
from transformers.models.clip.image_processing_clip import CLIPImageProcessor

from diffusers import UniPCMultistepScheduler, DDIMScheduler
from diffusers import AutoencoderKL
from diffusers import StableDiffusionPipeline
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker

from stablegarment.models import GarmentEncoderModel,ControlNetModel
from stablegarment.pipelines import StableGarmentPipeline
from stablegarment.pipelines import StableGarmentControlNetTryonPipeline

device = "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.bfloat16 if device=="cpu" else torch.float16
height = 512
width = 384

base_model_path = "SG161222/Realistic_Vision_V4.0_noVAE"
vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(dtype=torch_dtype,device=device)
scheduler = UniPCMultistepScheduler.from_pretrained("runwayml/stable-diffusion-v1-5",subfolder="scheduler")

pretrained_garment_encoder_path = "loooooong/StableGarment_text2img"
garment_encoder = GarmentEncoderModel.from_pretrained(pretrained_garment_encoder_path,torch_dtype=torch_dtype,subfolder="garment_encoder")
garment_encoder = garment_encoder.to(device=device,dtype=torch_dtype)

pipeline_t2i = StableGarmentPipeline.from_pretrained(base_model_path, vae=vae, torch_dtype=torch_dtype, use_safetensors=True,).to(device=device) #  variant="fp16"
# pipeline = StableDiffusionPipeline.from_pretrained("SG161222/Realistic_Vision_V4.0_noVAE", vae=vae, torch_dtype=torch_dtype).to(device=device)
pipeline_t2i.scheduler = scheduler
pipeline_t2i.safety_checker = StableDiffusionSafetyChecker.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch_dtype, subfolder="safety_checker").to(device=device)
pipeline_t2i.feature_extractor = CLIPImageProcessor.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch_dtype, subfolder="feature_extractor")

pipeline_tryon = None

pretrained_model_path = "loooooong/StableGarment_tryon"
controlnet = ControlNetModel.from_pretrained(pretrained_model_path,subfolder="controlnet")
garment_encoder_2 = GarmentEncoderModel.from_pretrained(pretrained_model_path,subfolder="garment_encoder").to(device=device,dtype=torch_dtype)
text_encoder = CLIPTextModel.from_pretrained(base_model_path, subfolder='text_encoder')
tokenizer = CLIPTokenizer.from_pretrained(base_model_path, subfolder='tokenizer')
scheduler = DDIMScheduler.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="scheduler")
pipeline_tryon = StableGarmentControlNetTryonPipeline.from_pretrained(
    base_model_path,
    vae=vae,
    text_encoder=text_encoder, 
    tokenizer=tokenizer,
    unet=pipeline_t2i.unet,
    controlnet=controlnet,
    scheduler=scheduler,
).to(device=device,dtype=torch_dtype)

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

def tryon(prompt,init_image,garment_top,garment_down,):
    basename = os.path.splitext(os.path.basename(init_image))[0]
    image_agn = Image.open(opj(parse_dir,basename+"_agn.jpg")).resize((width,height))
    image_agn_mask = Image.open(opj(parse_dir,basename+"_mask.png")).resize((width,height))
    densepose_image = Image.open(opj(parse_dir,basename+"_densepose.png")).resize((width,height))
    garment_top = Image.open(garment_top).resize((width,height))

    garment_images = [garment_top,]
    prompt = [prompt,]
    garment_prompt = ["",]
    controlnet_condition = prepare_controlnet_inputs([image_agn_mask],[densepose_image]).type(torch_dtype)

    images = pipeline_tryon(prompt, negative_prompt=[""]*len(garment_images),garment_prompt=garment_prompt, # negative_cloth_prompt = n_prompt,
                  control_image = [Image.new('RGB', (width, height), (0,0,0))]*len(garment_images),
                  height=height,width=width,num_inference_steps=25,guidance_scale=2.5,eta=0.0,
                  controlnet_condition=controlnet_condition,garment_image=garment_images,controlnet_conditioning_scale=1.0,
                  garment_encoder=garment_encoder_2,condition_extra=image_agn,num_images_per_prompt=1,
                  generator=None,fusion_blocks="full",).images
    return images[0]

def text2image(prompt,init_image,garment_top,garment_down,style_fidelity=1.):

    garment_top = Image.open(garment_top).resize((width,height))
    garment_top = transforms.CenterCrop((height,width))(transforms.Resize(max(height, width))(garment_top))

    # always enable classifier-free-guidance as it is related to garment
    cfg = 4 # if prompt else 0 
    garment_images = [garment_top,]
    prompt = [prompt,]
    cloth_prompt = ["",]
    n_prompt = "nsfw, unsaturated, abnormal, unnatural, artifact"
    negative_prompt = [n_prompt]
    
    images = pipeline_t2i(prompt,negative_prompt=negative_prompt,garment_prompt=cloth_prompt,height=height,width=width,
                    num_inference_steps=30,guidance_scale=cfg,num_images_per_prompt=1,style_fidelity=style_fidelity,
                    garment_encoder=garment_encoder,garment_image=garment_images,).images
    return images[0]

# def text2image(prompt,init_image,garment_top,garment_down,*args,**kwargs):
#     return pipeline(prompt).images[0]

def infer(prompt,init_image,garment_top,garment_down,t2i_only,style_fidelity):
    if t2i_only:
        return text2image(prompt,init_image,garment_top,garment_down,style_fidelity)
    else:
        return tryon(prompt,init_image,garment_top,garment_down)

init_state,prompt_state = None,""
t2i_only_state = False
def set_mode(t2i_only,person_condition,prompt):
    global init_state, prompt_state, t2i_only_state
    t2i_only_state = not t2i_only_state
    init_state, prompt_state =  person_condition or init_state, prompt_state or prompt
    if t2i_only:
        return [gr.Image(sources='clipboard', type="filepath", label="model",value=None, interactive=False),
                gr.Textbox(placeholder="", label="prompt(for t2i)", value=prompt_state, interactive=True),
                ]
    else:
        return [gr.Image(sources='clipboard', type="filepath", label="model",value=init_state, interactive=False),
                gr.Textbox(placeholder="", label="prompt(for t2i)", value="", interactive=False),
                ]

def example_fn(inputs,):
    if t2i_only_state:
        return gr.Image(sources='clipboard', type="filepath", label="model", value=None, interactive=False)
    return gr.Image(sources='clipboard', type="filepath", label="model",value=inputs, interactive=False)

gr.set_static_paths(paths=[opj(os.path.dirname(__file__), "assets/images/model")])
model_dir = opj(os.path.dirname(__file__), "assets/images/model")
garment_dir = opj(os.path.dirname(__file__), "assets/images/garment")
parse_dir = opj(os.path.dirname(__file__), "assets/images/image_parse")

model = opj(model_dir, "13987_00.jpg")
all_person = [opj(model_dir,fname) for fname in os.listdir(model_dir) if fname.endswith(".jpg")]
with gr.Blocks(css = ".output-image, .input-image, .image-preview {height: 400px !important} ") as gradio_app:
    gr.Markdown("# StableGarment")
    gr.Markdown("Demo for [StableGarment: Garment-Centric Generation via Stable Diffusion](https://arxiv.org/abs/2403.10783).")
    with gr.Row():
        with gr.Column():
            init_image = gr.Image(sources='clipboard', type="filepath", label="model", value=model, interactive=False,)
            example = gr.Examples(inputs=init_image, #gr.Image(visible=False), #
                                  examples_per_page=4,
                                  examples=all_person,
                                  run_on_click=True,
                                  outputs=init_image,
                                  fn=example_fn,
                                  cache_examples=False,)
        with gr.Column():
            with gr.Row():
                images_top = [opj(garment_dir,fname) for fname in os.listdir(garment_dir) if fname.endswith(".jpg")]
                garment_top = gr.Image(sources='upload', type="filepath", label="top garment",value=images_top[0]) # ,interactive=False
                example_top = gr.Examples(inputs=garment_top,
                                            examples_per_page=4,
                                            examples=images_top)
                images_down = []
                garment_down = gr.Image(sources='upload', type="filepath", label="lower garment",interactive=False, visible=False)
                example_down = gr.Examples(inputs=garment_down,
                                            examples_per_page=4,
                                            examples=images_down)
            prompt = gr.Textbox(placeholder="a photo of model", label="prompt(for t2i)",) # interactive=False
            with gr.Row():
                t2i_only = gr.Checkbox(label="t2i with garment", info="Only text and garment.", elem_id="t2i_switch", value=False, interactive=True,)
                run_button = gr.Button(value="Run")
                t2i_only.change(fn=set_mode,inputs=[t2i_only,init_image,prompt],outputs=[init_image,prompt,])
            with gr.Accordion("advance options", open=False):
                gr.Markdown("Garment fidelity control(Tune down it to reduce white edge).")
                style_fidelity = gr.Slider(0, 1, value=1, label="fidelity(only for t2i)") # , info=""
        with gr.Column():
            gallery = gr.Image()
            run_button.click(fn=infer, 
                            inputs=[
                                    prompt,
                                    init_image,
                                    garment_top,
                                    garment_down,
                                    t2i_only,
                                    style_fidelity,
                                    ], 
                            outputs=[gallery],)
    gr.Markdown("We borrow some code from [OutfitAnyone](https://huggingface.co/spaces/HumanAIGC/OutfitAnyone), thanks. This demo is not safe for all audiences, which may reflect implicit bias and other defects of base model.")
    
if __name__ == "__main__":
    gradio_app.launch()
