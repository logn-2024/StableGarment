import os
from os.path import join as opj
import imageio
import numpy as np

import torch
import torchvision

from PIL import Image
from typing import Union
from tqdm import tqdm
from einops import rearrange

import random
import cv2
import json

def tokenize_prompt(tokenizer, prompt):
    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids
    return text_input_ids

def get_val_data(data_dir, img_H=512, img_W=384, data_name="vitonhd", data_type="paired",use_text=False,**kwargs):
    '''
        return empty model prompt as it is not suitable for tryon
    '''
    data_name_list = ["vitonhd","dresscode"]
    assert data_name in data_name_list
    assert data_type in ["paired","unpaired"]

    ## load text
    img_text,gmt_text,img_text_neg,gmt_text_neg = "","","",""
    
    if data_name == "vitonhd":
        data_list = []
        with open(opj(data_dir, "test_pairs.txt"), "r") as f:
            inference_data = f.read().splitlines()[:40]
            data_list = [line.split() for line in inference_data]
        if data_type=="paired":
            data_list = [[pair[0],pair[0]] for pair in data_list]
        img_name, gmt_name = random.choice(data_list)
        test_dir = opj(data_dir,"test")
        img_path = opj(test_dir, "image", img_name)
        gmt_path = opj(test_dir, "cloth", gmt_name)
        densepose_path = opj(test_dir, "image-densepose", img_name)
        agn_mask_path = opj(test_dir, "agnostic-mask", os.path.splitext(img_name)[0]+"_mask.png")
        agn_path = opj(test_dir, "agnostic-v3.2", img_name)
        if use_text:
            image_caption_json = json.load(open(opj(data_dir, "test_image_text.json"), "r"))
            img_name_ = gmt_name # always use paired img_text # only use img_text for text2img, not tryon
            img_text = image_caption_json[img_name_] if img_name_ in image_caption_json else ""
            garment_caption_json = json.load(open(opj(data_dir, "test_cloth_text.json"), "r"))
            gmt_text = garment_caption_json[gmt_name] if gmt_name in garment_caption_json else ""
    elif data_name == "dresscode":
        category = kwargs.get("category",random.choice(["upper_body","dresses","lower_body",]))
        data_dir = opj(data_dir,category)
        data_list = []
        with open(opj(data_dir, f"test_pairs_{data_type}.txt"), "r") as f:
            inference_data = list(f.readlines())[:40]
            data_list = [line.split() for line in inference_data]
        img_name, gmt_name = random.choice(data_list)
        img_path = os.path.join(data_dir, "images", img_name)
        gmt_path = os.path.join(data_dir, "images", gmt_name)
        densepose_path = os.path.join(data_dir, "densepose", img_name)
        agn_mask_path = os.path.join(data_dir, "agnostic-mask-v2", img_name)
        agn_path = os.path.join(data_dir, "agnostic-v2", img_name)
        if use_text:
            image_caption_json = json.load(open(opj(data_dir, "img_cnt.json"), "r"))
            img_name_ = gmt_name.replace("_1","_0") # always use paired img_text
            img_text = image_caption_json[img_name_] if img_name_ in image_caption_json else ""
            garment_caption_json = json.load(open(opj(data_dir, "cloth_cnt.json"), "r"))
            gmt_text = garment_caption_json[gmt_name] if gmt_text in garment_caption_json else ""
    else:
        raise ValueError(f"data_name must be one of {data_name_list}")
    ## load image
    img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    gmt = cv2.cvtColor(cv2.imread(gmt_path), cv2.COLOR_BGR2RGB)
    img_densepose = cv2.cvtColor(cv2.imread(densepose_path), cv2.COLOR_BGR2RGB)
    img_agn = cv2.cvtColor(cv2.imread(agn_path), cv2.COLOR_BGR2RGB)
    # gray mask
    img_agn_mask = cv2.cvtColor(cv2.imread(agn_mask_path),cv2.COLOR_BGR2GRAY)
    img_agn_mask = np.expand_dims(img_agn_mask, axis=-1)
    img_agn_mask = (img_agn_mask >= 128).astype(np.float32)  # 0 or 1
    img_agn_mask = (1 - img_agn_mask) * 255
    # cat cond
    img_cond = np.concatenate([img_agn_mask, img_densepose], axis=-1) # np.concatenate([img_agn_mask, img_agn, img_densepose], axis=-1)

    img = cv2.resize(img, (img_W, img_H))
    gmt = cv2.resize(gmt, (img_W, img_H))
    img_cond = cv2.resize(img_cond, (img_W, img_H))
    img_agn = cv2.resize(img_agn, (img_W, img_H))
    
    return img,gmt,img_name,gmt_name, \
        img_text,gmt_text,img_text_neg,gmt_text_neg, \
        img_cond,img_agn
