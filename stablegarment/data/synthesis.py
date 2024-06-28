import torch
from torch.utils.data import Dataset

import numpy as np
from PIL import Image
from torchvision import transforms
import torchvision.transforms.functional as TF

import json
import os
from os.path import join as opj

from .utils import imread,ImageAugmentation
from ..utils.util import tokenize_prompt

# Not for tryon
class SyntheticDataset(Dataset):
    def __init__(
            self, 
            data_root_dir, 
            img_H, 
            img_W, 
            tokenizer,
            tokenizer_2=None,
            # pair or test make no difference here
            # is_paired=True, 
            # is_test=False, 
            is_sorted=False,      
            **kwargs
        ):
        self.drd = data_root_dir
        self.img_H = img_H
        self.img_W = img_W
        self.data_type = "train"
        
        im_names = []
        c_names = []
        with open(opj(self.drd, "ids_select.txt"),"r") as f:
            im_names = f.read().splitlines()
        # print("last names: ", im_names[-10:])
         
        for im_name in im_names:
            c_names.append(im_name.split("@@")[0]+".jpg")
        
        self.img_text_data,self.garment_text_data = {},{}
        image_caption_json = opj(self.drd, "total_fake.json")
        if os.path.exists(image_caption_json):
            with open(image_caption_json, "r") as f:
                self.img_text_data = json.load(f)
        garment_caption_json = opj(self.drd, f"train_cloth_text.json")
        if os.path.exists(garment_caption_json):
            with open(garment_caption_json, "r") as f:
                self.garment_text_data = json.load(f)
        print(f"img_text_data: {len(self.img_text_data)}, garment_text_data: {len(self.garment_text_data)}")
            
        if is_sorted:
            im_names, c_names = zip(*sorted(zip(im_names, c_names)))
        self.im_names = im_names
        self.c_names = c_names
        self.tokenizer = tokenizer
        self.tokenizer_2 = tokenizer_2

        self.null_id = tokenize_prompt(self.tokenizer, "")
        if self.tokenizer_2 is not None:
            self.null_id_2 = tokenize_prompt(self.tokenizer_2, "")

        self.augmentation = ImageAugmentation(**kwargs)

        self.start_signal = True
    def __len__(self):
        return len(self.im_names)
        
    def __getitem__(self, idx):
        if not self.start_signal:
            return {}
        img_fn = self.im_names[idx]
        garment_fn = self.c_names[idx]

        img_text_cnt = self.img_text_data[img_fn]
        img_text_token_ids = tokenize_prompt(self.tokenizer,img_text_cnt)[0]
        garment_text_cnt = self.garment_text_data[garment_fn]
        garment_text_token_ids = tokenize_prompt(self.tokenizer,garment_text_cnt)[0]
        if self.tokenizer_2 is not None:
            img_text_token_ids_2 = tokenize_prompt(self.tokenizer_2, img_text_cnt)[0]
            garment_text_token_ids_2 = tokenize_prompt(self.tokenizer_2, garment_text_cnt)[0]

        # *********************************************************************************************************************
        # there are bugs in data, but fortunely we don't need them for generation
        agn = imread(opj(self.drd, self.data_type, "agnostic-v3.2", garment_fn), self.img_H, self.img_W)
        # Note: [agnostic_mask_official -> agnostic-mask, 00006_00.jpg -> 00006_00_mask.png]
        mask_name = garment_fn # os.path.splitext(garment_fn)[0]+"_mask.png" # self.im_names[idx]
        agn_mask = imread(opj(self.drd, self.data_type, "agnostic_mask", mask_name), self.img_H, self.img_W, is_mask=True)
    
        image_densepose = imread(opj(self.drd, self.data_type, "image-densepose", garment_fn), self.img_H, self.img_W)
        # ####################################################################################################################

        garment = imread(opj(self.drd, self.data_type, "cloth", garment_fn), self.img_H, self.img_W)
        image = imread(opj(self.drd, self.data_type, "image_variation_v2", img_fn), self.img_H, self.img_W)
        
        # augmentation
        agn, agn_mask, garment, image, image_densepose = self.augmentation(agn, agn_mask, garment, image, image_densepose)
        
        agn_mask = np.array(agn_mask)
        agn_mask = (agn_mask >= 128).astype(np.float32)  # 0 or 1
        agn_mask = agn_mask[:,:,None]
        agn_mask = 1. - agn_mask

        # normalize
        agn = (np.array(agn).astype(np.float32) / 127.5) - 1.
        garment = (np.array(garment).astype(np.float32) / 127.5) - 1.
        image = (np.array(image).astype(np.float32) / 127.5) - 1.
        image_densepose = (np.array(image_densepose).astype(np.float32) / 127.5) - 1.

        # np to tensor
        agn = torch.from_numpy(agn.transpose(2, 0, 1)).float()
        agn_mask = torch.from_numpy(agn_mask.transpose(2, 0, 1)).float()
        garment = torch.from_numpy(garment.transpose(2, 0, 1)).float()
        image = torch.from_numpy(image.transpose(2, 0, 1)).float()
        image_densepose = torch.from_numpy(image_densepose.transpose(2, 0, 1)).float()

        example = dict(
            agn=agn,
            agn_mask=agn_mask,
            garment=garment,
            image=image,
            image_densepose=image_densepose,
            img_text_token_ids=img_text_token_ids,
            garment_text_token_ids=garment_text_token_ids,
            img_fn=img_fn,
            garment_fn=garment_fn,
        )
        if self.tokenizer_2 is not None:
            example["img_text_token_ids_2"] = img_text_token_ids_2
            example["garment_text_token_ids_2"] = garment_text_token_ids_2
        # if self.is_test:
        #     example["null_token_id"] = tokenize_prompt(self.tokenizer, "")[0]
        #     if self.tokenizer_2 is not None:
        #         example["null_token_id"] = tokenize_prompt(self.tokenizer_2, "")[0]
        return example