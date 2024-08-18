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


category2text = {
    "upper_body": "upper body",
    "lower_body": "lower body",
    "dresses": "dress",
}

class DressCodeDataset(Dataset):
    def __init__(
            self, 
            data_root_dir, 
            img_H, 
            img_W, 
            tokenizer,
            tokenizer_2=None,
            is_paired=True, 
            is_test=False, 
            is_sorted=False,
            category="upper_body",
            repeat=1,
            **kwargs
        ):
        self.drd = data_root_dir
        self.img_H = img_H
        self.img_W = img_W
        self.pair_key = "paired" if is_paired else "unpaired"
        self.data_type = "train" if not is_test else "test"
        self.is_test = is_test
        self.category = category
        self.repeat = repeat
       
        assert not (self.data_type == "train" and self.pair_key == "unpaired"), "train must use paired dataset"
        self.drd = opj(self.drd,category)

        im_names = []
        c_names = []
        pairs_txt_name = f"test_pairs_{self.pair_key}.txt" if self.data_type=="test" else "train_pairs.txt"
        with open(opj(self.drd, pairs_txt_name), "r") as f:
            for line in f.readlines():
                im_name, c_name = line.strip().split()
                im_names.append(im_name)
                c_names.append(c_name)

        self.img_text_data,self.garment_text_data = {},{}
        image_caption_json = opj(self.drd, "img_cnt.json")
        if os.path.exists(image_caption_json):
            with open(image_caption_json, "r") as f:
                self.img_text_data = json.load(f)
        garment_caption_json = opj(self.drd, "cloth_cnt.json")
        if os.path.exists(garment_caption_json):
            with open(garment_caption_json, "r") as f:
                self.garment_text_data = json.load(f)
        print(f"img_text_data: {len(self.img_text_data)}, garment_text_data: {len(self.garment_text_data)}")
        for im_name in im_names:
            if im_name not in self.img_text_data:
                self.img_text_data[im_name] = ""
        for c_name in c_names:
            if c_name not in self.garment_text_data:
                self.garment_text_data[c_name] = ""
    
        if is_sorted:
            im_names, c_names = zip(*sorted(zip(im_names, c_names)))
        self.im_names = im_names
        # self.text_names = new_text_dict
        self.tokenizer = tokenizer
        self.tokenizer_2 = tokenizer_2
        self.c_names = dict()
        self.c_names["paired"] = c_names
        self.c_names["unpaired"] = c_names
        self.null_id = tokenize_prompt(self.tokenizer, "")
        if self.tokenizer_2 is not None:
            self.null_id_2 = tokenize_prompt(self.tokenizer_2, "")
        
        self.augmentation = ImageAugmentation(**kwargs)

        self.start_signal = True
    def __len__(self):
        return int(len(self.im_names)*self.repeat)

    def __getitem__(self, idx):
        if not self.start_signal:
            return {}
        idx = idx%len(self.im_names)
        img_fn = self.im_names[idx]
        garment_fn = self.c_names[self.pair_key][idx]

        img_text_cnt = "best quality, a photo of a woman wearing fashion garment" if self.is_test else self.img_text_data[img_fn]
        img_text_token_ids = tokenize_prompt(self.tokenizer,img_text_cnt)[0]
        garment_text_cnt = f"{category2text[self.category]} cloth" if self.is_test else self.garment_text_data[garment_fn]
        garment_text_token_ids = tokenize_prompt(self.tokenizer,garment_text_cnt)[0]
        if self.tokenizer_2 is not None:
            img_text_token_ids_2 = tokenize_prompt(self.tokenizer_2, img_text_cnt)[0]
            garment_text_token_ids_2 = tokenize_prompt(self.tokenizer_2, garment_text_cnt)[0]

        version_suffix = ""
        agn = imread(opj(self.drd, "agnostic"+version_suffix, img_fn), self.img_H, self.img_W)
        agn_mask = imread(opj(self.drd, "agnostic-mask"+version_suffix, img_fn), self.img_H, self.img_W, is_mask=True)
        garment = imread(opj(self.drd, "images", garment_fn), self.img_H, self.img_W)

        image = imread(opj(self.drd, "images", img_fn), self.img_H, self.img_W)
        image_densepose = imread(opj(self.drd, "densepose", img_fn), self.img_H, self.img_W)

        if not self.is_test:
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
        negative_prompt = "nsfw, bad quality, worst quality, bad anatomy, wrong anatomy, extra limb, missing limb, floating limbs, disconnected limbs, mutation, mutated, ugly, disgusting, amputation"
        if self.tokenizer_2 is not None:
            example["img_text_token_ids_2"] = img_text_token_ids_2
            example["garment_text_token_ids_2"] = garment_text_token_ids_2
        if self.is_test:
            example["null_token_id"] = tokenize_prompt(self.tokenizer, negative_prompt)[0]
            if self.tokenizer_2 is not None:
                example["null_token_id_2"] = tokenize_prompt(self.tokenizer_2, negative_prompt)[0]
        return example