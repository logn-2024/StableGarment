import torch
from torch.utils.data import Dataset

import cv2
import numpy as np

import json
import os
from os.path import join as opj

def imread(p, h, w, is_mask=False, img=None):
    if img is None:
        img = cv2.imread(p)
    if not is_mask:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (w,h))
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (w,h))
    return img



class VITONHDInferenceDataset(Dataset):
    def __init__(
            self, 
            data_root_dir, 
            img_H, 
            img_W, 
            tokenizer,
            is_paired=True, 
            is_test=False, 
            is_sorted=False,      
            **kwargs
        ):
        self.drd = data_root_dir
        self.img_H = img_H
        self.img_W = img_W
        self.pair_key = "paired" if is_paired else "unpaired"
        self.data_type = "train" if not is_test else "test"
        self.is_test = is_test
       
        assert not (self.data_type == "train" and self.pair_key == "unpaired"), "train must use paired dataset"
        
        im_names = []
        c_names = []
        with open(opj(self.drd, "test_pairs.txt"), "r") as f:
            for line in f.readlines():
                im_name, c_name = line.strip().split()
                im_names.append(im_name)
                c_names.append(c_name)
        
        self.img_text_data,self.cloth_text_data = {},{}
        image_caption_json = opj(self.drd, f"{self.data_type}_image_text_.json")
        if os.path.exists(image_caption_json):
            with open(image_caption_json, "r") as f:
                self.img_text_data = json.load(f)
        cloth_caption_json = opj(self.drd, f"{self.data_type}_cloth_text_.json")
        if os.path.exists(cloth_caption_json):
            with open(cloth_caption_json, "r") as f:
                self.cloth_text_data = json.load(f)
        print(f"img_text_data: {len(self.img_text_data)}, cloth_text_data: {len(self.cloth_text_data)}")
        for im_name in im_names:
            if im_name not in self.img_text_data:
                self.img_text_data[im_name] = ""
        for c_name in c_names:
            if c_name not in self.cloth_text_data:
                self.cloth_text_data[c_name] = ""
            
        if is_sorted:
            im_names, c_names = zip(*sorted(zip(im_names, c_names)))
        self.im_names = im_names
        self.tokenizer = tokenizer
        self.c_names = dict()
        self.c_names["paired"] = im_names
        self.c_names["unpaired"] = c_names
        self.token_id = tokenizer.encode(
            "",
            padding="max_length",
            max_length=77,
            truncation=True,
            return_tensors="pt",
        )

    def __len__(self):
        return len(self.im_names)
        
    def __getitem__(self, idx):
        img_fn = self.im_names[idx]
        cloth_fn = self.c_names[self.pair_key][idx]
        img_text_cnt = self.img_text_data[img_fn]
        
        img_text_token_ids = self.tokenizer.encode(img_text_cnt, padding="max_length", max_length=77, truncation=True, return_tensors="pt")

        cloth_text_cnt = self.cloth_text_data[cloth_fn]
        cloth_text_token_ids = self.tokenizer.encode(cloth_text_cnt, padding="max_length", max_length=77, truncation=True, return_tensors="pt")

        agn = imread(opj(self.drd, self.data_type, "agnostic-v3.2", self.im_names[idx]), self.img_H, self.img_W)
        # Note: [agnostic_mask_official -> agnostic-mask, 00006_00.jpg -> 00006_00_mask.png]
        mask_name = os.path.splitext(self.im_names[idx])[0]+"_mask.png" # self.im_names[idx]
        agn_mask = imread(opj(self.drd, self.data_type, "agnostic-mask", mask_name), self.img_H, self.img_W, is_mask=True)

        cloth = imread(opj(self.drd, self.data_type, "cloth", cloth_fn), self.img_H, self.img_W)
        image = imread(opj(self.drd, self.data_type, "image", self.im_names[idx]), self.img_H, self.img_W)
        image_densepose = imread(opj(self.drd, self.data_type, "image-densepose", self.im_names[idx]), self.img_H, self.img_W)
        
        agn_mask = np.array(agn_mask)
        agn_mask = (agn_mask >= 128).astype(np.float32)  # 0 or 1
        agn_mask = agn_mask[:,:,None]
        agn_mask = 1. - agn_mask

        # normalize
        agn = (np.array(agn).astype(np.float32) / 127.5) - 1.
        cloth = (np.array(cloth).astype(np.float32) / 127.5) - 1.
        image = (np.array(image).astype(np.float32) / 127.5) - 1.
        image_densepose = (np.array(image_densepose).astype(np.float32) / 127.5) - 1.

        # np to tensor
        agn = torch.from_numpy(agn.transpose(2, 0, 1)).float()
        agn_mask = torch.from_numpy(agn_mask.transpose(2, 0, 1)).float()
        cloth = torch.from_numpy(cloth.transpose(2, 0, 1)).float()
        image = torch.from_numpy(image.transpose(2, 0, 1)).float()
        image_densepose = torch.from_numpy(image_densepose.transpose(2, 0, 1)).float()

        return dict(
            agn=agn,
            agn_mask=agn_mask,
            cloth=cloth,
            image=image,
            image_densepose=image_densepose,
            img_text_token_ids=img_text_token_ids.squeeze(0),
            cloth_text_token_ids=cloth_text_token_ids.squeeze(0),
            img_fn=img_fn,
            cloth_fn=cloth_fn,
            null_token_id=self.token_id[0],
        )
    
if __name__ == "__main__":
    from transformers import CLIPTokenizer
    tokenizer = CLIPTokenizer.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="tokenizer")

    root_data_dir = "/home/ghl/workspace/data/VITON-HD/zalando-hd-resized"
    img_H = 512
    img_W = 384
    is_pair = False
    is_test = True
    is_sorted = True

    inference_data = VITONHDInferenceDataset(
        data_root_dir = root_data_dir,
        img_H = img_H,
        img_W = img_W,
        tokenizer = tokenizer,
        is_paired = is_pair,
        is_test = is_test,
        is_sorted = is_sorted,
    )

    inference_data_loader = torch.utils.data.DataLoader(
        inference_data,
        batch_size=2,
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )

    for idx, batch in enumerate(inference_data_loader):
        print(batch["null_token_id"].shape,)
        print(batch["img_fn"],batch["img_text_token_ids"].shape,)
        print(batch["cloth_fn"],batch["cloth_text_token_ids"].shape,)
        print(batch["agn"].shape,batch["agn"].max(),batch["agn"].min(),)
        print(batch["agn_mask"].shape,batch["agn_mask"].max(),batch["agn_mask"].min(),)
        print(batch["cloth"].shape,batch["cloth"].max(),batch["cloth"].min(),)
        print(batch["image"].shape,batch["image"].max(),batch["image"].min(),)
        print(batch["image_densepose"].shape,batch["image_densepose"].max(),batch["image_densepose"].min(),)


    