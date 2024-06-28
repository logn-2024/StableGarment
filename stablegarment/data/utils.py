import torch
import numpy as np
import random

import cv2
from PIL import Image
from torchvision import transforms
import torchvision.transforms.functional as TF

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

class ImageAugmentation:
    def __init__(self, p_flip=0.5, p_crop=0.3, p_rmask=0.3):
        self.p_flip = p_flip
        self.p_crop = p_crop
        self.p_rmask = p_rmask
        print(f"setting aug prob: flip({p_flip}), crop({p_crop}), rmask({p_rmask})")

    def __call__(self, agn, agn_mask, garment, image, image_densepose):
        '''
            input : numpy.darray
            output: numpy.darray
        '''
        garment = Image.fromarray(garment)
        image = Image.fromarray(image)
        image_densepose = Image.fromarray(image_densepose)

        # random mask
        if torch.rand(1) < self.p_rmask:
            # Image.fromarray(agn_mask, 'L').save("000.jpg")
            sh,sw = np.random.randint(agn_mask.shape[0]),np.random.randint(agn_mask.shape[1])
            eh,ew = np.random.randint(sh,agn_mask.shape[0]),np.random.randint(sw,agn_mask.shape[1])
            agn_mask[agn_mask<128]=0
            if eh-sh>=agn_mask.shape[0]//20 and ew-sw>=agn_mask.shape[1]//20 and agn_mask[sh:eh,sw:ew].sum()>0:
                agn_mask[sh:eh,sw:ew] = 255
                mask = (agn_mask>0).astype(np.uint8)[:,:,None]
                agn = agn * (1-mask) + mask * 128
                # Image.fromarray(agn).save("001.jpg")
        
        # # erase garment
        # erase_p = 0.3
        # erase = transforms.RandomErasing(p=erase_p, scale=(0.02, 0.25), value=1)
        # garment = erase(garment)

        agn = Image.fromarray(agn)
        agn_mask = Image.fromarray(agn_mask, 'L')

        # Random horizontal flipping
        if torch.rand(1) < self.p_flip:
            garment = TF.hflip(garment)
            image = TF.hflip(image)
            image_densepose = TF.hflip(image_densepose)
            agn = TF.hflip(agn)
            agn_mask = TF.hflip(agn_mask)
        
        # Random crop
        if torch.rand(1) < self.p_crop:
            crop_rate = 0.7+random.random()*0.3
            crop_height, crop_width = int(image.size[1]*crop_rate), int(image.size[0]*crop_rate)
            params = transforms.RandomCrop.get_params(image,output_size=(crop_height,crop_width))
            resize = transforms.Resize(size=(image.size[1],image.size[0]))
            
            image = resize(TF.crop(image,*params))
            agn = resize(TF.crop(agn,*params))
            agn_mask = resize(TF.crop(agn_mask,*params))
            image_densepose = resize(TF.crop(image_densepose,*params))

            # # garment crop
            # crop_rate = 0.7+random.random()*0.3
            # crop_height, crop_width = int(image.size[1]*crop_rate), int(image.size[0]*crop_rate)
            # params = transforms.RandomCrop.get_params(garment,output_size=(crop_height,crop_width))
            # resize = transforms.Resize(size=(garment.size[1],garment.size[0]))
            # garment = resize(TF.crop(garment,*params))
        
        return agn, agn_mask, garment, image, image_densepose