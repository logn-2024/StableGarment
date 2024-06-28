import bisect
from torch.utils.data import Dataset

from .vitonhd import VITONHDDataset
from .dresscode import DressCodeDataset
from .synthesis import SyntheticDataset
from ..utils.util import tokenize_prompt



class PackedDataset(Dataset):
    def __init__(
            self, 
            vitonhd_dir,
            dresscode_dir, 
            img_H, 
            img_W, 
            tokenizer,
            tokenizer_2=None,
            is_sorted=False,      
            **kwargs
        ):
        vthd_repeat,dcup_repeat,dcdr_repeat,dclw_repeat = 1.5,1.5,1,2
        self.vitonhd_dataset = VITONHDDataset(data_root_dir = vitonhd_dir, img_H = img_H, img_W = img_W, tokenizer = tokenizer, tokenizer_2 = tokenizer_2, 
            is_paired = True, is_test = False, is_sorted = is_sorted, repeat=vthd_repeat, **kwargs)
        self.dresscode_upper = DressCodeDataset(data_root_dir = dresscode_dir, img_H = img_H, img_W = img_W, tokenizer = tokenizer, tokenizer_2 = tokenizer_2, 
            is_paired = True, is_test = False, is_sorted = is_sorted, category="upper_body", repeat=dcup_repeat, **kwargs)
        self.dresscode_dress = DressCodeDataset(data_root_dir = dresscode_dir, img_H = img_H, img_W = img_W, tokenizer = tokenizer, tokenizer_2 = tokenizer_2, 
            is_paired = True, is_test = False, is_sorted = is_sorted, category="dresses", repeat=dcdr_repeat, **kwargs)
        self.dresscode_lower = DressCodeDataset(data_root_dir = dresscode_dir, img_H = img_H, img_W = img_W, tokenizer = tokenizer, tokenizer_2 = tokenizer_2, 
            is_paired = True, is_test = False, is_sorted = is_sorted, category="lower_body", repeat=dclw_repeat, **kwargs)
        
        self.all_datasets = [self.vitonhd_dataset,self.dresscode_upper,self.dresscode_dress,self.dresscode_lower,]

        self.lens_cum = [0]
        for dataset in self.all_datasets:
            self.lens_cum.append(self.lens_cum[-1] + len(dataset)) 
        self.lens_cum = self.lens_cum[1:]

        self.null_id = tokenize_prompt(tokenizer, "")
        if tokenizer_2 is not None:
            self.null_id_2 = tokenize_prompt(tokenizer_2, "")

        self.start_signal = True

    def __len__(self):
        return self.lens_cum[-1]

    def __getitem__(self, idx):
        if not self.start_signal:
            return {}

        id_ds = bisect.bisect_right(self.lens_cum,idx)
        idx = idx if id_ds==0 else idx-self.lens_cum[id_ds-1]
        return self.all_datasets[id_ds][idx]

class GenerationDataset(Dataset):
    def __init__(
            self, 
            vitonhd_dir,
            dresscode_dir,
            synthesis_dir, 
            img_H, 
            img_W, 
            tokenizer,
            tokenizer_2=None,
            is_sorted=False,      
            **kwargs
        ):
        self.vitonhd_dataset = VITONHDDataset(data_root_dir = vitonhd_dir, img_H = img_H, img_W = img_W, tokenizer = tokenizer, tokenizer_2 = tokenizer_2, is_paired = True, is_test = False, is_sorted = is_sorted, **kwargs)
        self.dresscode_upper = DressCodeDataset(data_root_dir = dresscode_dir, img_H = img_H, img_W = img_W, tokenizer = tokenizer, tokenizer_2 = tokenizer_2, is_paired = True, is_test = False, is_sorted = is_sorted, category="upper_body", **kwargs)
        self.dresscode_dress = DressCodeDataset(data_root_dir = dresscode_dir, img_H = img_H, img_W = img_W, tokenizer = tokenizer, tokenizer_2 = tokenizer_2, is_paired = True, is_test = False, is_sorted = is_sorted, category="dresses", **kwargs)
        self.dresscode_lower = DressCodeDataset(data_root_dir = dresscode_dir, img_H = img_H, img_W = img_W, tokenizer = tokenizer, tokenizer_2 = tokenizer_2, is_paired = True, is_test = False, is_sorted = is_sorted, category="lower_body", **kwargs)
        self.synthesis_dataset = SyntheticDataset(data_root_dir = synthesis_dir, img_H = img_H, img_W = img_W, tokenizer = tokenizer, tokenizer_2 = tokenizer_2, is_sorted = is_sorted, **kwargs)

        self.all_datasets = [self.vitonhd_dataset,self.dresscode_upper,self.dresscode_dress,self.dresscode_lower,self.synthesis_dataset,]

        self.lens_cum = [0]
        for dataset in self.all_datasets:
            self.lens_cum.append(self.lens_cum[-1] + len(dataset)) 
        self.lens_cum = self.lens_cum[1:]
        
        self.null_id = tokenize_prompt(tokenizer, "")
        if tokenizer_2 is not None:
            self.null_id_2 = tokenize_prompt(tokenizer_2, "")

        self.start_signal = True

    def __len__(self):
        return self.lens_cum[-1]

    def __getitem__(self, idx):
        if not self.start_signal:
            return {}

        id_ds = bisect.bisect_right(self.lens_cum,idx)
        idx = idx if id_ds==0 else idx-self.lens_cum[id_ds-1]
        return self.all_datasets[id_ds][idx]