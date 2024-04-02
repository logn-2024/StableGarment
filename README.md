# StableGarment: Garment-Centric Generation via Stable Diffusion
This repository is the official implementation of [StableGarment](http://arxiv.org/abs/2403.10783).

[[Arxiv Paper](http://arxiv.org/abs/2403.10783)]&nbsp;
[[Website Page](https://raywang335.github.io/stablegarment.github.io/)]&nbsp;

![teaser](assets/teaser.jpg)&nbsp;

## Environments
```bash
git clone https://github.com/logn-2024/StableGarment
cd StableGarment

conda create --name StableGarment python=3.11 -y
conda activate StableGarment

pip3 install -r requirements.txt
```

## Demos, Models and Data
You can follow [VITON-HD](https://github.com/shadow2496/VITON-HD) and [Dress Code](https://github.com/aimagelab/dress-code) to get VITON-HD and Dress Code dataset respectively. You may run the following command to generate mask for Dress Code dataset and place it in corresponding directory before test.
```bash
python stablegarment/data/generate_mask.py
```
You can get pretrained garment encoder for text2img from [this](https://huggingface.co/loooooong/StableGarment_text2img) huggingface Repository. Our huggingface demo is available here [![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/loooooong/StableGarment). You can also run the demo locally:
```bash
python app.py
```

## Inference
Run following command to perform text2img task with garment condition. You can change base model to get different style.
```bash
python infer_t2i.py
```
Try-on task requires more inputs, mostly related to human. You can find corresponding extra inputs from VITON-HD and Dress Code dataset. If you want to perform virtual try-on on arbitrary images, you should get densepose and agnostic mask as these in the VITON-HD dataset. To perform virtual try-on application, Run following code for example:
```bash
python infer_tryon.py
```

## Test
To test StableGarment for VITON-HD dataset, run following command:
```bash
python test.py
```
You can change paired and unpaired setting by changeing is_pair variable. To test Dress Code dataset, just replace related variables and load target dataset in the test.py. 

**Acknowledgements** 

Thanks to [magic-animate](https://github.com/magic-research/magic-animate/), our code is heavily based on it. 

## Citation
If you find our work useful for your research, please cite us:
```
@article{wang2024stablegarment,
  title={StableGarment: Garment-Centric Generation via Stable Diffusion},
  author={Wang, Rui and Guo, Hailong and Liu, Jiaming and Li, Huaxia and Zhao, Haibo and Tang, Xu and Hu, Yao and Tang, Hao and Li, Peipei},
  journal={arXiv preprint arXiv:2403.10783},
  year={2024}
}
```

## License
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).