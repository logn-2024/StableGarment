import torch
import torch.nn as nn

def _tie_weights(source_module: nn.Module, target_module: nn.Module, only_freeze=False):
    weight_names = [name for name, _ in source_module.named_parameters()]
    for weight_name in weight_names:
        branches = weight_name.split('.')
        base_weight_name = branches.pop(-1)
        source_parent_module = source_module
        target_parent_module = target_module
        for branch in branches:
            source_parent_module = getattr(source_parent_module, branch)
            target_parent_module = getattr(target_parent_module, branch)
        weight = getattr(source_parent_module, base_weight_name)
        if only_freeze:
            if base_weight_name.endswith("bias") or base_weight_name.endswith("weight"):
                weight_real = getattr(target_parent_module, base_weight_name)
                weight_real.to(dtype=weight.dtype)
                weight_real.requires_grad_(False)
            else:
                print("Warning! Skip unknown params: "+base_weight_name)
        else:
            setattr(target_parent_module, base_weight_name, weight)


# From https://github.com/facebookresearch/detectron2/blob/main/detectron2/layers/batch_norm.py # noqa
# Itself from https://github.com/facebookresearch/ConvNeXt/blob/d1fa8f6fef0a165b27399986cc2bdacc92777e40/models/convnext.py#L119  # noqa
class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x

def zero_module(module):
    for p in module.parameters():
        nn.init.zeros_(p)
    return module