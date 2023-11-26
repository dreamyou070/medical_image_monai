import torch
from torch import nn
import numpy as np
"""
pdist = nn.PairwiseDistance(p=2, keepdim=True)
input1 = torch.randn(3,3)
print(input1)
input2 = torch.randn(3,3)
print(input2)
output = torch.nn.functional.mse_loss(input1, input2, reduction="none")
print(output)
"""
a = torch.randn((1,4,32,32))
b = torch.randn((1,4,32,32))
latent_distance = [a,b]
c = torch.stack(latent_distance, dim=0).mean(dim=0).mean(dim=0).mean(dim=0)
print(c.shape)
#word_map = expand_image(word_map, 512, 512)
import torch.nn.functional as F

def expand_image(im: torch.Tensor, h = 512, w = 512,
                 absolute: bool = False, threshold: float = None) -> torch.Tensor:
    im = im.unsqueeze(0).unsqueeze(0)
    im = F.interpolate(im.float().detach(), size=(h, w), mode='bicubic')
    if not absolute:
        im = (im - im.min()) / (im.max() - im.min() + 1e-8)
    if threshold:
        im = (im > threshold).float()
    # im = im.cpu().detach()
    return im.squeeze()