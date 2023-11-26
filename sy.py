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
loss = torch.randn((3,4,32,32))
loss = loss.sum([1,2,3])
print(loss)
small_mask_info = torch.randn((3,4,32,32))

pixel_num = torch.sum(small_mask_info, dim=(1,2,3))
print(pixel_num)
loss = loss / pixel_num
print(loss)