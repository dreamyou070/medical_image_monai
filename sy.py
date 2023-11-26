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

print(c.shape)