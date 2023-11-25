import torch
loss_diff  = torch.randn((32,1,3,3))

timestep = torch.Tensor([3]).repeat(1)
print(timestep)