import torch
loss_diff  = torch.randn((32,1,3,3))

for t in range(150 - 1, -1, -1):
    print(t)