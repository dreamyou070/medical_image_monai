import torch
loss_diff  = torch.randn((32,1,3,3))

loss_diff = (loss_diff).mean([1,2,3])
print(loss_diff)
loss_diff = torch.where(loss_diff > 0, loss_diff, 0)
print(loss_diff)
