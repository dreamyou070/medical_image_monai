import torch

a = torch.tensor(1).fill_(10.0)
print(a)
#label_tensor = torch.tensor(1).fill_(filling_label).type(input.type()).to(input[0].device)