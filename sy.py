import torch
anormal_score = torch.randn((2,1,3,3))
print(anormal_score)
#max_value = anormal_score.max(dim=0)
#print(max_value.values.shape)
max = torch.max(anormal_score, dim=-1)
max = torch.max(max.values, dim=-1)
max = max.values.unsqueeze(-1).unsqueeze(-1)
anormal_score = anormal_score / max
print(anormal_score)

