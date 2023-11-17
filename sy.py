import torch
import numpy as np
normal = True


recon_attn = torch.randn((64,1,6,6))

print(recon_attn.shape)
                    #generator_loss = mse_loss(recon_attn.float(), recon_attn.float())
                    #print(f'recon_attn : {recon_attn}')