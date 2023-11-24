import torch
import torch.nn.functional as F
from matplotlib import cm
import numpy as np

def _convert_heat_map_colors(heat_map: torch.Tensor):
    def get_color(value):
        return np.array(cm.turbo(value / 255)[0:3])
    # ------------------------------------------------------------------------------------------------------------------
    # color_map = torch, [256,3]
    color_map = torch.tensor(np.array([get_color(i) * 255 for i in range(256)]),device=heat_map.device)

    # ------------------------------------------------------------------------------------------------------------------
    # before, heat_map is from 0 to 1
    heat_map = (heat_map * 255).long()
    print(f'after m')
    # after, heat_map is from 0 to 255
    return color_map[heat_map]

def expand_image(im: torch.Tensor,
                 h = 512, w = 512,
                 absolute: bool = False,
                 threshold: float = None) -> torch.Tensor:
    im = im.unsqueeze(0).unsqueeze(0)
    im = F.interpolate(im.float().detach(), size=(h, w), mode='bicubic')
    if not absolute:
        im = (im - im.min()) / (im.max() - im.min() + 1e-8)
    if threshold:
        im = (im > threshold).float()
    # im = im.cpu().detach()
    return im.squeeze()
