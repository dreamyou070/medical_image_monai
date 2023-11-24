import torch
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
