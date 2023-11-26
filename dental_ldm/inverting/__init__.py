import torch
from typing import Optional, Union, Tuple, List, Callable, Dict
import numpy as np


class Inversor (object):

    def __init__(self, unet, scheduler, vae, scale_factor):
        self.unet = unet
        self.scheduler = scheduler
        self.vae = vae
        self.scale_factor = scale_factor
    @torch.no_grad()
    def img2latent(self, img: torch.Tensor):

        posterior = self.vae.encode(img).latent_dist
        latents = posterior.mode() * self.scale_factor
        return latents

    @torch.no_grad()
    def latent2img(self, latent: torch.Tensor, return_type='np'):
        img = self.vae.decode(latent/self.scale_factor, return_dict=True, generator=None).sample
        if return_type == 'np':
            image = (img / 2 + 0.5).clamp(0, 1)
            image = image.cpu().permute(0, 2, 3, 1).numpy()[0]
            img = (image * 255).astype(np.uint8)
        return img

    @torch.no_grad()
    def ddim_loop(self, img, inversion_steps):
        latent = self.img2latent(img)
        all_latents = [latent]
        for t in range(inversion_steps):
            noise_pred = self.unet(latent, t).sample
            latent = self.one_step_noising(noise_pred, t, latent)
            all_latents.append(latent)
        return all_latents
    def one_step_noising(self,
                         model_output: Union[torch.FloatTensor, np.ndarray], timestep: int,
                         sample: Union[torch.FloatTensor, np.ndarray]):
        timestep, next_timestep = timestep , timestep+1
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep] #if timestep >= 0 else self.scheduler.final_alpha_cumprod
        alpha_prod_t_next = self.scheduler.alphas_cumprod[next_timestep]
        beta_prod_t = 1 - alpha_prod_t
        next_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
        next_sample_direction = (1 - alpha_prod_t_next) ** 0.5 * model_output
        next_sample = alpha_prod_t_next ** 0.5 * next_original_sample + next_sample_direction
        return next_sample

    def prev_step(self,
                  model_output: Union[torch.FloatTensor, np.ndarray],
                  timestep: int,
                  sample: Union[torch.FloatTensor, np.ndarray]):
        prev_sample = self.scheduler.step(model_output, timestep, sample=sample)
        return prev_sample

    def gen_loop(self, latent, timestep):
        all_latents = [latent]
        for t in range(timestep- 1, -1, -1) :
            print(f'latent type = {type(latent)}')
            noise_pred = self.unet(latent, t).sample
            latent = self.prev_step(noise_pred, t, latent)
            all_latents.append(latent)
        return all_latents

