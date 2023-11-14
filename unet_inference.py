import argparse
import torch
from model_module.nets import AutoencoderKL, DiffusionModelUNet
from model_module.schedulers import DDPMScheduler
from utils.set_seed import set_determinism
from model_module.inferers import LatentDiffusionInferer

def main(args) :

    print(f' \n step 0. device')
    device = torch.device(args.device)

    print(f' \n step 1. model')
    print(f' (1.1) autoencoder')
    autoencoderkl = AutoencoderKL(spatial_dims=2, in_channels=1, out_channels=1,
                                  num_channels=(128, 128, 256), latent_channels=3,
                                  num_res_blocks=2, attention_levels=(False, False, False),
                                  with_encoder_nonlocal_attn=False, with_decoder_nonlocal_attn=False, )
    autoencoderkl = autoencoderkl.to(device)
    autoencoderkl.eval()
    state_dict = torch.load(args.autoencoder_pretrained_dir, map_location='cpu')['model']
    msg = autoencoderkl.load_state_dict(state_dict, strict=False)

    print(f' (1.2) unet')
    unet = DiffusionModelUNet(spatial_dims=2,
                              in_channels=3,
                              out_channels=3,
                              num_res_blocks=2,
                              num_channels=(128, 256, 512),
                              attention_levels=(False, True, True),
                              num_head_channels=(0, 256, 512), )
    unet = unet.to(device)
    unet.eval()
    unet_state_dict = torch.load(args.unet_pretrained_dir, map_location='cpu')['model']
    msg = unet.load_state_dict(unet_state_dict, strict=False)

    print(f' (1.3) scheduler')
    scheduler = DDPMScheduler(num_train_timesteps=1000, schedule="linear_beta", beta_start=0.0015, beta_end=0.0195)
    scheduler.set_timesteps(num_inference_steps=1000)

    print(f' \n step 2. inference')
    print(f' (2.1) pipeline')
    scale_factor
    pipeline = LatentDiffusionInferer(scheduler,
                                      scale_factor=scale_factor)
    set_determinism(args.seed)

    print(f' (1.3) scheduler')
    noise = torch.randn((1, 3, 16, 16))
    noise = noise.to(device)

    with torch.no_grad():
        image, intermediates = inferer.sample(input_noise=noise,
                                              diffusion_model=unet,
                                              scheduler=scheduler,
                                              save_intermediates=True,
                                              intermediate_steps=100,
                                              autoencoder_model=autoencoderkl,)

if __name__ == '__main__' :
    parser = argparse.ArgumentParser()
    # step 0. device
    parser.add_argument('--device', type=str, default='cuda:4')
    parser.add_argument('--autoencoder_pretrained_dir', type=str,
                        default='/data7/sooyeon/medical_image/experiment_result/vae_model/vae_checkpoint_100.pth')
    parser.add_argument('--unet_pretrained_dir', type=str,
                        default='/data7/sooyeon/medical_image/experiment_result/unet_model/unet_checkpoint_200.pth')
    args = parser.parse_args()
    main(args)
