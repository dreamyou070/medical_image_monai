import numpy as np
from matplotlib import pyplot as plt
import torch
import argparse
from generative.networks.nets import AutoencoderKL


def main(args) :

    print(f' \n step 0. device')
    device = torch.device(args.device)

    print(f' \n step 1. make empty model')
    autoencoderkl = AutoencoderKL(spatial_dims=2,in_channels=1,out_channels=1,
                                  num_channels=(128, 128, 256),latent_channels=3,
                                  num_res_blocks=2,attention_levels=(False, False, False),
                                  with_encoder_nonlocal_attn=False,with_decoder_nonlocal_attn=False,)
    autoencoderkl = autoencoderkl.to(device)

    print(f' \n step 2. model loading')
    state_dict = torch.load(args.pretrained_dir, map_location='cpu')['model']
    msg = autoencoderkl.load_state_dict(state_dict, strict=False)

    """
    with autocast(enabled=True):
        reconstruction, z_mu, z_sigma = autoencoderkl(images)
    # Plot last 5 evaluations
    # np.linspace means,
    n_epochs = 100
    val_interval = 10
    spave = int(n_epochs / val_interval)
    val_samples = np.linspace(n_epochs, val_interval, int(n_epochs / val_interval))
    print(val_samples)
    fig, ax = plt.subplots(nrows=args.infer_num, ncols=1, sharey=True)
    for image_n in range(5):
        reconstructions = torch.reshape(intermediary_images[image_n], (image_size * num_example_images, image_size)).T
        ax[image_n].imshow(reconstructions.cpu(), cmap="gray")
        ax[image_n].set_xticks([])
        ax[image_n].set_yticks([])
        ax[image_n].set_ylabel(f"Epoch {val_samples[image_n]:.0f}")
    

    
    """


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    # step 0. device
    parser.add_argument('--device', type=str, default='cuda:0')
    # step 1.
    parser.add_argument('--infer_num', type=int, default=5)
    # step 2. model loading
    parser.add_argument('--pretrained_dir', type=str, default='/data7/sooyeon/medical_image/model/checkpoint_25.pth')

    args = parser.parse_args()
    main(args)
