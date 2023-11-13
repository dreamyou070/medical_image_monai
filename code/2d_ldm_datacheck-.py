import os
import shutil
import tempfile
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from monai import transforms
from monai.apps import MedNISTDataset
from monai.config import print_config
from monai.data import DataLoader, Dataset
from monai.utils import first, set_determinism
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
from generative.inferers import LatentDiffusionInferer
from generative.losses.adversarial_loss import PatchAdversarialLoss
from generative.losses.perceptual import PerceptualLoss
from generative.networks.nets import AutoencoderKL, DiffusionModelUNet, PatchDiscriminator
from generative.networks.schedulers import DDPMScheduler
import argparse


def get_transform(image_size):
    train_transforms = transforms.Compose([transforms.LoadImaged(keys=["image"]),
                                           transforms.EnsureChannelFirstd(keys=["image"]),
                                           transforms.ScaleIntensityRanged(keys=["image"], a_min=0.0, a_max=255.0,
                                                                           b_min=0.0, b_max=1.0, clip=True),
                                           transforms.RandAffined(keys=["image"],
                                                                  rotate_range=[(-np.pi / 36, np.pi / 36),
                                                                                (-np.pi / 36, np.pi / 36)],
                                                                  translate_range=[(-1, 1), (-1, 1)],
                                                                  scale_range=[(-0.05, 0.05), (-0.05, 0.05)],
                                                                  spatial_size=[image_size, image_size],
                                                                  padding_mode="zeros",
                                                                  prob=0.5, ), ])
    val_transforms = transforms.Compose([transforms.LoadImaged(keys=["image"]),
                                         transforms.EnsureChannelFirstd(keys=["image"]),
                                         transforms.ScaleIntensityRanged(keys=["image"], a_min=0.0, a_max=255.0,
                                                                         b_min=0.0, b_max=1.0, clip=True), ])
    return train_transforms, val_transforms


def main(args):
    print(f'\n step 1. print version related')
    print_config()

    print(f' (1.1) set deterministic training for reproducibility')
    set_determinism(args.seed)

    print(f' (1.2) data directory and download dataset')
    # there is no datadirectory ...
    environment_argument = os.environ
    directory = environment_argument.get("MONAI_DATA_DIRECTORY")
    # make temp directory
    root_dir = args.root_dir
    os.makedirs(root_dir, exist_ok=True)

    print(f'\n step 2. dataset and dataloader')
    print(f' (2.1) train dataset')
    train_data = MedNISTDataset(root_dir=root_dir, section="training",  # download=True,
                                seed=0)
    train_datalist = []
    for item in train_data.data:
        if item["class_name"] == "Hand":
            train_datalist.append({"image": item["image"]})
    train_transforms, val_transforms = get_transform(args.image_size)
    train_ds = Dataset(data=train_datalist, transform=train_transforms)

    """
    print(f' (2.1.2) load dataloader')
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4,
                              persistent_workers=True)

    print(f' (2.1.3) visualise examples from the training set')
    print(f' (2.1.3.1) get first datas')
    check_data = first(train_loader)
    vis_num_images = args.vis_num_images
    fig, ax = plt.subplots(nrows=1, ncols=vis_num_images)
    for image_n in range(vis_num_images):
        ax[image_n].imshow(check_data["image"][image_n, 0, :, :], cmap="gray")
        ax[image_n].axis("off")
    plt.show()

    print(f' (2.2) validation dataset')
    val_data = MedNISTDataset(root_dir=root_dir, section="validation", download=True, seed=0)
    val_datalist = [{"image": item["image"]} for item in val_data.data if item["class_name"] == "Hand"]
    val_ds = Dataset(data=val_datalist, transform=val_transforms)
    print(f' (2.2.2) validation dataloader')
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, persistent_workers=True)

    print(f'\n step 3. model')
    print(f' (3.0) device')
    device = torch.device("cuda")
    print(f' (3.1) vae(autoencoder)')
    autoencoderkl = AutoencoderKL(spatial_dims=2, in_channels=1, out_channels=1,
                                  num_channels=(128, 128, 256), latent_channels=3, num_res_blocks=2,
                                  attention_levels=(False, False, False),
                                  with_encoder_nonlocal_attn=False, with_decoder_nonlocal_attn=False, ).to(device)

    print(f' (3.2) discriminator')
    # what is patchDiscriminator ?
    discriminator = PatchDiscriminator(spatial_dims=2, num_layers_d=3, num_channels=64,
                                       in_channels=1, out_channels=1).to(device)

    print(f' (3.3) perceptual_loss')
    perceptual_loss = PerceptualLoss(spatial_dims=2, network_type="alex").to(device)
    perceptual_weight = 0.001

    print(f' (3.4) patch adversarial loss')
    adv_loss = PatchAdversarialLoss(criterion="least_squares")
    adv_weight = 0.01

    print(f' (3.5) optimizer (for generator and discriminator)')
    optimizer_g = torch.optim.Adam(autoencoderkl.parameters(), lr=1e-4)
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=5e-4)

    print(f' (3.6) mixed precision (for generator and discriminator)')
    scaler_g = torch.cuda.amp.GradScaler()
    scaler_d = torch.cuda.amp.GradScaler()

    print(f'step 4. training (takes about one hour)')
    kl_weight = 1e-6
    n_epochs = 100
    val_interval = 10
    autoencoder_warm_up_n_epochs = 10
    epoch_recon_losses = []
    epoch_gen_losses = []
    epoch_disc_losses = []
    val_recon_losses = []
    intermediary_images = []
    num_example_images = 4

    for epoch in range(n_epochs):

        print(f' epoch {epoch + 1}/{n_epochs}')

        autoencoderkl.train()
        discriminator.train()

        epoch_loss = 0
        gen_epoch_loss = 0
        disc_epoch_loss = 0

        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), ncols=110)
        progress_bar.set_description(f"Epoch {epoch}")

        for step, batch in progress_bar:
            print(batch.__dict__)

    
    """


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--root_dir", type=str, default='../experiment')
    parser.add_argument("--image_size", type=int, default=64)
    parser.add_argument("--vis_num_images", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device", type=str, default='cuda')
    args = parser.parse_args()
    main(args)