import os
import wandb
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
import argparse
from generative.inferers import LatentDiffusionInferer
from generative.losses.adversarial_loss import PatchAdversarialLoss
from generative.losses.perceptual import PerceptualLoss
from generative.networks.nets import AutoencoderKL, DiffusionModelUNet, PatchDiscriminator
from generative.networks.schedulers import DDPMScheduler
from data_module import get_transform
def main(args) :

    print(f'\n step 1. wandb login')
    wandb.login(key=args.wandb_api_key)
    wandb.init(project=args.wandb_project_name, name=args.wandb_run_name)

    print(f'\n step 2. setting')
    print_config()
    set_determinism(args.seed)

    print(f'\n step 3. dataset')
    print(f' (3.1) train dataset and dataloader')
    train_datas = os.listdir(args.data_folder)
    train_datalist = [{"image": os.path.join(args.data_folder, train_data)} for train_data in train_datas]
    train_transforms, valid_transforms = get_transform(args.img_size)
    train_ds = Dataset(data=train_datalist, transform=train_transforms)
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=4, persistent_workers=True)
    check_data = first(train_loader)

    print(f'\n step 4. autoencoder')
    device = args.device
    print(f' (4.1) autoencoder')
    autoencoderkl = AutoencoderKL(spatial_dims=2,in_channels=1,out_channels=1,num_channels=(64, 128, 128, 128),
                                  latent_channels=3,num_res_blocks=2,attention_levels=(False, False, False, False),
                                  with_encoder_nonlocal_attn=False,with_decoder_nonlocal_attn=False,)
    autoencoderkl = autoencoderkl.to(device)
    print(f' (4.2) discriminator')
    discriminator = PatchDiscriminator(spatial_dims=2, num_layers_d=3, num_channels=64, in_channels=1, out_channels=1)
    discriminator = discriminator.to(device)
    print(f' (4.3) loss 1. perceptual loss')
    perceptual_loss = PerceptualLoss(spatial_dims=2, network_type="squeeze")
    perceptual_loss.to(device)
    perceptual_weight = 0.001
    print(f' (4.4) loss 2. adversarial loss')
    adv_loss = PatchAdversarialLoss(criterion="least_squares")
    adv_weight = 0.01

    print(f'\n step 5. optimizer')
    optimizer_g = torch.optim.Adam(autoencoderkl.parameters(), lr=1e-4)
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=0.0001)
    print(f' (5.2) precision')
    scaler_g = torch.cuda.amp.GradScaler()
    scaler_d = torch.cuda.amp.GradScaler()

    print(f'\n step 6. autoencoder training')
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
        autoencoderkl.train()
        discriminator.train()
        epoch_loss = 0
        gen_epoch_loss = 0
        disc_epoch_loss = 0
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), ncols=110)
        progress_bar.set_description(f"Epoch {epoch}")
        for step, batch in progress_bar:
            images = batch["image"].to(device)
            optimizer_g.zero_grad(set_to_none=True)
            with autocast(enabled=True):
                reconstruction, z_mu, z_sigma = autoencoderkl(images)
                recons_loss = F.l1_loss(reconstruction.float(), images.float())
                p_loss = perceptual_loss(reconstruction.float(), images.float())
                kl_loss = 0.5 * torch.sum(z_mu.pow(2) + z_sigma.pow(2) - torch.log(z_sigma.pow(2)) - 1, dim=[1, 2, 3])
                kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]
                loss_g = recons_loss + (kl_weight * kl_loss) + (perceptual_weight * p_loss)
                if epoch > autoencoder_warm_up_n_epochs:
                    logits_fake = discriminator(reconstruction.contiguous().float())[-1]
                    generator_loss = adv_loss(logits_fake, target_is_real=True, for_discriminator=False)
                    loss_g += adv_weight * generator_loss
            scaler_g.scale(loss_g).backward()
            scaler_g.step(optimizer_g)
            scaler_g.update()
            if epoch > autoencoder_warm_up_n_epochs:
                with autocast(enabled=True):
                    optimizer_d.zero_grad(set_to_none=True)
                    logits_fake = discriminator(reconstruction.contiguous().detach())[-1]
                    loss_d_fake = adv_loss(logits_fake, target_is_real=False, for_discriminator=True)
                    logits_real = discriminator(images.contiguous().detach())[-1]
                    loss_d_real = adv_loss(logits_real, target_is_real=True, for_discriminator=True)
                    discriminator_loss = (loss_d_fake + loss_d_real) * 0.5
                    loss_d = adv_weight * discriminator_loss
                scaler_d.scale(loss_d).backward()
                scaler_d.step(optimizer_d)
                scaler_d.update()
            epoch_loss += recons_loss.item()
            if epoch > autoencoder_warm_up_n_epochs:
                gen_epoch_loss += generator_loss.item()
                disc_epoch_loss += discriminator_loss.item()
            progress_bar.set_postfix({"recons_loss": epoch_loss / (step + 1),
                                      "gen_loss": gen_epoch_loss / (step + 1),
                                      "disc_loss": disc_epoch_loss / (step + 1),})
        epoch_recon_losses.append(epoch_loss / (step + 1))
        epoch_gen_losses.append(gen_epoch_loss / (step + 1))
        epoch_disc_losses.append(disc_epoch_loss / (step + 1))
    progress_bar.close()
    del discriminator
    del perceptual_loss
    torch.cuda.empty_cache()
    print(f' (6.2) autoencoder saving')
    experiment_basic_dir = args.experiment_basic_dir
    os.makedirs(experiment_basic_dir, exist_ok=True)
    save_obj = {'model': autoencoderkl.state_dict(), }
    torch.save(save_obj, os.path.join(experiment_basic_dir, f'vae_checkpoint_{epoch + 1}.pth'))

    print(f'\n step 7. make diffusion model')
    unet = DiffusionModelUNet(spatial_dims=2,in_channels=3,out_channels=3,num_res_blocks=2,
                              num_channels=(256, 512, 768),attention_levels=(False, True, True),num_head_channels=(0, 512, 768),)
    scheduler = DDPMScheduler(num_train_timesteps=1000, schedule="scaled_linear",
                              beta_start=0.0015, beta_end=0.0205)
    with torch.no_grad():
        with autocast(enabled=True):
            z = autoencoderkl.encode_stage_2_inputs(check_data["image"].to(device))
    print(f"  scaling factor set to {1 / torch.std(z)}")
    scale_factor = 1 / torch.std(z)
    inferer = LatentDiffusionInferer(scheduler, scale_factor=scale_factor)

    print(f'\n step 6. diffusion training')
    optimizer = torch.optim.Adam(unet.parameters(), lr=0.0001)
    unet = unet.to(device)
    n_epochs = 200
    val_interval = 40
    epoch_losses = []
    val_losses = []
    scaler = GradScaler()
    """
    for epoch in range(n_epochs):
        unet.train()
        autoencoderkl.eval()
        epoch_loss = 0
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), ncols=70)
        progress_bar.set_description(f"Epoch {epoch}")
        for step, batch in progress_bar:
            images = batch["image"].to(device)
            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=True):
                z_mu, z_sigma = autoencoderkl.encode(images)
                z = autoencoderkl.sampling(z_mu, z_sigma)
                noise = torch.randn_like(z).to(device)
                timesteps = torch.randint(0, inferer.scheduler.num_train_timesteps, (z.shape[0],),
                                          device=z.device).long()
                noise_pred = inferer(
                    inputs=images, diffusion_model=unet, noise=noise, timesteps=timesteps,
                    autoencoder_model=autoencoderkl
                )
                loss = F.mse_loss(noise_pred.float(), noise.float())

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()

            progress_bar.set_postfix({"loss": epoch_loss / (step + 1)})
        epoch_losses.append(epoch_loss / (step + 1))

        if (epoch + 1) % val_interval == 0:
            unet.eval()
            val_loss = 0
            with torch.no_grad():
                for val_step, batch in enumerate(val_loader, start=1):
                    images = batch["image"].to(device)

                    with autocast(enabled=True):
                        z_mu, z_sigma = autoencoderkl.encode(images)
                        z = autoencoderkl.sampling(z_mu, z_sigma)

                        noise = torch.randn_like(z).to(device)
                        timesteps = torch.randint(
                            0, inferer.scheduler.num_train_timesteps, (z.shape[0],), device=z.device
                        ).long()
                        noise_pred = inferer(
                            inputs=images,
                            diffusion_model=unet,
                            noise=noise,
                            timesteps=timesteps,
                            autoencoder_model=autoencoderkl,
                        )

                        loss = F.mse_loss(noise_pred.float(), noise.float())

                    val_loss += loss.item()
            val_loss /= val_step
            val_losses.append(val_loss)
            print(f"Epoch {epoch} val loss: {val_loss:.4f}")

            # Sampling image during training
            z = torch.randn((1, 3, 16, 16))
            z = z.to(device)
            scheduler.set_timesteps(num_inference_steps=1000)
            with autocast(enabled=True):
                decoded = inferer.sample(
                    input_noise=z, diffusion_model=unet, scheduler=scheduler, autoencoder_model=autoencoderkl
                )

            plt.figure(figsize=(2, 2))
            plt.style.use("default")
            plt.imshow(decoded[0, 0].detach().cpu(), vmin=0, vmax=1, cmap="gray")
            plt.tight_layout()
            plt.axis("off")
            plt.show()
    progress_bar.close()
    """


if __name__ == '__main__' :

    parser = argparse.ArgumentParser()
    # step 1. wandb login
    parser.add_argument("--wandb_api_key", type=str, default='3a3bc2f629692fa154b9274a5bbe5881d47245dc')
    parser.add_argument("--wandb_project_name", type=str, default='dental_experiment')
    parser.add_argument("--wandb_run_name", type=str, default='dental_total_training')
    # step 2. setting
    parser.add_argument("--seed", type=int, default=42)
    # step 3. dataset
    parser.add_argument("--data_folder", type=str, default='/data7/sooyeon/medical_image/experiment_data/dental/panoramic_data_high/original')
    parser.add_argument("--img_size", type=str, default='128,128')
    # step 4.
    parser.add_argument("--device", type=str, default='cuda:7')
    # step 6. autoencoder saving
    parser.add_argument("--experiment_basic_dir", type=str,
                        default='/data7/sooyeon/medical_image/experiment_result/dental_20231117_newscript')
    args = parser.parse_args()

    main(args)