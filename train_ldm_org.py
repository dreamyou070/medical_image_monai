import os, PIL, wandb
import argparse
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from monai import transforms
import torchvision.transforms as torch_transforms
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

def main(args) :

    print(f'\n step 1. wandb login')
    wandb.login(key=args.wandb_api_key)
    wandb.init(project=args.wandb_project_name, name=args.wandb_run_name)
    print_config()
    set_determinism(args.seed)

    print(f' step 2. data')
    data_dir =  args.data_folder
    total_datas = os.listdir(data_dir)
    train_num = int(0.9 * len(total_datas))
    train_datas, val_datas = total_datas[:train_num], total_datas[train_num:]
    train_datalist = [{"image": os.path.join(data_dir, train_data)} for train_data in train_datas]
    image_size = 64
    train_transforms = transforms.Compose([transforms.LoadImaged(keys=["image"]),
                                           transforms.EnsureChannelFirstd(keys=["image"]),
                                           transforms.ScaleIntensityRanged(keys=["image"], a_min=0.0, a_max=255.0, b_min=0.0, b_max=1.0, clip=True),
                                           transforms.RandAffined(keys=["image"],
                                                                  rotate_range=[(-np.pi / 36, np.pi / 36), (-np.pi / 36, np.pi / 36)],
                                                                  translate_range=[(-1, 1), (-1, 1)],
                                                                  scale_range=[(-0.05, 0.05), (-0.05, 0.05)],
                                                                  spatial_size=[image_size, image_size],
                                                                  padding_mode="zeros",
                                                                  prob=0.5,),])
    train_ds = Dataset(data=train_datalist, transform=train_transforms)
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=4, persistent_workers=True)
    check_data = first(train_loader)
    # ## Prepare validation set data loader
    val_datalist = [{"image": os.path.join(data_dir, val_data)} for val_data in val_datas]
    val_transforms = transforms.Compose([transforms.LoadImaged(keys=["image"]),
                                         transforms.EnsureChannelFirstd(keys=["image"]),
                                         transforms.ScaleIntensityRanged(keys=["image"],
                                                                         a_min=0.0, a_max=255.0, b_min=0.0, b_max=1.0, clip=True),])
    val_ds = Dataset(data=val_datalist, transform=val_transforms)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=True, num_workers=4, persistent_workers=True)

    print(f' step 3. Autoencoder KL')
    device = args.device
    autoencoderkl = AutoencoderKL(spatial_dims=2,
                                  in_channels=1,
                                    out_channels=1,
                                    num_channels=(128, 128, 256),
                                    latent_channels=3,
                                    num_res_blocks=2,
                                    attention_levels=(False, False, False),
                                    with_encoder_nonlocal_attn=False,
                                    with_decoder_nonlocal_attn=False,)
    autoencoderkl = autoencoderkl.to(device)
    perceptual_loss = PerceptualLoss(spatial_dims=2, network_type="alex")
    perceptual_loss.to(device)
    perceptual_weight = 0.001
    discriminator = PatchDiscriminator(spatial_dims=2, num_layers_d=3, num_channels=64, in_channels=1, out_channels=1)
    discriminator = discriminator.to(device)
    adv_loss = PatchAdversarialLoss(criterion="least_squares")
    adv_weight = 0.01
    optimizer_g = torch.optim.Adam(autoencoderkl.parameters(), lr=1e-4)
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=5e-4)
    scaler_g = torch.cuda.amp.GradScaler()
    scaler_d = torch.cuda.amp.GradScaler()

    print(f' step 4. Autoencoder KL training')
    kl_weight = 1e-6
    n_epochs = 100
    autoencoder_warm_up_n_epochs = 10
    epoch_recon_losses = []
    epoch_gen_losses = []
    epoch_disc_losses = []
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
            progress_bar.set_postfix({"recons_loss": epoch_loss / (step + 1), "gen_loss": gen_epoch_loss / (step + 1),
                                      "disc_loss": disc_epoch_loss / (step + 1),})
        epoch_recon_losses.append(epoch_loss / (step + 1))
        epoch_gen_losses.append(gen_epoch_loss / (step + 1))
        epoch_disc_losses.append(disc_epoch_loss / (step + 1))
    progress_bar.close()

    del discriminator
    del perceptual_loss
    torch.cuda.empty_cache()

    print(f' step 6. Autoencoder KL saving')
    experiment_basic_dir = args.experiment_basic_dir
    os.makedirs(experiment_basic_dir, exist_ok=True)
    save_obj = {'model': autoencoderkl.state_dict(), }
    torch.save(save_obj, os.path.join(experiment_basic_dir, f'vae_checkpoint_{epoch + 1}.pth'))

    print(f' step 7. autoencoder inference')
    autoencoder_inference_num = args.autoencoder_inference_num
    random_idx = np.random.randint(0, len(val_ds), size=autoencoder_inference_num)
    for idx in random_idx:
        # org shape is [1615,840]
        org_img_ = val_ds[idx]['image']
        org_pil = torch_transforms.ToPILImage()(org_img_)
        with torch.no_grad():
            org_img = org_img_.unsqueeze(0).to(device)  # [channel=1, width, height], torch type
            recon_img, z_mu, z_sigma = autoencoderkl(org_img)
            batch, channel, width, height = recon_img.shape
            reconstructions = torch.reshape(recon_img, (width, height))  # .T  # height, width
            recon_pil = torch_transforms.ToPILImage()(reconstructions)
        new_image = PIL.Image.new('RGB', (2 * org_pil.size[0], recon_pil.size[1]), (250, 250, 250))
        new_image.paste(org_pil, (0, 0))
        new_image.paste(recon_pil, (recon_pil.size[0], 0))
        loading_image = wandb.Image(new_image, caption=f"autokl_val_image_{idx}")
        wandb.log({"autoencoder inference": loading_image})
        new_image.save(os.path.join(experiment_basic_dir, f'autoencoderkl_{idx}.png'))

    print(f' step 8. unet')
    unet = DiffusionModelUNet(spatial_dims=2,
                              in_channels=3,
                              out_channels=3,
                              num_res_blocks=2,
                              num_channels=(128, 256, 512),
                              attention_levels=(False, True, True),
                              num_head_channels=(0, 256, 512),)
    scheduler = DDPMScheduler(num_train_timesteps=1000, schedule="linear_beta", beta_start=0.0015, beta_end=0.0195)
    with torch.no_grad():
        with autocast(enabled=True):
            z = autoencoderkl.encode_stage_2_inputs(check_data["image"].to(device))
    print(f"Scaling factor set to {1 / torch.std(z)}")
    scale_factor = 1 / torch.std(z)
    inferer = LatentDiffusionInferer(scheduler, scale_factor=scale_factor)
    optimizer = torch.optim.Adam(unet.parameters(), lr=1e-4)
    unet = unet.to(device)
    n_epochs = 200
    val_interval = 40
    epoch_losses = []
    val_losses = []
    scaler = GradScaler()

    print(f' step 9. unet training')

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
                noise_pred = inferer(inputs=images, diffusion_model=unet, noise=noise, timesteps=timesteps,
                                     autoencoder_model=autoencoderkl)
                loss = F.mse_loss(noise_pred.float(), noise.float())
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item()
            progress_bar.set_postfix({"loss": epoch_loss / (step + 1)})
        epoch_losses.append(epoch_loss / (step + 1))
        if (epoch + 1) % args.unet_val_interval == 0:
            unet.eval()
            z = torch.randn((1, 3, 16, 16))
            z = z.to(device)
            scheduler.set_timesteps(num_inference_steps=1000)
            with autocast(enabled=True):
                decoded = inferer.sample(input_noise=z, diffusion_model=unet, scheduler=scheduler,
                                         autoencoder_model=autoencoderkl)
            # save image -----------------------------------------------------------------------------------------------
            torch_img = decoded.detach().squeeze().cpu()
            pil_img = torch_transforms.ToPILImage()(torch_img)
            infer_save_basic_dir = os.path.join(args.experiment_basic_dir, 'unet_inference')
            os.makedirs(infer_save_basic_dir, exist_ok=True)
            pil_img.save(os.path.join(infer_save_basic_dir, f'unet_generated_epoch_{epoch + 1}_pil.png'))
            # wandb save image -----------------------------------------------------------------------------------------------
            loading_image = wandb.Image(pil_img, caption=f"epoch : {epoch + 1}")
            wandb.log({"Unet Generating": loading_image})
    progress_bar.close()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # step 1. wandb login
    parser.add_argument("--wandb_api_key", type=str, default='3a3bc2f629692fa154b9274a5bbe5881d47245dc')
    parser.add_argument("--wandb_project_name", type=str, default='dental_experiment')
    parser.add_argument("--wandb_run_name", type=str, default='hand_1000_64res')
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str)
    parser.add_argument("--data_folder", type=str)
    parser.add_argument("--experiment_basic_dir", type=str, default="experiments")
    parser.add_argument("--autoencoder_inference_num", type=int)
    args = parser.parse_args()
    main(args)
