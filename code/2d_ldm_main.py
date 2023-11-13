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

def get_transform(image_size) :
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
    val_transforms = transforms.Compose([transforms.LoadImaged(keys=["image"]),
                                         transforms.EnsureChannelFirstd(keys=["image"]),
                                         transforms.ScaleIntensityRanged(keys=["image"], a_min=0.0, a_max=255.0, b_min=0.0, b_max=1.0, clip=True),])
    return train_transforms, val_transforms

def main(args) :

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
    total_datas = os.listdir(args.data_folder)
    total_num = len(total_datas)
    train_num = int(0.7 * total_num)
    train_datas, val_datalist = total_datas[:train_num], total_datas[train_num:]
    train_datalist = [{"image": os.path.join(args.data_folder, train_data)} for train_data in train_datas]

    train_transforms, val_transforms = get_transform(args.image_size)
    train_ds = Dataset(data=train_datalist,transform=train_transforms)
    print(f' (2.1.2) load dataloader')
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, persistent_workers=True)

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
    val_ds = Dataset(data=val_datalist, transform=val_transforms)
    print(f' (2.2.2) validation dataloader')
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, persistent_workers=True)

    print(f'\n step 3. model')
    print(f' (3.0) device')
    device = torch.device("cuda")
    print(f' (3.1) vae(autoencoder)')
    autoencoderkl = AutoencoderKL(spatial_dims=2,in_channels=1,out_channels=1,
                                  num_channels=(128, 128, 256),latent_channels=3,num_res_blocks=2,
                                  attention_levels=(False, False, False),
                                  with_encoder_nonlocal_attn=False,with_decoder_nonlocal_attn=False,).to(device)

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
            images = batch["image"].to(device)
            optimizer_g.zero_grad(set_to_none=True)
            with autocast(enabled=True):

                reconstruction, z_mu, z_sigma = autoencoderkl(images)
                # ------------------------------------------------------------------------------------------------------------
                # (1) reconstruction loss (L1)
                recons_loss = F.l1_loss(reconstruction.float(),
                                        images.float())
                # ------------------------------------------------------------------------------------------------------------
                # (2) preceptual loss
                p_loss = perceptual_loss(reconstruction.float(),
                                         images.float())

                # ------------------------------------------------------------------------------------------------------------
                # (3)
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

            progress_bar.set_postfix(
                {
                    "recons_loss": epoch_loss / (step + 1),
                    "gen_loss": gen_epoch_loss / (step + 1),
                    "disc_loss": disc_epoch_loss / (step + 1),
                }
            )
        epoch_recon_losses.append(epoch_loss / (step + 1))
        epoch_gen_losses.append(gen_epoch_loss / (step + 1))
        epoch_disc_losses.append(disc_epoch_loss / (step + 1))

        if (epoch + 1) % val_interval == 0:
            autoencoderkl.eval()
            val_loss = 0
            with torch.no_grad():
                for val_step, batch in enumerate(val_loader, start=1):
                    images = batch["image"].to(device)

                    with autocast(enabled=True):
                        reconstruction, z_mu, z_sigma = autoencoderkl(images)
                        # Get the first reconstruction from the first validation batch for visualisation purposes
                        if val_step == 1:
                            intermediary_images.append(reconstruction[:num_example_images, 0])

                        recons_loss = F.l1_loss(images.float(), reconstruction.float())

                    val_loss += recons_loss.item()

            val_loss /= val_step
            val_recon_losses.append(val_loss)
            print(f"epoch {epoch + 1} val loss: {val_loss:.4f}")
    progress_bar.close()
    """
    
    
    
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

            progress_bar.set_postfix(
                {
                    "recons_loss": epoch_loss / (step + 1),
                    "gen_loss": gen_epoch_loss / (step + 1),
                    "disc_loss": disc_epoch_loss / (step + 1),
                }
            )
        epoch_recon_losses.append(epoch_loss / (step + 1))
        epoch_gen_losses.append(gen_epoch_loss / (step + 1))
        epoch_disc_losses.append(disc_epoch_loss / (step + 1))

        if (epoch + 1) % val_interval == 0:
            autoencoderkl.eval()
            val_loss = 0
            with torch.no_grad():
                for val_step, batch in enumerate(val_loader, start=1):
                    images = batch["image"].to(device)

                    with autocast(enabled=True):
                        reconstruction, z_mu, z_sigma = autoencoderkl(images)
                        # Get the first reconstruction from the first validation batch for visualisation purposes
                        if val_step == 1:
                            intermediary_images.append(reconstruction[:num_example_images, 0])

                        recons_loss = F.l1_loss(images.float(), reconstruction.float())

                    val_loss += recons_loss.item()

            val_loss /= val_step
            val_recon_losses.append(val_loss)
            print(f"epoch {epoch + 1} val loss: {val_loss:.4f}")
    progress_bar.close()

    del discriminator
    del perceptual_loss
    torch.cuda.empty_cache()
    # -

    # ### Visualise the results from the autoencoderKL

    # Plot last 5 evaluations
    val_samples = np.linspace(n_epochs, val_interval, int(n_epochs / val_interval))
    fig, ax = plt.subplots(nrows=5, ncols=1, sharey=True)
    for image_n in range(5):
        reconstructions = torch.reshape(intermediary_images[image_n], (image_size * num_example_images, image_size)).T
        ax[image_n].imshow(reconstructions.cpu(), cmap="gray")
        ax[image_n].set_xticks([])
        ax[image_n].set_yticks([])
        ax[image_n].set_ylabel(f"Epoch {val_samples[image_n]:.0f}")

    # ## Diffusion Model
    #
    # ### Define diffusion model and scheduler
    #
    # In this section, we will define the diffusion model that will learn data distribution of the latent representation of the autoencoder. Together with the diffusion model, we define a beta scheduler responsible for defining the amount of noise tahat is added across the diffusion's model Markov chain.

    # +
    unet = DiffusionModelUNet(
        spatial_dims=2,
        in_channels=3,
        out_channels=3,
        num_res_blocks=2,
        num_channels=(128, 256, 512),
        attention_levels=(False, True, True),
        num_head_channels=(0, 256, 512),
    )

    scheduler = DDPMScheduler(num_train_timesteps=1000, schedule="linear_beta", beta_start=0.0015, beta_end=0.0195)
    # -

    # ### Scaling factor
    #
    # As mentioned in Rombach et al. [1] Section 4.3.2 and D.1, the signal-to-noise ratio (induced by the scale of the latent space) can affect the results obtained with the LDM, if the standard deviation of the latent space distribution drifts too much from that of a Gaussian. For this reason, it is best practice to use a scaling factor to adapt this standard deviation.
    #
    # _Note: In case where the latent space is close to a Gaussian distribution, the scaling factor will be close to one, and the results will not differ from those obtained when it is not used._
    #

    # +
    with torch.no_grad():
        with autocast(enabled=True):
            z = autoencoderkl.encode_stage_2_inputs(check_data["image"].to(device))

    print(f"Scaling factor set to {1 / torch.std(z)}")
    scale_factor = 1 / torch.std(z)
    # -

    # We define the inferer using the scale factor:

    inferer = LatentDiffusionInferer(scheduler, scale_factor=scale_factor)

    # ### Train diffusion model
    #
    # It takes about ~80 min to train the model.

    # +
    optimizer = torch.optim.Adam(unet.parameters(), lr=1e-4)

    unet = unet.to(device)
    n_epochs = 200
    val_interval = 40
    epoch_losses = []
    val_losses = []
    scaler = GradScaler()

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

    # -

    # ### Plot learning curves

    plt.figure()
    plt.title("Learning Curves", fontsize=20)
    plt.plot(np.linspace(1, n_epochs, n_epochs), epoch_losses, linewidth=2.0, label="Train")
    plt.plot(
        np.linspace(val_interval, n_epochs, int(n_epochs / val_interval)), val_losses, linewidth=2.0, label="Validation"
    )
    plt.yticks(fontsize=12)
    plt.xticks(fontsize=12)
    plt.xlabel("Epochs", fontsize=16)
    plt.ylabel("Loss", fontsize=16)
    plt.legend(prop={"size": 14})

    # ### Plotting sampling example
    #
    # Finally, we generate an image with our LDM. For that, we will initialize a latent representation with just noise. Then, we will use the `unet` to perform 1000 denoising steps. For every 100 steps, we store the noisy intermediary samples. In the last step, we decode all latent representations and plot how the image looks like across the sampling process.

    # +
    unet.eval()
    scheduler.set_timesteps(num_inference_steps=1000)
    noise = torch.randn((1, 3, 16, 16))
    noise = noise.to(device)

    with torch.no_grad():
        image, intermediates = inferer.sample(
            input_noise=noise,
            diffusion_model=unet,
            scheduler=scheduler,
            save_intermediates=True,
            intermediate_steps=100,
            autoencoder_model=autoencoderkl,
        )

    # -

    # Decode latent representation of the intermediary images
    decoded_images = []
    for image in intermediates:
        with torch.no_grad():
            decoded_images.append(image)
    plt.figure(figsize=(10, 12))
    chain = torch.cat(decoded_images, dim=-1)
    plt.style.use("default")
    plt.imshow(chain[0, 0].cpu(), vmin=0, vmax=1, cmap="gray")
    plt.tight_layout()
    plt.axis("off")

    # ### Clean-up data directory

    if directory is None:
        shutil.rmtree(root_dir)
    """

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--root_dir", type=str, default='../experiment')
    parser.add_argument("--data_folder", type=str, default='../experiment/MedNIST/Hand' )
    parser.add_argument("--image_size", type=int, default=64)
    parser.add_argument("--vis_num_images", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device", type=str, default='cuda')
    args = parser.parse_args()
    main(args)