import argparse, wandb
import copy
import numpy as np
from PIL import Image
from random import seed
from torch import optim
from helpers import *
from tqdm import tqdm
import torch.nn.functional as F
from torchvision import transforms
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import torch.multiprocessing
import torchvision.transforms as torch_transforms
import PIL
from data_module import SYDataLoader, SYDataset
from monai.utils import first
from setproctitle import *
from generative.networks.nets import AutoencoderKL, PatchDiscriminator
from loss_module import PerceptualLoss, PatchAdversarialLoss
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
from diffusers import AutoencoderKL

torch.multiprocessing.set_sharing_strategy('file_system')
torch.cuda.empty_cache()



def save(final, unet, optimiser, args, ema, loss=0, epoch=0):
    model_save_base_dir = os.path.join(args.experiment_dir,'diffusion-models')
    os.makedirs(model_save_base_dir, exist_ok=True)
    if final:
        save_dir = os.path.join(model_save_base_dir,f'unet_final.pt')
        torch.save({'n_epoch':              args.train_epochs,
                    'model_state_dict':     unet.state_dict(),
                    'optimizer_state_dict': optimiser.state_dict(),
                    "ema":                  ema.state_dict(),
                    "args":                 args},save_dir)
    else:
        save_dir = os.path.join(model_save_base_dir, f'unet_epoch_{epoch}.pt')
        torch.save({'n_epoch':              epoch,
                    'model_state_dict':     unet.state_dict(),
                    'optimizer_state_dict': optimiser.state_dict(),
                    "args":                 args,
                    "ema":                  ema.state_dict(),
                    'loss':                 loss,},save_dir)

def training_outputs(args, test_data, scheduler, is_train_data, device, model, vae, scale_factor, epoch):

    if is_train_data == 'training_data':
        train_data = 'training_data'
    else :
        train_data = 'test_data'

    video_save_dir = os.path.join(args.experiment_dir, 'diffusion-videos')
    image_save_dir = os.path.join(args.experiment_dir, 'diffusion-training-images')
    os.makedirs(video_save_dir, exist_ok=True)
    os.makedirs(image_save_dir, exist_ok=True)

    # 1) make random noise
    x = test_data["image_info"].to(device)  # batch, channel, w, h
    normal_info = test_data['normal']  # if 1 = normal, 0 = abnormal
    mask_info = test_data['mask']  # if 1 = normal, 0 = abnormal

    with torch.no_grad():
        z_mu, z_sigma = vae.encode(x)
        latent = vae.sampling(z_mu, z_sigma) * scale_factor
    # 2) select random int
    t = torch.randint(args.sample_distance - 1, args.sample_distance, (latent.shape[0],), device=x.device)
    # 3) noise
    noise = torch.rand_like(latent).float().to(x.device)
    # 4) noise image generating
    with torch.no_grad() :
        noisy_latent = scheduler.add_noise(original_samples=latent, noise=noise, timesteps=t)
        latent = noisy_latent.clone().detach()

    # 5) denoising
    for t in range(int(args.sample_distance) , -1, -1):
        with torch.no_grad() :
            # 5-1) model prediction
            model_output = model(latent, torch.Tensor((t,)).to(device), None)
        # 5-2) update latent
        latent, _ = scheduler.step(model_output, t, latent)
    #latents =
    #image = self.vae.decode(latent / scale_factor).sample
    #image = (image / 2 + 0.5).clamp(0, 1)
    # we always cast to float32 as this does not cause significant overhead and is compatible with bfloa16
    #image = image.cpu().permute(0, 2, 3, 1).float().numpy()

    with torch.no_grad() :
        recon_image = vae.decode_stage_2_outputs(latent/scale_factor)

    for img_index in range(x.shape[0]):
        normal_info_ = normal_info[img_index]
        if normal_info_ == 1:
            is_normal = 'normal'
        else :
            is_normal = 'abnormal'

        real = x[img_index].squeeze()
        real = torch_transforms.ToPILImage()(real.unsqueeze(0))

        recon = recon_image[img_index].squeeze()
        recon = torch_transforms.ToPILImage()(recon.unsqueeze(0))

        mask_np = mask_info[img_index].squeeze().to('cpu').detach().numpy().copy().astype(np.uint8)
        mask_np = mask_np * 255
        mask = Image.fromarray(mask_np).convert('L')  # [128, 128, 3]

        new_image = PIL.Image.new('L', (3 * real.size[0], real.size[1]),250)
        new_image.paste(real,  (0, 0))
        new_image.paste(recon, (real.size[0], 0))
        new_image.paste(mask,  (real.size[0]+recon.size[0], 0))
        new_image.save(os.path.join(image_save_dir,
                                    f'real_recon_answer_{train_data}_epoch_{epoch}_{img_index}.png'))
        loading_image = wandb.Image(new_image,
                                    caption=f"(real_recon_answer) epoch {epoch + 1} | {is_normal}")
        if train_data == 'training_data' :
            wandb.log({"training data inference" : loading_image})
        else :
            wandb.log({"test data inference" : loading_image})


def main(args) :

    print(f'\n step 1. setting')
    if args.process_title:
        setproctitle(args.process_title)
    else:
        setproctitle('parksooyeon')
    print(f' (1.1) wandb')
    wandb.login(key=args.wandb_api_key)
    wandb.init(project=args.wandb_project_name, name=args.wandb_run_name)

    print(f' (1.2) seed and device')
    seed(args.seed)
    device = args.device

    print(f' (1.3) saving configuration')
    experiment_dir = args.experiment_dir
    os.makedirs(experiment_dir, exist_ok=True)
    var_args = vars(args)
    with open(os.path.join(experiment_dir, "config.txt"), "w") as f:
        for key in sorted(var_args.keys()):
            f.write(f"{key}: {var_args[key]}\n")

    print(f'\n step 2. dataset and dataloatder')
    w,h = int(args.img_size.split(',')[0].strip()),int(args.img_size.split(',')[1].strip())
    train_transforms = transforms.Compose([#transforms.ToPILImage(),
                                           transforms.Resize((w,h), transforms.InterpolationMode.BILINEAR),
                                           transforms.ToTensor()])
    train_ds = SYDataset(data_folder=args.train_data_folder,
                         transform=train_transforms,
                         base_mask_dir=args.train_mask_dir,
                         image_size=(w,h))
    training_dataset_loader = SYDataLoader(train_ds,
                                           batch_size=args.batch_size,
                                           shuffle=True,
                                           num_workers=4,
                                           persistent_workers=True)
    check_data = first(training_dataset_loader)
    # ## Prepare validation set data loader
    val_transforms = transforms.Compose([#transforms.ToPILImage(),
                                           transforms.Resize((w,h), transforms.InterpolationMode.BILINEAR),
                                           transforms.ToTensor()])
    val_ds = SYDataset(data_folder=args.val_data_folder,
                         transform=val_transforms,
                         base_mask_dir=args.val_mask_dir,image_size=(w,h))
    test_dataset_loader = SYDataLoader(val_ds,
                                       batch_size=args.batch_size,
                                       shuffle=False,
                                       num_workers=4,
                                       persistent_workers=True)

    print(f'\n step 3. latent_model')
    #vae = AutoencoderKL(in_channels = 1,
    #                    out_channels = 1,
    #                    latent_channels = 4,
    #                    norm_num_groups = 32,
    #                    sample_size = 128,
    #                    scaling_factor = 0.18215)
    vae = AutoencoderKL(in_channels = 1,
                  out_channels = 1,
                  down_block_types = ["DownEncoderBlock2D","DownEncoderBlock2D",
                                      "DownEncoderBlock2D","DownEncoderBlock2D"],
                  up_block_types = ["UpDecoderBlock2D","UpDecoderBlock2D","UpDecoderBlock2D","UpDecoderBlock2D"],
                  block_out_channels = [128,256,512,512],
                  layers_per_block = 2,
                  act_fn = "silu",
                  latent_channels = 4,
                  norm_num_groups = 32,
                  sample_size = 512,
                  scaling_factor = 0.18215,)
    vae = vae.to(device)
    perceptual_loss = PerceptualLoss(spatial_dims=2,
                                     network_type="alex",
                                     cache_dir='/data7/sooyeon/medical_image/pretrained')
    perceptual_loss.to(device)
    perceptual_weight = 0.001

    discriminator = PatchDiscriminator(spatial_dims=2, num_layers_d=3, num_channels=64,
                                       in_channels=1, out_channels=1)
    discriminator = discriminator.to(device)

    adv_loss = PatchAdversarialLoss(criterion="least_squares")
    adv_weight = 0.01

    optimizer_g = torch.optim.Adam(vae.parameters(), lr=1e-4)
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=5e-4)

    scaler_g = torch.cuda.amp.GradScaler()
    scaler_d = torch.cuda.amp.GradScaler()

    print(f'\n step 4. model training')
    kl_weight = 1e-6
    n_epochs = 100
    val_interval = 10
    autoencoder_warm_up_n_epochs = 10
    records = []
    for epoch in range(n_epochs):
        vae.train()
        discriminator.train()
        epoch_loss = 0
        gen_epoch_loss = 0
        disc_epoch_loss = 0
        progress_bar = tqdm(enumerate(training_dataset_loader),
                            total=len(training_dataset_loader), ncols=110)
        progress_bar.set_description(f"Epoch {epoch}")
        for step, batch in progress_bar:
            images = batch["image_info"].to(device)
            optimizer_g.zero_grad(set_to_none=True)
            with autocast(enabled=True):

                #latents = vae.encode(images).latent_dist.sample()
                #latents = latents * 0.18215
                # (1) reconstruction loss
                reconstruction = vae(images).sample
                recons_loss = F.l1_loss(reconstruction.float(), images.float())
                p_loss = perceptual_loss(reconstruction.float(), images.float())

                # (2) KL loss
                latents = vae.encode(images).latent_dist.sample()
                print(f'latents : {latents.shape}')
                # ---------------------------------------------------------
                posterior = vae.encode(images).latent_dist
                z_mu, z_sigma = posterior.mean, posterior.std
                kl_loss = 0.5 * torch.sum(z_mu.pow(2) + z_sigma.pow(2) - torch.log(z_sigma.pow(2)) - 1, dim=[1, 2, 3])
                kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]
                loss_g = recons_loss + (kl_weight * kl_loss) + (perceptual_weight * p_loss)
                if epoch > autoencoder_warm_up_n_epochs:
                    logits_fake = discriminator(reconstruction.contiguous().float())[-1]
                    generator_loss = adv_loss(logits_fake, target_is_real=True, for_discriminator=False)
                    loss_g += adv_weight * generator_loss
            wandb.log({"reconstruction_loss": recons_loss.item(), })
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
                wandb.log({"discriminator_loss": loss_d.item(), })
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
                })
        vae.eval()
        val_loss = 0
        with torch.no_grad():
            for val_step, batch in enumerate(test_dataset_loader, start=1):
                images = batch["image_info"].to(device)
                with autocast(enabled=True):
                    reconstruction = vae(images).sample
                    z_mu, z_sigma = posterior.mean, posterior.std
                    if val_step == 1:
                        import torchvision.transforms as torch_transforms
                        from PIL import Image
                        # real = torch_transforms.ToPILImage()(real)
                        org_img = images[0].squeeze()
                        org_img = torch_transforms.ToPILImage()(org_img.unsqueeze(0))
                        recon = reconstruction[0].squeeze()
                        recon = torch_transforms.ToPILImage()(recon.unsqueeze(0))
                        new = Image.new('RGB', (org_img.width + recon.width, org_img.height))
                        new.paste(org_img, (0, 0))
                        new.paste(recon, (org_img.width, 0))
                        loading_image = wandb.Image(new,
                                                    caption=f"(real-recon) epoch {epoch + 1} ")
                        wandb.log({"vae inference": loading_image})
                    recons_loss = F.l1_loss(images.float(), reconstruction.float())
                val_loss += recons_loss.item()
        val_loss /= val_step
        wandb.log({"val_loss": val_loss, })
        line = f"epoch {epoch + 1} val loss: {val_loss:.4f}"
        records.append(line)
        # save model
        if epoch > args.model_save_base_epoch:
            model_save_dir = os.path.join(experiment_dir, f'vae')
            os.makedirs(model_save_dir, exist_ok=True)
            torch.save(vae.state_dict(),
                       os.path.join(model_save_dir, f'vae_{epoch}.pth'))

    progress_bar.close()

    del discriminator
    del perceptual_loss
    torch.cuda.empty_cache()

    # save records
    with open(os.path.join(experiment_dir, f'records.txt'), 'w') as f:
        for line in records:
            f.write(line + '\n')

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # step 1. wandb login
    parser.add_argument("--process_title", type=str, default='parksooyeon')
    parser.add_argument("--wandb_api_key", type=str, default='3a3bc2f629692fa154b9274a5bbe5881d47245dc')
    parser.add_argument("--wandb_project_name", type=str, default='dental_experiment')
    parser.add_argument("--wandb_run_name", type=str, default='hand_1000_64res')
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str)
    parser.add_argument('--experiment_dir', type=str,
                        default = f'/data7/sooyeon/medical_image/anoddpm_result/20231119_dental_test')

    # step 2. dataset and dataloatder
    parser.add_argument('--train_data_folder', type=str)
    parser.add_argument('--train_mask_dir', type=str)
    parser.add_argument('--val_data_folder', type=str)
    parser.add_argument('--val_mask_dir', type=str)
    parser.add_argument('--img_size', type=str)
    parser.add_argument('--batch_size', type=int)

    # step 3. model
    parser.add_argument('--pretrained_vae_dir', type=str,
                        default=f'/data7/sooyeon/medical_image/anoddpm_result_vae/1_first_training/autoencoderkl/autoencoderkl_99.pth')
    parser.add_argument('--latent_channels', type=int, default=3)

    # step 4. model
    parser.add_argument('--timestep', type=int, default=1000)
    parser.add_argument('--schedule_type', type=str, default="linear_beta")

    # step 5. optimizer
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.0)

    # step 6. training
    parser.add_argument('--start_epoch', type=int, default=0)
    parser.add_argument('--train_epochs', type=int, default=3000)
    parser.add_argument('--only_normal_training', action='store_true')
    parser.add_argument('--sample_distance', type=int, default=150)
    parser.add_argument('--use_simplex_noise', action='store_true')
    # --------------------------------------------------------------------------------------------------------------
    parser.add_argument('--masked_loss_latent', action='store_true')
    parser.add_argument('--masked_loss', action='store_true')
    # --------------------------------------------------------------------------------------------------------------
    parser.add_argument('--info_nce_loss', action='store_true')
    # --------------------------------------------------------------------------------------------------------------
    parser.add_argument('--pos_info_nce_loss', action='store_true')
    parser.add_argument('--reg_loss_scale', type=float, default=1.0)
    # --------------------------------------------------------------------------------------------------------------
    parser.add_argument('--anormal_scoring', action='store_true')
    parser.add_argument('--min_max_training', action='store_true')

    parser.add_argument('--inference_freq', type=int, default=50)
    parser.add_argument('--inference_num', type=int, default=4)

    # step 7. save
    parser.add_argument('--model_save_freq', type=int, default=1000)
    parser.add_argument('--model_save_base_epoch', type=int, default=50)
    args = parser.parse_args()
    main(args)
