import argparse, wandb
from PIL import Image
from random import seed
from helpers import *
import torch.nn.functional as F
from torchvision import transforms
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import torch.multiprocessing
import torchvision.transforms as torch_transforms
from data_module import SYDataLoader, SYDataset
from setproctitle import *
from generative.networks.nets import AutoencoderKL, PatchDiscriminator
from loss_module import PerceptualLoss, PatchAdversarialLoss
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
from diffuser_module import AutoencoderKL

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
    train_transforms = transforms.Compose([transforms.Resize((w,h), transforms.InterpolationMode.BILINEAR),
                                           transforms.ToTensor()])
    train_ds = SYDataset(data_folder=args.train_data_folder, transform=train_transforms,
                         base_mask_dir=args.train_mask_dir, image_size=(w,h))
    training_dataset_loader = SYDataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4,
                                           persistent_workers=True)

    # ## Prepare validation set data loader
    val_transforms = transforms.Compose([transforms.Resize((w,h), transforms.InterpolationMode.BILINEAR),
                                         transforms.ToTensor()])
    val_ds = SYDataset(data_folder=args.val_data_folder, transform=val_transforms,
                       base_mask_dir=args.val_mask_dir,image_size=(w,h))
    test_dataset_loader = SYDataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                                       num_workers=4, persistent_workers=True)

    print(f'\n step 3. latent_model')
    vae_config_dir = args.vae_config_dir
    with open(vae_config_dir, "r") as f:
        vae_config = json.load(f)
    vae = AutoencoderKL.from_config(config=vae_config)
    vae = vae.to(device)

    perceptual_loss = PerceptualLoss(spatial_dims=2, network_type="alex", cache_dir='/data7/sooyeon/medical_image/pretrained')
    perceptual_loss.to(device)

    discriminator = PatchDiscriminator(spatial_dims=2, num_layers_d=3, num_channels=64, in_channels=1, out_channels=1)
    discriminator = discriminator.to(device)

    adv_loss = PatchAdversarialLoss(criterion="least_squares")

    optimizer_g = torch.optim.Adam(vae.parameters(), lr=args.lr)
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=5 * args.lr)

    scaler_g = torch.cuda.amp.GradScaler()
    scaler_d = torch.cuda.amp.GradScaler()

    print(f'\n step 4. model training')
    autoencoder_warm_up_n_epochs = 10
    records = []
    for epoch in range(args.n_epochs):
        vae.train()
        discriminator.train()
        epoch_loss = 0
        gen_epoch_loss = 0
        disc_epoch_loss = 0
        progress_bar = tqdm(enumerate(training_dataset_loader), total=len(training_dataset_loader), ncols=110)
        progress_bar.set_description(f"Epoch {epoch}")
        for step, batch in progress_bar:
            images = batch["image_info"].to(device)
            normal_info = batch['normal']  # if 1 = normal, 0 = abnormal
            if args.only_normal_training:
                images = images[normal_info == 1]
                mask_info = mask_info[normal_info == 1]
            b_size = images.shape[0]
            if b_size > 0 :
                optimizer_g.zero_grad(set_to_none=True)
                with autocast(enabled=True):
                    # (1) reconstruction loss
                    if args.sample_posterior :
                        reconstruction = vae(images, sample_posterior=True).sample
                    else :
                        reconstruction = vae(images,).sample
                    if args.loss_type == 'l1':
                        recons_loss = F.l1_loss(reconstruction.float(), images.float())
                    elif args.loss_type == 'l2':
                        recons_loss = F.mse_loss(reconstruction.float(), images.float())
                    p_loss = perceptual_loss(reconstruction.float(), images.float())
                    ########################################################################################################
                    latents = vae.encode(images).latent_dist.sample()
                    # ---------------------------------------------------------
                    # (2) KL loss
                    #posterior = vae.encode(images).latent_dist
                    # gaussian priot
                    #z_mu, z_sigma = posterior.mean, posterior.std
                    #kl_loss = 0.5 * torch.sum(z_mu.pow(2) + z_sigma.pow(2) - torch.log(z_sigma.pow(2)) - 1, dim=[1, 2, 3])
                    #kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]
                    wandb.log({"perceptual_loss": p_loss.item(),
                               #"kldivergence_loss": kl_loss.item(),
                               "reconstruction_loss": recons_loss.item(), })
                    #loss_g = recons_loss + (args.kl_weight * kl_loss) + (args.perceptual_weight * p_loss)
                    loss_g = recons_loss +  (args.perceptual_weight * p_loss)
                    if epoch > autoencoder_warm_up_n_epochs:
                        logits_fake = discriminator(reconstruction.contiguous().float())[-1]
                        generator_loss = adv_loss(logits_fake, target_is_real=True, for_discriminator=False)
                        loss_g += args.adv_weight * generator_loss
                        wandb.log({"generator_loss": generator_loss.item()})
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
                        wandb.log({"discriminator_loss": discriminator_loss.item(),})
                        loss_d = args.adv_weight * discriminator_loss
                    scaler_d.scale(loss_d).backward()
                    scaler_d.step(optimizer_d)
                    scaler_d.update()
                epoch_loss += recons_loss.item()
                if epoch > autoencoder_warm_up_n_epochs:
                    gen_epoch_loss += generator_loss.item()
                    disc_epoch_loss += discriminator_loss.item()

                progress_bar.set_postfix(
                    {"recons_loss": epoch_loss / (step + 1),
                        "gen_loss": gen_epoch_loss / (step + 1),
                        "disc_loss": disc_epoch_loss / (step + 1),
                    })
        vae.eval()
        val_loss = 0
        with torch.no_grad():
            for val_step, batch in enumerate(test_dataset_loader, start=1):
                images = batch["image_info"].to(device)
                normal_info = batch['normal']  # if 1 = normal, 0 = abnormal
                if args.only_normal_training:
                    images = images[normal_info == 1]
                    mask_info = mask_info[normal_info == 1]
                b_size = images.shape[0]
                if b_size > 0:
                    with autocast(enabled=True):
                        if args.sample_posterior:
                            reconstruction = vae(images, sample_posterior=True).sample
                        else:
                            reconstruction = vae(images,).sample
                        if val_step == 1:
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
                        if args.loss_type == 'l1':
                            recons_loss = F.l1_loss(images.float(), reconstruction.float())
                        elif args.loss_type == 'l2':
                            recons_loss = F.mse_loss(images.float(), reconstruction.float())
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
    parser.add_argument('--vae_config_dir', type=str, )
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.0)

    # step 4. training
    parser.add_argument('--n_epochs', type=int, default=100)
    parser.add_argument('--only_normal_training', action='store_true')
    parser.add_argument('--sample_posterior', action='store_true')
    parser.add_argument('--loss_type', type=str, default='l2')
    parser.add_argument('--perceptual_weight', type=float, default=0.001)
    parser.add_argument('--kl_weight', type=float, default=0.001)
    parser.add_argument('--adv_weight', type=float, default=0.01)
    parser.add_argument('--inference_freq', type=int, default=50)
    parser.add_argument('--inference_num', type=int, default=4)
    parser.add_argument('--model_save_freq', type=int, default=1000)
    parser.add_argument('--model_save_base_epoch', type=int, default=50)

    args = parser.parse_args()
    main(args)
