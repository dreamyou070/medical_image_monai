import argparse, wandb
import copy
import time
from random import seed
from torch import optim
from helpers import *
from tqdm import tqdm
from torchvision import transforms
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from data_module import SYDataLoader, SYDataset
from monai.utils import first
from nets import AutoencoderKL, DiffusionModelUNet
from nets.utils import update_ema_params
import torch.multiprocessing
import torchvision.transforms as torch_transforms
import PIL
from setproctitle import *
from schedulers import DDPMScheduler
from inferers import LatentDiffusionInferer
from torch.cuda.amp import GradScaler, autocast



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

    if is_train_data :
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
    for t in range(int(args.sample_distance) - 1, -1, -1):
        with torch.no_grad() :
            # 5-1) model prediction
            model_output = model(latent, torch.Tensor((t,)).to(device), None)
        # 5-2) update latent
        latent, _ = scheduler.step(model_output, t, latent)
    with torch.no_grad() :
        recon_image = vae.decode_stage_2_outputs(latent / scale_factor)

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

        mask = mask_info[img_index].squeeze()
        mask = torch_transforms.ToPILImage()(mask.unsqueeze(0))

        new_image = PIL.Image.new('L', (3 * real.size[0], real.size[1]),250)
        new_image.paste(real,  (0, 0))
        new_image.paste(recon, (real.size[0], 0))
        new_image.paste(mask,  (real.size[0]+recon.size[0], 0))
        new_image.save(os.path.join(image_save_dir,
                                    f'real_recon_answer_{train_data}_epoch_{epoch}_{img_index}.png'))
        loading_image = wandb.Image(new_image,
                                    caption=f"(real_recon_answer) epoch {epoch + 1} | {is_normal}")
        if train_data == 'training_data' :
            wandb.log({"training data inference": loading_image})
        elif train_data == 'test_data' :
            wandb.log({"test data inference": loading_image})


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
    autoencoderkl = AutoencoderKL(spatial_dims=2,
                                  in_channels=1,
                                  out_channels=1,
                                  num_channels=(128, 128, 256),
                                  latent_channels=args.latent_channels,
                                  num_res_blocks=2,
                                  attention_levels=(False, False, False),
                                  with_encoder_nonlocal_attn=False,
                                  with_decoder_nonlocal_attn=False, )
    autoencoderkl.load_state_dict(torch.load(args.pretrained_vae_dir))
    autoencoderkl = autoencoderkl.to(device)

    print(f'\n step 4. model')
    model = DiffusionModelUNet(spatial_dims=2,
                               in_channels=args.latent_channels,
                               out_channels=args.latent_channels,
                               num_res_blocks=2,
                               num_channels=(128, 256, 512),
                               attention_levels=(False, True, True),
                               num_head_channels=(0, 256, 512),)
    ema = copy.deepcopy(model)
    model.to(device)
    ema.to(device)

    # (2) scaheduler
    scheduler = DDPMScheduler(num_train_timesteps=args.timestep,
                              schedule=args.schedule_type, beta_start=0.0015, beta_end=0.0195)

    # (3) scaheduler
    with torch.no_grad():
        with autocast(enabled=True):
            z = autoencoderkl.encode_stage_2_inputs(check_data["image_info"].to(device))
    scale_factor = 1 / torch.std(z)
    inferer = LatentDiffusionInferer(scheduler,
                                       scale_factor=scale_factor)

    print(f'\n step 5. optimizer')
    optimiser = optim.AdamW(model.parameters(),
                            lr=args.lr,
                            weight_decay=args.weight_decay,
                            betas=(0.9, 0.999))

    print(f'\n step 6. training')
    tqdm_epoch = range(args.start_epoch, args.train_epochs + 1)

    for epoch in tqdm_epoch:
        progress_bar = tqdm(enumerate(training_dataset_loader),
                            total=len(training_dataset_loader),
                            ncols=200)
        progress_bar.set_description(f"Epoch {epoch}")
        for step, data in progress_bar:
            model.train()
            # -----------------------------------------------------------------------------------------
            # 0) data check
            x_0 = data["image_info"].to(device)  # batch, channel, w, h
            normal_info = data['normal'] # if 1 = normal, 0 = abnormal
            mask_info = data['mask'].unsqueeze(dim=1)    # if 1 = normal, 0 = abnormal
            if args.only_normal_training :
                x_0 = x_0[normal_info == 1]
                mask_info = mask_info[normal_info == 1]
            with torch.no_grad():
                z_mu, z_sigma = autoencoderkl.encode(x_0)
                z_0 = autoencoderkl.sampling(z_mu, z_sigma)
            # x_0 = [batch, 3, 128/4, 128/4]
            z_0 = z_0 * scale_factor

            # 1) check random t
            if z_0.shape[0] != 0 :
                t = torch.randint(0, args.sample_distance, (z_0.shape[0],), device =device)
                noise = torch.rand_like(z_0).float().to(device)
                # 2) make noisy latent
                noisy_latent = scheduler.add_noise(original_samples=z_0,noise=noise,timesteps=t)
                # 3) model prediction
                noise_pred = model(x=noisy_latent,timesteps=t,context=None)
                #with torch.no_grad():
                noise_pred_p = autoencoderkl.decode_stage_2_outputs(noise_pred/scale_factor)
                target_p = autoencoderkl.decode_stage_2_outputs(noise/scale_factor)
                # ------------------------------------------------------------------------------------------------------
                if args.masked_loss:
                    noise_pred_p = noise_pred_p * mask_info.to(device)
                    target_p = target_p * mask_info.to(device)
                loss = torch.nn.functional.mse_loss(noise_pred_p.float(),
                                                    target_p.float(),
                                                    reduction="none")
                if args.pos_neg_loss:
                    pos_loss = torch.nn.functional.mse_loss((noise_pred_p * mask_info.to(device)).float(),
                                                            (target_p * mask_info.to(device)).float(),
                                                            reduction="none")
                    neg_loss = torch.nn.functional.mse_loss((noise_pred_p * (1 - mask_info).to(device)).float(),
                                                            (target_p * (1 - mask_info).to(device)).float(),
                                                            reduction="none")
                    loss = pos_loss + args.pos_neg_loss_scale * (pos_loss - neg_loss)
                loss = loss.mean()

                wandb.log({"training loss": loss.item()})
                optimiser.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                optimiser.step()
                # ----------------------------------------------------------------------------------------- #
                # EMA model updating
                update_ema_params(ema, model)

                # ----------------------------------------------------------------------------------------- #
                # Inference
                if epoch % args.inference_freq == 0 and step == 0:
                    for i, test_data in enumerate(test_dataset_loader):
                        if i == 0:
                            ema.eval()
                            model.eval()
                            training_outputs(args, test_data, scheduler, 'true',  device, model, autoencoderkl, scale_factor, epoch+1)
                            training_outputs(args, data, scheduler, 'falce', device, model, autoencoderkl, scale_factor, epoch+1)


    """
        # ----------------------------------------------------------------------------------------- #
        # vlb loss calculating
        print(f'vlb loss calculating ... ')
        if epoch % args.vlb_freq == 0:
            for i, test_data in enumerate(test_dataset_loader) :
                if i == 0 :
                    x = test_data["image_info"].to(device)
                    normal_info_ = test_data['normal']  # if 1 = normal, 0 = abnormal
                    mask_info_ = test_data['mask']  # if 1 = normal, 0 = abnormal
                    normal_x_ = x[normal_info_ == 1]
                    abnormal_x_ = x[normal_info_ != 1]
                    # ----------------------------------------------------------------------------------------- #
                    # [mask] 1 = normal, 0 = abnormal
                    # abnormal_mask = [Batch, W, H]
                    abnormal_mask = mask_info_[normal_info_ != 1].to(device)
                    # --------------------------------------------------------------------------------------------------
                    # calculate vlb loss
                    # x = [Batch, Channel, 128, 128]
                    if normal_x_.shape[0] != 0 :
                        # ---------------------------------------------------------------------------------------------
                        # should i calculate whole timestep ???
                        # normal and abnormal ...
                        vlb_terms = diffusion.calc_total_vlb_in_sample_distance(normal_x_, model, args)
                        vlb = vlb_terms["whole_vb"]          # [batch, 1000, 1, W, H]
                        # ---------------------------------------------------------------------------
                        # timewise averaging ...
                        whole_vb = vlb.squeeze(dim=2).mean(dim=1) # batch, W, H
                        efficient_pixel_num = whole_vb.shape[-2] * whole_vb.shape[-1]
                        whole_vb = whole_vb.flatten(start_dim=1)  # batch, W*H
                        batch_vb = whole_vb.sum(dim=-1)           # batch
                        whole_vb = batch_vb / efficient_pixel_num # shape = [batch]
                        wandb.log({"total_vlb (test data normal sample)": whole_vb.mean().cpu().item()})
                    # --------------------------------------------------------------------------------------------------
                    if abnormal_x_.shape[0] != 0 :
                        ab_vlb_terms = diffusion.calc_total_vlb_in_sample_distance(abnormal_x_, model, args)
                        ab_whole_vb = ab_vlb_terms["whole_vb"].squeeze(dim=2).mean(dim=1)  # [Batch, W, H]
                        # ----------------------------------------------------------------------------------------------
                        normal_efficient_pixel_num = abnormal_mask.sum(dim=-1).sum(dim=-1).to(device)
                        # abnormal_mask = [batch, w, h]
                        # ab_whole_vb =   [batch, w, h]
                        normal_portion_ab_whole_vb = abnormal_mask * ab_whole_vb

                        normal_portion_ab_whole_vb = normal_portion_ab_whole_vb.sum(dim=-1).sum(dim=-1)
                        normal_portion_ab_whole_vb = normal_portion_ab_whole_vb / normal_efficient_pixel_num
                        wandb.log({"normal portion of *ab*normal sample kl":  normal_portion_ab_whole_vb.mean().cpu().item()})

                        # --------------------------------------------------------------------------------------------------
                        inverse_abnormal_mask = 1 - abnormal_mask
                        efficient_pixel_num = (inverse_abnormal_mask).sum(dim=-1).sum(dim=-1).to(device)
                        ab_portion_ab_whole_vb = inverse_abnormal_mask * ab_whole_vb

                        ab_portion_ab_whole_vb = ab_portion_ab_whole_vb.sum(dim=-1).sum(dim=-1)

                        ab_portion_ab_whole_vb = ab_portion_ab_whole_vb / efficient_pixel_num
                        wandb.log({"abnormal portion of *ab*normal sample kl" : ab_portion_ab_whole_vb.mean().cpu().item()})
                    # --------------------------------------------------------------------------------------------------
                    # collecting total vlb in deque collections
        if epoch % args.model_save_freq == 0 and epoch >= 0:
            save(unet=model, args=args, optimiser=optimiser, final=False, ema=ema, epoch=epoch)
    save(unet=model, args=args, optimiser=optimiser, final=True, ema=ema)
    """

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
    parser.add_argument('--masked_loss', action='store_true')
    parser.add_argument('--pos_neg_loss', action='store_true')
    parser.add_argument('--pos_neg_loss_scale', type=float, default=1.0)

    parser.add_argument('--inference_freq', type=int, default=50)
    parser.add_argument('--inference_num', type=int, default=4)
    args = parser.parse_args()
    main(args)