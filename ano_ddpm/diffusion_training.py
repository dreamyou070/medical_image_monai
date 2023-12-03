import argparse, wandb
import collections
import copy
import time
from random import seed
from torch import optim
from GaussianDiffusion import GaussianDiffusionModel, get_beta_schedule
from helpers import *
from tqdm import tqdm
from torchvision import transforms
import numpy  as np
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from data_module import SYDataLoader, SYDataset
from monai.utils import first
from UNet import UNetModel, update_ema_params
import torch.multiprocessing
import torchvision.transforms as torch_transforms
import PIL
from setproctitle import *

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
                    "args":                 args},
                   save_dir)
    else:
        save_dir = os.path.join(model_save_base_dir, f'unet_epoch_{epoch}.pt')
        torch.save({'n_epoch':              epoch,
                    'model_state_dict':     unet.state_dict(),
                    'optimizer_state_dict': optimiser.state_dict(),
                    "args":                 args,
                    "ema":                  ema.state_dict(),
                    'loss':                 loss,},save_dir)

def training_outputs(diffusion, test_data, epoch, num_images, ema, args,
                     save_imgs=False, is_train_data=True, device='cuda'):

    if is_train_data :
        train_data = 'training inference'
    else :
        train_data = 'test inference'

    video_save_dir = os.path.join(args.experiment_dir, 'diffusion-videos')
    image_save_dir = os.path.join(args.experiment_dir, 'diffusion-training-images')
    os.makedirs(video_save_dir, exist_ok=True)
    os.makedirs(image_save_dir, exist_ok=True)

    if save_imgs:
        # 1) make random noise
        x_0 = test_data["image_info"].to(device)
        weight_dtype = x_0.dtype
        normal_info = test_data['normal']
        mask_info = test_data['mask'].to(device).unsqueeze(1)
        t = torch.Tensor([args.sample_distance]).repeat(x_0.shape[0], ).long().to(x_0.device)
        if args.use_simplex_noise:
            noise = diffusion.noise_fn(x=x_0, t=t, octave=6, frequency=64).float()
        else:
            noise = torch.rand_like(x_0.float().to(x_0.device))
        x_t = diffusion.sample_q(x_0, t, noise)
        normal_scores, abnormal_scores = [], []
        if x_0.shape[0] != normal_info.sum() and x_0.dim() == 4 :
            if args.onestep_inference :
                output = diffusion.sample_p(ema, x_0, t, denoise_fn="gauss")
                recon = output["pred_x_0"]
            else :
                with torch.no_grad():
                    for t in range(args.sample_distance, -1, -1):
                        if t > 0:
                            if args.recon_with_standard_gaussian:
                                model_output = torch.rand_like(x_0.float().to(x_0.device))
                            else :
                                model_output = ema(x_t, torch.Tensor([t]).repeat(x_0.shape[0], ).long().to(x_0.device))
                            pred_x_0 = diffusion.predict_x_0_from_eps(x_t,
                                                                      torch.Tensor([t]).repeat(x_0.shape[0], ).long().to(x_0.device),
                                                                      model_output)
                            x_t = diffusion.q_posterior_mean_variance(pred_x_0, x_t,
                                                                      torch.Tensor([t]).repeat(x_0.shape[0], ).long().to(x_0.device),)[0]
                            """
                            kl_div = out["whole_kl"]  # batch, 1, W, H
                            # normal portion kl divergence
                            normal_kl_div = (kl_div * mask_info).sum([1, 2, 3])
                            normal_pixel_num = mask_info.sum([1, 2, 3])
                            normal_pixel_num = torch.where(normal_pixel_num == 0, 1, normal_pixel_num)
                            normal_kl_score = normal_kl_div / normal_pixel_num

                            # abnormal portaion kl divergence
                            abnormal_sample_kl = kl_div[normal_info != 1]

                            if abnormal_sample_kl.shape[0] > 0 and abnormal_sample_kl.dim() == 4:
                                abnormal_mask = mask_info[normal_info != 1]

                                abnormal_mask_abnormal_portion = 1 - abnormal_mask
                                abnormal_sample_kl = (abnormal_sample_kl * abnormal_mask_abnormal_portion).sum(
                                    [1, 2, 3])
                                abnormal_pixel_num = abnormal_mask_abnormal_portion.sum([1, 2, 3])
                                abnormal_pixel_num = torch.where(abnormal_pixel_num == 0, 1, abnormal_pixel_num)
                                abnormal_kl_score = abnormal_sample_kl / abnormal_pixel_num

                                normal_kl_score = normal_kl_score.mean()
                                abnormal_kl_score = abnormal_kl_score.mean()
                                normal_scores.append(normal_kl_score)
                                abnormal_scores.append(abnormal_kl_score)
                            """
                    """
                    normal_score = torch.stack(normal_scores).mean()
                    abnormal_score = torch.stack(abnormal_scores).mean()
                    wandb.log({f"[{train_data}] normal kl": normal_score, f"[{train_data}] abnormal kl": abnormal_score, })
                    """
                    recon = x_t
            real_images = x_0[:num_images, ...].cpu()#.permute(0,1,3,2) # [Batch, 1, W, H]
            #sample_images = x_t[:num_images, ...].cpu()#.permute(0, 1, 3, 2)  # [Batch, 1, W, H]
            recon_images = recon[:num_images, ...].cpu()
            mask_images = mask_info[:num_images, ...].cpu()
            for img_index in range(num_images):
                normal_info_ = normal_info[img_index]
                if normal_info_ == 1:
                    is_normal = 'normal'
                else :
                    is_normal = 'abnormal'

                real = real_images[img_index,...].squeeze()
                #real= real.unsqueeze(0)
                real = torch_transforms.ToPILImage()(real)

                #sample = sample_images[img_index,...].squeeze()
                sample = recon_images[img_index,...].squeeze()
                #sample = sample.unsqueeze(0)
                sample = torch_transforms.ToPILImage()(sample)

                #mask = mask_images[img_index,...].squeeze()
                #mask = mask.unsqueeze(0).to(weight_dtype)
                #mask = torch_transforms.ToPILImage()(mask)

                #recon_mask = PIL.Image.blend(sample, mask, 0.5)

                new_image = PIL.Image.new('RGB', (2 * real.size[0], real.size[1]),(250,250,250))
                new_image.paste(real,   (0, 0))
                new_image.paste(sample, (real.size[0], 0))
                #new_image.paste(recon_mask, (real.size[0]+sample.size[0], 0))
                new_image.save(os.path.join(image_save_dir, f'real_recon_mask_epoch_{epoch}_{train_data}_{is_normal}_{img_index}.png'))
                loading_image = wandb.Image(new_image,
                                            caption=f"(real-noisy-recon) epoch {epoch + 1} | {is_normal}")
                wandb.log({train_data : loading_image})

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
    w, h = int(args.img_size.split(',')[0].strip()), int(args.img_size.split(',')[1].strip())
    train_transforms = transforms.Compose([transforms.Resize((w, h), transforms.InterpolationMode.BILINEAR),
                                           transforms.ToTensor()])
    train_ds = SYDataset(data_folder=args.train_data_folder, transform=train_transforms,
                         base_mask_dir=args.train_mask_dir, image_size=(w, h))
    training_dataset_loader = SYDataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                                           num_workers=4, persistent_workers=True)
    # ## Prepare validation set data loader
    val_transforms = transforms.Compose([transforms.Resize((w, h), transforms.InterpolationMode.BILINEAR),
                                         transforms.ToTensor()])
    val_ds = SYDataset(data_folder=args.val_data_folder, transform=val_transforms,
                       base_mask_dir=args.val_mask_dir, image_size=(w, h))
    test_dataset_loader = SYDataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                                       num_workers=4, persistent_workers=True)

    print(f'\n step 3. model')
    in_channels = args.in_channels
    model = UNetModel(img_size=int(w),
                      base_channels=args.base_channels,
                      dropout=args.dropout,
                      n_heads=args.num_heads,
                      n_head_channels=args.num_head_channels,
                      in_channels=in_channels)
    ema = copy.deepcopy(model)
    model.to(device)
    ema.to(device)
    # (2) scaheduler
    betas = get_beta_schedule(args.timestep,
                              args.beta_schedule)
    # (3) scaheduler
    diffusion = GaussianDiffusionModel([w, h],  # [128, 128]
                                       betas,  # 1
                                       img_channels=in_channels,
                                       loss_type=args.loss_type,  # l2
                                       loss_weight=args.loss_weight,  # none
                                       noise='simplex' )  # 1

    print(f'\n step 5. optimizer')
    optimiser = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.999))

    print(f'\n step 6. training')
    tqdm_epoch = range(args.start_epoch, args.train_epochs + 1)
    for epoch in tqdm_epoch:
        progress_bar = tqdm(enumerate(training_dataset_loader), total=len(training_dataset_loader), ncols=200)
        progress_bar.set_description(f"Epoch {epoch}")
        for step, data in progress_bar:
            model.train()
            x_0 = data["image_info"].to(device)  # batch, channel, w, h
            normal_info = data['normal'] # if 1 = normal, 0 = abnormal
            mask_info = data['mask'].unsqueeze(dim=1)    # if 1 = normal, 0 = abnormal
            if args.only_normal_training :
                x_0 = x_0[normal_info == 1]
                mask_info = mask_info[normal_info == 1]

            if x_0.shape[0] != 0 and x_0.dim() == 4 :
                t = torch.randint(0, args.sample_distance, (x_0.shape[0],), device =device)
                if args.use_simplex_noise:
                    noise = diffusion.noise_fn(x=x_0, t=t, octave=6, frequency=64).float()
                else:
                    noise = torch.rand_like(x_0).float().to(device)
                with torch.no_grad():
                    x_t = diffusion.sample_q(x_0, t, noise)                # 3) model prediction

                if args.simple_loss :
                    noise_pred = model(x_t, t)
                    target = noise
                    loss = torch.nn.functional.mse_loss(noise_pred.float(), target.float(), reduction="none").mean(dim=(1, 2, 3))
                elif args.kl_loss :
                    negative_log_likelihood = diffusion.calc_vlb_xt(model, x_0, x_t, t)['output']
                    loss = negative_log_likelihood.mean
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
                if epoch % args.inference_freq == 0 and epoch > 0:
                    for i, test_data in enumerate(test_dataset_loader):
                        if i == step :
                            ema.eval()
                            model.eval()
                            inference_num = min(args.inference_num, args.batch_size)
                            training_outputs(diffusion, test_data, epoch, inference_num, save_imgs=args.save_imgs,
                                             ema=ema, args=args, is_train_data = False, device = device)
                            training_outputs(diffusion, data, epoch, inference_num, save_imgs=args.save_imgs,
                                             ema=ema, args=args, is_train_data=True, device = device)
        if epoch % args.model_save_freq == 0 and epoch > 0 :
            save(unet=model, args=args, optimiser=optimiser, final=False, ema=ema, epoch=epoch)
    save(unet=model, args=args, optimiser=optimiser, final=True, ema=ema)

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
                        default=f'/data7/sooyeon/medical_image/anoddpm_result/20231119_dental_test')

    # step 2. dataset and dataloatder
    parser.add_argument('--train_data_folder', type=str)
    parser.add_argument('--train_mask_dir', type=str)
    parser.add_argument('--val_data_folder', type=str)
    parser.add_argument('--val_mask_dir', type=str)
    parser.add_argument('--img_size', type=str)
    parser.add_argument('--batch_size', type=int)

    # step 3. model
    parser.add_argument('--in_channels', type=int, default = 1)
    parser.add_argument('--base_channels', type=int, default = 256)
    parser.add_argument('--channel_mults', type=str, default = '1,2,3,4')
    parser.add_argument('--dropout', type=float, default = 0.0)
    parser.add_argument('--num_heads', type=int, default = 2)
    parser.add_argument('--num_head_channels', type=int, default = -1)

    # (2) scaheduler
    parser.add_argument('--timestep', type=int, default=1000)
    parser.add_argument('--beta_schedule', type=str, default='linear')
    # (3) diffusion
    parser.add_argument('--loss_weight', type=str, default = "none")
    parser.add_argument('--loss_type', type=str, default="l2")
    # step 5. optimizer
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    # step 6. training
    parser.add_argument('--start_epoch', type=int, default=0)
    parser.add_argument('--train_epochs', type=int, default=3000)
    parser.add_argument('--train_start', action = 'store_true')
    parser.add_argument('--use_simplex_noise', action='store_true')
    parser.add_argument('--sample_distance', type=int, default = 800)

    parser.add_argument('--only_normal_training', action='store_true')
    parser.add_argument('--use_vlb_loss', action='store_true')
    parser.add_argument('--kl_loss_weight', type=float, default=1.0)
    parser.add_argument('--simple_loss', action='store_true')
    parser.add_argument('--kl_loss', action='store_true')

    # step 7. inference
    parser.add_argument('--inference_num', type=int, default=4)
    parser.add_argument('--inference_freq', type=int, default=10)
    parser.add_argument('--model_save_freq', type=int, default=1000)
    parser.add_argument('--save_imgs', action='store_true')
    parser.add_argument('--save_vids', action='store_true')

    parser.add_argument('--onestep_inference', action='store_true')
    parser.add_argument('--recon_with_standard_gaussian', action='store_true')

    args = parser.parse_args()
    main(args)