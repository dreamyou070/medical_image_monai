import argparse, wandb
import collections
import copy
import matplotlib.pyplot as plt
from matplotlib import animation
import time
from random import seed
from torch import optim
from GaussianDiffusion import GaussianDiffusionModel, get_beta_schedule
from helpers import *
from tqdm import tqdm
from monai import transforms
import numpy  as np
from monai.data import DataLoader, Dataset
from monai.utils import first
from UNet import UNetModel, update_ema_params
import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy('file_system')
torch.cuda.empty_cache()

def save(final, unet, optimiser, args, ema, loss=0, epoch=0):
    model_save_base_dir = os.path.join(args['experiment_dir'], 'diffusion-models')
    os.makedirs(model_save_base_dir, exist_ok=True)
    if final:
        torch.save({'n_epoch':              args["EPOCHS"],
                    'model_state_dict':     unet.state_dict(),
                    'optimizer_state_dict': optimiser.state_dict(),
                    "ema":                  ema.state_dict(),
                    "args":                 args},
            os.path.join(model_save_base_dir, f'diff-params-ARGS={args["arg_num"]}_params-final.pt'))
    else:
        torch.save({'n_epoch':              epoch,
                    'model_state_dict':     unet.state_dict(),
                    'optimizer_state_dict': optimiser.state_dict(),
                    "args":                 args,
                    "ema":                  ema.state_dict(),
                    'loss':                 loss,},
            os.path.join(model_save_base_dir, f'diff-params-ARGS={args["arg_num"]}_diff_epoch={epoch}.pt'))
def training_outputs(diffusion, x, est, noisy, epoch, row_size, ema, args,
                     save_imgs=False, save_vids=False,):

    video_save_dir = os.path.join(args['experiment_dir'], 'diffusion-videos')
    image_save_dir = os.path.join(args['experiment_dir'], 'diffusion-training-images')
    os.makedirs(video_save_dir, exist_ok=True)
    os.makedirs(image_save_dir, exist_ok=True)

    if save_imgs:
        if epoch % 100 == 0 or epoch < 2:

            # 1) make random noise
            noise = torch.rand_like(x)

            # 2) select random int
            t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=x.device)

            # 3) q sampling = noising & p sampling = denoising
            x_t  = diffusion.sample_q(x, t, noise)
            temp = diffusion.sample_p(ema, x_t, t)

            # 4)
            out = torch.cat((x[:row_size, ...].cpu(),                             # real number
                             temp["sample"][:row_size, ...].cpu(),                # t-1 sample
                             temp["pred_x_0"][:row_size, ...].cpu()))             # prediction (reconstruction)
            plt.title(f'real,sample(one step upmode?) , prediction x_0-{epoch}epoch')
        else:
            out = torch.cat((x[:row_size, ...].cpu(),
                             noisy[:row_size, ...].cpu(),                         # just random noise
                             est[:row_size, ...].cpu(),                           # estimated noise
                             (est - noisy).square().cpu()[:row_size, ...]))       # the difference between estimated and random noise (should be zero)
            plt.title(f'real,noisy,noise prediction,mse-{epoch}epoch')
        plt.rcParams['figure.dpi'] = 150
        plt.grid(False)
        plt.imshow(gridify_output(out, row_size), cmap='gray')
        plt.axis('off')
        img_save_dir = os.path.join(image_save_dir, f'ARGS={args["arg_num"]}_EPOCH={epoch}.png')
        print(f'saving image to {img_save_dir}')
        plt.savefig(img_save_dir)
        plt.clf()
    if save_vids:
        fig, ax = plt.subplots()
        if epoch % 500 == 0 or epoch < 2 :
            plt.rcParams['figure.dpi'] = 200
            if epoch % 1000 == 0:
                out = diffusion.forward_backward(ema, x, "half", args['sample_distance'] // 2, denoise_fn="noise_fn")
            else:
                out = diffusion.forward_backward(ema, x, "half", args['sample_distance'] // 4, denoise_fn="noise_fn")
            imgs = [[ax.imshow(gridify_output(x, row_size), animated=True)] for x in out]
            ani = animation.ArtistAnimation(fig, imgs, interval=50, blit=True,repeat_delay=1000)
            ani_save_dir = os.path.join(video_save_dir, f'ARGS={args["arg_num"]}_EPOCH={epoch}.mp4')
            ani.save(ani_save_dir)

    plt.close('all')

def main(args) :

    print(f'\n step 1. setting')
    print(f' (1.1) wandb')
    wandb.login(key=args.wandb_api_key)
    wandb.init(project=args.wandb_project_name, name=args.wandb_run_name)

    print(f' (1.2) seed and device')
    seed(args.seed)
    device = args.device

    print(f' (1.3) saving configuration')
    experiment_dir = args.experiment_dir
    os.makedirs(experiment_dir, exist_ok=True)
    with open(os.path.join(experiment_dir, "config.txt"), "w") as f:
        for key in sorted(args.keys()):
            f.write(f"{key}: {args[key]}\n")


    print(f'\n step 2. dataset and dataloatder')
    train_datas = os.listdir(args.train_data_folder)
    val_datas = os.listdir(args.val_data_folder)
    train_datalist = [{"image": os.path.join(args.train_data_folder, train_data)} for train_data in train_datas]
    w,h = int(args.img_size.split(',')[0].strip()), int(args.img_size.split(',')[1].strip())
    train_transforms = transforms.Compose([transforms.LoadImaged(keys=["image"]),
                                           transforms.EnsureChannelFirstd(keys=["image"]),
                                           transforms.ScaleIntensityRanged(keys=["image"], a_min=0.0, a_max=255.0, b_min=0.0, b_max=1.0, clip=True),
                                           transforms.RandAffined(keys=["image"],
                                                                  rotate_range=[(-np.pi / 36, np.pi / 36),(-np.pi / 36, np.pi / 36)],
                                                                  translate_range=[(-1, 1), (-1, 1)],
                                                                  scale_range=[(-0.05, 0.05), (-0.05, 0.05)],
                                                                  spatial_size=[w,h],
                                                                  padding_mode="zeros",
                                                                  prob=0.5, ), ])
    train_ds = Dataset(data=train_datalist, transform=train_transforms)
    training_dataset_loader = DataLoader(train_ds,
                                         batch_size=args.batch_size,
                                         shuffle=True, num_workers=4, persistent_workers=True)
    check_data = first(training_dataset_loader)
    # ## Prepare validation set data loader
    val_datalist = [{"image": os.path.join(args.val_data_folder, val_data)} for val_data in val_datas]
    val_transforms = transforms.Compose([transforms.LoadImaged(keys=["image"]),
                                         transforms.EnsureChannelFirstd(keys=["image"]),
                                         transforms.ScaleIntensityRanged(keys=["image"],
                                                                         a_min=0.0, a_max=255.0, b_min=0.0, b_max=1.0,
                                                                         clip=True), ])
    val_ds = Dataset(data=val_datalist, transform=val_transforms)
    test_dataset_loader = DataLoader(val_ds,batch_size=args.batch_size,
                                     shuffle=True, num_workers=4, persistent_workers=True)

    print(f'\n step 3. resume or not')
    loaded_model = {}

    print(f'\n step 4. model')
    in_channels = args.in_channels
    model = UNetModel(w,
                      args.base_channels,
                      channel_mults=args.channel_mults,
                      dropout=args.dropout,
                      n_heads=args.num_heads,
                      n_head_channels=args.num_head_channels,
                      in_channels=in_channels)
    ema = copy.deepcopy(model)
    model.to(device)
    ema.to(device)
    # (2) scaheduler
    betas = get_beta_schedule(args.timestep, args.beta_schedule)
    # (3) scaheduler
    diffusion = GaussianDiffusionModel([w,h], #  [128, 128]
                                       betas,            #  1
                                       img_channels=in_channels,
                                       loss_weight=args.loss_weight, # none
                                       loss_type=args.loss_type,     # l2
                                       noise= args.noise_fn,)        # 1

    print(f'\n step 5. optimizer')
    optimiser = optim.AdamW(model.parameters(),
                            lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.999))

    print(f'\n step 6. training')
    start_epoch = args.start_epoch
    train_epoch = args.train_epochs
    tqdm_epoch = range(start_epoch, train_epoch + 1)
    start_time = time.time()
    losses = []
    vlb = collections.deque([], maxlen=10)
    for epoch in tqdm_epoch:
        mean_loss = []
        progress_bar = tqdm(enumerate(training_dataset_loader),
                            total=len(training_dataset_loader),
                            ncols=70)
        progress_bar.set_description(f"Epoch {epoch}")
        for step, data in progress_bar:
            x = data["image"].to(device)  # batch, channel, w, h
            # GaussianDiffusionModel.p_loss
            loss, estimates = diffusion.p_loss(model, x, args)
            noisy_latent, unet_estimate_noise = estimates[1], estimates[2]
            optimiser.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimiser.step()
            # ----------------------------------------------------------------------------------------- #
            # EMA model updating
            update_ema_params(ema, model)
            mean_loss.append(loss.data.cpu())
            if epoch % 50 == 0 and step == 0:
                row_size = min(8, args.batch_size)
                training_outputs(diffusion, x, unet_estimate_noise, noisy_latent, epoch, row_size,
                                 save_imgs=args['save_imgs'],  # true
                                 save_vids=args['save_vids'],  # true
                                 ema=ema, args=args)
        losses.append(np.mean(mean_loss))
        if epoch % 200 == 0:
            time_taken = time.time() - start_time
            remaining_epochs = args['EPOCHS'] - epoch
            time_per_epoch = time_taken / (epoch + 1 - start_epoch)
            hours = remaining_epochs * time_per_epoch / 3600
            mins = (hours % 1) * 60
            hours = int(hours)
            # ------------------------------------------------------------------------
            # calculate vlb loss
            vlb_terms = diffusion.calc_total_vlb(x, model, args)
            vlb.append(vlb_terms["total_vlb"].mean(dim=-1).cpu().item())
            print(f"epoch: {epoch}, most recent total VLB: {vlb[-1]} mean total VLB:"
                  f" {np.mean(vlb):.4f}, "
                  f"prior vlb: {vlb_terms['prior_vlb'].mean(dim=-1).cpu().item():.2f}, vb: "
                  f"{torch.mean(vlb_terms['vb'], dim=list(range(2))).cpu().item():.2f}, x_0_mse: "
                  f"{torch.mean(vlb_terms['x_0_mse'], dim=list(range(2))).cpu().item():.2f}, mse: "
                  f"{torch.mean(vlb_terms['mse'], dim=list(range(2))).cpu().item():.2f}"
                  f" time elapsed {int(time_taken / 3600)}:{((time_taken / 3600) % 1) * 60:02.0f}, "
                  f"est time remaining: {hours}:{mins:02.0f}\r")
        if epoch % 1000 == 0 and epoch >= 0:
            save(unet=model, args=args, optimiser=optimiser, final=False, ema=ema, epoch=epoch)
    save(unet=model, args=args, optimiser=optimiser, final=True, ema=ema)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # step 1. wandb login
    parser.add_argument("--wandb_api_key", type=str, default='3a3bc2f629692fa154b9274a5bbe5881d47245dc')
    parser.add_argument("--wandb_project_name", type=str, default='dental_experiment')
    parser.add_argument("--wandb_run_name", type=str, default='hand_1000_64res')
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str)
    parser.add_argument('--experiment_dir', type=str,
                        default = f'/data7/sooyeon/medical_image/anoddpm_result/20231119_dental_test')

    # step 2. dataset and dataloatder
    parser.add_argument('--train_data_folder', type=str)
    parser.add_argument('--val_data_folder', type=str)
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
    parser.add_argument('--loss_weight', type=float)
    parser.add_argument('--loss_type', type=str, default='l2')
    parser.add_argument('--noise_fn', type=str, default='simplex')

    # step 5. optimizer
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.0)

    # step 6. training
    parser.add_argument('--start_epoch', type=int, default=0)
    parser.add_argument('--train_epochs', type=int, default=3000)

    args = parser.parse_args()
    main(args)

