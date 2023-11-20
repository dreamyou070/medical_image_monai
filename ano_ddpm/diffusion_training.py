import argparse, wandb
import collections
import copy
import time
from random import seed
from torch import optim
from GaussianDiffusion import GaussianDiffusionModel, get_beta_schedule
from helpers import *
from tqdm import tqdm
from monai import transforms
import numpy  as np
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from data_module import SYDataLoader, SYDataset_masking
from monai.utils import first
from UNet import UNetModel, update_ema_params
import torch.multiprocessing
import torchvision.transforms as torch_transforms
import PIL

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

def training_outputs(diffusion, test_data, epoch, num_images, ema, args,
                     save_imgs=False, is_train_data=True, device='cuda'):

    if is_train_data :
        train_data = 'training_data'
    else :
        train_data = 'test_data'
    video_save_dir = os.path.join(args.experiment_dir, 'diffusion-videos')
    image_save_dir = os.path.join(args.experiment_dir, 'diffusion-training-images')
    os.makedirs(video_save_dir, exist_ok=True)
    os.makedirs(image_save_dir, exist_ok=True)

    if save_imgs:

        # 1) make random noise
        x = test_data["image_info"]['image'].to(device)  # batch, channel, w, h
        normal_info = test_data['normal']  # if 1 = normal, 0 = abnormal
        mask_info = test_data['mask']  # if 1 = normal, 0 = abnormal

        noise = torch.rand_like(x).to(x.device)

        # 2) select random int
        t = torch.Tensor([args.sample_distance]).repeat(x.shape[0]).to(x.device)
        time_step = t[0].item()

        # 3) q sampling = noising & p sampling = denoising
        print(f'Let sample q ')
        x_t = diffusion.sample_q(x, t, noise)
        time.sleep(10)

        print(f'Let sample p ')
        temp = diffusion.sample_p(ema, x_t, t)

        # 4)
        real_images = x[:num_images, ...].cpu().permute(0,1,3,2) # [Batch, 1, W, H]
        sample_images = temp["sample"][:num_images, ...].cpu().permute(0, 1, 3, 2)  # [Batch, 1, W, H]
        pred_images = temp["pred_x_0"][:num_images, ...].cpu().permute(0,1,3,2)

        merge_images = []
        for img_index in range(num_images):
            normal_info = normal_info[img_index]
            real = real_images[img_index,...].squeeze()
            real= real.unsqueeze(0)
            real = torch_transforms.ToPILImage()(real)
            sample = sample_images[img_index,...].squeeze()
            sample = sample.unsqueeze(0)
            sample = torch_transforms.ToPILImage()(sample)

            pred = pred_images[img_index,...].squeeze()
            pred = pred.unsqueeze(0)
            pred = torch_transforms.ToPILImage()(pred)

            new_image = PIL.Image.new('RGB', (3 * real.size[0], real.size[1]), (250, 250, 250))
            new_image.paste(real, (0, 0))
            new_image.paste(sample, (real.size[0], 0))
            new_image.paste(pred, (real.size[0]+sample.size[0], 0))
            merge_images.append(new_image)

        new_image = PIL.Image.new('RGB', (merge_images[0].size[0], len(merge_images) * merge_images[0].size[1]), (250, 250, 250))
        for i, img in enumerate(merge_images) :
            new_image.paste(img, (0, i * img.size[1]))
        img_save_dir = os.path.join(image_save_dir, f'epoch_{epoch}_{train_data}.png')
        new_image.save(img_save_dir)
        print(f'saving image to {img_save_dir}')
        loading_image = wandb.Image(new_image,
                                    caption=f"epoch : {epoch + 1}")
        wandb.log({"train_data": loading_image})



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
    var_args = vars(args)
    with open(os.path.join(experiment_dir, "config.txt"), "w") as f:
        for key in sorted(var_args.keys()):
            f.write(f"{key}: {var_args[key]}\n")

    print(f'\n step 2. dataset and dataloatder')
    train_datas = os.listdir(args.train_data_folder)
    val_datas = os.listdir(args.val_data_folder)
    train_datalist = [{"image": os.path.join(args.train_data_folder, train_data)} for train_data in train_datas]
    w,h = int(args.img_size.split(',')[0].strip()), int(args.img_size.split(',')[1].strip())
    train_transforms = transforms.Compose([transforms.LoadImaged(keys=["image"]),
                                           transforms.EnsureChannelFirstd(keys=["image"]),
                                           transforms.ScaleIntensityRanged(keys=["image"],
                                                                           a_min=0.0, a_max=255.0, b_min=0.0, b_max=1.0, clip=True),
                                           transforms.RandAffined(keys=["image"],
                                                                  spatial_size=[w, h], # output image spatial size ...........
                                                                  rotate_range=[(-np.pi / 36, np.pi / 36),(-np.pi / 36, np.pi / 36)],
                                                                  translate_range=[(-1, 1), (-1, 1)],
                                                                  scale_range=[(-0.05, 0.05), (-0.05, 0.05)],
                                                                  padding_mode="zeros",
                                                                  prob=0.5, ), ])
    train_ds = SYDataset_masking(data=train_datalist,transform=train_transforms,
                                 base_mask_dir=args.train_mask_dir,image_size = args.img_size)
    training_dataset_loader = SYDataLoader(train_ds,batch_size=args.batch_size,
                                         shuffle=True, num_workers=4, persistent_workers=True)
    check_data = first(training_dataset_loader)

    # ## Prepare validation set data loader
    val_datalist = [{"image": os.path.join(args.val_data_folder, val_data)} for val_data in val_datas]
    val_transforms = transforms.Compose([transforms.LoadImaged(keys=["image"]),
                                         transforms.EnsureChannelFirstd(keys=["image"]),
                                         transforms.ScaleIntensityRanged(keys=["image"],
                                                                         a_min=0.0, a_max=255.0, b_min=0.0, b_max=1.0,
                                                                         clip=True),
                                         transforms.RandAffined(keys=["image"], spatial_size=[w, h], ), ])
    val_ds = SYDataset_masking(data=val_datalist, transform=val_transforms,
                               base_mask_dir=args.val_mask_dir, image_size=args.img_size)
    test_dataset_loader = SYDataLoader(val_ds,batch_size=args.batch_size,
                                     shuffle=True, num_workers=4, persistent_workers=True)

    print(f'\n step 3. resume or not')


    print(f'\n step 4. model')
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
    betas = get_beta_schedule(args.timestep, args.beta_schedule)
    # (3) scaheduler
    diffusion = GaussianDiffusionModel([w,h], #  [128, 128]
                                       betas,            #  1
                                       img_channels=in_channels,
                                       loss_type=args.loss_type,  # l2
                                       loss_weight=args.loss_weight, # none
                                       noise= args.noise_fn,)        # 1

    print(f'\n step 5. optimizer')
    optimiser = optim.AdamW(model.parameters(),
                            lr=args.lr,
                            weight_decay=args.weight_decay,
                            betas=(0.9, 0.999))

    print(f'\n step 6. training')
    tqdm_epoch = range(args.start_epoch, args.train_epochs + 1)
    start_time = time.time()
    vlb = collections.deque([], maxlen=10)

    for epoch in tqdm_epoch:
        progress_bar = tqdm(enumerate(training_dataset_loader), total=len(training_dataset_loader), ncols=300)
        progress_bar.set_description(f"Epoch {epoch}")
        for step, data in progress_bar:
            model.train()
            loss_dict = {}
            # -----------------------------------------------------------------------------------------
            # 0) data check
            x = data["image_info"]['image'].to(device)  # batch, channel, w, h
            normal_info = data['normal'] # if 1 = normal, 0 = abnormal
            mask_info = data['mask']     # if 1 = normal, 0 = abnormal
            if args.only_normal_training :
                x = x[normal_info == 1]
                mask_info = mask_info[normal_info == 1]
            # -----------------------------------------------------------------------------------------
            # 1) check random t
            t = torch.randint(0, args.sample_distance, (x.shape[0],), device = x.device)
            # 2) make noisy latent
            noise = diffusion.noise_fn(x, t).float()
            x_t = diffusion.sample_q(x, t, noise)
            # 3) model prediction
            noise_pred = model(x_t, t)
            target = noise
            if args.masked_loss:
                noise_pred = noise_pred * mask_info
                target = target * mask_info
            # 4) loss
            loss = torch.nn.functional.mse_loss(noise_pred.float(), target.float(), reduction="none")
            loss = loss.mean()

            def mean_flat(tensor):
                return torch.mean(tensor, dim=list(range(1, len(tensor.shape))))
            ddpm_loss = mean_flat((noise_pred - target).square()).mean()
            print(f'loss : {loss.item()} | ddpm_loss : {ddpm_loss.item()}')

            wandb.log({"loss": loss.item()})
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
                        training_outputs(diffusion, test_data, epoch, args.inference_num, save_imgs=args.save_imgs,
                                         ema=ema, args=args, is_train_data = False, device = device)
                        training_outputs(diffusion, data, epoch, args.inference_num, save_imgs=args.save_imgs,
                                         ema=ema, args=args, is_train_data=True, device = device)
            
        if epoch % args.vlb_freq == 0:
            for i, test_data in enumerate(test_dataset_loader) :
                if i == 0 :
                    x = test_data["image_info"]['image'].to(device)
                    time_taken = time.time() - start_time
                    remaining_epochs = args.train_epochs - epoch
                    time_per_epoch = time_taken / (epoch + 1 - args.start_epoch)
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
        if epoch % args.model_save_freq == 0 and epoch >= 0:
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
    parser.add_argument('--loss_type', type=str, default='l2')
    parser.add_argument('--noise_fn', type=str, default='simplex')

    # step 5. optimizer
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.0)

    # step 6. training
    parser.add_argument('--start_epoch', type=int, default=0)
    parser.add_argument('--train_epochs', type=int, default=3000)
    parser.add_argument('--train_start', action = 'store_true')
    parser.add_argument('--sample_distance', type=int, default = 800)
    parser.add_argument('--only_normal_training', action='store_true')
    parser.add_argument('--masked_loss', action='store_true')


    # step 7. inference
    parser.add_argument('--inference_num', type=int, default=4)
    parser.add_argument('--inference_freq', type=int, default=50)
    parser.add_argument('--vlb_freq', type=int, default=200)
    parser.add_argument('--model_save_freq', type=int, default=1000)
    parser.add_argument('--save_imgs', action='store_true')
    parser.add_argument('--save_vids', action='store_true')
    args = parser.parse_args()
    main(args)

