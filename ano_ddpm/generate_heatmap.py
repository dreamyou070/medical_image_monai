import numpy as np
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
from monai.utils import first
from UNet import UNetModel, update_ema_params
import torch.multiprocessing
import torchvision.transforms as torch_transforms
import PIL
from setproctitle import *
import sys, os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from data_module import SYDataLoader, SYDataset
from heatmap_module import _convert_heat_map_colors, expand_image
from PIL import Image
from matplotlib import cm

#torch.multiprocessing.set_sharing_strategy('file_system')
#torch.cuda.empty_cache()

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
        x = test_data["image_info"].to(device)  # batch, channel, w, h
        normal_info = test_data['normal']  # if 1 = normal, 0 = abnormal
        mask_info = test_data['mask']  # if 1 = normal, 0 = abnormal

        t = torch.randint(args.sample_distance - 1, args.sample_distance, (x.shape[0],), device=x.device)
        time_step = t[0].item()

        if args.use_simplex_noise:
            noise = diffusion.noise_fn(x=x, t=t, octave=6, frequency=64).float()
        else:
            noise = torch.rand_like(x).float().to(x.device)
        # 2) select random int

        with torch.no_grad():
            # 3) q sampling = noising & p sampling = denoising
            x_t = diffusion.sample_q(x, t, noise)
            temp = diffusion.sample_p(ema, x_t, t)

        # 4) what is sample_p do ?
        real_images = x[:num_images, ...].cpu()#.permute(0,1,3,2) # [Batch, 1, W, H]
        sample_images = temp["sample"][:num_images, ...].cpu()#.permute(0, 1, 3, 2)  # [Batch, 1, W, H]
        pred_images = temp["pred_x_0"][:num_images, ...].cpu()#.permute(0,1,3,2)
        merge_images = []
        #num_images = min(len(normal_info), num_images)
        for img_index in range(num_images):
            normal_info_ = normal_info[img_index]
            if normal_info_ == 1:
                is_normal = 'normal'
            else :
                is_normal = 'abnormal'
            real = real_images[img_index,...].squeeze()
            real= real.unsqueeze(0)
            real = torch_transforms.ToPILImage()(real)
            sample = sample_images[img_index,...].squeeze()
            sample = sample.unsqueeze(0)
            sample = torch_transforms.ToPILImage()(sample)
            pred = pred_images[img_index,...].squeeze()
            pred = pred.unsqueeze(0)
            pred = torch_transforms.ToPILImage()(pred)
            new_image = PIL.Image.new('L', (3 * real.size[0], real.size[1]),250)
            new_image.paste(real, (0, 0))
            new_image.paste(sample, (real.size[0], 0))
            new_image.paste(pred, (real.size[0]+sample.size[0], 0))
            new_image.save(os.path.join(image_save_dir, f'real_noisy_recon_epoch_{epoch}_{train_data}_{is_normal}_{img_index}.png'))
            loading_image = wandb.Image(new_image,
                                        caption=f"(real-noisy-recon) epoch {epoch + 1} | {is_normal} | {train_data}")
            wandb.log({"inference": loading_image})

def generate_heatmap_image(data, device, diffusion, model, args,save_base_dir, is_train ) :
    caption = 'train'
    if not is_train :
        caption = 'test'
    print(f'generate heatmap image {caption} data')
    contents = []
    x = data["image_info"].to(device)
    img_dirs = data['image_dir']
    mask_info = data['mask']  # if 1 = normal, 0 = abnormal -> shape [ batch, 128, 128 ]
    with torch.no_grad():
        vlb_terms = diffusion.calc_total_vlb_in_sample_distance(x, model, args)
    vlb = vlb_terms["whole_vb"]                # [batch, 1000, 1, W, H]
    pixelwise_anormal_score = vlb.squeeze(dim=2).mean(dim=1)  # [batch, W, H]
    w,h = pixelwise_anormal_score.shape[-2], pixelwise_anormal_score.shape[-1]
    for img_index in range(x.shape[0]):
        img_dir = img_dirs[img_index]
        img_name = os.path.split(img_dir)[-1]
        abnormal_pixel_num = 0
        score_patch = pixelwise_anormal_score[img_index].squeeze()
        anormal_detect_background = torch.zeros_like(score_patch)
        for i in range(w):
            for j in range(h):
                abnormal_score = score_patch[i, j]
                if abnormal_score < args.thredhold :
                    anormal_detect_background[i, j] = 0
                else :
                    abnormal_pixel_num += 1
                    anormal_detect_background[i, j] = 1.0 #abnormal_score
        line = f'[train] {img_name} | abnormal pixel num : {abnormal_pixel_num}'
        contents.append(line)
        # --------------------------------------------------------------------------------------------------------------
        image = data["image_info"][img_index].squeeze()                           # [1, 128, 128]
        original_img = torch_transforms.ToPILImage()(image).convert('RGB')
        # --------------------------------------------------------------------------------------------------------------
        # [128,128] torch
        heat_map = expand_image(im=anormal_detect_background, h=h, w=w, absolute=False).to('cpu').detach()
        max_value = heat_map.max()
        np_heatmap = cm.turbo(heat_map)[:, :, :-1]
        np_heatmap = np.uint8(np_heatmap * 255)
        heat_map_img = Image.fromarray(np_heatmap)
        blended_img = Image.blend(original_img, heat_map_img, 0.5)
        # --------------------------------------------------------------------------------------------------------------
        mask_np = mask_info[img_index].squeeze().to('cpu').detach().numpy().copy().astype(np.uint8)
        mask_np = mask_np * 255
        mask_img = Image.fromarray(mask_np).convert('RGB')  # [128, 128, 3]
        # --------------------------------------------------------------------------------------------------------------
        new_image = PIL.Image.new('RGB', (4 * w, h), (0,0,0))
        new_image.paste(original_img, (0, 0))
        new_image.paste(heat_map_img, (w, 0))
        new_image.paste(blended_img, (2*w, 0))
        new_image.paste(mask_img, (3*w, 0))
        new_image.save(os.path.join(save_base_dir, f'heatmap_{caption}_data_{img_index}.png'))
    return contents


def main(args):

    print(f'\n step 1. setting')
    if args.process_title:
        setproctitle(args.process_title)
    else:
        setproctitle('parksooyeon')

    print(f' (1.2) seed and device')
    seed(args.seed)
    device = args.device

    print(f' (1.3) experiment_dir')
    experiment_dir = args.experiment_dir
    os.makedirs(experiment_dir, exist_ok=True)

    print(f'\n step 2. dataset and dataloatder')
    w, h = int(args.img_size.split(',')[0].strip()), int(args.img_size.split(',')[1].strip())
    train_transforms = transforms.Compose([transforms.Resize((w, h), transforms.InterpolationMode.BILINEAR),
                                           transforms.ToTensor(),])
    train_ds = SYDataset(data_folder=args.train_data_folder,
                         transform=train_transforms,
                         base_mask_dir=args.train_mask_dir,
                         image_size=(w, h))
    from torch.utils.data import DataLoader
    training_dataset_loader = DataLoader(train_ds,
                                           batch_size=args.batch_size,
                                           shuffle=True,)
                                           #num_workers=0,)
                                           #persistent_workers=True)
    # ## Prepare validation set data loader
    val_transforms = transforms.Compose([transforms.Resize((w, h), transforms.InterpolationMode.BILINEAR),
                                         transforms.ToTensor(),])
    val_ds = SYDataset(data_folder=args.val_data_folder,
                       transform=val_transforms,
                       base_mask_dir=args.val_mask_dir, image_size=(w, h))
    test_dataset_loader = SYDataLoader(val_ds,
                                       batch_size=args.batch_size,
                                       shuffle=False,)
                                       #num_workers=4,
                                      # persistent_workers=True)

    print(f'\n step 3. unet model')
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
    print(f' (3.1) model load state dict')
    state_dict_path = os.path.join(args.experiment_dir, 'diffusion-models', args.model_name)
    model_epoch = str(int(os.path.splitext(args.model_name)[0].split('_')[-1]))

    if os.path.exists(state_dict_path):
        state_dict = torch.load(state_dict_path)
        model.load_state_dict(state_dict['model_state_dict'])
        #ema.load_state_dict(state_dict['ema'])

    print(f'\n step 4. diffusion')
    betas = get_beta_schedule(args.timestep, args.beta_schedule)
    diffusion = GaussianDiffusionModel([w, h],  # [128, 128]
                                       betas,  # 1
                                       img_channels=in_channels,
                                       loss_type=args.loss_type,  # l2
                                       loss_weight=args.loss_weight,  # none
                                       noise='simplex')  # 1

    print(f'\n step 5. inference')
    img_base_dir = os.path.join(args.experiment_dir, 'inference')
    os.makedirs(img_base_dir, exist_ok=True)
    save_base_dir = os.path.join(img_base_dir, f'model_epoch_{model_epoch}_thredhold_{args.thredhold}')
    os.makedirs(save_base_dir, exist_ok=True)

    print(f' (5.1) where to write')
    abnormal_pixel_num_file = os.path.join(save_base_dir, f'abnormal_pixel_num.txt')

    print(f' (5.2) inferencing')
    for i, train_data in training_dataset_loader :
        if i == 0 :
            train_content = generate_heatmap_image(train_data, device, diffusion, model, args,save_base_dir, is_train= 'true' )
    """
    test_data = first(training_dataset_loader)
    test_content = generate_heatmap_image(test_data, device, diffusion, model, args, save_base_dir, is_train= 'false')
    total_content = train_content + test_content
    with open(abnormal_pixel_num_file, 'w') as f:
        for content in total_content:
            f.write(content)
    """


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # step 1. wandb login
    parser.add_argument("--process_title", type=str, default='parksooyeon')
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str)
    parser.add_argument('--experiment_dir', type=str,
                        default=f'/data7/sooyeon/medical_image/anoddpm_result/7_gaussian_linear_pos_infonce')

    # step 2. dataset and dataloatder
    parser.add_argument('--train_data_folder', type=str)
    parser.add_argument('--train_mask_dir', type=str)
    parser.add_argument('--val_data_folder', type=str)
    parser.add_argument('--val_mask_dir', type=str)
    parser.add_argument('--img_size', type=str)
    parser.add_argument('--batch_size', type=int)

    # step 3. model
    parser.add_argument('--in_channels', type=int, default=1)
    parser.add_argument('--base_channels', type=int, default=256)
    parser.add_argument('--channel_mults', type=str, default='1,2,3,4')
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--num_heads', type=int, default=2)
    parser.add_argument('--num_head_channels', type=int, default=-1)
    parser.add_argument('--model_name', type=str, default='unet_epoch_500.pt')

    # step 4. model
    parser.add_argument('--timestep', type=int, default=1000)
    parser.add_argument('--beta_schedule', type=str, default='linear')
    parser.add_argument('--loss_weight', type=str, default="none")
    parser.add_argument('--loss_type', type=str, default='l2')
    parser.add_argument('--sample_distance', type=int, default=800)
    parser.add_argument('--thredhold', type=float, default=0.05)

    args = parser.parse_args()
    main(args)