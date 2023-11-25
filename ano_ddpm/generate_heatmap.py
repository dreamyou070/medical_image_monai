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

def reconstructing(data, device, diffusion, model, args, save_base_dir, is_train) :
    caption = 'train'
    if is_train == 'false':
        caption = 'test'

    x = data["image_info"].to(device)
    normal_info = data['normal']
    recon = diffusion.dental_forward_backward(model,x,args,device,args.sample_distance)
    for img_index in range(x.shape[0]):
        is_normal = normal_info[img_index]
        if is_normal == 1 :
            normal_caption = 'normal'
        else :
            normal_caption = 'abnormal'
        image = x[img_index].squeeze()  # [1, 128, 128]
        original_img = torch_transforms.ToPILImage()(image).convert('RGB')
        w,h = original_img.size
        recon_img = torch_transforms.ToPILImage()(recon[img_index]).convert('RGB')
        new_image = PIL.Image.new('RGB', (2 * w, h), (0, 0, 0))
        new_image.paste(original_img, (0, 0))
        new_image.paste(recon_img, (w, 0))
        new_image.save(os.path.join(save_base_dir, f'{caption}_{normal_caption}_{img_index}.png'))


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

    pixelwise_anormal_score = vlb_terms["whole_vb"].squeeze(dim=2).mean(dim=1)     # [batch, W, H]
    w,h = pixelwise_anormal_score.shape[-2], pixelwise_anormal_score.shape[-1]
    for img_index in range(x.shape[0]):
        img_dir = img_dirs[img_index]
        img_name = os.path.split(img_dir)[-1]
        mask_ = mask_info[img_index]
        abnormal_pixel_num = 0
        score_patch = pixelwise_anormal_score[img_index].squeeze()
        anormal_detect_background = torch.zeros_like(score_patch)
        for i in range(w):
            for j in range(h):
                abnormal_score = score_patch[i, j]
                print(f'abnormal score : {abnormal_score}')
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
    train_data = first(training_dataset_loader)
    #train_content = generate_heatmap_image(train_data, device, diffusion, model, args,save_base_dir, is_train= 'true' )

    print(f' (5.3) reconstructing')
    recon_save_base_dir = os.path.join(save_base_dir, 'reconstruct')
    os.makedirs(recon_save_base_dir, exist_ok=True)
    recon_save_base_dir = os.path.join(recon_save_base_dir, f'model_epoch_{model_epoch}')
    os.makedirs(recon_save_base_dir, exist_ok=True)
    reconstructing(train_data, device, diffusion, model, args, recon_save_base_dir, is_train='true')

    #test_data = first(test_dataset_loader)
    #test_content = generate_heatmap_image(test_data, device, diffusion, model, args, save_base_dir, is_train= 'false')

    #total_content = train_content #+ test_content

    #with open(abnormal_pixel_num_file, 'w') as f:
    #    for content in total_content:
    #        f.write(f'{content}\n')


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
    parser.add_argument('--use_simplex_noise', action='store_true')

    # step 4. model
    parser.add_argument('--timestep', type=int, default=1000)
    parser.add_argument('--beta_schedule', type=str, default='linear')
    parser.add_argument('--loss_weight', type=str, default="none")
    parser.add_argument('--loss_type', type=str, default='l2')
    parser.add_argument('--sample_distance', type=int, default=150)
    parser.add_argument('--thredhold', type=float, default=0.00000216)

    args = parser.parse_args()
    main(args)