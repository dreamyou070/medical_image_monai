import random
import time
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
import dataset
import evaluation
from GaussianDiffusion import GaussianDiffusionModel, get_beta_schedule
from helpers import *
from UNet import UNetModel
import sys
from matplotlib import font_manager
import argparse

def anomalous_validation_1(args):

    print(f'\n step 1. device')
    device = args.device

    print(f'\n step 2. model')
    print(f' (2.1) unet model')
    w, h = int(args.img_size.split(',')[0].strip()), int(args.img_size.split(',')[1].strip())
    in_channels = args.in_channels
    unet = UNetModel(img_size=int(w),
                      base_channels=args.base_channels,
                      dropout=args.dropout,
                      n_heads=args.num_heads,
                      n_head_channels=args.num_head_channels,
                      in_channels=in_channels)
    unet_state_dict_dir = args.unet_state_dict_dir
    unet.load_state_dict(unet_state_dict_dir)
    unet.to(device)
    unet.eval()

    print(f' (2.2) scheduler')
    betas = get_beta_schedule(args.timestep, args.beta_schedule)

    print(f' (2.3) diffusion model')
    diffusion = GaussianDiffusionModel([w, h],  # [128, 128]
                                       betas,  # 1
                                       img_channels=in_channels,
                                       loss_type=args.loss_type,  # l2
                                       loss_weight=args.loss_weight,  # none
                                       noise=args.noise_fn, )  # 1

    print(f'\n step 3. dataset')
    data_dir = args.dataset_path
    mask_dir = os.path.join(os.path.split(data_dir)[0], 'mask')
    ano_dataset = dataset.DentalDataset(img_dir=args.dataset_path,
                                        mask_dir = mask_dir,
                                        transform=None,
                                        img_size=(w,h))
    loader = torch.utils.data.DataLoader(ano_dataset,
                                         batch_size=args.batch_size,
                                         shuffle=False,
                                         num_workers=0,
                                         drop_last=True)



if __name__ == "__main__" :
    parser = argparse.ArgumentParser()
    # step 1
    parser.add_argument('--device', type=str, default='cuda:1')

    # step 2
    parser.add_argument('--img_size', type=str,default='128,128,')
    parser.add_argument('--in_channels', type=int, default=1)
    parser.add_argument('--base_channels', type=int, default=256)
    parser.add_argument('--channel_mults', type=str, default='1,2,3,4')
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--num_heads', type=int, default=2)
    parser.add_argument('--num_head_channels', type=int, default=-1)
    # (2) scaheduler
    parser.add_argument('--timestep', type=int, default=1000)
    parser.add_argument('--beta_schedule', type=str, default='linear')
    # (3) diffusion
    parser.add_argument('--loss_weight', type=str, default="none")
    parser.add_argument('--loss_type', type=str, default='l2')
    parser.add_argument('--noise_fn', type=str, default='simplex')

    # step 3 dataset
    parser.add_argument('--unet_state_dict_dir', type=str,
                        default='/data7/sooyeon/medical_image/experiment_data/dental/panoramic_data_res_128/valid/original')
    parser.add_argument('--dataset_path', type=str,
                        default='/data7/sooyeon/medical_image/experiment_data/dental/panoramic_data_res_128/valid/original')

    args = parser.parse_args()

    anomalous_validation_1(args)