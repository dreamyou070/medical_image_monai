import random
import time
from data_module import SYDataLoader, SYDataset_masking
import dataset
from GaussianDiffusion import GaussianDiffusionModel, get_beta_schedule
from helpers import *
from UNet import UNetModel
from monai import transforms
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
    unet_state_dict = torch.load(args.unet_state_dict_dir,
                                     map_location='cpu')
    unet_state_weight = unet_state_dict['model_state_dict']
    unet.load_state_dict(unet_state_weight)
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
    datas = os.listdir(args.dataset_path)
    datalist = [{"image": os.path.join(args.dataset_path, data)} for data in datas]
    data_transforms = transforms.Compose([transforms.LoadImaged(keys=["image"]),
                                     transforms.EnsureChannelFirstd(keys=["image"]),
                                     transforms.ScaleIntensityRanged(keys=["image"],
                                                                     a_min=0.0, a_max=255.0, b_min=0.0, b_max=1.0,
                                                                     clip=True),
                                     transforms.RandAffined(keys=["image"],
                                                            spatial_size=[w, h], ), ])
    mask_dir = os.path.join(os.path.split(args.dataset_path)[0], 'mask')
    val_ds = SYDataset_masking(data=datalist,
                               transform=data_transforms,
                               base_mask_dir=mask_dir,
                               image_size=args.img_size)
    test_dataset_loader = SYDataLoader(val_ds,
                                       batch_size=args.batch_size,
                                       shuffle=True, num_workers=4, persistent_workers=True)



    data_dir = args.dataset_path
    mask_dir = os.path.join(os.path.split(data_dir)[0], 'mask')
    val_transforms = transforms.Compose([transforms.LoadImaged(keys=["image"]),
                                         transforms.EnsureChannelFirstd(keys=["image"]),
                                         transforms.ScaleIntensityRanged(keys=["image"],
                                                                         a_min=0.0, a_max=255.0, b_min=0.0, b_max=1.0,
                                                                         clip=True),
                                         transforms.RandAffined(keys=["image"],
                                                                spatial_size=[w, h], ), ])
    ano_dataset = dataset.DentalDataset(img_dir = args.dataset_path,
                                        mask_dir = mask_dir,
                                        jaw_mask_dir=None,
                                        transform=val_transforms,
                                        img_size=(w,h))
    loader = torch.utils.data.DataLoader(ano_dataset,
                                         batch_size=args.batch_size,
                                         shuffle=False,
                                         num_workers=0,
                                         drop_last=True)

    print(f'\n step 4. ... ')
    dice_data = []
    start_time = time.time()
    for data in loader :

        x_0 = data["image_info"]['image'][0].to(device)  # batch, channel, w, h
        x_0 = x_0.unsqueeze(0).unsqueeze(0)
        normal_info = data['normal'][0]  # if 1 = normal, 0 = abnormal
        mask_info = data['mask'][0].unsqueeze(0).unsqueeze(0)  # if 1 = normal, 0 = abnormal
        print(f'x_0 (1,1,128,128) : {x_0.shape}')
        print(f'mask_info (1,1,128,128) : {mask_info.shape}')
        # -----------------------------------------------------------------------------------------------------------
        # 1) make noisy image
        t = random.randint(int(args.sample_distance * 0.3), int(args.sample_distance * 0.8))  # random timestep 45~120
        noise = torch.randn_like(x_0)
        x_t = diffusion.sample_q(x_0, t, noise)

        # -----------------------------------------------------------------------------------------------------------
        # 2) unet iteratively recon image
        for time_step in reversed(range(t)):
            print(f'time_step : {time_step}')
        break



        #print(f'img  (1,1,128,128) : {img.shape}')
        #print(f'abnormal_mask (1,1,128,128) : {abnormal_mask.shape}')
        # -----------------------------------------------------------------------------------------------------------
        """
        
        output = diffusion.forward_backward(unet,
                                            img[slice_number, ...].reshape(1, 1, *args["img_size"]),
                                            see_whole_sequence="whole",
                                            t_distance=timestep,
                                            denoise_fn=args["noise_fn"])
        """




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
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--sample_distance', type=int, default=150)

    args = parser.parse_args()

    anomalous_validation_1(args)