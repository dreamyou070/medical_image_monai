import random
import time, sys, os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from data_module import SYDataLoader, SYDataset_masking
import dataset
from GaussianDiffusion import GaussianDiffusionModel, get_beta_schedule
from helpers import *
from UNet import UNetModel
from monai import transforms
import argparse
import torchvision.transforms as torch_transforms
import PIL

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
    loader = SYDataLoader(val_ds,batch_size=args.batch_size,shuffle=True, num_workers=4, persistent_workers=True)

    print(f'\n step 4. ... ')
    dice_data = []
    start_time = time.time()
    for data in loader :
        batch_size = data["image_info"]['image'].shape[0]
        for i in range(batch_size) :
            x_0 = data["image_info"]['image'][i].to(device)  # batch, channel, w, h
            x_0 = x_0.unsqueeze(0)
            normal_info = data['normal'][i]  # if 1 = normal, 0 = abnormal
            mask_info = data['mask'][i].unsqueeze(0).unsqueeze(0)  # if 1 = normal, 0 = abnormal
            if normal_info != 1 :
                # -----------------------------------------------------------------------------------------------------------
                # 1) make noisy image
                t = torch.randint(int(args.sample_distance * 0.3), int(args.sample_distance * 0.8), (x_0.shape[0],), device=x_0.device)  # random timestep 45~120
                noise = torch.randn_like(x_0).to(device)
                x_t = diffusion.sample_q(x_0, t, noise)
                # -----------------------------------------------------------------------------------------------------------
                # 2) unet iteratively recon image
                for time_step in reversed(range(t)):
                    with torch.no_grad():
                        temp = diffusion.sample_p(unet, x_t, t)
                        x_t = temp['sample']
                pred_x_0 = x_t
                real = torch_transforms.ToPILImage()(x_0.permute(0, 1, 3, 2)[0])
                pred_x_0 = torch_transforms.ToPILImage()(pred_x_0.permute(0, 1, 3, 2)[0])
                mask = torch_transforms.ToPILImage()(mask_info.permute(0, 1, 3, 2)[0].float())
                new_image = PIL.Image.new('L', (3 * real.size[0], real.size[1]), 250)
                new_image.paste(real, (0, 0))
                new_image.paste(pred_x_0, (real.size[0], 0))
                new_image.paste(mask, (real.size[0] + pred_x_0.size[0], 0))
                new_image.save('test.png')




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