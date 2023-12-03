import argparse, wandb
from random import seed
from GaussianDiffusion import GaussianDiffusionModel, get_beta_schedule
from helpers import *
from torchvision import transforms
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from data_module import SYDataLoader, SYDataset
from UNet import UNetModel, update_ema_params
import torch.multiprocessing
from setproctitle import *
import torchvision.transforms as torch_transforms
import PIL
from diffuser_module import AutoencoderKL, UNet2DModel, DDPMScheduler,StableDiffusionPipeline

torch.multiprocessing.set_sharing_strategy('file_system')
torch.cuda.empty_cache()

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
    image_save_dir = os.path.join(experiment_dir, 'scheduling_images')
    print(f'  - image_save_dir : {image_save_dir}')
    os.makedirs(image_save_dir, exist_ok=True)

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
    model.load_state_dict(torch.load(args.unet_pretrained_dir, map_location='cpu')['model_state_dict'])
    model.to(device)
    model.eval()

    print(f'\n step 4. scheduler')
    betas = get_beta_schedule(args.timestep,
                              args.beta_schedule)
    scheduler = GaussianDiffusionModel([w, h],  # [128, 128]
                                       betas,  # 1
                                       img_channels=in_channels,
                                       loss_type=args.loss_type,  # l2
                                       loss_weight=args.loss_weight,  # none
                                       noise='simplex' )  # 1
    print(f'\n step 5. inference')
    train_data = 'train_data'
    for i, data in enumerate(training_dataset_loader):
        if i == 0:
            x_0 = data["image_info"].to(device)  # batch, channel, w, h
            normal_info = data['normal']  # if 1 = normal, 0 = abnormal
            mask_info = data['mask'].unsqueeze(dim=1)  # if 1 = normal, 0 = abnormal
            t = torch.Tensor([args.sample_distance]).repeat(x_0.shape[0], ).long().to(x_0.device)
            noise = torch.rand_like(x_0).float().to(device)
            # 2) select random int
            x_t = scheduler.sample_q(x_0, t, noise)
            #x_t = scheduler.add_noise(x_0, noise, t)
            one_step_recon = scheduler.sample_p(model, x_t, t, denoise_fn="gauss")['pred_x_0']
            torch_transforms.ToPILImage()(one_step_recon.squeeze()).save(os.path.join(image_save_dir, f'one_step_inference.png'))
            with torch.no_grad():
                noise_pred = model(x_t, t)
                #pred_images = scheduler.step(noise_pred,args.sample_distance,x_t, return_dict=True)['pred_original_sample']
                for i in range(args.sample_distance-1, -1, -1):
                    # sample = sample.unsqueeze(0)
                    sample = torch_transforms.ToPILImage()(x_t.squeeze())
                    sample.save(os.path.join(image_save_dir, f'scheduling_{i}.png'))
                    if i > 0 :
                        # Check Posterior
                        #x_t = scheduler.sample_p(model, x_t,
                        #                         torch.Tensor([t]).repeat(x_0.shape[0], ).long().to(x_0.device),)['sample']
                        x_t = scheduler.q_sample(x_0,
                                                 torch.Tensor([i]).repeat(x_0.shape[0], ).long().to(x_0.device),
                                                 noise,
                                                 denoise_fn='gauss')['sample']
                        #sample = x_t.squeeze()
                        # sample = sample.unsqueeze(0)
                        #sample = torch_transforms.ToPILImage()(sample)
                        #sample.save(os.path.join(image_save_dir,
                        #                        f'inference_time_{i}_after.png'))

                        """
                        is_normal = 'normal'
                        #else:
                        #    is_normal = 'abnormal'
                        # 1) one step inference
                        real = pred_images[img_index, ...].squeeze()
                        #real = real.unsqueeze(0)
                        real = torch_transforms.ToPILImage()(real)
        
                        # 2) one step inference
                        sample = final_pred[img_index, ...].squeeze()
                        #sample = sample.unsqueeze(0)
                        sample = torch_transforms.ToPILImage()(sample)
        
                        new_image = PIL.Image.new('RGB', (2 * real.size[0], real.size[1]), 250)
                        new_image.paste(real, (0, 0))
                        new_image.paste(sample, (real.size[0], 0))
                        new_image.save(os.path.join(image_save_dir,
                                                    f'once_stepping_{train_data}_{is_normal}_{img_index}.png'))
                        loading_image = wandb.Image(new_image,
                                                    caption=f"once_stepping_{train_data}_{is_normal}_{img_index}")
                        wandb.log({"inference": loading_image})
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
                        default=f'/data7/sooyeon/medical_image/anoddpm_result/20231119_dental_test')

    # step 2. dataset and dataloatder
    parser.add_argument('--train_data_folder', type=str)
    parser.add_argument('--train_mask_dir', type=str)
    parser.add_argument('--val_data_folder', type=str)
    parser.add_argument('--val_mask_dir', type=str)
    parser.add_argument('--img_size', type=str)
    parser.add_argument('--batch_size', type=int)

    # step 3. model
    parser.add_argument('--unet_pretrained_dir', type=str,
                        default='/data7/sooyeon/medical_image/anoddpm_result_ddpm/2_compare_allsample_nonmasked_loss/diffusion-models/unet_epoch_200.pt')
    parser.add_argument('--in_channels', type=int, default = 1)
    parser.add_argument('--base_channels', type=int, default = 256)
    parser.add_argument('--channel_mults', type=str, default = '1,2,3,4')
    parser.add_argument('--dropout', type=float, default = 0.0)
    parser.add_argument('--num_heads', type=int, default = 2)
    parser.add_argument('--num_head_channels', type=int, default = -1)
    parser.add_argument('--timestep', type=int, default=1000)
    parser.add_argument('--beta_schedule', type=str, default='linear')
    parser.add_argument('--loss_weight', type=str, default = "none")
    parser.add_argument('--loss_type', type=str, default='l2')

    # step 4. inference
    parser.add_argument('--sample_distance', type=int, default=150)
    parser.add_argument('--use_simplex_noise', action='store_true')
    args = parser.parse_args()
    main(args)

