import argparse, wandb
import copy
import numpy as np
from PIL import Image
from random import seed

from torch import optim
from helpers import *
from tqdm import tqdm
import torch.nn.functional as F
from torchvision import transforms
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from diffuser_module.models.vae import DiagonalGaussianDistribution
import torch.multiprocessing
import torchvision.transforms as torch_transforms
import PIL
from data_module import SYDataLoader, SYDataset
from monai.utils import first
from setproctitle import *
from generative.networks.nets import AutoencoderKL, PatchDiscriminator
from loss_module import PerceptualLoss, PatchAdversarialLoss
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
from diffuser_module import AutoencoderKL

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

    print(f'\n step 3. latent_model')
    vae_config_dir = args.vae_config_dir
    with open(vae_config_dir, "r") as f:
        vae_config = json.load(f)
    vae = AutoencoderKL.from_config(config=vae_config)
    vae.load_state_dict(torch.load(args.pretrained_vae_dir), strict=True)
    vae = vae.to(device)
    vae.eval()

    print(f'\n step 4. model inference')
    training_inference_dir = os.path.join(experiment_dir, 'training_inference')
    os.makedirs(training_inference_dir, exist_ok=True)
    global_num = 0
    for step, batch in enumerate(training_dataset_loader) :
        images = batch["image_info"].to(device)
        normal_info = batch['normal']  # if 1 = normal, 0 = abnormal
        with torch.no_grad():
            reconstruction = vae(images).sample
        batch_size = images.shape[0]
        for i in range(batch_size) :
            normal = normal_info[i]
            if normal == 1 :
                normal = 'normal'
            else :
                normal = 'abnormal'
            org_img = images[i].squeeze()
            org_img = torch_transforms.ToPILImage()(org_img.unsqueeze(0))
            recon = reconstruction[i].squeeze()
            recon = torch_transforms.ToPILImage()(recon.unsqueeze(0))
            new = Image.new('RGB', (org_img.width + recon.width, org_img.height))
            new.paste(org_img, (0, 0))
            new.paste(recon, (org_img.width, 0))
            new.save(os.path.join(training_inference_dir, f'infer_check_training_{global_num}_{normal}.png'))
            global_num += 1

    test_inference_dir = os.path.join(experiment_dir, 'test_inference')
    os.makedirs(test_inference_dir, exist_ok=True)
    global_num = 0
    for step, batch in enumerate(test_dataset_loader):
        images = batch["image_info"].to(device)
        normal_info = batch['normal']  # if 1 = normal, 0 = abnormal
        with torch.no_grad():
            reconstruction = vae(images).sample
        batch_size = images.shape[0]
        for i in range(batch_size):
            normal = normal_info[i]
            if normal == 1:
                normal = 'normal'
            else:
                normal = 'abnormal'
            org_img = images[i].squeeze()
            org_img = torch_transforms.ToPILImage()(org_img.unsqueeze(0))
            recon = reconstruction[i].squeeze()
            recon = torch_transforms.ToPILImage()(recon.unsqueeze(0))
            new = Image.new('RGB', (org_img.width + recon.width, org_img.height))
            new.paste(org_img, (0, 0))
            new.paste(recon, (org_img.width, 0))
            new.save(os.path.join(training_inference_dir, f'infer_check_test_{global_num}_{normal}.png'))
            global_num += 1


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
    parser.add_argument('--vae_config_dir', type=str,)
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
    # --------------------------------------------------------------------------------------------------------------
    parser.add_argument('--masked_loss_latent', action='store_true')
    parser.add_argument('--masked_loss', action='store_true')
    # --------------------------------------------------------------------------------------------------------------
    parser.add_argument('--info_nce_loss', action='store_true')
    # --------------------------------------------------------------------------------------------------------------
    parser.add_argument('--pos_info_nce_loss', action='store_true')
    parser.add_argument('--reg_loss_scale', type=float, default=1.0)
    # --------------------------------------------------------------------------------------------------------------
    parser.add_argument('--anormal_scoring', action='store_true')
    parser.add_argument('--min_max_training', action='store_true')

    parser.add_argument('--inference_freq', type=int, default=50)
    parser.add_argument('--inference_num', type=int, default=4)

    # step 7. save
    parser.add_argument('--model_save_freq', type=int, default=1000)
    parser.add_argument('--model_save_base_epoch', type=int, default=50)
    args = parser.parse_args()
    main(args)
