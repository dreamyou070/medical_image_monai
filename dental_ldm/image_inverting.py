import argparse, wandb
import copy
import numpy as np
from PIL import Image
from random import seed
from torch import optim
from helpers import *
from tqdm import tqdm
from torchvision import transforms
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from nets.utils import update_ema_params
from data_module import SYDataLoader, SYDataset
from monai.utils import first
import torch.multiprocessing
import torchvision.transforms as torch_transforms
import PIL
from setproctitle import *
from diffuser_module import AutoencoderKL, UNet2DModel, DDPMScheduler,StableDiffusionPipeline
from inverting import Inversor
import json

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
                    "args":                 args},save_dir)
    else:
        save_dir = os.path.join(model_save_base_dir, f'unet_epoch_{epoch}.pt')
        torch.save({'n_epoch':              epoch,
                    'model_state_dict':     unet.state_dict(),
                    'optimizer_state_dict': optimiser.state_dict(),
                    "args":                 args,
                    "ema":                  ema.state_dict(),
                    'loss':                 loss,},save_dir)

# training_outputs(args, batch,     scheduler, 'training_data', device, ema, vae, scale_factor, epoch + 1)
def training_outputs(args, test_data, scheduler, is_train_data, device, model, vae, scale_factor, epoch):

    if is_train_data == 'training_data':
        train_data = 'training_data'
    else :
        train_data = 'test_data'

    video_save_dir = os.path.join(args.experiment_dir, 'diffusion-videos')
    image_save_dir = os.path.join(args.experiment_dir, 'diffusion-training-images')
    os.makedirs(video_save_dir, exist_ok=True)
    os.makedirs(image_save_dir, exist_ok=True)

    # 1) make random noise
    x = test_data["image_info"].to(device)  # batch, channel, w, h
    normal_info = test_data['normal']  # if 1 = normal, 0 = abnormal
    mask_info = test_data['mask']  # if 1 = normal, 0 = abnormal
    with torch.no_grad():
        latent = vae.encode(x).latent_dist.mode()  # [Batch, 4, 32, 32]
        latent = latent * scale_factor
    # 2) select random int
    b_size = latent.shape[0]
    t = torch.randint(args.sample_distance - 1, args.sample_distance, (b_size,), device=device).long()
    # 3) noise
    noise = torch.rand_like(latent).float().to(x.device)
    # 4) noise image generating
    with torch.no_grad() :
        noisy_latent = scheduler.add_noise(original_samples=latent, noise=noise, timesteps=t)
        latents = noisy_latent.clone().detach()
        # 5) denoising
        for t in range(int(args.sample_distance)-1 , -1, -1):
            with torch.no_grad() :
                timestep = torch.Tensor([t]).repeat(b_size).long()
                model_output = model(latents, timestep.to(device), None).sample
                latents = scheduler.step(model_output, t, sample=latents).prev_sample
        recon_image = vae.decode(latents / scale_factor,return_dict=True,generator=None).sample

    for img_index in range(b_size):
        normal_info_ = normal_info[img_index]
        if normal_info_ == 1:
            is_normal = 'normal'
        else :
            is_normal = 'abnormal'

        real = x[img_index].squeeze()
        real = torch_transforms.ToPILImage()(real.unsqueeze(0))
        # wrong in here ...
        recon = recon_image[img_index].squeeze()
        recon = torch_transforms.ToPILImage()(recon.unsqueeze(0))

        mask_np = mask_info[img_index].squeeze().to('cpu').detach().numpy().copy().astype(np.uint8)
        mask_np = mask_np * 255
        mask = Image.fromarray(mask_np).convert('L')  # [128, 128, 3]

        new_image = PIL.Image.new('L', (3 * real.size[0], real.size[1]),250)
        new_image.paste(real,  (0, 0))
        new_image.paste(recon, (real.size[0], 0))
        new_image.paste(mask,  (real.size[0]+recon.size[0], 0))
        new_image.save(os.path.join(image_save_dir,
                                   f'real_recon_answer_{train_data}_epoch_{epoch}_{img_index}.png'))
        loading_image = wandb.Image(new_image,
                                    caption=f"(real_recon_answer) epoch {epoch + 1} | {is_normal}")
        if train_data == 'training_data' :
            wandb.log({"training data inference" : loading_image})
        else :
            wandb.log({"test data inference" : loading_image})


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
    w,h = int(args.img_size.split(',')[0].strip()),int(args.img_size.split(',')[1].strip())
    train_transforms = transforms.Compose([transforms.Resize((w,h), transforms.InterpolationMode.BILINEAR),
                                           transforms.ToTensor()])
    train_ds = SYDataset(data_folder=args.train_data_folder, transform=train_transforms,
                         base_mask_dir=args.train_mask_dir, image_size=(w,h))
    training_dataset_loader = SYDataLoader(train_ds,batch_size=args.batch_size,shuffle=True,
                                           num_workers=4,persistent_workers=True)
    # ## Prepare validation set data loader
    val_transforms = transforms.Compose([transforms.Resize((w,h), transforms.InterpolationMode.BILINEAR),
                                         transforms.ToTensor()])
    val_ds = SYDataset(data_folder=args.val_data_folder,transform=val_transforms,
                       base_mask_dir=args.val_mask_dir,image_size=(w,h))
    test_dataset_loader = SYDataLoader(val_ds,batch_size=args.batch_size,shuffle=False,
                                       num_workers=4,persistent_workers=True)

    print(f'\n step 3. latent_model')
    vae_config_dir = args.vae_config_dir
    with open(vae_config_dir, "r") as f :
        vae_config = json.load(f)
    vae = AutoencoderKL.from_config(config = vae_config)
    vae.load_state_dict(torch.load(args.pretrained_vae_dir), strict=True)
    vae = vae.to(device)
    vae.eval()

    print(f'\n step 4. unet model')
    with open(args.unet_config_dir, "r") as f :
        unet_config_dir = json.load(f)
    unet = UNet2DModel.from_config(config=unet_config_dir)
    state_dict = torch.load(args.pretrained_unet_dir)["ema"]
    unet.load_state_dict(state_dict, strict=True)
    unet = unet.to(device)
    unet.eval()

    """
    print(f'\n step 5. scheduler')
    scheduler = DDPMScheduler(num_train_timesteps = 1000,
                              beta_start = 0.0001,
                              beta_end = 0.02,
                              beta_schedule = "linear",
                              variance_type = "fixed_small",
                              steps_offset = 1,
                              clip_sample = False)

    print(f' \n step 6. infererence scheduler pipeline')
    pipeline = StableDiffusionPipeline(vae = vae,
                                       text_encoder = None,
                                       tokenizer = None,
                                       unet =unet,
                                       scheduler = scheduler,
                                       safety_checker = None,
                                       feature_extractor=None)

    print(f' \n step 7. invertor')
    invertor = Inversor(unet = unet,scheduler = scheduler, vae = vae,scale_factor = scale_factor)

    print(f' \n step 8. noising image')
    noising_save_base_dir = os.path.join(args.experiment_dir, 'noising_image')
    os.makedirs(noising_save_base_dir, exist_ok=True)
    check_data = first(training_dataset_loader)
    img = check_data['image_info'].to(device)
    all_latents = invertor.ddim_loop(img = img,
                                     inversion_steps = args.sample_distance)

    for i in range(len(all_latents)) :
        latent = all_latents[i]
        img = invertor.latent2img(latent, return_type = 'torch')
        img = img[0].squeeze()
        img = torch_transforms.ToPILImage()(img.unsqueeze(0))
        img_save_dir = os.path.join(noising_save_base_dir, f'noising_{i}.png')
        img.save(img_save_dir)

    last_latent = all_latents[-1]
    all_latents_as_generating = invertor.gen_loop(latent = last_latent,
                                                   timestep = args.sample_distance)
    for i in range(len(all_latents_as_generating)) :
        naming_time = args.sample_distance - i
        latent = all_latents_as_generating[i]
        img = invertor.latent2img(latent, return_type = 'torch')
        img = img[0].squeeze()
        img = torch_transforms.ToPILImage()(img.unsqueeze(0))
        img_save_dir = os.path.join(noising_save_base_dir, f'generating_process_{naming_time}.png')
        img.save(img_save_dir)

    latent_distance = [torch.nn.functional.mse_loss(org_img.float(), reversed_img.float(), reduction="none")
                       for org_img, reversed_img in zip(all_latents, all_latents_as_generating[::-1])]

    latent_distance = torch.stack(latent_distance, dim=0).mean(dim=0).mean(dim=0).mean(dim=0)
    import torch.nn.functional as F

    def expand_image(im: torch.Tensor, h=512, w=512,
                     absolute: bool = False, threshold: float = None) -> torch.Tensor:
        im = im.unsqueeze(0).unsqueeze(0)
        im = F.interpolate(im.float().detach(), size=(h, w), mode='bicubic')
        if not absolute:
            im = (im - im.min()) / (im.max() - im.min() + 1e-8)
        if threshold:
            im = (im > threshold).float()
        # im = im.cpu().detach()
        return im.squeeze()
    heatmap = expand_image(latent_distance, h=256, w=256, absolute=False)
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
                        default = f'/data7/sooyeon/medical_image/anoddpm_result/20231119_dental_test')

    # step 2. dataset and dataloatder
    parser.add_argument('--train_data_folder', type=str)
    parser.add_argument('--train_mask_dir', type=str)
    parser.add_argument('--val_data_folder', type=str)
    parser.add_argument('--val_mask_dir', type=str)
    parser.add_argument('--img_size', type=str)
    parser.add_argument('--batch_size', type=int)

    # step 3. model
    parser.add_argument('--vae_config_dir', type=str,
                        default = '/data7/sooyeon/medical_image/pretrained/vae/config.json')
    parser.add_argument('--pretrained_vae_dir', type=str)

    # step 4. model
    parser.add_argument('--unet_config_dir', type=str,
                        default='/data7/sooyeon/medical_image/pretrained/unet/config.json')
    parser.add_argument('--pretrained_unet_dir', type=str)
    parser.add_argument('--latent_channels', type=int, default=3)


    parser.add_argument('--timestep', type=int, default=1000)
    parser.add_argument('--schedule_type', type=str, default="linear_beta")

    # step 5. optimizer
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.0)

    # step 6. training
    parser.add_argument('--sample_distance', type=int, default=150)
    parser.add_argument('--use_simplex_noise', action='store_true')

    args = parser.parse_args()
    main(args)
