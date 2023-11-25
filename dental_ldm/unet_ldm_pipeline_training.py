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
from diffusers import AutoencoderKL, UNet2DModel, DDPMScheduler,StableDiffusionPipeline
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

# training_outputs(args, test_data, scheduler, 'training_data', device, ema, autoencoderkl, scale_factor, epoch + 1)
def training_outputs(args, test_data, scheduler, is_train_data, device, model, vae,
                     vae_scale_factor, epoch):

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
        latents = vae.encode(x).latent_dist.sample()
        latents = (latents * vae_scale_factor).to(device)
    # 2) select random int
    t = torch.randint(args.sample_distance - 1, args.sample_distance, (latents.shape[0],), device=x.device)
    # 3) noise
    noise = torch.rand_like(latents).float().to(x.device)
    # 4) noise image generating
    with torch.no_grad() :
        noisy_latent = scheduler.add_noise(original_samples=latents,
                                           noise=noise,
                                           timesteps=t)
        latent = noisy_latent.clone().detach()
        # 5) denoising
        for t in range(int(args.sample_distance) , -1, -1):
            with torch.no_grad() :
                # 5-1) model prediction
                model_output = model(latent, torch.Tensor((t,)).to(device), None)
            # 5-2) update latent
            latent, _ = scheduler.step(model_output, t, latent)
        recon_image = vae.decode(latents / vae_scale_factor,return_dict=False,generator=None)[0]
        print(f'recon_image : {recon_image}')
    """
    for img_index in range(x.shape[0]):
        normal_info_ = normal_info[img_index]
        if normal_info_ == 1:
            is_normal = 'normal'
        else :
            is_normal = 'abnormal'

        real = x[img_index].squeeze()
        real = torch_transforms.ToPILImage()(real.unsqueeze(0))

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
    """

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
    train_ds = SYDataset(data_folder=args.train_data_folder,
                         transform=train_transforms,
                         base_mask_dir=args.train_mask_dir,
                         image_size=(w,h))
    training_dataset_loader = SYDataLoader(train_ds,
                                           batch_size=args.batch_size,
                                           shuffle=True,
                                           num_workers=4,
                                           persistent_workers=True)
    check_data = first(training_dataset_loader)
    # ## Prepare validation set data loader
    val_transforms = transforms.Compose([transforms.Resize((w,h), transforms.InterpolationMode.BILINEAR),
                                         transforms.ToTensor()])
    val_ds = SYDataset(data_folder=args.val_data_folder,
                         transform=val_transforms,
                         base_mask_dir=args.val_mask_dir,image_size=(w,h))
    test_dataset_loader = SYDataLoader(val_ds,
                                       batch_size=args.batch_size,
                                       shuffle=False,
                                       num_workers=4,
                                       persistent_workers=True)

    print(f'\n step 3. latent_model')
    vae = AutoencoderKL(in_channels=1,
                        out_channels=1,
                        latent_channels=4,
                        norm_num_groups=32,
                        sample_size=32,
                        scaling_factor=0.18215)
    vae.load_state_dict(torch.load(args.pretrained_vae_dir))
    vae = vae.to(device)
    vae_scale_factor = 0.18215

    print(f'\n step 4. unet model')
    unet = UNet2DModel(sample_size = 32,in_channels = 4,out_channels =4,)
    ema = copy.deepcopy(unet)
    ema.to(device)
    unet = unet.to(device)
    #  unet.config.sample_size = 32

    print(f'\n step 5. scheduler')
    scheduler = DDPMScheduler(num_train_timesteps = 1000,
                              beta_start = 0.0001,
                              beta_end = 0.02,
                              beta_schedule = "linear",
                              variance_type = "fixed_small",steps_offset = 1)

    print(f' \n step 6. infererence scheduler pipeline')
    pipeline = StableDiffusionPipeline(vae = vae,
                                       text_encoder = None,
                                       tokenizer = None,
                                       unet =unet,
                                       scheduler = scheduler,
                                       safety_checker = None,
                                       feature_extractor=None)
    scale_factor_ = pipeline.vae_scale_factor #= 2 ** (len(self.vae.config.block_out_channels) - 1) # 1

    print(f' \n step 7. optimizer')
    optimizer = torch.optim.Adam(unet.parameters(), lr=1e-4)

    print(f' \n step 8. training')
    for epoch in range(args.n_epochs):
        unet.train()
        vae.eval()
        epoch_loss = 0
        progress_bar = tqdm(enumerate(training_dataset_loader), total=len(training_dataset_loader), ncols=70)
        progress_bar.set_description(f"Epoch {epoch}")
        for step, batch in progress_bar:
            x_0 = batch["image_info"].to(device)  # [Batch, 1, 128, 128]
            mask_info = batch["mask"].unsqueeze(dim=1)
            #small_mask_info = batch['small_mask'].unsqueeze(dim=1)
            normal_info = batch['normal']  # if 1 = normal, 0 = abnormal
            if args.only_normal_training:
                x_0 = x_0[normal_info == 1]
                mask_info = mask_info[normal_info == 1]
            if x_0.shape[0] > 0:
                latents = vae.encode(x_0).latent_dist.sample()
                latents = (latents * vae_scale_factor).to(device)
                # 2) t
                timesteps = torch.randint(0, args.sample_distance, (latents.shape[0],),device=device).long()
                # 3) noise
                noise = torch.randn_like(latents).to(device)
                # 4) x_t
                noisy_samples = scheduler.add_noise(original_samples = latents,noise = noise,timesteps = timesteps,)
                # 5) unet inference
                noise_pred = pipeline.unet(noisy_samples,timesteps).sample
                target = noise
                if args.masked_loss :
                    noise_pred = noise_pred * mask_info
                    target = target * mask_info
                    print(f'target : {target.shape} | mask_info : {mask_info.shape}')
                loss = torch.nn.functional.mse_loss(noise_pred.float(), target.float(), reduction="none").mean([1, 2, 3])

                loss = loss.mean()

                wandb.log({"training loss": loss.item()})
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(unet.parameters(), 1)
                optimizer.step()
                # ----------------------------------------------------------------------------------------- #
                # EMA model updating
                update_ema_params(ema, unet)
        # inference ?
        if epoch % args.inference_freq == 0 and epoch == 0:
            for i, test_data in enumerate(test_dataset_loader):
                if i == 0:
                    ema.eval()
                    unet.eval()
                    training_outputs(args, test_data, scheduler, 'test_data', device, ema, vae,
                                     vae_scale_factor, epoch + 1)
                    training_outputs(args, batch, scheduler, 'training_data', device, ema, vae,
                                     vae_scale_factor, epoch + 1)



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
    parser.add_argument('--pretrained_vae_dir', type=str)
    parser.add_argument('--latent_channels', type=int, default=3)

    # step 4. model
    parser.add_argument('--timestep', type=int, default=1000)
    parser.add_argument('--schedule_type', type=str, default="linear_beta")

    # step 5. optimizer
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.0)

    # step 6. training
    parser.add_argument('--sample_distance', type=int, default=150)
    parser.add_argument('--use_simplex_noise', action='store_true')
    # --------------------------------------------------------------------------------------------------------------
    parser.add_argument('--masked_loss_latent', action='store_true')

    # --------------------------------------------------------------------------------------------------------------
    parser.add_argument('--info_nce_loss', action='store_true')
    # --------------------------------------------------------------------------------------------------------------
    parser.add_argument('--pos_info_nce_loss', action='store_true')
    parser.add_argument('--reg_loss_scale', type=float, default=1.0)
    # --------------------------------------------------------------------------------------------------------------
    parser.add_argument('--anormal_scoring', action='store_true')
    parser.add_argument('--min_max_training', action='store_true')


    parser.add_argument('--inference_num', type=int, default=4)

    # step 7. save
    parser.add_argument('--model_save_freq', type=int, default=1000)


    # step 8. training
    parser.add_argument('--n_epochs', type=int, default=3000)
    parser.add_argument('--only_normal_training', action='store_true')
    parser.add_argument('--masked_loss', action='store_true')
    # inference
    parser.add_argument('--inference_freq', type=int, default=50)

    args = parser.parse_args()
    main(args)
