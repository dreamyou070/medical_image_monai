import argparse, torch, os
from model_module.nets import AutoencoderKL, DiffusionModelUNet, PatchDiscriminator
from data_module import get_transform, SYDataset, SYDataLoader
from model_module.schedulers import DDPMScheduler
from model_module.inferers import LatentDiffusionInferer
from torch.cuda.amp import GradScaler, autocast
from utils.check import first
from tqdm import tqdm
import torch.nn.functional as F

def main(args) :

    print(f' \n step 1. device')
    device = torch.device(args.device)

    print(f' \n step 2. load dataset')
    total_datas = os.listdir(args.data_folder)
    print(f' (2.0) data_module transform')
    train_transforms, val_transforms = get_transform(args.image_size)
    print(f' (2.1.1) train dataset')
    train_num = int(0.7 * len(total_datas))
    train_datas, val_datas = total_datas[:train_num], total_datas[train_num:]
    train_datalist = [{"image": os.path.join(args.data_folder, train_data)} for train_data in train_datas]
    train_ds = SYDataset(data=train_datalist, transform=train_transforms)
    print(f' (2.1.2) train dataloader')
    train_loader = SYDataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4,
                                persistent_workers=True)
    check_data = first(train_loader)

    print(f' (2.2.1) val dataset')
    val_datalist = [{"image": os.path.join(args.data_folder, val_data)} for val_data in val_datas]
    val_ds = SYDataset(data=val_datalist, transform=val_transforms)
    print(f' (2.2.2) val dataloader')
    val_loader = SYDataLoader(val_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, persistent_workers=True)

    print(f' \n step 3. load model')
    print(f' (3.1) autoencoder')
    autoencoderkl = AutoencoderKL(spatial_dims=2, in_channels=1, out_channels=1,
                                  num_channels=(128, 128, 256), latent_channels=3,
                                  num_res_blocks=2, attention_levels=(False, False, False),
                                  with_encoder_nonlocal_attn=False, with_decoder_nonlocal_attn=False, )
    autoencoderkl = autoencoderkl.to(device)
    autoencoderkl.eval()
    state_dict = torch.load(args.autoencoder_pretrained_dir, map_location='cpu')['model']
    msg = autoencoderkl.load_state_dict(state_dict, strict=False)

    print(f' (3.2) unet')
    unet = DiffusionModelUNet(spatial_dims=2,
                              in_channels=3,
                              out_channels=3,
                              num_res_blocks=2,
                              num_channels=(128, 256, 512),
                              attention_levels=(False, True, True),
                              num_head_channels=(0, 256, 512),)
    unet = unet.to(device)
    print(f' (3.3) scheduler')
    scheduler = DDPMScheduler(num_train_timesteps=1000, schedule="linear_beta", beta_start=0.0015, beta_end=0.0195)

    print(f' \n step 4. scaling factor')
    with torch.no_grad():
        with autocast(enabled=True):
            # ------------------------------------------------------------------------------------------------
            # result of sampling (of first image, that is the first latent)
            z = autoencoderkl.encode_stage_2_inputs(check_data["image"].to(device))
    scale_factor = 1 / torch.std(z)
    print(f'scale_factor : {scale_factor}')

    print(f' \n step 5. infererence scheduler pipeline')
    inferer = LatentDiffusionInferer(scheduler, scale_factor=scale_factor)

    print(f' \n step 6. Training Diffusion Unet Model')
    print(f' (6.1) optimizer')
    optimizer = torch.optim.Adam(unet.parameters(), lr=1e-4)

    n_epochs = 200
    epoch_losses = []
    scaler = GradScaler()
    for epoch in range(n_epochs):
        unet.train()
        autoencoderkl.eval()
        epoch_loss = 0
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), ncols=70)
        progress_bar.set_description(f"Epoch {epoch}")
        for step, batch in progress_bar:
            # ------------------------------------------------------------------------------------------------
            # image = [Batch=64, channel=1, W=1600, H=800] -> after compression, [Batch=64, channel=3, W=160, H=80]
            images = batch["image"].to(device)         # [64,1,160,80]
            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=True):
                # ------------------------------------------------------------------------------------------------
                # 1) auto encoder : [Batch, output channel =3 , 160/f, 80/f] f = 4
                z_mu, z_sigma = autoencoderkl.encode(images)
                z = autoencoderkl.sampling(z_mu, z_sigma)
                # 2) get random noise
                noise = torch.randn_like(z).to(device) # [64,3,40,20]
                # 3) timestep condition
                timesteps = torch.randint(0, inferer.scheduler.num_train_timesteps, (z.shape[0],), device=z.device).long()
                # 4) noise prediction (prediction = diffusion_model(x=noisy_image, timesteps=timesteps, context=condition))
                # how can be without context?
                noise_pred = inferer(inputs=images, # batch, 1, W, H
                                     diffusion_model=unet,
                                     noise=noise,
                                     timesteps=timesteps,
                                     autoencoder_model=autoencoderkl)
                # ------------------------------------------------------------------------------------------------
                # (2) VLB Loss
                loss = F.mse_loss(noise_pred.float(),
                                  noise.float())
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item()
        # ------------------------------------------------------------------------------------------------
        # (3) unet saving
        #print(f' model saving ... ')
        #model_save_dir = os.path.join(args.model_save_baic_dir, 'unet_model')
        #os.makedirs(model_save_dir, exist_ok=True)
        #save_obj = {'model': unet.state_dict(), }
        #torch.save(save_obj, os.path.join(model_save_dir, f'unet_checkpoint_{epoch + 1}.pth'))
        #progress_bar.set_postfix({"loss": epoch_loss / (step + 1)})
        #epoch_losses.append(epoch_loss / (step + 1))
        # ------------------------------------------------------------------------------------------------
        # saving unet model
    progress_bar.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # step 1. device
    parser.add_argument('--device', type=str, default='cuda:1')
    # step 1. load dataset
    parser.add_argument("--data_folder", type=str,
                        default='/data7/sooyeon/medical_image/experiment_data/dental/Radiographs_L')
    parser.add_argument("--image_size", type=str, default='160,80')
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument('--autoencoder_pretrained_dir', type=str,
                        default='/data7/sooyeon/medical_image/experiment_result/vae_model/vae_checkpoint_100.pth')
    parser.add_argument('--model_save_baic_dir', type=str,
                        default='/data7/sooyeon/medical_image/experiment_result')

    args = parser.parse_args()
    main(args)