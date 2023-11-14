import argparse, torch, os
from model_module.nets import AutoencoderKL, DiffusionModelUNet, PatchDiscriminator
from data_module import get_transform, SYDataset, SYDataLoader
from model_module.schedulers import DDPMScheduler
from model_module.inferers import LatentDiffusionInferer
from torch.cuda.amp import autocast
from utils.check import first

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
                              num_res_blocks=2,num_channels=(128, 256, 512),
                              attention_levels=(False, True, True),
                              num_head_channels=(0, 256, 512),)
    print(f' (3.3) scheduler')
    scheduler = DDPMScheduler(num_train_timesteps=1000, schedule="linear_beta", beta_start=0.0015, beta_end=0.0195)

    print(f' \n step 4. scaling factor')
    with torch.no_grad():
        with autocast(enabled=True):
            z = autoencoderkl.encode_stage_2_inputs(check_data["image"].to(device))
    scale_factor = 1 / torch.std(z)
    print(f"  scaling factor set to {scale_factor}")

    print(f' \n step 5. infererence scheduler')
    inferer = LatentDiffusionInferer(scheduler, scale_factor=scale_factor)

    print(f' \n step 6. Training Diffusion Unet Model')




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # step 0. device
    parser.add_argument('--device', type=str, default='cuda:0')
    # step 1. load dataset
    parser.add_argument("--data_folder", type=str, default='/data7/sooyeon/medical_image/experiment_data/dental/Radiographs_L')
    parser.add_argument("--image_size", type=str, default='160,84')
    parser.add_argument('--autoencoder_pretrained_dir', type=str, default='/data7/sooyeon/medical_image/experiment_result/vae_model/checkpoint_100.pth')
    args = parser.parse_args()
    main(args)
