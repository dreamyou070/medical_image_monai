from monai.config import print_config
from utils import set_determinism
from data_module import get_transform, SYDataset, SYDataLoader, SYDataset_masking
import os
from monai.utils import first
from utils.set_seed import set_determinism
import argparse
import torch
from model_module.nets import AutoencoderKL, PatchDiscriminator
from loss_module import PatchAdversarialLoss, PerceptualLoss
from tqdm import tqdm
from torch.cuda.amp import GradScaler, autocast
import torch.nn.functional as F
import torchvision.transforms as torch_transforms


def main(args):

    print(f'\n step 1. print version and set seed')
    print_config()
    set_determinism(args.seed)

    print(f'\n step 2. dataset and dataloader')
    data_base_dir = os.path.join(args.data_folder, 'original')
    base_mask_dir = os.path.join(args.data_folder, 'mask')
    total_datas = os.listdir(data_base_dir)
    train_transforms, val_transforms = get_transform(args.image_size)
    train_num = int(0.7 * len(total_datas))
    train_datas, val_datas = total_datas[:train_num], total_datas[train_num:]
    train_datalist = [{"image": os.path.join(data_base_dir, train_data)} for train_data in train_datas]
    train_ds = SYDataset_masking(data=train_datalist,
                                 transform=train_transforms,
                                 base_mask_dir = base_mask_dir)

    print(f' (2.1.2) train dataloader')
    train_loader = SYDataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4,
                                persistent_workers=True)
    check_data = first(train_loader)

    print(f' (2.2.1) normal val dataset and dataloader')
    val_datalist = [{"image": os.path.join(data_base_dir, val_data)} for val_data in val_datas]
    val_ds = SYDataset_masking(data=val_datalist,
                               transform=val_transforms,
                               base_mask_dir = base_mask_dir)
    val_loader = SYDataLoader(val_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, persistent_workers=True)


    # -----------------------------------------------------------------------------------------------------------------------------------
    print(f'\n step 3. pretrain autoencoder model')
    print(f' (3.0) device')
    device = torch.device("cuda")
    print(f' (3.1) generator (vae autoencoder)')
    autoencoderkl = AutoencoderKL(spatial_dims=2, in_channels=1, out_channels=1, num_channels=(128, 128, 256),
                                  latent_channels=3, num_res_blocks=2,
                                  attention_levels=(False, False, False), with_encoder_nonlocal_attn=False,
                                  with_decoder_nonlocal_attn=False, ).to(device)
    print(f' (3.2) discriminator')
    discriminator = PatchDiscriminator(spatial_dims=2, num_layers_d=3, num_channels=64, in_channels=1,
                                       out_channels=1).to(device)
    print(f' (3.3) perceptual_loss')
    perceptual_loss = PerceptualLoss(spatial_dims=2, network_type="alex").to(device)
    perceptual_weight = 0.001
    print(f' (3.4) patch adversarial loss')
    adv_loss = PatchAdversarialLoss(criterion="least_squares")
    adv_weight = 0.01
    print(f' (3.5) optimizer (for generator and discriminator)')
    optimizer_g = torch.optim.Adam(autoencoderkl.parameters(), lr=1e-4)
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=5e-4)

    print(f' (3.6) mixed precision (for generator and discriminator)')
    scaler_g = torch.cuda.amp.GradScaler()
    scaler_d = torch.cuda.amp.GradScaler()

    print(f'step 4. training (takes about one hour)')
    kl_weight = 1e-6
    n_epochs = 100
    val_interval = 1
    autoencoder_warm_up_n_epochs = 10
    #weight_dtype = autoencoderkl.dtype#encoder.weight.dtype
    #text_encoder.to(dtype=weight_dtype)

    save_basic_dir = args.save_basic_dir
    os.makedirs(save_basic_dir, exist_ok=True)
    inf_save_basic_dir = os.path.join(args.save_basic_dir, 'inference_20231115')
    os.makedirs(inf_save_basic_dir, exist_ok=True)
    for epoch in range(n_epochs):
        print(f' epoch {epoch + 1}/{n_epochs}')
        autoencoderkl.train()
        discriminator.train()
        epoch_loss = 0
        gen_epoch_loss = 0
        disc_epoch_loss = 0
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), ncols=110)
        progress_bar.set_description(f"Epoch {epoch}")
        for step, batch in progress_bar:

            normal_info = batch['normal']
            normal_index = torch.where(normal_info == 1)
            ood_index = torch.where(normal_info != 1)

            img_info = batch['image_info']['image'].to(device)
            weight_dtype = img_info.dtype
            normal_img_info = img_info[normal_index]
            ood_img_info = img_info[ood_index]

            mask_info = batch['mask'].to(device, weight_dtype)
            mask_info = mask_info.unsqueeze(1)
            masked_img_info = img_info * mask_info
            # 0black = 0 -> 1 ->
            normal_mask_info = mask_info[normal_index]
            ood_mask_info = mask_info[ood_index]

            masked_img_info = img_info * masked_img_info
            optimizer_g.zero_grad(set_to_none=True)
            with autocast(enabled=True):
                reconstruction, z_mu, z_sigma = autoencoderkl(masked_img_info)
                recons_loss = F.l1_loss(reconstruction.float(),
                                        img_info.float())
                                        #masked_img_info.float())
                p_loss = perceptual_loss(reconstruction.float(),
                                         img_info.float())
                                         # masked_img_info.float())
                kl_loss = 0.5 * (torch.sum(z_mu.pow(2) + z_sigma.pow(2) - torch.log(z_sigma.pow(2)) - 1, dim=[1, 2, 3]))
                kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]
                loss_g = recons_loss + (kl_weight * kl_loss) + (perceptual_weight * p_loss)
                if epoch > autoencoder_warm_up_n_epochs:
                    discrimator_output = discriminator(reconstruction.contiguous().float())
                    logits_fake = discrimator_output[-1]
                    generator_loss = adv_loss(logits_fake, target_is_real=True, for_discriminator=False)
                    loss_g += adv_weight * generator_loss
            scaler_g.scale(loss_g).backward()
            scaler_g.step(optimizer_g)
            scaler_g.update()
            # ------------------------------------------------------------------------------------------------------------
            # (3) discriminator training
            if epoch > autoencoder_warm_up_n_epochs:
                with autocast(enabled=True):
                    # make smart discriminator
                    optimizer_d.zero_grad(set_to_none=True)
                    # ---------------------------------------------------------------------------------------------------------------------
                    # reconstruction.contiguous().detach() 를 discriminator 은 가짜라고 판별해야 한다
                    # 즉 target 은 가짜이므로, target_is_real 은 False 가 되어야 한다.
                    # 구분자를 학습시키기 위한 것이므로 for_discriminator 는 True 로 한다.
                    loss_d_fake = adv_loss(discriminator(reconstruction.contiguous().detach())[-1],
                                           target_is_real=False,
                                           for_discriminator=True)
                    # ---------------------------------------------------------------------------------------------------------------------
                    # discriminator 가 real 을 real 이라고 판별해야 한다.\
                    # 즉, target 은 real 이므로, target_is_real 은 True 가 되어야 한다.
                    # 구분자를 학습시키기 위한 것이므로 for_discriminator 는 True 로 한다.
                    loss_d_real = adv_loss(discriminator(img_info.contiguous().detach())[-1],
                                           target_is_real=True,
                                           for_discriminator=True)
                    discriminator_loss = (loss_d_fake + loss_d_real) * 0.5
                    loss_d = adv_weight * discriminator_loss
                scaler_d.scale(loss_d).backward()
                scaler_d.step(optimizer_d)
                scaler_d.update()
            #epoch_loss += recons_loss.item()
            if epoch > autoencoder_warm_up_n_epochs:
                gen_epoch_loss += generator_loss.item()
                disc_epoch_loss += discriminator_loss.item()
            progress_bar.set_postfix({"recons_loss": epoch_loss / (step + 1),
                                      "gen_loss": gen_epoch_loss / (step + 1),
                                      "disc_loss": disc_epoch_loss / (step + 1), })
        #
        if (epoch + 1) % val_interval == 0:
            autoencoderkl.eval()
            for val_step, batch in enumerate(val_loader, start=1):

                normal_info = batch['normal']
                normal_index = torch.where(normal_info == 1)
                ood_index = torch.where(normal_info != 1)

                img_info = batch['image_info']['image'].to(device)
                weight_dtype = img_info.dtype
                normal_img_info = img_info[normal_index]
                ood_img_info = img_info[ood_index]

                mask_info = batch['mask']
                normal_mask_info = mask_info[normal_index]
                ood_mask_info = mask_info[ood_index]

                for norm_img in normal_img_info :
                    with torch.no_grad():
                        # batch, channel, height, width (64)
                        recon_img, z_mu, z_sigma = autoencoderkl(normal_img_info)
                        batch, channel, width, height = recon_img.shape
                        # ------------------- save image ------------------- #
                        for i in range(batch):
                            normal_origins = torch.reshape(normal_mask_info[i], (width, height)).T
                            normal_origins = normal_origins.detach().squeeze().cpu()
                            pil_normal = torch_transforms.ToPILImage()(normal_origins)
                            pil_normal.save(os.path.join(inf_save_basic_dir, f'epoch_{epoch + 1}_normal.png'))

                        # ------------------- save image ------------------- #
                        #reconstructions = torch.reshape(recon_img, (width, height)).T  # height, width
                        #reconstructions = reconstructions.detach().squeeze().cpu()
                        #pil_recon = torch_transforms.ToPILImage()(reconstructions)
                        #pil_recon.save(os.path.join(inf_save_basic_dir, f'epoch_{epoch + 1}_recon.png'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # step 1. print version and set seed
    parser.add_argument("--seed", type=int, default=42)

    # step 2. dataset and dataloader
    parser.add_argument("--data_folder", type=str,
                        default='/data7/sooyeon/medical_image/experiment_data/dental/panoramic_data')

    parser.add_argument("--ood_data_folder", type=str,
                        default='/data7/sooyeon/medical_image/experiment_data/dental/Radiographs_L_ood_lowres')
    parser.add_argument("--image_size", type=str, default='64,64')
    parser.add_argument("--vis_num_images", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device", type=str, default='cuda:4')
    # step 5. saving autoencoder model
    parser.add_argument("--save_basic_dir", type=str,
                        default='/data7/sooyeon/medical_image/experiment_result_idea_20231116')
    args = parser.parse_args()
    main(args)
