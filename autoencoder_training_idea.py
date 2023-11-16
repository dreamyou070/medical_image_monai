from monai.config import print_config
from utils import set_determinism
from data_module import get_transform, SYDataset, SYDataLoader
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


def main(args):

    print(f'\n step 1. print version and set seed')
    print_config()
    set_determinism(args.seed)

    print(f'\n step 2. dataset and dataloader')
    total_datas = os.listdir(args.norm_data_folder)
    print(f' (2.0) data_module transform')
    train_transforms, val_transforms = get_transform(args.image_size)
    print(f' (2.1.1) train dataset')
    train_num = int(0.7 * len(total_datas))
    train_datas, norm_val_datas = total_datas[:train_num], total_datas[train_num:]
    train_datalist = [{"image": os.path.join(args.norm_data_folder, train_data)} for train_data in train_datas]
    train_ds = SYDataset(data=train_datalist, transform=train_transforms)
    print(f' (2.1.2) train dataloader')
    train_loader = SYDataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4,
                                persistent_workers=True)
    check_data = first(train_loader)

    print(f' (2.2.1) normal val dataset and dataloader')
    norm_val_datalist = [{"image": os.path.join(args.norm_data_folder, val_data)} for val_data in norm_val_datas]
    norm_val_ds = SYDataset(data=norm_val_datalist, transform=val_transforms)
    norm_val_loader = SYDataLoader(norm_val_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, persistent_workers=True)

    print(f' (2.2.2) val dataloader')
    ood_datas = os.listdir(args.ood_data_folder)
    ood_val_datalist = [{"image": os.path.join(args.ood_data_folder, val_data)} for val_data in ood_datas]
    ood_val_ds = SYDataset(data=ood_val_datalist, transform=val_transforms)
    ood_val_loader = SYDataLoader(ood_val_ds, batch_size=args.batch_size, shuffle=True, num_workers=4,
                                  persistent_workers=True)

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
    autoencoder_warm_up_n_epochs = -1
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
            images = batch["image"].to(device)
            optimizer_g.zero_grad(set_to_none=True)
            with autocast(enabled=True):
                reconstruction, z_mu, z_sigma = autoencoderkl(images)
                # ------------------------------------------------------------------------------------------------------------
                # (1.1) reconstruction loss (L1)
                recons_loss = F.l1_loss(reconstruction.float(), images.float())
                # (1.2) preceptual loss (it measure following perceptual loss function)
                p_loss = perceptual_loss(reconstruction.float(), images.float())
                # (1.3) KL loss
                kl_loss = 0.5 * (torch.sum(z_mu.pow(2) + z_sigma.pow(2) - torch.log(z_sigma.pow(2)) - 1, dim=[1, 2, 3]))
                kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]
                # (2) total loss
                loss_g = recons_loss + (kl_weight * kl_loss) + (perceptual_weight * p_loss)
                # ------------------------------------------------------------------------------------------------------------
                # (3) generator loss
                if epoch > autoencoder_warm_up_n_epochs:
                    # ---------------------------------------------------------------------------------------------------------
                    # there are five length from the output of discriminator, if i choose just the last one, that is the final result
                    # logits_fake = [Batch, 1, /8 -2 , /8 -2] (This show multi scale discriminator, that is patch....)
                    # 생성자를 학습시키기 위한 것
                    # 생성자가 판별한 이미지가 마치 real 인 것처럼 판정이 나야 한다. 즉, 생성자가 생성한 이미지 input 은 target 으로 True 가 되어야 하고,
                    # 이는 생성자를 학습하기 위한 것이므로, for_discriminator 는 False 로 한다.
                    discrimator_output = discriminator(reconstruction.contiguous().float())
                    logits_fake = discrimator_output[-1] # one or zero
                    # how does adversarial loss do...
                    generator_loss = adv_loss(logits_fake,
                                              target_is_real=True,
                                              for_discriminator=False)
                    loss_g += adv_weight * generator_loss
            scaler_g.scale(loss_g).backward()
            scaler_g.step(optimizer_g)
            scaler_g.update()
            # ------------------------------------------------------------------------------------------------------------
            # (3) discriminator training
            if epoch > autoencoder_warm_up_n_epochs:
                with autocast(enabled=True):
                    # 구분자를 학습시키기 위한 것
                    optimizer_d.zero_grad(set_to_none=True)
                    # ---------------------------------------------------------------------------------------------------------------------
                    # reconstruction.contiguous().detach() 를 discriminator 은 가짜라고 판별해야 한다
                    # 즉 target 은 가짜이므로, target_is_real 은 False 가 되어야 한다.
                    # 구분자를 학습시키기 위한 것이므로 for_discriminator 는 True 로 한다.


                    discriminator_result = discriminator(reconstruction.contiguous().detach())[-1]


                    loss_d_fake = adv_loss(discriminator_result,
                                           target_is_real=False,
                                           for_discriminator=True)

                    print(f'shape of logits_fake : {logits_fake.shape}')

                    # ---------------------------------------------------------------------------------------------------------------------
                    # discriminator 가 real 을 real 이라고 판별해야 한다.\
                    # 즉, target 은 real 이므로, target_is_real 은 True 가 되어야 한다.
                    # 구분자를 학습시키기 위한 것이므로 for_discriminator 는 True 로 한다.
                    loss_d_real = adv_loss(discriminator(images.contiguous().detach())[-1],
                                           target_is_real=True,
                                           for_discriminator=True)
                    discriminator_loss = (loss_d_fake + loss_d_real) * 0.5
                    loss_d = adv_weight * discriminator_loss
                scaler_d.scale(loss_d).backward()
                scaler_d.step(optimizer_d)
                scaler_d.update()
            epoch_loss += recons_loss.item()
            if epoch > autoencoder_warm_up_n_epochs:
                gen_epoch_loss += generator_loss.item()
                disc_epoch_loss += discriminator_loss.item()
            progress_bar.set_postfix({"recons_loss": epoch_loss / (step + 1),
                                      "gen_loss": gen_epoch_loss / (step + 1),
                                      "disc_loss": disc_epoch_loss / (step + 1), })
        """
        epoch_recon_losses.append(epoch_loss / (step + 1))
        epoch_gen_losses.append(gen_epoch_loss / (step + 1))
        epoch_disc_losses.append(disc_epoch_loss / (step + 1))

        if (epoch + 1) % val_interval == 0:
            autoencoderkl.eval()
            val_loss = 0
            with torch.no_grad():
                for val_step, batch in enumerate(val_loader, start=1):
                    images = batch["image"].to(device)
                    with autocast(enabled=True):
                        reconstruction, z_mu, z_sigma = autoencoderkl(images)
                        # Get the first reconstruction from the first validation batch for visualisation purposes
                        if val_step == 1:
                            intermediary_images.append(reconstruction[:num_example_images, 0])
                        recons_loss = F.l1_loss(images.float(), reconstruction.float())
                    val_loss += recons_loss.item()
            val_loss /= val_step
            val_recon_losses.append(val_loss)
            print(f"epoch {epoch + 1} val loss: {val_loss:.4f}")
            """

        # ------------------------------------------------------------------------------------------------------------
        print(f' model saving ... ')
        model_save_dir = os.path.join(args.model_save_baic_dir, 'model')
        os.makedirs(model_save_dir, exist_ok=True)
        save_obj = {'model': autoencoderkl.state_dict(),}
        torch.save(save_obj, os.path.join(model_save_dir, f'vae_checkpoint_{epoch+1}.pth'))
    progress_bar.close()
    del discriminator
    del perceptual_loss
    torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # step 1. print version and set seed
    parser.add_argument("--seed", type=int, default=42)

    # step 2. dataset and dataloader
    parser.add_argument("--norm_data_folder", type=str,
                        default='/data7/sooyeon/medical_image/experiment_data/dental/Radiographs_L_normal_lowres')
    parser.add_argument("--ood_data_folder", type=str,
                        default='/data7/sooyeon/medical_image/experiment_data/dental/Radiographs_L_ood_lowres')
    parser.add_argument("--image_size", type=str, default='64,64')
    parser.add_argument("--vis_num_images", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device", type=str, default='cuda:4')
    # step 5. saving autoencoder model
    parser.add_argument("--model_save_baic_dir", type=str,
                        default='/data7/sooyeon/medical_image/experiment_result_idea_20231116/vae_model')

    args = parser.parse_args()
    main(args)