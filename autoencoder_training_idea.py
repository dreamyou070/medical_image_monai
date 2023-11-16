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
import wandb
import matplotlib.pyplot as plt
from PIL import Image
import io
import json

def main(args):

    print(f'\n step 1. wandb login')
    wandb.login(key=args.wandb_api_key)
    wandb.init(project=args.wandb_project_name,
               name=args.wandb_run_name)

    print(f'\n step 2. print version and set seed')
    print_config()
    set_determinism(args.seed)

    print(f'\n confiv_save')
    os.makedirs(args.save_basic_dir, exist_ok=True)
    with open(os.path.join(args.save_basic_dir, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)


    print(f'\n step 3. dataset and dataloader')
    data_base_dir = os.path.join(args.data_folder, 'original')
    base_mask_dir = os.path.join(args.data_folder, 'mask')
    total_datas = os.listdir(data_base_dir)
    train_transforms, val_transforms = get_transform(args.image_size)
    train_num = int(0.8 * len(total_datas))
    train_datas, val_datas = total_datas[:train_num], total_datas[train_num:]
    train_datalist = [{"image": os.path.join(data_base_dir, train_data)} for train_data in train_datas]
    train_ds = SYDataset_masking(data=train_datalist,transform=train_transforms,base_mask_dir = base_mask_dir)

    print(f' (3.1) train dataloader')
    train_loader = SYDataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4,persistent_workers=True)
    print(f' (3.2) validdataloader')
    val_datalist = [{"image": os.path.join(data_base_dir, val_data)} for val_data in val_datas]
    val_ds = SYDataset_masking(data=val_datalist,transform=val_transforms,base_mask_dir = base_mask_dir)
    val_loader = SYDataLoader(val_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, persistent_workers=True)


    print(f'\n step 4. model')
    device = torch.device("cuda")
    print(f' (4.1) generator (vae autoencoder)')
    autoencoderkl = AutoencoderKL(spatial_dims=2, in_channels=1, out_channels=1, num_channels=(128, 128, 256),
                                  latent_channels=3, num_res_blocks=2, attention_levels=(False, False, False), with_encoder_nonlocal_attn=False,
                                  with_decoder_nonlocal_attn=False, ).to(device)
    print(f' (4.2) discriminator')
    discriminator = PatchDiscriminator(spatial_dims=2,
                                       num_layers_d=3,
                                       num_channels=64,
                                       in_channels=1,
                                       out_channels=1).to(device)



    print(f'\n step 5. loss function')
    mse_loss = torch.nn.MSELoss(reduction='mean')

    print(f' (5.3) perceptual_loss')
    perceptual_loss = PerceptualLoss(spatial_dims=2, network_type="alex").to(device)
    perceptual_weight = 0.001
    print(f' (5.4) patch adversarial loss')
    adv_loss = PatchAdversarialLoss(criterion="least_squares")
    adv_weight = 0.01



    print(f'\n step 5. optimizer')
    optimizer_g = torch.optim.Adam(autoencoderkl.parameters(), lr=1e-4)
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=5e-4)

    print(f'step 6. Training')
    kl_weight = 1e-6
    n_epochs = 100
    autoencoder_warm_up_n_epochs = args.autoencoder_warm_up_n_epochs

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

            mask_info = batch['mask'].to(device, weight_dtype).unsqueeze(1)
            normal_mask_info = mask_info[normal_index]
            ood_mask_info = mask_info[ood_index]

            optimizer_g.zero_grad(set_to_none=True)
            with autocast(enabled=True):
                loss_dict = {}
                # -----------------------------------------------------------------------------------------------------
                # autoencoder must reconstruct the normal image
                reconstruction, z_mu, z_sigma = autoencoderkl(normal_img_info)
                recons_loss = F.l1_loss(reconstruction.float(),
                                        normal_img_info.float())
                loss_dict["loss/recons_loss"] = recons_loss.item()
                p_loss = perceptual_loss(reconstruction.float(),normal_img_info)
                loss_dict["loss/perceptual_loss"] = p_loss.item()
                kl_loss = 0.5 * (torch.sum(z_mu.pow(2) + z_sigma.pow(2) - torch.log(z_sigma.pow(2)) - 1, dim=[1, 2, 3]))
                kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]
                loss_dict["loss/kl_loss"] = kl_loss.item()
                loss_g = recons_loss + (kl_weight * kl_loss) + (perceptual_weight * p_loss)
                if epoch > autoencoder_warm_up_n_epochs:
                    # -----------------------------------------------------------------------------------------------------
                    # normal image adversarial loss
                    recon_attn = discriminator(reconstruction.contiguous().float())[-1]
                    recon_attn_target = normal_mask_info.contiguous().float()
                    generator_loss = torch.mean(torch.stack([mse_loss(recon_attn.float(),
                                                                      recon_attn_target.float())]))
                    loss_g += adv_weight * generator_loss
                    loss_dict["loss/adversarial_normal_generator_loss"] = generator_loss.item()
                    # -----------------------------------------------------------------------------------------------------
                    # ood image adversarial loss
                    ood_reconstruction, z_mu, z_sigma = autoencoderkl(ood_img_info)
                    ood_recon_attn = discriminator(ood_reconstruction.contiguous().float())[-1]
                    ood_generator_loss = torch.mean(torch.stack([mse_loss(ood_recon_attn.float(),
                                                                          ood_mask_info.contiguous().float())]))
                    loss_g += adv_weight * ood_generator_loss
                    loss_dict["loss/adversarial_ood_generator_loss"] = generator_loss.item()
            loss_g.backward()
            optimizer_g.step()

            # ------------------------------------------------------------------------------------------------------------
            # (3) discriminator training
            if epoch > autoencoder_warm_up_n_epochs:
                with autocast(enabled=True):
                    optimizer_d.zero_grad(set_to_none=True)
                    # -----------------------------------------------------------------------------------------------------
                    # 1) real normal
                    normal_img_attn = discriminator(normal_img_info.contiguous().float())[-1]
                    normal_real_loss = torch.mean(torch.stack([mse_loss(normal_img_attn.float(),
                                                                        normal_mask_info.contiguous().float())]))
                    loss_dict["loss/adversarial_discriminator_normal_real"] = normal_real_loss.item()
                    # -----------------------------------------------------------------------------------------------------
                    # 2) synthesized normal
                    synthesized_normal_attn = discriminator(reconstruction.contiguous().detach())[-1]
                    normal_synthesized_loss = torch.mean(torch.stack([mse_loss(synthesized_normal_attn.float(),
                                                                               normal_mask_info.contiguous().float())]))
                    loss_dict["loss/adversarial_discriminator_normal_synthesized"] = normal_synthesized_loss.item()
                    # -----------------------------------------------------------------------------------------------------
                    # 3) real ood
                    ood_img_attn = discriminator(ood_img_info.contiguous().float())[-1]
                    ood_real_loss = torch.mean(torch.stack([mse_loss(ood_img_attn.float(),
                                                                     ood_mask_info.float())]))
                    loss_dict["loss/adversarial_discriminator_ood_real"] = ood_real_loss.item()
                    # -----------------------------------------------------------------------------------------------------
                    # 4) synthesized ood
                    ood_img_attn = discriminator(ood_reconstruction.contiguous().detach())[-1]
                    ood_synthesized_loss = torch.mean(torch.stack([mse_loss(ood_img_attn.float(),
                                                                            ood_mask_info.float())]))
                    loss_dict["loss/adversarial_discriminator_ood_synthesized"] = ood_synthesized_loss.item()

                    discriminator_loss = (normal_real_loss + normal_synthesized_loss + ood_real_loss + ood_synthesized_loss) * 0.25
                    loss_d = adv_weight * discriminator_loss
                loss_d.backward()
                optimizer_d.step()
            #epoch_loss += recons_loss.item()
            if epoch > autoencoder_warm_up_n_epochs:
                gen_epoch_loss += generator_loss.item()
                disc_epoch_loss += discriminator_loss.item()
            progress_bar.set_postfix({"recons_loss": epoch_loss / (step + 1),
                                      "gen_loss": gen_epoch_loss / (step + 1),
                                      "disc_loss": disc_epoch_loss / (step + 1), })
            wandb.log(loss_dict)
        # -------------------------------------------------------------------------------------------------------------------------------
        if (epoch + 1) % args.val_interval == 0:
            autoencoderkl.eval()
            max_norm, max_ood = 4, 4
            norm_num, ood_num = 0, 0
            for val_step, batch in enumerate(val_loader, start=1):
                normal_info = batch['normal']
                img_info = batch['image_info']['image'].to(device)
                mask_info = batch['mask'].to(device, img_info.dtype).unsqueeze(1)
                masked_img_info = img_info * mask_info

                with torch.no_grad():
                    recon_info, z_mu, z_sigma = autoencoderkl(img_info)
                    batch, channel, width, height = recon_info.shape
                    index = 0
                    for img_, masked_img_, recon_ in zip(img_info, masked_img_info, recon_info) :
                        is_normal = normal_info[index]
                        if is_normal == 1 :
                            caption = 'Normal Image'
                            norm_num += 1
                        else :
                            caption = 'OOD Image'
                            ood_num += 1
                        if (is_normal == 1 and norm_num < max_norm) or (is_normal != 1 and ood_num < max_ood):
                            fig, ax = plt.subplots(nrows=1, ncols=3, sharey=True)

                            ax[0].imshow(torch.reshape(img_, (width, height)).T.cpu(), cmap="gray")
                            ax[0].set_title('original')
                            ax[0].axis("off")

                            ax[1].imshow(torch.reshape(masked_img_, (width, height)).T.cpu(), cmap="gray")
                            ax[1].set_title('masked image')
                            ax[1].axis("off")

                            ax[2].imshow(torch.reshape(recon_, (width, height)).T.cpu(), cmap='gray')
                            ax[2].set_title('recon image')
                            ax[2].axis("off")

                            fig.suptitle(caption)
                            fig_save_dir = os.path.join(inf_save_basic_dir, f'epoch_{epoch + 1}')
                            os.makedirs(fig_save_dir, exist_ok=True)
                            plt.savefig(os.path.join(fig_save_dir, f'{caption}_epoch_{epoch + 1}_{index}.png'))
                            index += 1
                            buf = io.BytesIO()
                            fig.savefig(buf)
                            buf.seek(0)
                            pil = Image.open(buf)
                            w,h = pil.size
                            wandb.log({f"epoch : {epoch + 1} index : {index} : " : wandb.Image(pil.resize((w*4, h*4)), caption=caption)})
                            plt.close()
        # -------------------------------------------------------------------------------------------------------------------------------
        # model save
        if epoch > args.model_save_num :
            model_save_dir = os.path.join(inf_save_basic_dir, 'vae_model')
            os.makedirs(model_save_dir, exist_ok=True)
            torch.save({'model': autoencoderkl.state_dict(), },
                       os.path.join(model_save_dir, f'vae_checkpoint_{epoch + 1}.pth'))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # step 1. wandb login
    parser.add_argument("--wandb_api_key", type=str, default='3a3bc2f629692fa154b9274a5bbe5881d47245dc')
    parser.add_argument("--wandb_project_name", type=str, default='test')
    parser.add_argument("--wandb_run_name", type=str, default='test')

    # step 2. print version and set seed
    parser.add_argument("--seed", type=int, default=42)

    # step 3. dataset and dataloader
    parser.add_argument("--data_folder", type=str,
                        default='/data7/sooyeon/medical_image/experiment_data/dental/panoramic_data')
    parser.add_argument("--image_size", type=str, default='64,64')
    parser.add_argument("--batch_size", type=int, default=64)

    # step 4. make model
    parser.add_argument("--device", type=str, default='cuda:4')

    # step 6. Training
    parser.add_argument("--kl_weight", type=float, default=1e-6)
    parser.add_argument("--n_epochs",  type=int, default=100)
    parser.add_argument("--val_interval", type=int, default=20)
    parser.add_argument("--autoencoder_warm_up_n_epochs", type=int, default=1)

    # step 7. saving autoencoder model
    parser.add_argument("--save_basic_dir", type=str,
                        default='/data7/sooyeon/medical_image/experiment_result_idea_20231116')
    parser.add_argument("--model_save_num", type=int, default=90)
    args = parser.parse_args()
    main(args)
