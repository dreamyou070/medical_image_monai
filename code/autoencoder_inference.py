import numpy as np
from matplotlib import pyplot as plt
import torch, os
import argparse
from generative.networks.nets import AutoencoderKL
from data_module import val_transform, SYDataset, SYDataLoader


def main(args) :

    print(f' \n step 0. device')
    device = torch.device(args.device)

    print(f' \n step 1. make empty model')
    autoencoderkl = AutoencoderKL(spatial_dims=2,in_channels=1,out_channels=1,
                                  num_channels=(128, 128, 256),latent_channels=3,
                                  num_res_blocks=2,attention_levels=(False, False, False),
                                  with_encoder_nonlocal_attn=False,with_decoder_nonlocal_attn=False,)
    autoencoderkl = autoencoderkl.to(device)
    autoencoderkl.eval()

    print(f' \n step 2. model loading')
    state_dict = torch.load(args.pretrained_dir, map_location='cpu')['model']
    msg = autoencoderkl.load_state_dict(state_dict, strict=False)

    print(f' \n step 3. get original image for reconstruct')
    total_datas = os.listdir(args.data_folder)
    val_transforms = val_transform()
    train_num = int(0.7 * len(total_datas))
    train_datas, val_datas = total_datas[:train_num], total_datas[train_num:]
    val_datalist = [{"image": os.path.join(args.data_folder, val_data)} for val_data in val_datas]
    val_ds = SYDataset(data=val_datalist, transform=val_transforms)
    print(f' (3.0) random select numbers')
    inference_num = args.inference_num
    random_idx = np.random.randint(0, len(val_ds), size=inference_num)
    recon_img_list = []
    for idx in random_idx :

        fig, ax = plt.subplots(nrows=1, ncols=2, sharey=True)
        # ------------------------------------------------------------------------------------------------
        # org shape is [1615,840]
        org_img_ = val_ds[idx]['image']
        org_img = org_img_.unsqueeze(0).to(device)
        with torch.no_grad():
            recon_img, z_mu, z_sigma = autoencoderkl(org_img)
            batch, channel, width, height = recon_img.shape
            # recon_img = [Batch, Channel=1, Width, Height]
            recon_img_list.append(recon_img[:1, 0])
            reconstructions = torch.reshape(recon_img, (width, height)).T # height, width

            org_img_ = org_img_.squeeze().T
            ax[0].imshow(org_img_.cpu(), cmap="gray")
            ax[0].set_title('original')
            ax[0].axis("off")

            ax[1].imshow(reconstructions.cpu(), cmap='gray')
            ax[1].set_title('reconstruction')
            ax[1].axis("off")
            print(f'org_img_.shape : {org_img_.shape} / reconstructions.shape : {reconstructions.shape}')
            plt.savefig(f'./reconstructions_{idx}.png')
            plt.close()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    # step 0. device
    parser.add_argument('--device', type=str, default='cuda:0')
    # step 1.
    parser.add_argument('--infer_num', type=int, default=5)
    # step 2. model loading
    parser.add_argument('--pretrained_dir', type=str, default='/data7/sooyeon/medical_image/model/checkpoint_100.pth')
    # step 3. get original image for reconstruct
    parser.add_argument("--data_folder", type=str, default='../experiment/dental/Radiographs_L')
    parser.add_argument("--inference_num", type=int, default=5)
    args = parser.parse_args()
    main(args)
