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
    first_data = val_ds.__getitem__(0)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    # step 0. device
    parser.add_argument('--device', type=str, default='cuda:0')
    # step 1.
    parser.add_argument('--infer_num', type=int, default=5)
    # step 2. model loading
    parser.add_argument('--pretrained_dir', type=str, default='/data7/sooyeon/medical_image/model/checkpoint_25.pth')
    # step 3. get original image for reconstruct
    parser.add_argument("--data_folder", type=str, default='../experiment/dental/Radiographs_L')
    args = parser.parse_args()
    main(args)
