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
    val_ds = SYDataset_masking(data=val_datalist, transform=val_transforms)
    norm_val_loader = SYDataLoader(val_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, persistent_workers=True)
    for i, batch in train_loader :
        img_info = batch['image_info']
        mask_info = batch['mask']
        normal_info = batch['nonrmal']
        print(f'normal_info : {normal_info}')
        break



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
    parser.add_argument("--model_save_baic_dir", type=str,
                        default='/data7/sooyeon/medical_image/experiment_result_idea_20231116/vae_model')

    args = parser.parse_args()
    main(args)