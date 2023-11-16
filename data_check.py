import os, argparse, ast
import json
from PIL import Image


def main(args) :

    base_dir = r'../medical_Data/experiment/dental/panoramic_data'

    print(f'\n step 1. image check')
    img_base_dir = r'../medical_Data/experiment/dental/Radiographs_L'
    images = os.listdir(img_base_dir)
    data_save_base_dir = os.path.join(base_dir, 'original')
    os.makedirs(data_save_base_dir, exist_ok=True)

    mask_base_dir = r'../medical_Data/experiment/dental/Expert/mask'
    mask_save_base_dir = os.path.join(base_dir, 'mask')
    os.makedirs(mask_save_base_dir, exist_ok=True)
    for image_name in images :
        image_dir = os.path.join(img_base_dir, image_name)
        pil_image = Image.open(image_dir)
        pil_image = pil_image.convert('L')
        pil_image = pil_image.resize((64,64))
        pil_image.save(os.path.join(data_save_base_dir, f'{image_name}'))

        mask_dir = os.path.join(mask_base_dir, image_name)
        pil_mask = Image.open(mask_dir)
        pil_mask = pil_mask.convert('L')
        pil_mask = pil_mask.resize((64, 64))
        pil_mask.save(os.path.join(mask_save_base_dir, f'{image_name}'))





if __name__ == '__main__' :
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    main(args)
