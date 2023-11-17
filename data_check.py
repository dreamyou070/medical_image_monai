import os, argparse, ast
import json
from PIL import Image


def main(args) :

    #base_dir = r'../medical_Data/experiment/dental/panoramic_data_high'

    print(f'\n step 1. image check')
    img_base_dir = r'../medical_data/experiment/dental/panoramic_data_high/mask'
    images = os.listdir(img_base_dir)

    mask_base_dir = r'../medical_data/experiment/dental/panoramic_data_high/mask'
    #mask_save_base_dir = os.path.join(base_dir, 'mask')
    #os.makedirs(mask_save_base_dir, exist_ok=True)
    for image_name in images :
        image_dir = os.path.join(img_base_dir, image_name)
        pil_image = Image.open(image_dir)
        print(f'pil_image : {pil_image.size}')
        #pil_image = pil_image.convert('L')
        #pil_image = pil_image.resize((256, 256))
        #pil_image.save(image_dir)
        """
        
        

        mask_dir = os.path.join(mask_base_dir, image_name)
        pil_mask = Image.open(mask_dir)
        pil_mask = pil_mask.convert('L')
        pil_mask = pil_mask.resize((H,H))
        pil_mask.save(os.path.join(mask_save_base_dir, f'{image_name}'))
        """





if __name__ == '__main__' :
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    main(args)
