import os, argparse, ast
import json
from PIL import Image


def main(args) :

    base_folder = '/data7/sooyeon/medical_image/experiment_data/dental/panoramic_data_res_256/original'
    mask_folder = '/data7/sooyeon/medical_image/experiment_data/dental/panoramic_data_res_256/mask'
    images = os.listdir(base_folder)
    for image in images :
        image_dir = os.path.join(base_folder, image)
        mask_dir = os.path.join(mask_folder, image)
        pil = Image.open(image_dir)
        try :
            mask_pil = Image.open(mask_dir)
        except :
            print(f'{image} does not have mask')





if __name__ == '__main__' :
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    main(args)
