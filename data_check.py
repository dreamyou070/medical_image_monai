import os, argparse, ast
import json
from PIL import Image


def main(args) :

    base_folder = '../medical_data/experiment/dental/Radiographs'
    images = os.listdir(base_folder)
    #mask_folder = '/data7/sooyeon/medical_image/experiment_data/dental/panoramic_data_res_256/mask'

    new_folder = '../medical_data/experiment/dental/panoramic_data_high/original'
    os.makedirs(new_folder, exist_ok=True)
    for image in images :
        image_dir = os.path.join(base_folder, image)
        pil = Image.open(image_dir)
        pil = pil.resize((256, 256))
        pil = pil.convert('L')
        pil.save(os.path.join(new_folder, image))
    """"""




if __name__ == '__main__' :
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    main(args)
