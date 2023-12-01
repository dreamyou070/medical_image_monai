import os, json
from PIL import Image


import numpy as np
from PIL import Image
import os
import argparse

def main(args) :

    img_base_dir = r"../../medical_data/bagel/train/good/rgb"
    masked_img_base_dir = r"../../medical_data/bagel/train/good/xyz"
    os.makedirs(masked_img_base_dir, exist_ok=True)
    mask_base_dir = r"../../medical_data/dental/turf/Expert/mask"
    images = os.listdir(img_base_dir)
    for image in images :
        image_path = os.path.join(img_base_dir, image)
        img = Image.open(image_path)#.convert("L")
        print(img.size)
        """
        np_img = np.array(img)


        mask_path = os.path.join(mask_base_dir, image)
        mask = Image.open(mask_path).convert("L")
        np_mask = np.array(mask)
        np_mask = np.where(np_mask < 50, 0, 255)

        blended_img = Image.blend(img, mask, 1)
        np_blended_img = np.array(blended_img)
        new = np.where(np_mask == 0, np_img, np_blended_img)
        new = Image.fromarray(new).convert("L")
        new_dir = os.path.join(masked_img_base_dir, image)
        new.save(new_dir)
        """

    #img_dir = '139.JPG'
    #mask_dir = '139_masked.JPG'


    #new.show()
    #
    """
    mask_base_dir = r"../../medical_data/dental/turf/Expert/mask"
    binary_mask_base_dir = r"../../medical_data/dental/turf/Expert/binary_mask"
    os.makedirs(binary_mask_base_dir, exist_ok=True)
    images = os.listdir(mask_base_dir)
    for image in images :
        image_path = os.path.join(mask_base_dir, image)
        img = Image.open(image_path).convert("L")
        img = np.array(img)
        img = np.where(img < 50, 0, 255)
        img = Image.fromarray(img).convert("L")
        d = os.path.join(binary_mask_base_dir, image)
        print(d)
        img.save(d)
    """


if __name__ == '__main__' :

    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    main(args)


