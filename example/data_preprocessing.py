import numpy as np
from PIL import Image
import os
import argparse

def main(args) :

    base_img_dir  = r'../medical_data/dental/turf/panoramic_data_res_256/train/original'
    base_mask_dir = r'../medical_data/dental/turf/panoramic_data_res_256/train/mask'
    images = os.listdir(base_img_dir)
    for image in images:

        # 1) image
        img_dir = os.path.join(base_img_dir, image)
        pil_img = Image.open(img_dir)
        pil_img = pil_img.convert("RGBA")
        np_img = np.array(pil_img)

        # 2) mask
        mask_dir = os.path.join(base_mask_dir, image)
        pil_mask = Image.open(mask_dir)
        np_mask = np.array(pil_mask)
        binary_mask_np = np.where(np_mask < 50, 0, 255)
        w, h = binary_mask_np
        w_index_list = []
        h_index_list = []
        for w_index in range(w):
            for h_index in range(h):
                if binary_mask_np[w_index, h_index] == 255:
                    w_index_list.append(w_index)
                    h_index_list.append(h_index)
        w_min = min(w_index_list)
        w_max = max(w_index_list)
        h_min = min(h_index_list)
        h_max = max(h_index_list)


        np_img[:, :, 3] = alpha_channel
        alpha_pil_img = Image.fromarray(np_img)
        np_img = np.array(alpha_pil_img)
        r, g, b, a = np_img[:, :, 0], np_img[:, :, 1], np_img[:, :, 2], np_img[:, :, 3]
        i, j = a.shape
        for i_index in range(i):
            for j_index in range(j):
                if np_img[i_index, j_index, 3] == 0:
                    np_img[i_index, j_index, 0] = 0
                    np_img[i_index, j_index, 1] = 0
                    np_img[i_index, j_index, 2] = 0
        alpha_pil_img = Image.fromarray(np_img)
        img = alpha_pil_img.convert('RGB')
        # ----------------------------------------------------------------------------------------------------------------------------
        # img crop
        im1 = img.crop((h_min, w_min, h_max, w_max))
        im1 = im1.resize((128, 128))
        im1 = im1.convert("L")
        im1.save(os.path.join(save_img_dir, image))

if __name__ == '__main__' :

    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    main(args)


