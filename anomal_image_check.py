import os
from PIL import Image
import numpy as np
import torch

#basic_dir = r'../medical_data/experiment/dental/panoramic_data'
#img_folder = os.path.join(basic_dir, 'original')
img_folder = r'../medical_data/experiment/dental/Radiographs'
mask_folder = r'../medical_data/experiment/dental/Expert/mask'
images = os.listdir(img_folder)
for image in images :
    mask_dir = os.path.join(mask_folder, image)
    mask_pil = Image.open(mask_dir)
    mask_np = np.array(mask_pil)
    criterion = np.sum(mask_np)
    normal = True
    if criterion > 0:
        normal = False
    mask_torch = torch.from_numpy(mask_np)
    if not normal :
        image_dir = os.path.join(img_folder, image)
        image_pil = Image.open(image_dir)
        image_np = np.array(image_pil)
        print(image_np.shape)

        mask_r_channel = mask_np
        mask_g_channel = np.zeros_like(mask_np)
        mask_b_channel = np.zeros_like(mask_np)
        alpha_channel = np.zeros_like(mask_np)
        mask_rgb = np.stack([mask_r_channel, mask_g_channel, mask_b_channel, alpha_channel], axis=2)
        red_mask = Image.fromarray(mask_rgb)
        red_mask.show()



        #new = Image.alpha_composite(image_pil, red_mask)
        #Image.merge("RGB", (image_pil, red_mask)).show()
        #new.show()
        #print(mask_rgb.shape)
        merged_img =  Image.alpha_composite(image_pil.convert("RGBA"), red_mask.convert("RGBA")).show()
        """
        W, H = image_pil.size
        mask_W, mask_H = mask_pil.size
        image_np = np.array(image_pil)
        masked_image_np = image_np * mask_np
        W, H = masked_image_np.shape
        masked_pil = Image.fromarray(masked_image_np)
        masked_pil.show()
        image_pil.show()
        """
        break