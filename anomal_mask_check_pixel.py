import os
from PIL import Image
import numpy as np

anomal_folder_mask_dir = r'../medical_data/experiment/dental/Radiographs_L_ood_lowres_mask_'
masks = os.listdir(anomal_folder_mask_dir)
for mask in masks :
    mask_dir = os.path.join(anomal_folder_mask_dir, mask)
    mask_pil = Image.open(mask_dir)
    mask_np = np_mask = np.array(mask_pil)
    print(mask)
    print(mask_np)
    break