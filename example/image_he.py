import cv2
import numpy as np
from matplotlib import pyplot as plt
import cv2
import numpy as np
from matplotlib import pyplot as plt
import os
from PIL import Image

base_img_dir = '../../medical_data/dental/turf/Radiographs_hist_crop'
bask_mask_dir = '../../medical_data/dental/turf/Expert/mask_crop'
images = os.listdir(base_img_dir)

train_base_dir = '../../medical_data/dental/turf/panoramic_data_res_256/train'
os.makedirs(train_base_dir, exist_ok=True)
valid_base_dir = '../../medical_data/dental/turf/panoramic_data_res_256/valid'
os.makedirs(valid_base_dir, exist_ok=True)
train_base_dir = '../../medical_data/dental/turf/panoramic_data_res_256/train/original'
os.makedirs(train_base_dir, exist_ok=True)
train_mask_dir = '../../medical_data/dental/turf/panoramic_data_res_256/train/mask'
os.makedirs(train_mask_dir, exist_ok=True)
valid_base_dir = '../../medical_data/dental/turf/panoramic_data_res_256/valid/original'
os.makedirs(valid_base_dir, exist_ok=True)
valid_mask_dir = '../../medical_data/dental/turf/panoramic_data_res_256/valid/mask'
os.makedirs(valid_mask_dir, exist_ok=True)

for i, image in enumerate(images) :
    img_dir = os.path.join(base_img_dir, image)
    mask_dir = os.path.join(bask_mask_dir, image)
    if i < 200 :

        os.rename(img_dir, os.path.join(valid_base_dir, image))
        os.rename(mask_dir, os.path.join(valid_mask_dir, image))
    else :
        os.rename(img_dir, os.path.join(train_base_dir, image))
        os.rename(mask_dir, os.path.join(train_mask_dir, image))


