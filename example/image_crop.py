import cv2
import numpy as np
from matplotlib import pyplot as plt
import cv2
import numpy as np
from matplotlib import pyplot as plt
import os
from PIL import Image

base_img_dir = '../../medical_data/dental/turf/Radiographs'
save_base_img_dir = '../../medical_data/dental/turf/Radiographs_hist_crop'
os.makedirs(save_base_img_dir, exist_ok=True)

bask_mask_dir = '../../medical_data/dental/turf/Expert/mask'
save_mask_dir = '../../medical_data/dental/turf/Expert/mask_crop'
os.makedirs(save_mask_dir, exist_ok=True)

images = os.listdir(base_img_dir)
for image in images :
    img_dir = os.path.join(base_img_dir, image)
    # 1. cropping
    pil_img = Image.open(img_dir).convert("L")
    width, height = pil_img.size
    left = 75
    top = 25
    right = width - left
    bottom = height - top
    resized_pil_img = pil_img.crop((left, top, right, bottom))
    np_img = np.array(resized_pil_img)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img2 = clahe.apply(np_img)
    # 2. resize
    img2 = Image.fromarray(img2).convert("L")
    img2 = img2.resize((256,256))
    img2.save(os.path.join(save_base_img_dir, image))

    mask_dir = os.path.join(bask_mask_dir, image)
    pil_mask = Image.open(mask_dir).convert("L")
    resized_pil_mask = pil_mask.crop((left, top, right, bottom))
    resized_pil_mask = resized_pil_mask.resize((256, 256))
    resized_pil_mask.save(os.path.join(save_mask_dir, image))
