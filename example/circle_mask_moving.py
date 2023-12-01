import os
from PIL import Image
base_dir = r"../../medical_data/dental/turf/Expert/mask_circle"
train_mask_base_dir = r"../../medical_data/dental/turf/panoramic_data_res_256/train/mask"
train_mask_base_dir_record = r"../../medical_data/dental/turf/panoramic_data_res_256/train/mask_rectangle_by_record"
os.makedirs(train_mask_base_dir_record, exist_ok=True)
masks = os.listdir(train_mask_base_dir)
for mask in masks:
    name, ext = os.path.splitext(mask)
    big_mask_dir = os.path.join(base_dir, mask)
    big_mask = Image.open(big_mask_dir).convert("L").resize((256, 256))
    big_mask.save(os.path.join(train_mask_base_dir_record, mask))

train_mask_base_dir = r"../../medical_data/dental/turf/panoramic_data_res_256/valid/mask"
train_mask_base_dir_record = r"../../medical_data/dental/turf/panoramic_data_res_256/valid/mask_rectangle_by_record"
os.makedirs(train_mask_base_dir_record, exist_ok=True)
masks = os.listdir(train_mask_base_dir)
for mask in masks:
    name, ext = os.path.splitext(mask)
    big_mask_dir = os.path.join(base_dir, mask)
    big_mask = Image.open(big_mask_dir).convert("L").resize((256, 256))
    big_mask.save(os.path.join(train_mask_base_dir_record, mask))
