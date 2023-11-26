import os
from PIL import Image
base_img_dir = r'../../medical_data/dental/turf/Radiographs'
base_mask_dir =r'../../medical_data/dental/turf/Expert/mask'
images = os.listdir(base_img_dir)

train_base_dir = r'../../medical_data/dental/turf/train'
os.makedirs(train_base_dir, exist_ok=True)
train_base_image_dir = r'../../medical_data/dental/turf/train/original'
os.makedirs(train_base_image_dir, exist_ok=True)
train_base_mask_dir = r'../../medical_data/dental/turf/train/mask'
os.makedirs(train_base_mask_dir, exist_ok=True)

test_base_dir = r'../../medical_data/dental/turf/valid'
os.makedirs(test_base_dir, exist_ok=True)
test_base_image_dir = r'../../medical_data/dental/turf/valid/original'
os.makedirs(test_base_image_dir, exist_ok=True)
test_base_mask_dir = r'../../medical_data/dental/turf/valid/mask'
os.makedirs(test_base_mask_dir, exist_ok=True)

for i, image in enumerate(images) :

    img_dir = os.path.join(base_img_dir,image)
    pil_img = Image.open(img_dir).convert('L')
    pil_img = pil_img.resize((512,512))
    mask_dir = os.path.join(base_mask_dir, image)
    pil_mask = Image.open(mask_dir).convert('L')
    pil_mask = pil_mask.resize((512,512))

    if i < 200 :
        pil_img.save(os.path.join(test_base_image_dir, image))
        pil_mask.save(os.path.join(test_base_mask_dir, image))

    else :
        pil_img.save(os.path.join(train_base_image_dir, image))
        pil_mask.save(os.path.join(train_base_mask_dir, image))
