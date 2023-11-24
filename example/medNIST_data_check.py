import os
from PIL import Image

hand_base_dir = r'../../medical_data/experiment/MedNIST/Hand_1000'
raw_images = os.listdir(hand_base_dir)
for raw_img in raw_images :
    raw_img_dir = os.path.join(hand_base_dir, raw_img)
    raw_img_pil = Image.open(raw_img_dir)
    
    print(raw_img_pil.size)

