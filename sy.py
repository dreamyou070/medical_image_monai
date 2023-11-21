import os
from PIL import Image

base_folder = '/data7/sooyeon/medical_image/experiment_data/dental/panoramic_data_res_128/train'
folder_list = os.listdir(base_folder)
save_folder = os.path.join('/data7/sooyeon/medical_image/experiment_data/dental/panoramic_data_res_128', 'valid')
os.makedirs(save_folder, exist_ok=True)

for folder in folder_list :
    folder_dir = os.path.join(base_folder, folder)
    new_folder_dir = os.path.join(save_folder, folder)
    os.makedirs(new_folder_dir, exist_ok=True)
    images = os.listdir(folder_dir)
    for i, image in enumerate(images) :
        if i < 200 :
            image_dir = os.path.join(folder_dir, image)
            new_image_dir = os.path.join(new_folder_dir, image)
            os.rename(image_dir, new_image_dir)