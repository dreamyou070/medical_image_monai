import os
from PIL import Image

base_folder = '/data7/sooyeon/medical_image/experiment_data/dental/panoramic_data_res_128'
folder_list = os.listdir(base_folder)
for folder in folder_list :
    folder_dir = os.path.join(base_folder, folder)
    images = os.listdir(folder_dir)
    for image in images :
        image_dir = os.path.join(folder_dir, image)
        pil = Image.open(image_dir)
        pil = pil.resize((128, 128))
        pil.save(image_dir)