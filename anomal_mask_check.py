import argparse, os
from PIL import Image
def main(args) :

    expert_mask_folder_dir = r'../medical_data/experiment/dental/Expert/mask'
    anomal_folder_dir = r'../medical_data/experiment/dental/Radiographs_L_ood_lowres'
    anomal_images = os.listdir(anomal_folder_dir)

    anomal_folder_mask_dir = r'../medical_data/experiment/dental/Radiographs_L_ood_lowres_mask_'
    os.makedirs(anomal_folder_mask_dir, exist_ok=True)
    for anomal_image in anomal_images :
        anomal_dir = os.path.join(anomal_folder_dir, anomal_image)
        mask_dir = os.path.join(expert_mask_folder_dir, anomal_image)
        mask_pil = Image.open(mask_dir)
        mask_pil = mask_pil.convert('L')
        reshaped_mask_pil = mask_pil.resize((64,64))
        name = anomal_image.split('.')[0]
        reshaped_mask_pil.save(os.path.join(anomal_folder_mask_dir, f'{name}_mask.JPG'))


if __name__ == '__main__' :
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    main(args)