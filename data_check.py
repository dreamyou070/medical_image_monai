import os, argparse, ast
import json
from PIL import Image


def main(args) :

    print(f'\n step 1. image check')
    img_base_dir = r'../medical_Data/experiment/dental/Radiographs_L'

    normal_img_base_dir = r'../medical_Data/experiment/dental/Radiographs_L_normal_lowres_'
    os.makedirs(normal_img_base_dir, exist_ok=True)

    odd_img_base_dir = r'../medical_Data/experiment/dental/Radiographs_L_odd_lowres'
    os.makedirs(odd_img_base_dir, exist_ok=True)

    print(f'\n step 2. writing check')
    ood_text = 'ood_text.txt'

    expert_memo_dir = r'../medical_Data/experiment/dental/Expert/expert.json'
    with open(expert_memo_dir, "r") as st_json:
        memo_data = json.load(st_json)
    ood_dict = {}
    for data in memo_data :
        description = data['Description'].strip()
        image_name = data['External ID']
        org_dir = os.path.join(img_base_dir, image_name)
        pil_img = Image.open(org_dir)
        W, H = pil_img.size
        resized_image = pil_img.resize((64,64))

        main_name = image_name.split('.')[0]
        #diff = int((W - H) / 2)
        #background = Image.new('L', (W, W), (0))
        #background.paste(pil_img, (0, diff))
        if description == 'Within normal limits' :

            new_dir = os.path.join(normal_img_base_dir, image_name)
            resized_image.save(new_dir)
            new_dir = os.path.join(normal_img_base_dir, f'{main_name}_11.JPG')
            resized_image.save(new_dir)
            new_dir = os.path.join(normal_img_base_dir, f'{main_name}_12.JPG')
            resized_image.save(new_dir)
            new_dir = os.path.join(normal_img_base_dir, f'{main_name}_13.JPG')
            resized_image.save(new_dir)
            #os.rename(org_dir, new_dir)
        #else :
        #    new_dir = os.path.join(odd_img_base_dir, image_name)
        #    resized_image.save(new_dir)
            #background.save(new_dir)
         #   name = image_name.split('.')[0]
          #  ood_dict[name] = description
    #sorted_ood_dict = sorted(ood_dict.items(), key=lambda x: x[0])
    #with open(ood_text, 'w') as f :
    #    for key, value in sorted_ood_dict :
    #        f.write(f'{key}.JPG : {value}\n')


if __name__ == '__main__' :
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    main(args)
