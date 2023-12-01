import json, os
import numpy as np
from PIL import Image
import cv2

expert_file = 'expert.json'
with open(expert_file, 'r') as f:
    json_data = json.load(f)

circles_mask_base_dir = r'../../medical_data/dental/turf/Expert/mask_circle'
os.makedirs(circles_mask_base_dir, exist_ok=True)
for data in json_data :
    # 1) image name
    img_info = data['External ID']
    name = img_info.split('.')[0]
    # 2) expert opinion
    expert_opinion = data['Description']
    # 3) label info
    label_info = data['Label']['objects']#[0]['polygons']

    mask_dir = os.path.join(r'../../medical_data/dental/turf/Expert/mask', name + '.JPG')
    mask_pil = Image.open(mask_dir)
    circled_mask = mask_pil
    if expert_opinion != 'Within normal limits' :
    #if name == '17' :
        new_mask_list = []
        for info in label_info :
            polygons = info['polygons']
            h, w = np.array(mask_pil).shape
            for polygon in polygons :
                h_index_list = []
                w_index_list = []
                for position in polygon :
                    pos_y, pos_x = position
                    h_index_list.append(pos_y)
                    w_index_list.append(pos_x)
                w_min = min(w_index_list)
                w_max = max(w_index_list)
                h_min = min(h_index_list)
                h_max = max(h_index_list)
                center_w = (w_min + w_max) // 2
                center_h = (h_min + h_max) // 2
                radius_x = (w_max - w_min) // 2
                radius_y = (h_max - h_min) // 2
                circle_mask = np.zeros((h,w), dtype=np.uint8)
                #circle_mask = cv2.ellipse(circle_mask,(center_h, center_w), (radius_y, radius_x), 0, 0, 360, 255, -1)
                circle_mask = cv2.rectangle(circle_mask,
                                            (h_min, w_min),
                                            (h_max, w_max), 255, -1)
                np_mask = np.where(circle_mask < 50, 0, 255).astype(np.uint8)
                #circle_mask = circle_mask.astype(np.uint8) #pil_mask = Image.fromarray(circle_mask) #np_mask = np.array(pil_mask)
                new_mask_list.append(np_mask)
        new_mask_list = np.array(new_mask_list)
        new_mask_np = np.sum(new_mask_list, axis=0)
        new_mask = np.where(new_mask_np < 50, 0, 255)
        circled_mask = Image.fromarray(new_mask)
        circled_mask = circled_mask.convert('L')
    circled_mask.save(os.path.join(circles_mask_base_dir, name + '.JPG'))