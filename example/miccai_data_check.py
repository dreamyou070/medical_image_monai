import os
from PIL import Image
import numpy as np
# {"height":1316,"width":2955,"id":54,"file_name":"train_0.png"}

data_base_dir = r'../../medical_data/DentalAI'

train_data_dir = os.path.join(data_base_dir, 'test')
train_images = os.listdir(train_data_dir)

train_save_data_dir = os.path.join(data_base_dir, 'test_preprocessed')
os.makedirs(train_save_data_dir, exist_ok=True)
thred  = 220

for image in train_images :
    name, ext = os.path.split(image)
    if ext != '.txt' :
        image_dir = os.path.join(train_data_dir, image)
        pil_img = Image.open(image_dir)
        np_img = np.array(pil_img)
        r_channel,g_channel,b_channel = np_img[:,:,0], np_img[:,:,1], np_img[:,:,2]
        h,w = r_channel.shape
        w_list = []
        h_list = []
        for h_index in range(h) :
            for w_index in range(w) :
                r = r_channel[h_index, w_index]
                g = g_channel[h_index, w_index]
                b = b_channel[h_index, w_index]
                if r < thred and g < thred and b < thred :
                    w_list.append(w_index)
                    h_list.append(h_index)
        w_min = min(w_list)
        w_max = max(w_list)
        h_min = min(h_list)
        h_max = max(h_list)
        """
        for h_index in range(h) :
            for w_index in range(w) :
                if r < thred and g < thred and b < thred:
                    pass
                else :
                    if h_index >= h_min and h_index <= h_max and w_index >= w_min and w_index <= w_max :
                        r_channel[h_index, w_index] = 0
                        g_channel[h_index, w_index] = 0
                        b_channel[h_index, w_index] = 0


                else :
        """
        #np_img[:,:,0] = r_channel
        #np_img[:,:,1] = g_channel
        #np_img[:,:,2] = b_channel
        #pil_img = Image.fromarray(np_img)
        im1 = pil_img.crop((w_min, h_min, w_max, h_max))
        im1 = im1.convert("L")
        im1 = im1.resize((64,64))
        im1.save(os.path.join(train_save_data_dir,f'{name}{ext}' ))
