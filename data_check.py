import os
from PIL import Image
base_dir = r'experiment/dental/Radiographs_L'

images = os.listdir(base_dir)
for image in images :
    img_dir = os.path.join(base_dir,image)
    pil_img = Image.open(img_dir)
    print(pil_img.size)
    #org_w, org_h = pil_img.size
    #new_w, new_h = 160,84
    #a = pil_img.resize((new_w, new_h))
    #L_pil_image = pil_img.convert('L')
    #L_pil_image.save(os.path.join(save_dir,image))

    # [160,84]
    # [5*323,84*2*5]
    # [323,84*2]
    # 1615,840
