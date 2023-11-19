import os
from PIL import Image
data_dir = '1.JPG'
pil = Image.open(data_dir)
print(pil.size)