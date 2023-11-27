import cv2
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image

org_img_dir = '48.JPG'

img = cv2.imread(org_img_dir, cv2.IMREAD_GRAYSCALE)
org_img = Image.open(org_img_dir).convert("L")
org_img = org_img.resize((256,256))
org_img.show()
clahe = cv2.createCLAHE(clipLimit=0.5, tileGridSize=(4,4))
img2 = clahe.apply(img)
re_img = Image.fromarray(img2)
re_img = re_img.resize((256,256))
re_img.show()
"""
# contrast limit가 2이고 title의 size는 8X8


img = cv2.resize(img,(400,400))
img2 = cv2.resize(img2,(400,400))
dst = np.hstack((img, img2))
cv2.imshow('img',dst)
cv2.waitKey()
cv2.destroyAllWindows()
"""