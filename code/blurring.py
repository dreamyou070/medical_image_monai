import numpy as np
import argparse
import cv2
from matplotlib import pyplot as plt
def main(args) :

    print(f'\n step 1. image check')
    image = cv2.imread('1.jpg')
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.show()

    print(f'\n step 2. kernel (sum = 1)')
    print(f' (2.1) basic kernel')
    size = 4
    basic_kernel = np.ones((size, size), dtype=np.float32) / (size * size)
    print(f' basic_kernel : {basic_kernel}')
    dst = cv2.filter2D(image, -1, basic_kernel)
    plt.imshow(cv2.cvtColor(dst, cv2.COLOR_BGR2RGB))
    plt.show()

    print(f' (2.2) gaussian kernel')
    gaussian_kernel = np.array([[1/16, 1/8, 1/16],
                                [1/8,  1/4, 1/8],
                                [1/16, 1/8, 1/16]])
    dst = cv2.filter2D(image, -1, gaussian_kernel)
    plt.imshow(cv2.cvtColor(dst, cv2.COLOR_BGR2RGB))
    plt.show()




if __name__ == '__main__' :
    parser = argparse.ArgumentParser()
    parser.add_argument('--mean', type=float, default=0.0)
    parser.add_argument('--var', type=float, default=1.0)
    args = parser.parse_args()
    main(args)

