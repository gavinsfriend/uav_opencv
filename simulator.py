import cv2
import numpy as np
from transform_perspective import simulate

# Read input image and create output image
src = cv2.imread("./images/test_1.jpg")
dst = np.zeros_like(src)
h, w = src.shape[:2]

warped = simulate(src, dst, w, h, -45, 0, 0, 0, -2500, -2000)

cv2.imwrite("./images/test_1_warped.jpg", warped)
