import cv2
import numpy as np
from transform_perspective import simulate

# Read input image
src = cv2.imread("./images/test_1.jpg")

# Warp image based on the rotation and translation values
warped = simulate(src, -45, 0, 0, 0, -2500, -2000)

# Create output image
cv2.imwrite("./images/test_1_warped.jpg", warped)

