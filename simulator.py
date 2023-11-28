import cv2
import numpy as np
from transform_perspective import simulate


# https://stackoverflow.com/questions/69050464/zoom-into-image-with-opencv
def move(img, angle=0, scale=1, coord=None):
    h, w = img.shape[:2]
    cy, cx = [i / 2 for i in img.shape[:-1]] if coord is None else coord[::-1]
    center = (cx, cy)
    rotation_matrix = cv2.getRotationMatrix2D(center=center, angle=angle, scale=scale)
    rotated = cv2.warpAffine(src=img, M=rotation_matrix, dsize=(w,h))
    return rotated


if __name__ == '__main__':
    # Read input image
    src = cv2.imread("./images/training/DJI_0034.JPG")

    # Warp image based on the rotation and translation values
    warped = simulate(src, -45, 0, 0, 0, -2500, -2000)

    cv2.imshow("warped", warped)
    cv2.waitKey(0)

    h, w = warped.shape[:2]
    zoom = move(warped, scale=1.3, coord=(w*.4, h*.5))

    # Create output image
    # cv2.imwrite("./warped.JPG", warped)
    cv2.imshow("zoom", zoom)
    cv2.waitKey(0)

