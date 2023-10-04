import cv2
import glob
import numpy as np
import warnings
import matplotlib.pyplot as plt


# remove black background and create transparent image
# https://www.geeksforgeeks.org/removing-black-background-and-make-transparent-using-python-opencv/#
def transparent(img):
    tmp = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, alpha = cv2.threshold(tmp, 0, 255, cv2.THRESH_BINARY)
    b, g, r = cv2.split(img)
    rgba = [b, g, r, alpha]
    transparent_img = cv2.merge(rgba, 4)
    return transparent_img


# https://youtu.be/Zs51cg4mb0k?si=oYuz-0Z1Q-kY5sRT
image_paths = glob.glob('images/training/*.JPG')  # '*.png')
images = []

for image in image_paths:
    img = cv2.imread(image)
    img = transparent(img)
    # cv2.imshow("original", img)

    # canny1 = cv2.Canny(img, 125, 175)
    # cv2.imshow("canny before", canny1)

    # blur = cv2.GaussianBlur(img, (3,3), cv2.BORDER_DEFAULT)
    # canny = cv2.Canny(blur, 125,175)
    # cv2.imshow("canny edges after",canny)

    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # cv2.imshow("after cvt", img)

    # cv2.waitKey(0)
    images.append(img)

imageStitcher = cv2.Stitcher.create()
error, stitched = imageStitcher.stitch(images)

if not error:
    # stitched = cv2.cvtColor(stitched, cv2.COLOR_RGB2BGRA)
    transImg = transparent(stitched)
    cv2.imwrite("stitchedImg.png", transImg)
    cv2.imshow("stitched", transImg)
    # img_height, img_width = stitched.shape[:2]
    # print("height: ", img_height, ", width: ", img_width)
    cv2.waitKey(0)
else:
    if error == 1:
        print("error: 1 ERR_NEED_MORE_IMGS")
    elif error == 2:
        print("error: 2 ERR_HOMOGRAPHY_EST_FAIL")
    elif error == 3:
        print("error: 3 ERR_CAMERA_PARAMS_ADJUST_FAIL")

