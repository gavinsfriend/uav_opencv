import cv2
import glob
import numpy as np
import warnings
import matplotlib.pyplot as plt
# import imageio
# cv2.ocl.setUseOpenCL(False)
# warnings.filterwarnings('ignore')

# https://youtu.be/Zs51cg4mb0k?si=oYuz-0Z1Q-kY5sRT
image_paths = glob.glob('images/training/*.JPG')#'*.png')
images = []

for image in image_paths:
    img = cv2.imread(image)
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

    # https://stackoverflow.com/questions/44595160/create-transparent-image-in-opencv-python
    img_height, img_width = 300, 300
    n_channels = 4
    transparent_img = np.zeros((img_height, img_width, n_channels), dtype=np.uint8)

    # Save the image for visualization
    cv2.imwrite("./transparent_img.png", transparent_img)

    cv2.imwrite("stitchedImg.png",stitched)
    cv2.imshow("stitched",stitched)
    cv2.waitKey(0)
else:
    if error == 1:
        print("error: 1 ERR_NEED_MORE_IMGS")
    elif error == 2:
        print("error: 2 ERR_HOMOGRAPHY_EST_FAIL")
    elif error == 3:
        print("error: 3 ERR_CAMERA_PARAMS_ADJUST_FAIL")

