import cv2
import glob
import numpy as np
import warnings
import matplotlib.pyplot as plt


# remove black background and create transparent image. transparent images do not work with stitcher
# https://www.geeksforgeeks.org/removing-black-background-and-make-transparent-using-python-opencv/#
def transparent(img):
    tmp = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, alpha = cv2.threshold(tmp, 0, 255, cv2.THRESH_BINARY)
    b, g, r = cv2.split(img)
    rgba = [b, g, r, alpha]
    transparent_img = cv2.merge(rgba, 4)
    return transparent_img


def kpAndDescription(imgs, feature_extraction_algo):
    kp_des_list = []
    for img in imgs:
        if feature_extraction_algo == "sift":
            algo = cv2.SIFT_create()    # xfeatures2d.SIFT_create()
        elif feature_extraction_algo == "surf":
            algo = cv2.xfeatures2d.SURF_create()
        elif feature_extraction_algo == "brisk":
            algo = cv2.xfeatures2d.BRISK_create()
        elif feature_extraction_algo == "orb":
            algo = cv2.xfeatures2d.ORB_create()

        (kp, des) = algo.detectAndCompute(img, None)
        kp_des_list.append((kp, des))
    return kp_des_list
    # TODO: kp_des_list element NOT TUPLES. PLEASE FIX


# https://pylessons.com/OpenCV-image-stiching
# kpdes1, kpdes2 are tuples
def match(kpdes1, kpdes2, method):
    if method != "flann" or method != "bf":
        print("invalid matching method")
        return

    matches = None
    print("kpdes1"+kpdes1)
    if method == "flann":
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        match = cv2.FlannBasedMatcher(index_params, search_params)
        matches = match.knnMatch(kpdes1[1], kpdes2[1], k=2)
    elif method == "bf":
        match = cv2.BFMatcher()
        matches = match.knnMatch(kpdes1[1], kpdes2[1], k=2)

    good = []
    for m, n in matches:
        if m.distance < 0.03 * n.distance:
            good.append(m)

    return good


# https://youtu.be/uMABRY8QPe0?si=IkYtBlwBAJHBPMTM
# https://docs.opencv.org/3.1.0/da/df5/tutorial_py_sift_intro.html
# https://pylessons.com/OpenCV-image-stiching
def manualStitch(imgs):
    gray_resized = []
    for img in imgs:
        img1 = cv2.resize(img, (0,0), fx=.6, fy=1)  # size down image
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray_resized.append(img1)

    feature_extraction_algo = "sift"
    kp_des_list = kpAndDescription(imgs, feature_extraction_algo)
    match_method = "bf"
    print(kp_des_list)
    good = match(kp_des_list[0], kp_des_list[1], match_method)

    draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                       singlePointColor=None,
                       flags=2)

    img3 = cv2.drawMatches(imgs[0], kp_des_list[0], imgs[1], kp_des_list[1], good, None, **draw_params)
    cv2.imshow("original_image_drawMatches.jpg", img3)
    cv2.waitKey(0)


# https://youtu.be/Zs51cg4mb0k?si=oYuz-0Z1Q-kY5sRT
image_paths = glob.glob('images/training/*.JPG')  # '*.png')
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

    # imageStitcher = cv2.Stitcher.create()
    # error, stitched = imageStitcher.stitch(images)
    manualStitch(images)

# if not error:
#     # stitched = cv2.cvtColor(stitched, cv2.COLOR_RGB2BGRA)
#     cv2.imwrite("stitchedImg.png", stitched)
#     cv2.imshow("stitched", stitched)
#     # img_height, img_width = stitched.shape[:2]
#     # print("height: ", img_height, ", width: ", img_width)
#     cv2.waitKey(0)
# else:
#     if error == 1:
#         print("error: 1 ERR_NEED_MORE_IMGS")
#     elif error == 2:
#         print("error: 2 ERR_HOMOGRAPHY_EST_FAIL")
#     elif error == 3:
#         print("error: 3 ERR_CAMERA_PARAMS_ADJUST_FAIL")

