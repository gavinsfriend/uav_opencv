# References:
# https://www.geeksforgeeks.org/python-opencv-bfmatcher-function/#
# https://www.geeksforgeeks.org/removing-black-background-and-make-transparent-using-python-opencv/#
# https://youtu.be/uMABRY8QPe0?si=IkYtBlwBAJHBPMTM
# https://docs.opencv.org/3.1.0/da/df5/tutorial_py_sift_intro.html
# https://pylessons.com/OpenCV-image-stiching
# https://youtu.be/Zs51cg4mb0k?si=oYuz-0Z1Q-kY5sRT
# https://www.geeksforgeeks.org/python-opencv-object-tracking-using-homography/#
import cv2
import glob
import numpy as np


# remove black background and create transparent image. transparent images do not work with stitcher
def transparent(img):
    tmp = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, alpha = cv2.threshold(tmp, 0, 255, cv2.THRESH_BINARY)
    b, g, r = cv2.split(img)
    rgba = [b, g, r, alpha]
    transparent_img = cv2.merge(rgba, 4)
    return transparent_img


def kpAndDescriptor(img, feature_extraction_algo):
    if feature_extraction_algo == "sift":
        algo = cv2.SIFT_create()    # xfeatures2d.SIFT_create()
    elif feature_extraction_algo == "surf":
        algo = cv2.SURF_create()
    elif feature_extraction_algo == "brisk":
        algo = cv2.BRISK_create()
    elif feature_extraction_algo == "orb":
        algo = cv2.ORB_create()

    kp, des = algo.detectAndCompute(img, None)
    return kp, des


def manualStitch(img1, img2):
    feature_extraction_algo = "sift"        # sift, surf, brisk, orb
    match_method = "bf"                     # flann, bf
    gray_resized1 = gray_resize(img1)
    gray_resized2 = gray_resize(img2)
    kp_des1 = kpAndDescriptor(gray_resized1, feature_extraction_algo)
    kp_des2 = kpAndDescriptor(gray_resized2, feature_extraction_algo)

    if match_method == "flann":
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        match = cv2.FlannBasedMatcher(index_params, search_params)
    elif match_method == "bf":
        match = cv2.BFMatcher()
    matches = match.knnMatch(kp_des1[1], kp_des2[1], k=2)

    good = []
    for m, n in matches:
        FILTER = 0.3
        if m.distance < FILTER * n.distance:
            good.append(m)

    # draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
    #                    singlePointColor=None,
    #                    flags=2)
    # img3 = cv2.drawMatches(gray_resized1, kp_des1[0], gray_resized2, kp_des2[0], good, None, **draw_params)
    # cv2.imshow("original_image_drawMatches.jpg", img3)
    # cv2.waitKey(0)

    MIN_MATCH_COUNT = 10
    if len(good) > MIN_MATCH_COUNT:
        src_pts = np.float32([kp_des1[0][m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp_des2[0][m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        h, w, l = img1.shape
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, M)
        pic = cv2.polylines(img2, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)
        # cv2.imshow("original_image_overlapping.jpg", pic)
        # cv2.waitKey(0)
    else:
        print("Not enough matches are found - %d/%d" % (len(good), MIN_MATCH_COUNT))

    dst = cv2.warpPerspective(img1, M, (img2.shape[1] + img1.shape[1], img2.shape[0]))
    dst[0:img2.shape[0], 0:img2.shape[1]] = img2
    cv2.imwrite("stitchedImg.png", dst)
    cv2.imshow("original_image_stitched.jpg", dst)
    cv2.waitKey(0)
    return dst


def trim(frame):
    # crop top
    if not np.sum(frame[0]):
        return trim(frame[1:])
    # crop top
    if not np.sum(frame[-1]):
        return trim(frame[:-2])
    # crop top
    if not np.sum(frame[:, 0]):
        return trim(frame[:, 1:])
    # crop top
    if not np.sum(frame[:, -1]):
        return trim(frame[:, :-2])
    return frame


def gray_resize(img_):
    img_ = cv2.resize(img_, (0,0), fx=.5, fy=.5)  # size down image
    img_ = cv2.cvtColor(img_, cv2.COLOR_BGR2GRAY)
    return img_


def autoStitch(imgs):
    imageStitcher = cv2.Stitcher.create()
    error, stitched = imageStitcher.stitch(images)

    if not error:
        cv2.imwrite("stitchedImg.png", stitched)
        cv2.imshow("stitched", stitched)
        cv2.waitKey(0)
    else:
        if error == 1:
            print("error: 1 ERR_NEED_MORE_IMGS")
        elif error == 2:
            print("error: 2 ERR_HOMOGRAPHY_EST_FAIL")
        elif error == 3:
            print("error: 3 ERR_CAMERA_PARAMS_ADJUST_FAIL")


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

manualStitch(images[0], images[1])
# autoStitch(images)
