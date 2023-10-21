# References:
# https://www.geeksforgeeks.org/python-opencv-bfmatcher-function/#
# https://www.geeksforgeeks.org/removing-black-background-and-make-transparent-using-python-opencv/#
# https://youtu.be/uMABRY8QPe0?si=IkYtBlwBAJHBPMTM
# https://docs.opencv.org/3.1.0/da/df5/tutorial_py_sift_intro.html
# https://pylessons.com/OpenCV-image-stiching
# https://youtu.be/Zs51cg4mb0k?si=oYuz-0Z1Q-kY5sRT
# https://www.geeksforgeeks.org/python-opencv-object-tracking-using-homography/#
# https://stackoverflow.com/questions/64659657/fast-and-robust-image-stitching-algorithm-for-many-images-in-python
import cv2 as cv
import glob
import numpy as np
from PIL import Image, ExifTags
# from exif import Image
from operator import sub


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


class ImgObj:
    def get_GPS(self, path):
        img = Image.open(path)
        # exif = img._getexif()
        # print(exif)
        exif = {
            ExifTags.TAGS[k]: v
            for k, v in img._getexif().items()
            if k in ExifTags.TAGS
        }
        # print(exif["GPSInfo"])
        return exif["GPSInfo"]

    # https://stackoverflow.com/questions/33997361/how-to-convert-degree-minute-second-to-degree-decimal
    def dms_to_dd(self, direction, lat_or_long):
        deg = lat_or_long[0]
        minutes = lat_or_long[1]
        seconds = lat_or_long[2]
        return (float(deg) + float(minutes) / 60 + float(seconds) / (60 * 60)) * (-1 if direction in ['W', 'S'] else 1)

    def gps_to_dds(self, gps):
        lat = self.dms_to_dd(gps[1], gps[2])    # gps[1]: N/S, gps[2]: d, m, s
        long = self.dms_to_dd(gps[3], gps[4])   # gps[3]: W/E, gps[4]: d, m, s
        return lat, long

    def __init__(self, path):
        self.path = path
        temp = cv.imread(path)
        self.image = cv.resize(temp, (0, 0), fx=.52, fy=.52)    # size down
        self.gpsInfo = self.get_GPS(path)
        self.lat_long = self.gps_to_dds(self.gpsInfo)
        # cv.imshow("img", self.image)
        # cv.waitKey(0)


def process_all(originals):
    imgs = []
    for img in originals:
        temp = cv.resize(img, (0, 0), fx=.7, fy=.7)  # size down image
        # temp = cv.cvtColor(temp, cv.COLOR_BGR2GRAY)
        imgs.append(temp)
    return imgs


def read_and_create_objs(paths):
    imgObjs = []
    for path in paths:
        img_obj = ImgObj(path)
        imgObjs.append(img_obj)
    return imgObjs


def geo_sort(imgObjs):
    imgObjs = np.array(imgObjs)

    imgObjs = sorted(imgObjs, reverse=True, key=lambda l: l.lat_long[1])
    return imgObjs


def manualStitch(img1, img2):
    def kpAndDescriptor(img_, feature_extraction_algo):
        if feature_extraction_algo == "sift":
            algo = cv.SIFT_create()  # xfeatures2d.SIFT_create()
        elif feature_extraction_algo == "surf":
            algo = cv.SURF_create()
        elif feature_extraction_algo == "brisk":
            algo = cv.BRISK_create()
        elif feature_extraction_algo == "orb":
            algo = cv.ORB_create()

        kp, des = algo.detectAndCompute(img_, None)
        return kp, des

    # find the ROI of a transformation result
    def warpRect(rect, H):
        x, y, w, h = rect
        corners = [[x, y], [x, y + h - 1], [x + w - 1, y], [x + w - 1, y + h - 1]]
        extremum = cv.transform(corners, H)
        minx, miny = np.min(extremum[:, 0]), np.min(extremum[:, 1])
        maxx, maxy = np.max(extremum[:, 0]), np.max(extremum[:, 1])
        xo = int(np.floor(minx))
        yo = int(np.floor(miny))
        wo = int(np.ceil(maxx - minx))
        ho = int(np.ceil(maxy - miny))
        outrect = (xo, yo, wo, ho)
        return outrect

    # homography matrix is translated to fit in the screen
    def coverH(rect, H):
        # obtain bounding box of the result
        x, y, _, _ = warpRect(rect, H)
        # shift amount to the first quadrant
        xpos = int(-x if x < 0 else 0)
        ypos = int(-y if y < 0 else 0)
        # correct the homography matrix so that no point is thrown out
        T = np.array([[1, 0, xpos], [0, 1, ypos], [0, 0, 1]])
        H_corr = T.dot(H)
        return H_corr, (xpos, ypos)

    # pad image to cover ROI, return the shift amount of origin
    def addBorder(img_, rect):
        x, y, w, h = rect
        tl = (x, y)
        br = (x + w, y + h)
        top = int(-tl[1] if tl[1] < 0 else 0)
        bottom = int(br[1] - img_.shape[0] if br[1] > img_.shape[0] else 0)
        left = int(-tl[0] if tl[0] < 0 else 0)
        right = int(br[0] - img_.shape[1] if br[0] > img_.shape[1] else 0)
        img_ = cv.copyMakeBorder(img_, top, bottom, left, right, cv.BORDER_CONSTANT, value=[0, 0, 0])
        orig = (left, top)
        return img_, orig

    def size2rect(size):
        return 0, 0, size[1], size[0]

    def warpImage(img_, H):
        # tweak the homography matrix to move the result to the first quadrant
        H_cover, pos = coverH(size2rect(img_.shape), H)
        # find the bounding box of the output
        x, y, w, h = warpRect(size2rect(img_.shape), H_cover)
        width, height = x + w, y + h
        # warp the image using the corrected homography matrix
        warped = cv.warpPerspective(img_, H_cover, (width, height))
        # make the external boundary solid black, useful for masking
        warped = np.ascontiguousarray(warped, dtype=np.uint8)
        gray = cv.cvtColor(warped, cv.COLOR_RGB2GRAY)
        _, bw = cv.threshold(gray, 1, 255, cv.THRESH_BINARY)
        # https://stackoverflow.com/a/55806272/12447766
        major = cv.__version__.split('.')[0]
        if major == '3':
            _, cnts, _ = cv.findContours(bw, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
        else:
            cnts, _ = cv.findContours(bw, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
        warped = cv.drawContours(warped, cnts, 0, [0, 0, 0], lineType=cv.LINE_4)
        return warped, pos

    def mean_blend(img1, img2):
        assert (img1.shape == img2.shape)
        locs1 = np.where(cv.cvtColor(img1, cv.COLOR_RGB2GRAY) != 0)
        blended1 = np.copy(img2)
        blended1[locs1[0], locs1[1]] = img1[locs1[0], locs1[1]]
        locs2 = np.where(cv.cvtColor(img2, cv.COLOR_RGB2GRAY) != 0)
        blended2 = np.copy(img1)
        blended2[locs2[0], locs2[1]] = img2[locs2[0], locs2[1]]
        blended = cv.addWeighted(blended1, 0.5, blended2, 0.5, 0)
        return blended

    def warpPano(prevPano, img, H, orig):
        # correct homography matrix
        T = np.array([[1, 0, -orig[0]], [0, 1, -orig[1]], [0, 0, 1]])
        H_corr = H.dot(T)
        # warp the image and obtain shift amount of origin
        result, pos = warpImage(prevPano, H_corr)
        xpos, ypos = pos
        # zero pad the result
        rect = (xpos, ypos, img.shape[1], img.shape[0])
        result, _ = addBorder(result, rect)
        # mean value blending
        idx = np.s_[ypos: ypos + img.shape[0], xpos: xpos + img.shape[1]]
        result[idx] = mean_blend(result[idx], img)
        # crop extra paddings
        x, y, w, h = cv.boundingRect(cv.cvtColor(result, cv.COLOR_RGB2GRAY))
        result = result[y: y + h, x: x + w]
        # return the resulting image with shift amount
        return result, (xpos - x, ypos - y)

    # base image is the last image in each iteration
    def blend_multiple_images(images, homographies):
        N = len(images)
        assert (N >= 2)
        assert (len(homographies) == N - 1)
        pano = np.copy(images[0])
        pos = (0, 0)
        for i in range(N - 1):
            img = images[i + 1]
            # get homography matrix
            H = homographies[i]
            # warp pano onto image
            pano, pos = warpPano(pano, img, H, pos)
        return pano, pos

    # no warping here, useful for combining two different stitched images
    # the image at given origin coordinates must be the same
    def patchPano(img1, img2, orig1=(0, 0), orig2=(0, 0)):
        # bottom right points
        br1 = (img1.shape[1] - 1, img1.shape[0] - 1)
        br2 = (img2.shape[1] - 1, img2.shape[0] - 1)
        # distance from orig to br
        diag2 = tuple(map(sub, br2, orig2))
        # possible pano corner coordinates based on img1
        extremum = np.array([(0, 0), br1,
                             tuple(map(sum, zip(orig1, diag2))),
                             tuple(map(sub, orig1, orig2))])
        bb = cv.boundingRect(extremum)
        # patch img1 to img2
        pano, shift = addBorder(img1, bb)
        orig = tuple(map(sum, zip(orig1, shift)))
        idx = np.s_[orig[1]: orig[1] + img2.shape[0] - orig2[1],
              orig[0]: orig[0] + img2.shape[1] - orig2[0]]
        subImg = img2[orig2[1]: img2.shape[0], orig2[0]: img2.shape[1]]
        pano[idx] = mean_blend(pano[idx], subImg)
        return pano, orig

    feature_extraction_algo = "sift"        # sift, surf, brisk, orb
    match_method = "bf"                     # flann, bf
    processed1 = process_single(img1)
    processed2 = cv.resize()
    kp_des1 = kpAndDescriptor(processed1, feature_extraction_algo)
    kp_des2 = kpAndDescriptor(processed2, feature_extraction_algo)

    # raw match
    if match_method == "flann":
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        match = cv.FlannBasedMatcher(index_params, search_params)
    elif match_method == "bf":
        match = cv.BFMatcher()
    matches = match.knnMatch(kp_des1[1], kp_des2[1], k=2)

    # narrow down matches
    good = []
    for m, n in matches:
        RATIO = 0.3
        if m.distance < RATIO * n.distance:
            good.append(m)

    # draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
    #                    singlePointColor=None,
    #                    flags=2)
    # img3 = cv.drawMatches(gray_resized1, kp_des1[0], gray_resized2, kp_des2[0], good, None, **draw_params)
    # cv.imshow("original_image_drawMatches.jpg", img3)
    # cv.waitKey(0)

    MIN_MATCH_COUNT = 10
    if len(good) <= MIN_MATCH_COUNT:
        print("Not enough matches are found - %d/%d" % (len(good), MIN_MATCH_COUNT))
        return

    src_pts = np.float32([kp_des1[0][m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp_des2[0][m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)

    h, w, l = img1.shape
    pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
    dst = cv.perspectiveTransform(pts, M)
    pic = cv.polylines(img2, [np.int32(dst)], True, 255, 3, cv.LINE_AA)
    # cv.imshow("original_image_overlapping.jpg", pic)
    # cv.waitKey(0)

    dst = cv.warpPerspective(img1, M, (img2.shape[1] + img1.shape[1], img2.shape[0]))
    dst[0:img2.shape[0], 0:img2.shape[1]] = img2
    cv.imwrite("stitchedImg.png", dst)
    cv.imshow("original_image_stitched.jpg", dst)
    cv.waitKey(0)
    return dst


def autoStitch(imgs):
    imageStitcher = cv.Stitcher.create()
    error, stitched = imageStitcher.stitch(imgs)

    if not error:
        final = stitched #cv.cvtColor(stitched, cv.COLOR_GRAY2BGR)
        cv.imwrite("stitchedImg.png", final)
        cv.imshow("stitched", final)
        cv.waitKey(0)
    else:
        if error == 1:
            print("error: 1 ERR_NEED_MORE_IMGS")
        elif error == 2:
            print("error: 2 ERR_HOMOGRAPHY_EST_FAIL")
        elif error == 3:
            print("error: 3 ERR_CAMERA_PARAMS_ADJUST_FAIL")


def main():
    image_paths = glob.glob('images/training/*.JPG')  # '*.png')
    img_objs = read_and_create_objs(image_paths)
    # geo_sort(img_objs)
    # processed = process_all(originals)

    # manualStitch(images[1], images[0])
    # autoStitch(processed)


if __name__ == '__main__':
    main()

