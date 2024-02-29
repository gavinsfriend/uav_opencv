import cv2 as cv
import imutils
import numpy as np
from imutils import perspective
from scipy.spatial.distance import euclidean
gray = [140, 130, 130]
red = [157, 133, 251]
# red roof size: 105 x 72 ft


# # https://github.com/computervisioneng/color-detection-opencv
def get_limits(color):
    if color == "gray" or color == "grey":
        lower = np.array([0,0,0])
        upper = np.array([255, 10, 255])
        return lower,upper

    c = np.uint8([[color]])  # BGR values
    hsvC = cv.cvtColor(c, cv.COLOR_BGR2HSV)

    hue = hsvC[0][0][0]  # Get the hue value

    if hue >= 165:  # Upper limit for divided red hue
        lowerLimit = np.array([hue - 10, 100, 100], dtype=np.uint8)
        upperLimit = np.array([180, 255, 255], dtype=np.uint8)
    elif hue <= 15:  # Lower limit for divided red hue
        lowerLimit = np.array([0, 100, 100], dtype=np.uint8)
        upperLimit = np.array([hue + 10, 255, 255], dtype=np.uint8)
    else:
        lowerLimit = np.array([hue - 10, 100, 100], dtype=np.uint8)
        upperLimit = np.array([hue + 10, 255, 255], dtype=np.uint8)

    return lowerLimit, upperLimit


# https://github.com/noorkhokhar99/Measure-size-of-objects-in-an-image-using-OpenCV/blob/main/size_object.py
def measure_size(path):
    # pre-process image
    img = cv.imread(path)
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

    # filter colors within range
    lower, upper = get_limits(red)
    mask = cv.inRange(hsv, lower, upper)
    filtered = cv.bitwise_and(img, img, mask=mask)

    # get edges and contours
    edges = cv.Canny(filtered, 50, 100)
    contour = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    contour = imutils.grab_contours(contour)
    # remove contours which are not big enough
    contour = [x for x in contour if cv.contourArea(x) > 3000]
    # sort by area
    sortedContours = sorted(contour, key=lambda x:cv.contourArea(x), reverse=True)

    # reference: large red roof
    reference = sortedContours[0]
    box = cv.minAreaRect(reference)
    box = cv.boxPoints(box)
    box = np.array(box, dtype="int")
    box = perspective.order_points(box)
    (tl, tr, br, bl) = box
    dist_in_pixel = euclidean(tl, tr)
    dist_in_ft = 105
    pixel_per_ft = dist_in_pixel / dist_in_ft

    # process all contours
    for cnt in sortedContours:
        # draw bounding box
        box = cv.minAreaRect(cnt)
        box = cv.boxPoints(box)
        box = np.array(box, dtype="int")
        box = perspective.order_points(box)
        (tl, tr, br, bl) = box
        cv.drawContours(img, [box.astype("int")], -1, (0, 255, 255), 20)

        mid_pt_horizontal = (tl[0] + int(abs(tr[0] - tl[0]) / 2), tl[1] + int(abs(tr[1] - tl[1]) / 2))
        mid_pt_verticle = (tr[0] + int(abs(tr[0] - br[0]) / 2), tr[1] + int(abs(tr[1] - br[1]) / 2))

        # find real distance with euclidean distance and reference pixel per ft
        wid = euclidean(tl, tr) / pixel_per_ft
        ht = euclidean(tr, br) / pixel_per_ft
        cv.putText(img, "{:.1f}ft".format(wid), (int(mid_pt_horizontal[0] - 15), int(mid_pt_horizontal[1] - 10)),
                    cv.FONT_HERSHEY_SIMPLEX, 5, (0, 0, 0), 10)
        cv.putText(img, "{:.1f}ft".format(ht), (int(mid_pt_verticle[0] + 10), int(mid_pt_verticle[1])),
                    cv.FONT_HERSHEY_SIMPLEX, 5, (0, 0, 0), 10)
        # TODO print area?

    cv.imshow('img', img)
    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == "__main__":
    # resultObj, resultName, imgsUsed = stitching.stitch("sift")
    measure_size("stitchedImg.JPG")