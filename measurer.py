import cv2 as cv
import numpy as np
import feature_detection
import glob, random
from imutils import perspective
from scipy.spatial.distance import euclidean
from segmenter import Segmenter
ROOF_RED = (157, 133, 251)
RED = (0, 0, 255)
BLUE = (255, 0, 0)
YELLOW = (0, 255, 255)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)


class Measurer:
    def __init__(self):
        # big red roof size: 105 x 72 ft
        self.dist_in_ft = 105

    # https://github.com/computervisioneng/color-detection-opencv
    def get_limits(self, color):
        if color == "gray" or color == "grey":
            lower = np.array([0,0,0])
            upper = np.array([255, 10, 255])
            return lower,upper

        c = np.uint8([[color]])  # BGR values
        hsvC = cv.cvtColor(c, cv.COLOR_BGR2HSV)

        hue = hsvC[0][0][0]  # Get the hue value

        if hue >= 165:
            lowerLimit = np.array([hue - 10, 100, 100], dtype=np.uint8)
            upperLimit = np.array([180, 255, 255], dtype=np.uint8)
        elif hue <= 15:
            lowerLimit = np.array([0, 100, 100], dtype=np.uint8)
            upperLimit = np.array([hue + 10, 255, 255], dtype=np.uint8)
        else:
            lowerLimit = np.array([hue - 10, 100, 100], dtype=np.uint8)
            upperLimit = np.array([hue + 10, 255, 255], dtype=np.uint8)

        return lowerLimit, upperLimit

    def get_reference(self, img):
        # pre-process image
        hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

        # filter colors within range
        lower, upper = self.get_limits(ROOF_RED)
        mask = cv.inRange(hsv, lower, upper)
        filtered = cv.bitwise_and(img, img, mask=mask)

        # get contours
        ret, thresh = cv.threshold(mask, 127, 255, 0)
        contours, hierarchy = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        # contours = imutils.grab_contours(contours)
        # remove contours which are not big enough
        contours = [x for x in contours if cv.contourArea(x) > 1500]
        # sort by area
        sortedContours = sorted(contours, key=lambda x: cv.contourArea(x), reverse=True)

        # reference: large red roof
        reference = sortedContours[0]
        box = cv.minAreaRect(reference)
        box = cv.boxPoints(box)
        box = np.array(box, dtype="int")
        box = perspective.order_points(box)
        (tl, tr, br, bl) = box
        dist_in_pixel = euclidean(tl, tr)
        pixel_per_ft = dist_in_pixel / self.dist_in_ft

        return pixel_per_ft

    def measure_size(self, img, xyxys, confidences, class_ids, mask_pts):
        pixel_per_ft = self.get_reference(img)
        copy = img.copy()

        for xyxy, conf, id, pts in zip(xyxys, confidences, class_ids, mask_pts):
            cv.fillPoly(copy, pts, YELLOW)  # paint mask
            x1, y1, x2, y2 = xyxy
            cv.rectangle(copy, (x1, y1), (x2, y2), RED, 30)
            w = abs(x1-x2) / pixel_per_ft
            h = abs(y1-y2) / pixel_per_ft
            cv.putText(copy, "{:.2f}ft".format(w), (int(abs(x1+x2)/2), y1-10),
                       cv.FONT_HERSHEY_SIMPLEX, 5, WHITE, 5)
            cv.putText(copy, "{:.2f}ft".format(h), (x2+10, int(abs(y1+y2)/2)),
                       cv.FONT_HERSHEY_SIMPLEX, 5, WHITE, 5)

        cv.imshow('img', copy)
        cv.waitKey(0)

    # https://github.com/noorkhokhar99/Measure-size-of-objects-in-an-image-using-OpenCV/blob/main/size_object.py
    def measure_size_by_contour(self, img):
        pixel_per_ft = self.get_reference(img)

        cntrs = feature_detection.process(path)

        # process all contours
        for cnt in cntrs: #sortedContours:
            # draw bounding box
            box = cv.minAreaRect(cnt)
            box = cv.boxPoints(box)
            box = np.array(box, dtype="int")
            box = perspective.order_points(box)
            (tl, tr, br, bl) = box
            cv.drawContours(img, [box.astype("int")], -1, (0, 255, 255), 5)

            mid_pt_horizontal = (tl[0] + int(abs(tr[0] - tl[0]) / 2), tl[1] + int(abs(tr[1] - tl[1]) / 2))
            mid_pt_vertical = (tr[0] + int(abs(tr[0] - br[0]) / 2), tr[1] + int(abs(tr[1] - br[1]) / 2))

            # find real distance with euclidean distance and reference pixel per ft
            wid = euclidean(tl, tr) / pixel_per_ft
            ht = euclidean(tr, br) / pixel_per_ft
            cv.putText(img, "{:.2f}ft".format(wid), (int(mid_pt_horizontal[0] - 15), int(mid_pt_horizontal[1] - 10)),
                        cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
            cv.putText(img, "{:.2f}ft".format(ht), (int(mid_pt_vertical[0] + 10), int(mid_pt_vertical[1])),
                        cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)

        cv.imshow('img', img)
        cv.waitKey(0)
        cv.destroyAllWindows()


if __name__ == "__main__":
    msr = Measurer()
    seg = Segmenter("best.pt")
    paths = glob.glob("../images/test/stitchedImg36.JPG")
    for path in paths:
        im = cv.imread(path)

        coords, confs, classes, masks = seg.extract(im, 0.4)
        msr.measure_size(im, coords, confs, classes, masks)
