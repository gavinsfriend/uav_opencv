import cv2 as cv
import numpy as np

if __name__ == '__main__':
    img = cv.imread("../stitchedImg34.JPG")
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    blank = np.zeros(img.shape, dtype='uint8')

    # 2 blurring methods. Bilateral filtering is preferred
    blur = cv.GaussianBlur(gray, (5, 5), cv.BORDER_DEFAULT)
    blur2 = cv.bilateralFilter(gray, 9, 75, 75)

    # canny edges
    canny = cv.Canny(blur2, 125, 175)
    contours, hierarchies = cv.findContours(canny, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
    cv.drawContours(blank, contours, -1, (0, 0, 255), 1)

    # threshold
    ret, thresh = cv.threshold(blur2, 125, 255, cv.THRESH_BINARY)
    contours1, hierarchies1 = cv.findContours(thresh, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
    cv.drawContours(blank, contours1, -1, (0, 0, 255), 1)
    cv.imwrite("../thresh_contours.jpg", blank)
