import glob, cv2
import numpy as np
num = 0


# crop and remove black background
# https://stackoverflow.com/questions/51656362/how-do-i-crop-the-black-background-of-the-image-using-opencv-in-python
def crop(path):
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)
    x, y, w, h = cv2.boundingRect(thresh)
    foreground = img[y:y+h, x:x+w]
    global num
    num += 1
    dst = './images/dewarped_cropped/img' + str(num) + '.JPG'
    cv2.imwrite(dst, foreground)
    return foreground


if __name__ == '__main__':
    src = glob.glob("./*.*")
    for i in src:
        crop(i)


