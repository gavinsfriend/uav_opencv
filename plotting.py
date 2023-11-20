from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image, ExifTags
import matplotlib.pyplot as plt
import stitching
import cv2 as cv
import glob
import numpy as np
coords = []
fig, ax = plt.subplots()


def getImage(path, zoom=1):
    img = Image.open(path)  # plt.imread(path)
    img = img.transpose(Image.ROTATE_180)
    return OffsetImage(img, zoom=0.02)


def plotImgs(paths):
    images = []
    for path in paths:
        imgObj = stitching.ImgObj(path=path)
        images.append(imgObj)

    lat = [i.lat_long[0] for i in images]
    long = [i.lat_long[1] for i in images]

    ax.scatter(long, lat)

    for x0, y0, path in zip(long, lat, paths):
        ab = AnnotationBbox(getImage(path), (x0, y0), frameon=False)
        ax.add_artist(ab)

    plt.show()


def plotStitched(path):
    imgObj = stitching.ImgObj(path=path, resize=1)
    print(stitching.read_exif(path))
    lat = imgObj.lat_long[0]
    long = imgObj.lat_long[1]

    ax.scatter(long, lat)
    ab = AnnotationBbox(getImage(path, zoom=1), (long, lat), frameon=False)
    ax.add_artist(ab)
    ax.set_ylim(ymin=imgObj.pos[3])
    plt.show()


# https://stackoverflow.com/questions/25521120/store-mouse-click-event-coordinates-with-matplotlib
def onclick(event):
    global ix, iy
    ix, iy = event.xdata, event.ydata
    print(f'x = {ix}, y = {iy}')
    global coords
    coords.append((ix, iy))


if __name__ == '__main__':
    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    image_path = glob.glob('images/training/*.*')
    plotImgs(image_path)
    fig.canvas.mpl_disconnect(cid)
