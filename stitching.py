# References:
# https://support.pix4d.com/hc/en-us/articles/202560249-TOOLS-GSD-calculator
# https://gis.stackexchange.com/questions/340301/determining-longitude-and-latitude-of-each-corner-of-image
import pickle, piexif, glob, cProfile, tomllib
import cv2 as cv
import numpy as np
import stitching_detailed
from PIL import Image
import image
# DJI FC3170: Mavic Air 2
# sensor size: 1/2" CMOS, 6.4 x 4.8 mm
# focal length: 24 mm

# image width: 8000, actual distance of width: ~700ft
# GSD = 0.00501    m/pixel or .501 cm/pix
# widthOffset = GSD * (8000/2)    # meters
# heightOffset = GSD * (6000/2)
# widthDis = 97   # m
# wOffset = 0.04090841289


# create position and GPS data for newly stitched images
def create_gps(imgObjs):
    left = min(imgObjs, key=lambda l: l.lat_long[1]).lat_long[1]
    right = max(imgObjs, key=lambda l: l.lat_long[1]).lat_long[1]
    top = max(imgObjs, key=lambda l: l.lat_long[0]).lat_long[0]
    bottom = min(imgObjs, key=lambda l: l.lat_long[0]).lat_long[0]
    lat, long = (top+bottom)/2, (left+right)/2
    # sum_alt = sum(imgObjs, key=lambda l: l.alt)
    # print(sum_alt)
    pos = (left, right, bottom, top)
    lat_long = lat, long
    return pos, lat_long

# modify exif data of an image
def modify_exif(imgObj: image.Img, lat_long, pos=None):
    tags = {"pos": pos, "lat_long": lat_long}
    data = pickle.dumps(tags)
    exif_ifd = {piexif.ExifIFD.UserComment: data}
    exif_dict = {"0th": {}, "Exif": exif_ifd, "1st": {}, "thumbnail": None, "GPS": {}}
    exif_dat = piexif.dump(exif_dict)
    img = Image.open(imgObj.path)
    img.save(imgObj.path, exif=exif_dat)
    imgObj.lat_long = lat_long
    imgObj.pos = pos


def format_bytes(size):
    """Converts bytes to KB/MB/GB/TB
    https://stackoverflow.com/questions/12523586/python-format-size-application-converting-b-to-kb-mb-gb-tb

    Args:
    -----
        size: file size in bytes
    Returns:
    --------
        tuple: size in KB/MB/GB/TB, and the corresponding unit
    """
    # 2**10 = 1024
    power = 2**10
    n = 0
    power_labels = {0 : '', 1: 'kilo', 2: 'mega', 3: 'giga', 4: 'tera'}
    while size > power:
        size /= power
        n += 1
    return size, power_labels[n]+'bytes'


def distance(imgObj1: image.Img, imgObj2: image.Img):
    """Calculate distance between locations where 2 images are taken

    Args:
    -----
        imgObj1 (Img.Img): Image object 1
        imgObj2 (Img.Img): Image object 2

    Returns:
    --------
        dis (): _description_
    """
    latLong1 = imgObj1.lat_long
    latLong2 = imgObj2.lat_long
    dis = np.sqrt((latLong1[0] - latLong2[0])**2 + (latLong1[1] - latLong2[1])**2)
    print("type of dis is: " + type(dis))
    return dis


def time_sort(imgObjs):
    """Sort images by time taken.

    Args:
    -----
        imgObjs (list): list of image objects to be sorted
    Returns:
    --------
        list: sorted list of image objects
    """
    newObjs = sorted(imgObjs, key=lambda l: l.timestamp)
    return newObjs


def geo_sort(imgObjs):
    """Sort images to aid stitching process. Sort by distance to the first image, stitching closer images together first

    Args:
    -----
        imgObjs (list): list of image objects to be sorted

    Returns:
    --------
        imgObjs (list): sorted list of image objects
    """
    # sort by longitude
    # imgObjs = sorted(imgObjs, reverse=True, key=lambda l: l.lat_long[1])
    
    head = imgObjs[0:1]
    rest = sorted(imgObjs[1:], reverse=False, key=lambda l: distance(imgObjs[0], l))
    newObjs = head + rest
    return newObjs


def stitch(paths: str, feature: str):
    """Given the paths to images and the selected feature extraction method, stitch multiple images together to
    create a single large image.

    Args:
    -----
        paths (str): path to the images to be stitched (e.g. './images/*.*')
        feature (str): feature extraction method (e.g. 'orb', 'sift', 'brisk', 'akaze')

    Returns:
    --------
        resultObj: the stitched image object, 
        resultName: file name of the stitched image, 
        imgsUsed: image objects used in the stitching process
    """
    image_paths = glob.glob(paths)

    # configure initial params
    with open("config.toml", "rb") as f:
        data = tomllib.load(f)
    ft = data[feature]["feature"]
    conf_thresh = data[feature]["conf_thresh"]
    match_conf = data[feature]["match_conf"]
    print("feature:", feature, ", conf_thresh:", conf_thresh, ", match_conf:", match_conf)

    try:
        # FIXME: stitcher generating split images?
        resultObj, resultName, imgsUsed = stitching_detailed.stitch(image_paths, conf_thresh, match_conf, ft)
        print("new coordinates:", resultObj.lat_long, resultObj.pos)
        return resultObj, resultName, imgsUsed
    except:
        print("stitching failed")

    cv.destroyAllWindows()


if __name__ == '__main__':
    # cProfile.run('stitch("sift")')
    # stitch(img_paths, "brisk")
    img_paths = "../images/imgs/*.*"
    stitch(img_paths, "sift")

