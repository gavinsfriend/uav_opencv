# References:
# https://www.geeksforgeeks.org/python-opencv-bfmatcher-function/#
# https://www.geeksforgeeks.org/removing-black-background-and-make-transparent-using-python-opencv/#
# https://youtu.be/uMABRY8QPe0?si=IkYtBlwBAJHBPMTM
# https://docs.opencv.org/3.1.0/da/df5/tutorial_py_sift_intro.html
# https://pylessons.com/OpenCV-image-stiching
# https://youtu.be/Zs51cg4mb0k?si=oYuz-0Z1Q-kY5sRT
# https://www.geeksforgeeks.org/python-opencv-object-tracking-using-homography/#
# https://stackoverflow.com/questions/64659657/fast-and-robust-image-stitching-algorithm-for-many-images-in-python
# https://support.pix4d.com/hc/en-us/articles/202560249-TOOLS-GSD-calculator
# https://gis.stackexchange.com/questions/340301/determining-longitude-and-latitude-of-each-corner-of-image
# https://youtu.be/HaGj0DjX8W8?si=Thlcu7J_n2KDM3ud
import pickle, piexif, glob, cProfile, tomllib
import cv2 as cv
import numpy as np
import stitching_detailed
from PIL import Image, ExifTags
# DJI FC3170: Mavic Air 2
# sensor size: 1/2" CMOS, 6.4 x 4.8 mm
# focal length: 24 mm

# image width: 8000, actual distance of width: ~700ft
# GSD = 0.00501    m/pixel or .501 cm/pix
# widthOffset = GSD * (8000/2)    # meters
# heightOffset = GSD * (6000/2)
# widthDis = 97   # m
# wOffset = 0.04090841289


class ImgObj:
    # extract GPS info if image contains it
    def get_GPS(self):
        img = Image.open(self.path)
        try:
            exif = {
                ExifTags.TAGS[k]: v
                for k, v in img._getexif().items()
                if k in ExifTags.TAGS
            }
            return exif["GPSInfo"]
        except:
            return None

    def info_dump(self):
        img = Image.open(self.path)
        try:
            exif = {
                ExifTags.TAGS[k]: v
                for k, v in img._getexif().items()
                if k in ExifTags.TAGS
            }
            return exif
        except:
            return None

    # https://stackoverflow.com/questions/33997361/how-to-convert-degree-minute-second-to-degree-decimal
    # convert coordinates from dms to decimal degree
    def dms_to_dd(self, direction, lat_or_long):
        deg = lat_or_long[0]
        minutes = lat_or_long[1]
        seconds = lat_or_long[2]
        return (float(deg) + float(minutes) / 60 + float(seconds) / (60 * 60)) * (-1 if direction in ['W', 'S'] else 1)

    # extract latitude and longitude from GPS info
    def gps_to_dds(self, gps):
        lat = self.dms_to_dd(gps[1], gps[2])    # gps[1]: N/S, gps[2]: d, m, s
        long = self.dms_to_dd(gps[3], gps[4])   # gps[3]: W/E, gps[4]: d, m, s
        return lat, long

    def __init__(self, path, resize=.7):
        self.pos = None     # left, right, top, bottom
        self.path = path
        temp = cv.imread(path)
        temp = cv.resize(temp, (0, 0), fx=resize, fy=resize)   # size down
        self.image = temp
        self.size = temp.shape[:2]
        self.gpsInfo = self.get_GPS()
        if self.gpsInfo is not None:    # image has embedded GPS info
            self.lat_long = self.gps_to_dds(self.gpsInfo)
            self.alt = self.gpsInfo[6]
        else:
            try:
                pos, lat_long = read_exif(path)
                self.lat_long = lat_long
                self.pos = pos
            except:
                pass
                print("file does not contain exif data")


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


# https://stackoverflow.com/questions/58311162/error-while-trying-to-get-the-exif-tags-of-the-image
# read exif data if image contains any
def read_exif(path):
    img = Image.open(path)
    exif_dict = img.info.get("exif")  # returns None if exif key does not exist
    if exif_dict:
        exif_data = piexif.load(exif_dict)
        raw = exif_data['Exif'][piexif.ExifIFD.UserComment]     # custom data is stored in UserComment
        tags = pickle.loads(raw)
        pos = tags.get("pos")
        lat_long = tags.get("lat_long")
        return pos, lat_long


# TODO: extract makers note
def read_makersNote(path):
    img = Image.open(path)
    exif_dict = img.info.get("exif")  # returns None if exif key does not exist
    if exif_dict:
        exif_data = piexif.load(exif_dict)
        raw = exif_data['Exif'][piexif.ExifIFD.MakerNote]
        newlist = []
        print(raw)
        for i in raw:
            print(i)
            tag = pickle.loads(i)
            newlist.append(tag)
        return newlist


# modify exif data of an image
def modify_exif(imgObj: ImgObj, lat_long, pos=None):
    tags = {"pos": pos, "lat_long": lat_long}
    data = pickle.dumps(tags)
    exif_ifd = {piexif.ExifIFD.UserComment: data}
    exif_dict = {"0th": {}, "Exif": exif_ifd, "1st": {}, "thumbnail": None, "GPS": {}}
    exif_dat = piexif.dump(exif_dict)
    img = Image.open(imgObj.path)
    img.save(imgObj.path, exif=exif_dat)
    imgObj.lat_long = lat_long
    imgObj.pos = pos


# https://stackoverflow.com/questions/12523586/python-format-size-application-converting-b-to-kb-mb-gb-tb
# convert byte to KB/MB/GB/TB
def format_bytes(size):
    # 2**10 = 1024
    power = 2**10
    n = 0
    power_labels = {0 : '', 1: 'kilo', 2: 'mega', 3: 'giga', 4: 'tera'}
    while size > power:
        size /= power
        n += 1
    return size, power_labels[n]+'bytes'


# calculate distance between locations where 2 images are taken
def distance(imgObj1: ImgObj, imgObj2: ImgObj):
    latLong1 = imgObj1.lat_long
    latLong2 = imgObj2.lat_long
    dis = np.sqrt((latLong1[0] - latLong2[0])**2 + (latLong1[1] - latLong2[1])**2)
    return dis


# sort images to aid stitching process
def geo_sort(imgObjs):
    # for i in imgObjs:
    #     print(i.lat_long, " dist: ", distance(imgObjs[0], i))

    # sort by longitude
    # imgObjs = sorted(imgObjs, reverse=True, key=lambda l: l.lat_long[1])

    # sort by distance to the first image, stitching closer images together first
    head = imgObjs[0:1]
    rest = sorted(imgObjs[1:], reverse=False, key=lambda l: distance(imgObjs[0], l))
    imgObjs = head + rest
    return imgObjs


def read_and_create_objs(paths):
    imgObjs = []
    for path in paths:
        img_obj = ImgObj(path)
        imgObjs.append(img_obj)
    return imgObjs


def stitch(feature):
    image_paths = glob.glob('../training/*.*')
    # image_paths = sorted(image_paths)

    # configure initial params
    with open("config.toml", "rb") as f:
        data = tomllib.load(f)
    ft = data[feature]["feature"]
    conf_thresh = data[feature]["conf_thresh"]
    match_conf = data[feature]["match_conf"]

    try:
        resultObj, resultName, imgsUsed = stitching_detailed.stitch(image_paths, conf_thresh=conf_thresh,
                                                                match_conf=match_conf, ft=ft)
        print("new coordinates", resultObj.lat_long, resultObj.pos)
        return resultObj, resultName, imgsUsed
    except:
        print("stitching failed")

    cv.destroyAllWindows()


if __name__ == '__main__':
    cProfile.run('stitch("sift")')
