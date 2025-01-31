# References:
# https://youtu.be/HaGj0DjX8W8?si=Thlcu7J_n2KDM3ud
import pickle, piexif, glob, logging
import cv2 as cv
from PIL import Image, ExifTags
from datetime import datetime


class Img:
    """ A class to represent an image, along with all its information.

    Methods:
    --------
    info_dump(): returns all exif data if image contains any
    get_GPS(): extracts and returns GPS info if image contains it
    get_timestamp(): returns timestamp of image as string
    dms_to_dd(direction, lat_or_long): converts coordinates from dms to decimal degree
    gps_to_dds(gps): extracts and returns latitude and longitude (in dd format) from GPS info (in exif)
    read_exif(): reads and returns custom exif data if image contains any
    """

    def info_dump(self) -> dict|None:
        """Returns the entire EXIF data if the image contains any.

        Returns:
        --------
            dict: EXIF data of the image, or
            None: if image does not contain any EXIF data
        """
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

    def get_GPS(self) -> dict|None:
        """ Extracts and returns GPS info if image contains it in EXIF data.

        Returns:
        --------
            dict: GPS information of the image, or
            None: if image does not contain GPS info
        """
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

    def get_timestamp(self) -> datetime|None:
        """Get timestamp of the image.

        Returns:
        --------
            datetime: timestamp of the image as a datetime object
        """
        img = Image.open(self.path)
        try:
            exif = {
                ExifTags.TAGS[k]: v
                for k, v in img._getexif().items()
                if k in ExifTags.TAGS
            }
            dt_obj = datetime.strptime(exif["DateTime"], "%Y:%m:%d %H:%M:%S")
            return dt_obj
        except:
            return None

    def dms_to_dd(self, direction: str, lat_or_long: tuple) -> float:
        """Convert coordinates from Degree, Minutes, Seconds to Decimal Degree.
        https://stackoverflow.com/questions/33997361/how-to-convert-degree-minute-second-to-degree-decimal

        Args:
        -----
            direction (str): 'N'/'S' or 'W'/'E'
            lat_or_long (tuple): latitude or longitude in DMS format

        Returns:
        --------
            float: coordinates in Decimal Degree format
        """
        deg = lat_or_long[0]
        minutes = lat_or_long[1]
        seconds = lat_or_long[2]
        return (float(deg) + float(minutes) / 60 + float(seconds) / (60 * 60)) * (-1 if direction in ['W', 'S'] else 1)

    def gps_to_dds(self, gps: dict):
        """Given GPS info, extract latitude and longitude, then return them as a tuple in Decimal Degree format.

        Args:
        -----
            gps (dict): GPS info

        Returns:
        --------
            float: latitude in Decimal Degree format
            float: longitude in Decimal Degree format
        """
        lat = self.dms_to_dd(gps[1], gps[2])    # gps[1]: N/S, gps[2]: d, m, s
        long = self.dms_to_dd(gps[3], gps[4])   # gps[3]: W/E, gps[4]: d, m, s
        return lat, long

    def read_exif(self) -> tuple|None:
        """Read custom EXIF data (in UserComment) if image contains any
        https://stackoverflow.com/questions/58311162/error-while-trying-to-get-the-exif-tags-of-the-image

        Returns:
        --------
            tuple: information stored in UserComment (i.e. pos, lat_long), or
            None: if image does not contain any custom EXIF data
        """
        img = Image.open(self.path)
        exif_dict = img.info.get("exif")  # returns None if exif key does not exist
        if exif_dict:
            exif_data = piexif.load(exif_dict)
            raw = exif_data['Exif'][piexif.ExifIFD.UserComment]  # custom data is stored in UserComment
            tags = pickle.loads(raw)
            pos = tags.get("pos")
            lat_long = tags.get("lat_long")
            return pos, lat_long
        else:
            return None

    def __init__(self, filepath: str, resize=1):
        """Initializes an image object with the given path and resize factor.

        Args:
        -----
            filepath (str): file path of the image
            resize (float, optional): resize factor if image size is too big. Set to 1 for no resizing.
        
        Class Attributes:
        -----------------
            path (str): file path
            timestamp (datetime): timestamp
            image (np.ndarray): image data
            pos (str): position (left, right, top, bottom)
            alt (float): altitude
            gpsInfo (dict): GPS information
            lat_long (tuple): latitude and longitude (in Decimal Degree format)
        """

        self.path = filepath
        print("initializing image object for", self.path)
        self.timestamp = self.get_timestamp()
        temp = cv.imread(self.path)
        temp = cv.resize(temp, (0, 0), fx=resize, fy=resize)
        self.image = temp
        self.pos = None
        self.alt = None
        self.gpsInfo = self.get_GPS()
        if self.gpsInfo is not None:    # check if image has embedded GPS info
            self.lat_long = self.gps_to_dds(self.gpsInfo)
            self.alt = self.gpsInfo[6]
        else:
            try:    # check if image has custom exif data
                pos, lat_long = self.read_exif()
                self.lat_long = lat_long
                self.pos = pos
            except:
                print("file does not contain exif data")
                self.lat_long = None


if __name__ == "__main__":
    paths = glob.glob("../images/imgs/DJI_0040.JPG")
    for path in paths:
        img_obj = Img(path)
        print(img_obj.timestamp.year, "\n", type(img_obj.timestamp.year))
