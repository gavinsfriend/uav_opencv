import cv2 as cv
import numpy as np


def warp(src: np.ndarray, rotXval, rotYval, rotZval, distXval, distYval, distZval) -> np.ndarray:
    """https://stackoverflow.com/questions/45811421/python-create-image-with-new-camera-position

    Args:
    -----
        src (np.ndarray): source image data
        rotXval: rotate along the x-axis. pitch(Y) -- dec->up, inc->down
        rotYval: rotate along the y-axis. yaw(Z) pan -- dec->right, inc->left
        rotZval: rotate along the z-axis. roll(X) tilt -- dec->left, inc->right
        distXval: X translation -- dec->left, inc->right
        distYval: Y translation -- dec->up, inc->down
        distZval: Z translation -- dec->forward/zoom in, inc->backward/zoom out

    Returns:
    --------
        np.ndarray: warped image data
    """

    # Initalize destination of transformation
    dst = np.zeros_like(src)
    
    # Initialize height and width
    h, w = src.shape[:2]
    
    # Initialize transformations
    f = (max(w, h)/2) / np.tan(np.pi*45/180)    # assumption: FOV = 90
    rotX = rotXval*np.pi/180
    rotY = rotYval*np.pi/180
    rotZ = rotZval*np.pi/180
    distX = distXval
    distY = distYval
    distZ = distZval

    # Camera intrinsic matrix (K)
    K = np.array([[f, 0, w/2, 0],
                  [0, f, h/2, 0],
                  [0, 0,   1, 0]])

    # Find K inverse
    Kinv = np.zeros((4,3))
    Kinv[:3,:3] = np.linalg.inv(K[:3,:3])*f
    Kinv[-1,:] = [0, 0, 1]

    # Rotation matrices around the X,Y,Z axis
    RX = np.array([[1,           0,            0, 0],
                   [0,np.cos(rotX),-np.sin(rotX), 0],
                   [0,np.sin(rotX), np.cos(rotX), 0],
                   [0,           0,            0, 1]])

    RY = np.array([[ np.cos(rotY), 0, np.sin(rotY), 0],
                   [            0, 1,            0, 0],
                   [-np.sin(rotY), 0, np.cos(rotY), 0],
                   [            0, 0,            0, 1]])

    RZ = np.array([[ np.cos(rotZ),-np.sin(rotZ), 0, 0],
                   [ np.sin(rotZ), np.cos(rotZ), 0, 0],
                   [            0,            0, 1, 0],
                   [            0,            0, 0, 1]])

    # Merge rotation matrices (RX,RY,RZ)
    R = np.linalg.multi_dot([RX, RY, RZ])

    # Translation matrix
    T = np.array([[1,0,0,distX],
                  [0,1,0,distY],
                  [0,0,1,distZ],
                  [0,0,0,    1]])

    # Homography matrix (all transformations combined)
    H = np.linalg.multi_dot([K, R, T, Kinv])

    # Apply matrix transformation
    cv.warpPerspective(src, H, (w, h), dst, cv.INTER_NEAREST, cv.BORDER_CONSTANT, 0)

    return dst


def move(img: np.ndarray, angle=0, scale=1, coord=None):
    """Rotate, zoom in/out of images, and/or set center to provided coordinates
    https://stackoverflow.com/questions/69050464/zoom-into-image-with-opencv

    Args:
    -----
        img (np.ndarray): image data
        angle (int, optional): rotation angle. Defaults to 0.
        scale (int, optional): sizing scale. Defaults to 1.
        coord (_type_, optional): coordinates to center. Expected in the format of (x, y) in pixels. Defaults to None.

    Returns:
    --------
        np.ndarray: modified image data
    """
    h, w = img.shape[:2]
    cy, cx = [i / 2 for i in img.shape[:-1]] if coord is None else coord[::-1]
    center = (cx, cy)
    rotation_matrix = cv.getRotationMatrix2D(center=center, angle=angle, scale=scale)
    rotated = cv.warpAffine(src=img, M=rotation_matrix, dsize=(w,h))
    return rotated


if __name__ == '__main__':
    # Read input image
    # org = cv.imread("../images/stitched from uav/stitchedImg34.JPG")
    org = cv.imread("../images/Eli_Field_Oct_8_2020 original/DJI_0022.JPG")
    cv.imshow("original", org)
    cv.waitKey(0)

    # for i in range(0, 360, 30):
    #     warped = warp(org, 0, 0, 0, 0, 0, 0)
    #     cv.imshow("warped", warped)
    #     cv.waitKey(0)

    # Warp image based on the rotation and translation values
    # zoomed = move(org, 0, 1.5)
    warped = warp(org, -45, 0, 0, 0, -2500, -2000)
    cv.imshow("warped", warped)
    cv.waitKey(0)
    cv.destroyAllWindows()
