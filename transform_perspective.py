import cv2
import numpy as np

def simulate(src, dst, w, h, rotXval, rotYval, rotZval, distXval, distYval, distZval):

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
    cv2.warpPerspective(src, H, (w, h), dst, cv2.INTER_NEAREST, cv2.BORDER_CONSTANT, 0)

    return dst
