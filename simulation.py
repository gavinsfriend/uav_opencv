import cv2 as cv
import glob as glob
import transform_perspective as tp
import stitching, image


class Simulator:
    def __init__(self, img: image.Img, rotX=0, rotY=0, rotZ=0, distX=0, distY=0, distZ=0):
        self.image = img
        self.rotX = rotX
        self.rotY = rotY
        self.rotZ = rotZ
        self.distX = distX
        self.distY = distY
        self.distZ = distZ


def start():
    path = "../images/stitched from uav/stitchedImg34.JPG"
    img = image.Img(path)
    sim = Simulator(img, -80, 0, -90, 700, -50, -600)
    print("height(y):", img.image.shape[0], ", width(x):", img.image.shape[1])
    cv.imshow("original", img.image)
    cv.waitKey(0)

    frame = tp.warp(sim.image.image, sim.rotZ, sim.rotY, sim.rotZ, sim.distX, sim.distY, sim.distZ)

    # display image and wait for key
    while True:
        cv.imshow("simulation", frame)
        key = cv.waitKey(0)

        if key == ord("w"):       # tilt up (pitch)
            sim.rotX -= 25
            frame = tp.warp(sim.image.image, sim.rotX, sim.rotY, sim.rotZ, sim.distX, sim.distY, sim.distZ)
        elif key == ord("s"):     # tilt down (pitch)
            sim.rotX += 25
            frame = tp.warp(sim.image.image, sim.rotX, sim.rotY, sim.rotZ, sim.distX, sim.distY, sim.distZ)
        elif key == ord("a"):     # pan left (yaw)
            sim.rotY += 25
            frame = tp.warp(sim.image.image, sim.rotX, sim.rotY, sim.rotZ, sim.distX, sim.distY, sim.distZ)
        elif key == ord("d"):     # pan right (yaw)
            sim.rotY -= 25
            frame = tp.warp(sim.image.image, sim.rotX, sim.rotY, sim.rotZ, sim.distX, sim.distY, sim.distZ)
        elif key == ord("q"):     # tilt left (roll)
            sim.rotZ -= 25
            frame = tp.warp(sim.image.image, sim.rotX, sim.rotY, sim.rotZ, sim.distX, sim.distY, sim.distZ)
        elif key == ord("e"):     # tilt right (roll)
            sim.rotZ += 25
            frame = tp.warp(sim.image.image, sim.rotX, sim.rotY, sim.rotZ, sim.distX, sim.distY, sim.distZ)
        elif key == ord(" "):     # next stop/frame
            break
        elif key == 27:           # (esc) exit simulation
            break


if __name__ == "__main__":
    start()

