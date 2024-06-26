import cv2 as cv
import transform_perspective as tp
import image


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

    # display image and wait for key
    while True:
        frame = tp.warp(sim.image.image, sim.rotX, sim.rotY, sim.rotZ, sim.distX, sim.distY, sim.distZ)
        cv.imshow("simulation", frame)
        key = cv.waitKeyEx(0)
        print(key)

        if key == ord("w"):       # tilt up (pitch)
            sim.rotX -= 25
        elif key == ord("s"):     # tilt down (pitch)
            sim.rotX += 25
        elif key == ord("a"):     # pan left (yaw)
            sim.rotZ += 25
        elif key == ord("d"):     # pan right (yaw)
            sim.rotZ -= 25
        elif key == ord("q"):     # tilt left (roll)
            sim.rotY -= 25
        elif key == ord("e"):     # tilt right (roll)
            sim.rotY += 25
        elif key == ord("r"):     # ascend
            sim.distZ += 25
        elif key == ord("f"):     # descend
            sim.distZ -= 25
        elif key == 63234:        # left key
            sim.distY += 25
        elif key == 63232:        # up key
            sim.distX -= 25
        elif key == 63235:        # right key
            sim.distY -= 25
        elif key == 63233:        # down key
            sim.distX += 25
        elif key == 27:           # (esc) exit
            break


if __name__ == "__main__":
    start()

