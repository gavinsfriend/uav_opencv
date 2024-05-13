# https://github.com/freedomwebtech/yolov8-custom-segment/tree/main
# https://www.youtube.com/watch?v=QtsI0TnwDZs
# https://github.com/niconielsen32/YOLOv8-Class/blob/main/YOLOv8InferenceClass.py
from ultralytics import YOLO
import numpy as np
import cv2 as cv
import random
RED = (0, 0, 255)
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)


class Segmenter:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.classes = list(self.model.names.values())
        self.ids = [self.classes.index(clas) for clas in self.classes]

    def extract(self, img, conf):
        results = self.model.predict(img.copy(), conf=conf)

        result = results[0]     # for result in results: ??
        # boxes
        boxes = result.boxes.cpu().numpy()
        xyxys = np.array(boxes.xyxy, dtype="int")
        confidences = np.array(boxes.conf, dtype="float").round(3)
        class_ids = np.array(boxes.cls, dtype="int")
        # masks
        masks = result.masks.xy
        points = [np.int32([mask]) for mask in masks]
        return xyxys, confidences, class_ids, points

    def display(self, img, xyxys, confidences, class_ids_res, mask_pts):
        original = img.copy()
        classes = self.classes
        class_ids = self.ids
        # generate random colors for each object in the frame/image
        colors = [random.choices(range(256), k=3) for _ in class_ids]

        for xyxy, conf, id, pts in zip(xyxys, confidences, class_ids_res, mask_pts):
            color_number = class_ids.index(id)              # pick a color
            cv.fillPoly(img, pts, colors[color_number])     # paint mask
            # draw bbox
            x1, y1, x2, y2 = xyxy
            cv.rectangle(img, (x1, y1), (x2, y2), RED, 20)
            # print object class and confidence score
            text = "{cls}, conf: {conf:.2f}".format(cls=classes[id], conf=conf)
            text_size, _ = cv.getTextSize(text, cv.FONT_HERSHEY_SIMPLEX, 5, 20)
            text_w, text_h = text_size
            cv.rectangle(img, (x1, y1 - text_h), (x1 + text_w, y1), WHITE, -1)
            cv.putText(img, text, (x1, y1-30), cv.FONT_HERSHEY_SIMPLEX, 5, BLACK, 20)

        cv.imshow("original", original)
        cv.imshow("prediction", img)
        cv.waitKey(0)


if __name__ == "__main__":
    seg = Segmenter("best.pt")
    img_path = "./stitchedImg.JPG"
    image = cv.imread(img_path)

    xyxys, confs, ids, pts = seg.extract(image, 0.2)
    seg.display(image, xyxys, confs, ids, pts)
    cv.destroyAllWindows()