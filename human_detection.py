import numpy as np
import cv2
import imutils

class DetectorAPI:
    def __init__(self, path_to_ckpt):
        self.path_to_ckpt = path_to_ckpt
        self.fgbgAdaptiveGaussain = cv2.createBackgroundSubtractorMOG2()

    def processFrame(self, image):
        # Expand dimensions since the trained_model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image, axis=0)

        frame = cv2.GaussianBlur(image, (7,7),0)
        fgbgAdaptiveGaussainmask = self.fgbgAdaptiveGaussain.apply(frame)
        thresh = cv2.dilate(fgbgAdaptiveGaussainmask, None, iterations=2)
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,	cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        # loop over the contours
        min_area = 600
        boxes_list = []
        scores = []
        classes = []
        num = 0
        for c in cnts:
            # if the contour is too small, ignore it
            if cv2.contourArea(c) < min_area:
                continue
            #print(cv2.contourArea(c ))
            # compute the bounding box for the contour, draw it on the frame,
            # and update the text
            (x, y, w, h) = cv2.boundingRect(c)
            boxes_list.append((y, x, y + h, x + w))
            scores.append(0.9)
            classes.append(1)
            num += 1

        return boxes_list, scores, classes, num

    def close(self):
        pass


