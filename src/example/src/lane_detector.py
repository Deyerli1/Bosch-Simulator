import cv2
import numpy as np
from line_detector import LineDetector
from region_of_interest import RegionOfInterest

FILENAME = 'test2.mp4'

PIXEL_PRECISSION = 2
RADIAN_PRECISSION = np.pi / 180
THRESHOLD = 100
MIN_LINE_LENGTH = 50
MAX_LINE_GAP = 50
END_OF_LINES = 3 / 5


class LaneDetector:

    def __init__(self, roi):
        self.line_detector = LineDetector(
            PIXEL_PRECISSION, RADIAN_PRECISSION, THRESHOLD, MIN_LINE_LENGTH, MAX_LINE_GAP, END_OF_LINES, roi)

    def detect_lane(self, image):
        return self.line_detector.detect_lines(image)

    def display_lines(self, lines):
        return self.line_detector.display_lines(lines)


if __name__ == '__main__':

    cap = cv2.VideoCapture(FILENAME)
 
    roi = RegionOfInterest((550, 330), (190, 720), (1100, 720), (600, 330))
    lane_detector = LaneDetector(roi)

    while cap.isOpened():
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        _, frame = cap.read()
        image_w_lines = lane_detector.display_lines(frame)
        cv2.imshow('image', image_w_lines)
        # cv2.imshow('warped', lane_detector.line_detector.warped_image.T)

    # image = cv2.imread('test1.jpg')
    # roi = RegionOfInterest((274, 184), (0, 337), (575, 337), (371, 184))
    # lane_detector = LaneDetector(roi)
    # image_w_lines = lane_detector.display_lines(image)
    # cv2.imshow('image', image_w_lines)
    # # cv2.imshow('warped', lane_detector.line_detector.warped_image.T)
    # cv2.waitKey(0)
