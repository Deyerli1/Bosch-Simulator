import cv2
import numpy as np


class LineDetector:

    def __init__(self, pixel_precission, radian_precission, threshold, min_line_length, max_line_length, end_of_lines, roi):
        self.pixel_precission = pixel_precission
        self.radian_precission = radian_precission
        self.threshold = threshold
        self.min_line_length = min_line_length
        self.max_line_length = max_line_length
        self.end_of_lines = end_of_lines
        self.roi = roi

    def _detect_edges(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        canny = cv2.Canny(blur, 50, 150)

        return canny

    def _to_coordinates(self, line_parameters):
        slope, intercept = line_parameters
        y1 = self.image.shape[0]
        y2 = int(y1 * self.end_of_lines)
        x1 = int((y1 - intercept) / slope)
        x2 = int((y2 - intercept) / slope)

        return np.array([x1, y1, x2, y2])

    def _average_slope_intercept(self, lines):
        left_fit = []
        right_fit = []

        # TODO: Handle case when lines is None i.e. no lines detected
        if lines is None:
            return np.array([])

        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            parameters = np.polyfit((x1, x2), (y1, y2), 1)
            slope = parameters[0]
            intercept = parameters[1]
            if slope < 0:
                left_fit.append((slope, intercept))
            else:
                right_fit.append((slope, intercept))

        # TODO: Handle case when left_fit or right_fit is empty i.e. no lines detected
        if not left_fit or not right_fit:
            return np.array([])

        left_fit_average = np.average(left_fit, axis=0)
        right_fit_average = np.average(right_fit, axis=0)
        left_line = self._to_coordinates(left_fit_average)
        right_line = self._to_coordinates(right_fit_average)

        return np.array([left_line, right_line])

    def detect_lines(self, image):
        self.image = np.copy(image)
        self.edged_image = self._detect_edges(self.image)
        self.cropped_image = self.roi.mask(self.edged_image)
        self.warped_image = self.roi.warp_image(self.cropped_image)
        lines = cv2.HoughLinesP(self.warped_image, self.pixel_precission, self.radian_precission, self.threshold, np.array(
            []), minLineLength=self.min_line_length, maxLineGap=self.max_line_length)
        newLines =  self.roi.unwarp_lines(lines)

        return newLines

    def display_lines(self, image):
        lines = self.detect_lines(image)
        avg_lines = self._average_slope_intercept(lines)
        line_image = np.zeros_like(image)
        if lines is not None and lines.all():
            for line in lines:
                x1, y1, x2, y2 = line.reshape(4)
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10)

            for avg_line in avg_lines:
                x1, y1, x2, y2 = avg_line
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 10)

        return cv2.addWeighted(image, 0.8, line_image, 1, 1)

  