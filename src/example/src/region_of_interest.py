import cv2
import numpy as np


class RegionOfInterest:
    # TODO: This class shoud calculate the mask needed for "_apply_roi" method in LineDetector class and also the matrix needed for "_warp_image" method in LineDetector class

    def __init__(self, top_left, bottom_left, bottom_right, top_right):
        self.vertices = np.array(
            [top_left, bottom_left, bottom_right, top_right])

    def mask(self, image):
        mask = np.zeros_like(image)
        cv2.fillPoly(mask, [self.vertices], 255)
        return cv2.bitwise_and(image, mask)

    def warp_image(self, image):
        # padding from side of the image in pixels
        padding = int(0.10 * image.shape[0])
        self.desired_roi_points = np.float32([
            [padding, 0],  # Top-left corner
            [padding, image.shape[0]],  # Bottom-left corner
            [image.shape[0]-padding, image.shape[1]],  # Bottom-right corner
            [image.shape[0]-padding, 0],  # Top-right corner
        ])
        # Calculate the transformation matrix
        self.transformation_matrix = cv2.getPerspectiveTransform(
            np.float32(self.vertices), self.desired_roi_points)
        # Warp the image
        self.warped_image = cv2.warpPerspective(
            image, self.transformation_matrix, image.shape, flags=(cv2.INTER_LINEAR))
        # Convert image to binary
        (_, binary_warped) = cv2.threshold(
            self.warped_image, 127, 255, cv2.THRESH_BINARY)
        self.warped_image = binary_warped

        return self.warped_image

    def unwarp_lines(self, lines):
        transformation_matrix = cv2.getPerspectiveTransform(
           self.desired_roi_points,  np.float32(self.vertices))
        unwarped_lines = []

        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            unwarped_line = cv2.perspectiveTransform(
                np.array([[[x1, y1], [x2, y2]]], dtype=np.float32), transformation_matrix)
            unwarped_lines.append(unwarped_line[0])

        return np.array(unwarped_lines)
    
    # def unwarp_image(self, image):
    #             # padding from side of the image in pixels
    #     padding = int(0.10 * image.shape[0])
    #     self.desired_roi_points = np.float32([
    #         [padding, 0],  # Top-left corner
    #         [padding, image.shape[0]],  # Bottom-left corner
    #         [image.shape[0]-padding, image.shape[1]],  # Bottom-right corner
    #         [image.shape[0]-padding, 0],  # Top-right corner
    #     ])
    #     # Calculate the transformation matrix
    #     transformation_matrix = cv2.getPerspectiveTransform(
    #        self.desired_roi_points,  np.float32(self.vertices))
    #     # Warp the image
    #     warped_image = cv2.warpPerspective(
    #         image, transformation_matrix, image.shape, flags=(cv2.INTER_LINEAR))
    #     # Convert image to binary
    #     (_, binary_warped) = cv2.threshold(
    #         self.warped_image, 127, 255, cv2.THRESH_BINARY)
    #     return binary_warped
