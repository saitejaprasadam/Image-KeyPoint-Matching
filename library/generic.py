import cv2 as opencv
import numpy as np
from copy import deepcopy


def get_gradient(image):
    dx = opencv.Sobel(image, ddepth=opencv.CV_64F, dx=1, dy=0)
    dy = opencv.Sobel(image, ddepth=opencv.CV_64F, dx=0, dy=1)
    return dx, dy, dx*dx, dx*dy, dy*dy


def stack_images(image1, image2):
    try:
        margin = np.full((len(image1), 1, 3), 255).astype(np.uint8)
        stacked_image = np.concatenate((image1, margin, image2), axis=1)
        opencv.imshow("Key Points", stacked_image)
        opencv.waitKey(0)

    except ValueError:
        opencv.imshow("Image 1", image1)
        opencv.imshow("Image 2", image2)
        opencv.waitKey(0)

# Custom drawKeypoints as the opencv version i installed has binding issue on that method
def draw_key_points(image, corner_points):

    image = deepcopy(image)
    for corner_point in corner_points:
        x, y = corner_point.pt
        opencv.circle(image, (int(x), int(y)), 4, (0, 0, 255))

    return image
