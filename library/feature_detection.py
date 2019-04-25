import cv2 as opencv
import numpy as np
from math import floor
from copy import deepcopy
from .generic import get_gradient

"""
Harris corner detector algorithm (Wikipedia)
    1. Color to grayscale
    2. Spatial derivative calculation
    3. Structure tensor setup
    4. Harris response calculation
    5. Non-maximum suppression
"""

THRESHOLD = 123325  # 325
WINDOW_SIZE = 5

def window_summation(image, x, y):
    half_window_size = floor(WINDOW_SIZE / 2)
    return image[x - half_window_size: x + half_window_size, y - half_window_size: y + half_window_size].sum()

def get_structure_tensor(dxx, dxy, dyy, x, y):
    half_window_size = floor(WINDOW_SIZE / 2)
    ixx = dxx[y - half_window_size: y + half_window_size, x - half_window_size: x + half_window_size].sum()
    ixy = dxy[y - half_window_size: y + half_window_size, x - half_window_size: x + half_window_size].sum()
    iyy = dyy[y - half_window_size: y + half_window_size, x - half_window_size: x + half_window_size].sum()
    return np.array([[ixx, ixy], [ixy, iyy]])

def check_max_neighbour(corner_strength, x, y):
    half_window_size = floor(WINDOW_SIZE / 2)
    largest = corner_strength[y - half_window_size: y + half_window_size, x - half_window_size: x + half_window_size].max()
    if corner_strength[y][x] >= largest:
        return True

    return False

def harris_edge_detection(image):

    image = deepcopy(image)
    image = opencv.cvtColor(image, opencv.COLOR_BGR2GRAY)
    dx, dy, dxx, dxy, dyy = get_gradient(image)

    image_border_offset = floor(16/2)
    height, width = image.shape
    corner_strength = np.empty(shape=(height, width))

    for y in range(image_border_offset, height - image_border_offset):
        for x in range(image_border_offset, width - image_border_offset):
            #summation_of_neighbouring_pixels = window_summation(image, x, y)
            tensor = get_structure_tensor(dxx, dxy, dyy, x, y)
            harris_matrix = tensor
            trace = harris_matrix[0][0] + harris_matrix[1][1]
            det = harris_matrix[0][0] * harris_matrix[1][1] - (harris_matrix[0][1] * harris_matrix[1][0])
            corner_strength[y][x] = det/trace

    corner_points = []

    for y in range(corner_strength.shape[0]):
        for x in range(corner_strength.shape[1]):
            if corner_strength[y][x] > THRESHOLD and check_max_neighbour(corner_strength, x, y):
                corner_points.append(opencv.KeyPoint(x, y, _size=1))

    return corner_points
