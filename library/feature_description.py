import cv2 as opencv
import numpy as np
from numpy import linalg
from copy import deepcopy
from .generic import get_gradient

"""
Feature Description
    1. Compute gradient matrix of the image (dx, dy, magnitude, angle)
    2. Iterate every key point of the image
        a. We get the 16x16 window of the key point
        b. we compute the rotation of the patch
        c. key_point_descriptions = compute_histogram(array of 4 sub patches of the 16x16 patch)
            (Each sub patch is 4x4 patch of 16 pixel, each histogram is 8 element array. (so total 16 pixels x 8 elements = 128)
"""

def spilt_patch(patch, x, y):
    return patch[y: y+4, x: x+4]

def get_patch(image, key_point, patch_size):
    x, y = key_point.pt
    x = int(x)
    y = int(y)
    return image[y - patch_size: y + patch_size, x - patch_size: x + patch_size]

def gradient(patch):
    dx, dy, _, _, _ = get_gradient(patch)
    height, width = patch.shape
    gradient_matrix = np.zeros((height, width, 4))

    for y in range(height):
        for x in range(width):
            gradient_matrix[y, x] = [dx[y, x], dy[y, x], linalg.norm([dx[y, x], dy[y, x]]), opencv.fastAtan2(dy[y, x], dx[y, x])]

    return gradient_matrix

def compute_rotation(image_gradient):
    hist_array = np.zeros(36)
    height, width, _ = image_gradient.shape

    for y in range(height):
        for x in range(width):
            keypoint_angle = image_gradient[y, x][3]
            bin_pos = int(keypoint_angle / 10)
            hist_array[bin_pos] += image_gradient[y, x][2]

    max_bin = np.argmax(hist_array)
    rotation_angle = max_bin * 10 + 5
    return rotation_angle


def compute_histogram(image_gradient, angle):
    hist_array = np.zeros(8)
    height, width, _ = image_gradient.shape

    for y in range(height):
        for x in range(width):
            keypoint = image_gradient[y, x]
            computed_angle = keypoint[3] - angle

            if computed_angle < 0: computed_angle += 360
            elif computed_angle > 360: computed_angle -= 360

            bin_pos = int(computed_angle / 45) % 8
            hist_array[bin_pos] += keypoint[2]

    return hist_array

def feature_description(image, key_points):

    image = deepcopy(image)
    image = opencv.cvtColor(image, opencv.COLOR_BGR2GRAY)
    image_gradient = gradient(image)

    key_points_descriptions = []

    for key_point in key_points:
        patch = get_patch(image_gradient, key_point, patch_size=8)
        height, width, _ = patch.shape
        angle = compute_rotation(patch)

        key_point_descriptions = []
        key_point_descriptions.append(compute_histogram(spilt_patch(patch, 0, 0), angle))
        key_point_descriptions.append(compute_histogram(spilt_patch(patch, 0, 4), angle))
        key_point_descriptions.append(compute_histogram(spilt_patch(patch, 0, 8), angle))
        key_point_descriptions.append(compute_histogram(spilt_patch(patch, 0, 12), angle))

        key_point_descriptions.append(compute_histogram(spilt_patch(patch, 4, 0), angle))
        key_point_descriptions.append(compute_histogram(spilt_patch(patch, 4, 4), angle))
        key_point_descriptions.append(compute_histogram(spilt_patch(patch, 4, 8), angle))
        key_point_descriptions.append(compute_histogram(spilt_patch(patch, 4, 12), angle))

        key_point_descriptions.append(compute_histogram(spilt_patch(patch, 8, 0), angle))
        key_point_descriptions.append(compute_histogram(spilt_patch(patch, 8, 4), angle))
        key_point_descriptions.append(compute_histogram(spilt_patch(patch, 8, 8), angle))
        key_point_descriptions.append(compute_histogram(spilt_patch(patch, 8, 12), angle))

        key_point_descriptions.append(compute_histogram(spilt_patch(patch, 12, 0), angle))
        key_point_descriptions.append(compute_histogram(spilt_patch(patch, 12, 4), angle))
        key_point_descriptions.append(compute_histogram(spilt_patch(patch, 12, 8), angle))
        key_point_descriptions.append(compute_histogram(spilt_patch(patch, 12, 12), angle))

        key_points_descriptions.append(key_point_descriptions)

    return np.array(key_points_descriptions)
