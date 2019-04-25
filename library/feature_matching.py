import cv2 as opencv
import numpy as np

"""
Feature Matching
    1. Iterate every keypoint with its descriptor
        a. Compute SSD of key point with every other key point in other image
        b. Find 1st and 2nd Best Key point
        c. If the 1st best SSD / 2nd best SSD ratio is < 0.5 and best SSD is < 0.3
                We can add to matches list

    2. We sort this list with respect to there distance
"""

def feature_vector(image_description):
    feature_histogram = image_description.flatten()
    feature_histogram = feature_histogram / (np.sqrt(np.sum(feature_histogram ** 2)) + 1e-8)
    return feature_histogram

def feature_matching(image1_descriptions, image2_descriptions):
    matches = []

    for image1_index, image1_description in enumerate(image1_descriptions):

        best_distance1 = 10
        best_distance2 = 10
        best_distance_image2_index = -1
        image1_feature_vector = feature_vector(image1_description)

        for image2_index, image2_description in enumerate(image2_descriptions):

            image2_feature_vector = feature_vector(image2_description)
            SSD = ((image1_feature_vector - image2_feature_vector) ** 2).sum()

            if SSD < best_distance1:
                best_distance2 = best_distance1
                best_distance1 = SSD
                best_distance_image2_index = image2_index

        if (best_distance1 / best_distance2) < 0.5 and float(best_distance1) < 0.3:
            matches.append(opencv.DMatch(image1_index, best_distance_image2_index, best_distance1))

    matches = sorted(matches, key=lambda x:x.distance)
    return matches
