import library
import cv2 as opencv

image1 = opencv.imread(r"F:\Notes\Master's notes\Semester VI\Computer Vision\Assignments\Assignment 2\image_set\yosemite\Yosemite1.jpg")
image2 = opencv.imread(r"F:\Notes\Master's notes\Semester VI\Computer Vision\Assignments\Assignment 2\image_set\yosemite\Yosemite2.jpg")

print("Fetching Image 1 & 2 key points")
image1_key_points = library.harris_edge_detection(image1)
image2_key_points = library.harris_edge_detection(image2)

library.stack_images(library.draw_key_points(image1, image1_key_points), library.draw_key_points(image2, image2_key_points))

print("Fetching Image 1 & 2 Feature Description")
image1_descriptions = library.feature_description(image1, image1_key_points)
image2_descriptions = library.feature_description(image2, image2_key_points)

print("Fetching Image 1 & 2 Feature Matches")
matches = library.feature_matching(image1_descriptions, image2_descriptions)

output = opencv.drawMatches(image1, image1_key_points, image2, image2_key_points, matches[:25], None, flags=2)
opencv.imshow("matches", output)
opencv.waitKey(0)
