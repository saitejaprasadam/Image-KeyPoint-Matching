# Image KeyPoint Matching
Please adhere to your organization's rules for using this code. For [Concordia University](http://www.concordia.ca), these rules can be found [here](http://www.concordia.ca/students/academic-integrity/offences.html).

Keypoint Matching between two images using OpenCV

### Feature Detection by Harris corner detector algorithm (Wikipedia)
* Color to grayscale
* Spatial derivative calculation
* Structure tensor setup
* Harris response calculation
* Non-maximum suppression

=> we display the keypoints using a custom drawKeypoints as the opencv version i installed has binding issue on that method

### Feature Description
* Compute gradient matrix of the image (dx, dy, magnitude, angle)
* Iterate every key point of the image
  * We get the 16x16 window of the key point
  * we compute the rotation of the patch
  * key_point_descriptions = compute_histogram(array of 4 sub patches of the 16x16 patch)
    (Each sub patch is 4x4 patch of 16 pixel, each histogram is 8 element array. (so total 16 pixels x 8 elements = 128)

### Feature Matching
* Iterate every keypoint with its descriptor
  * Compute SSD of key point with every other key point in other image
  * Find 1st and 2nd Best Key point
  * If the 1st best SSD / 2nd best SSD ratio is < 0.5 and best SSD is < 0.3, We can add to matches list

* We sort this list with respect to there distance

=> display the best 25 matches using drawMatches function
