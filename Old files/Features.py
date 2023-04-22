import cv2
import numpy as np

# Load the two images
img1 = cv2.imread("milkyway_001.jpg")
img2 = cv2.imread("milkyway_002.jpg")

# Create a SIFT object
sift = cv2.SIFT_create()

# Detect keypoints and compute descriptors for the two images
keypoints1, descriptors1 = sift.detectAndCompute(img1, None)
keypoints2, descriptors2 = sift.detectAndCompute(img2, None)

# Draw the keypoints on the two images
img1_keypoints = cv2.drawKeypoints(img1, keypoints1, outImage=np.array([]), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
img2_keypoints = cv2.drawKeypoints(img2, keypoints2, outImage=np.array([]), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# Resize the images to a smaller size to fit on screen
img1_keypoints = cv2.resize(img1_keypoints, (819, 614))
img2_keypoints = cv2.resize(img2_keypoints, (819, 614))

# Stack the two images horizontally
vis = np.concatenate((img1_keypoints, img2_keypoints), axis=1)

# Use the BFMatcher to find the best matches between the two images
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
matches = bf.match(descriptors1, descriptors2)

# Sort the matches by distance
matches = sorted(matches, key = lambda x:x.distance)

# Draw the matches on the stacked images
for i, match in enumerate(matches[:150]):
    img1_point = keypoints1[match.queryIdx].pt
    img2_point = (keypoints2[match.trainIdx].pt[0]+img1_keypoints.shape[1], keypoints2[match.trainIdx].pt[1])
    cv2.line(vis, (int(img1_point[0]), int(img1_point[1])), (int(img2_point[0]), int(img2_point[1])), (255, 0, 0), thickness=5)


    # Print the values of the img1_point and img2_point variables
    print(f"img1_point: {img1_point}, img2_point: {img2_point}")

# Display the stacked images with the matches
cv2.imshow("Keypoint Matches", vis)
cv2.waitKey(0)
cv2.destroyAllWindows()
