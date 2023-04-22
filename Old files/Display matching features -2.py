from exif import Image
from datetime import datetime
import cv2
import numpy as np

def get_time(image):
    """
    Returns the creation time of the image file.

    :param image: The file name of the image
    :type image: str
    :return: The creation time of the image
    :rtype: datetime
    """
    with open(image, 'rb') as image_file:
        img = Image(image_file)
        time_str = img.get("datetime_original")
        return datetime.strptime(time_str, '%Y:%m:%d %H:%M:%S')

def get_time_difference(image_1, image_2):
    """
    Returns the time difference between two image files.

    :param image_1: The file name of the first image
    :type image_1: str
    :param image_2: The file name of the second image
    :type image_2: str
    :return: The time difference between the two images
    :rtype: datetime.timedelta
    """
    time_1 = get_time(image_1)
    time_2 = get_time(image_2)
    return time_2 - time_1

def convert_to_cv(image_1, image_2):
    image_1_cv = cv2.imread(image_1, cv2.IMREAD_GRAYSCALE)
    image_2_cv = cv2.imread(image_2, cv2.IMREAD_GRAYSCALE)
    return image_1_cv, image_2_cv

from skimage.feature import SIFT

def calculate_features(image_1, image_2):
    sift = SIFT()
    keypoints_1 = sift.detect(image_1)
    keypoints_2 = sift.detect(image_2)
    descriptors_1 = sift.compute(image_1, keypoints_1)[1]
    descriptors_2 = sift.compute(image_2, keypoints_2)[1]
    return keypoints_1, keypoints_2, descriptors_1, descriptors_2

def calculate_matches(descriptors_1, descriptors_2):
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(descriptors_1, descriptors_2, k=2)
    return matches


def filter_matches(keypoints_1, keypoints_2, matches):
    good_matches = []
    for m,n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)
    src_pts = np.float32([ keypoints_1[m.queryIdx].pt for m in good_matches ]).reshape(-1,1,2)
    dst_pts = np.float32([ keypoints_2[m.trainIdx].pt for m in good_matches ]).reshape(-1,1,2)
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    matches_mask = mask.ravel().tolist()
    inlier_matches = [m for (i, m) in enumerate(good_matches) if matches_mask[i]]
    return inlier_matches

def display_matches(image_1_cv, keypoints_1, image_2_cv, keypoints_2, matches):
    match_img = cv2.drawMatches(image_1_cv, keypoints_1, image_2_cv, keypoints_2, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    resize = cv2.resize(match_img, (1600,600), interpolation = cv2.INTER_AREA)
    cv2.imshow('matches', resize)
    cv2.waitKey(0)
    cv2.destroyWindow('matches')


image_1 = 'milkyway_001.jpg'
image_2 = 'milkyway_390.jpg'


time_difference = get_time_difference(image_1, image_2)
image_1_cv, image_2_cv = convert_to_cv(image_1, image_2)
keypoints_1, keypoints_2, descriptors_1, descriptors_2 = calculate_features(image_1_cv, image_2_cv)
matches = calculate_matches(descriptors_1, descriptors_2)
inlier_matches = filter_matches(keypoints_1, keypoints_2, matches)
display_matches(image_1_cv, keypoints_1, image_2_cv, keypoints_2, inlier_matches)

