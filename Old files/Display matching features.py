from exif import Image
from datetime import datetime
import cv2
import math


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
"""
time_diff = get_time_difference('photo_07464.jpg', 'photo_07465.jpg')
print(time_diff)
"""

def convert_to_cv(image_1, image_2):
    image_1_cv = cv2.imread(image_1, 0)
    image_2_cv = cv2.imread(image_2, 0)
    return image_1_cv, image_2_cv


def calculate_features(image_1, image_2, feature_number):
    orb = cv2.ORB_create(nfeatures = feature_number)
    keypoints_1, descriptors_1 = orb.detectAndCompute(image_1_cv, None)
    keypoints_2, descriptors_2 = orb.detectAndCompute(image_2_cv, None)
    return keypoints_1, keypoints_2, descriptors_1, descriptors_2

def calculate_matches(descriptors_1, descriptors_2):
    brute_force = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = brute_force.match(descriptors_1, descriptors_2)
    matches = sorted(matches, key=lambda x: x.distance)
    return matches


def display_matches(image_1_cv, keypoints_1, image_2_cv, keypoints_2, matches):
    match_img = cv2.drawMatches(image_1_cv, keypoints_1, image_2_cv, keypoints_2, matches[:100], None)
    resize = cv2.resize(match_img, (1600,600), interpolation = cv2.INTER_AREA)
    cv2.imshow('matches', resize)
    cv2.waitKey(0)
    cv2.destroyWindow('matches')



image_1 = 'milkyway_001.jpg'
image_2 = 'milkyway_390.jpg'



time_difference = get_time_difference(image_1, image_2) #get time difference between images
image_1_cv, image_2_cv = convert_to_cv(image_1, image_2) #create opencfv images objects
keypoints_1, keypoints_2, descriptors_1, descriptors_2 = calculate_features(image_1_cv, image_2_cv, 1000) #get keypoints and descriptors
matches = calculate_matches(descriptors_1, descriptors_2) #match descriptors
display_matches(image_1_cv, keypoints_1, image_2_cv, keypoints_2, matches) #display matches

