from exif import Image
from datetime import datetime
import cv2
import math
import numpy as np
from pyproj import Geod, Transformer


def lat_lon_alt_to_ecef(lat, lon, alt):
    transformer = Transformer.from_crs(4326, 4978)
    x, y, z = transformer.transform(lat, lon, alt)
    return x, y, z

def ecef_distance(coords1, coords2):
    return np.linalg.norm(np.array(coords1) - np.array(coords2))

# Sample positional data (latitude, longitude, altitude)
location1 = (37.51861666697, 86.1550989816772, 416638)
location2 = (36.9042216736308, 87.0420874603178, 416573)

# Convert to ECEF coordinates
ecef1 = lat_lon_alt_to_ecef(*location1)
ecef2 = lat_lon_alt_to_ecef(*location2)

# Calculate 3D Euclidean distance
distance = ecef_distance(ecef1, ecef2)


def get_time(image):
    with open(image, 'rb') as image_file:
        img = Image(image_file)
        time_str = img.get("datetime_original")
        time = datetime.strptime(time_str, '%Y:%m:%d %H:%M:%S')
    return time
    
    
def get_time_difference(image_1, image_2):
    time_1 = get_time(image_1)
    time_2 = get_time(image_2)
    time_difference = time_2 - time_1
    return time_difference.seconds


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

def apply_ransac_filter(keypoints_1, keypoints_2, matches):
    src_pts = np.float32([keypoints_1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints_2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    matches = [m for m, m_mask in zip(matches, mask) if m_mask]
    return matches

def apply_ratio_filter(matches):
    ratio_filtered_matches = []
    for match in matches:
        if match.distance < 0.4 * max([m.distance for m in matches if m != match]):
            ratio_filtered_matches.append(match)
    return ratio_filtered_matches

def display_matches(image_1_cv, keypoints_1, image_2_cv, keypoints_2, matches):
    match_img = cv2.drawMatches(image_1_cv, keypoints_1, image_2_cv, keypoints_2, matches[:200], None)
    resize = cv2.resize(match_img, (1600,600), interpolation = cv2.INTER_AREA)
    cv2.imshow('matches', resize)
    cv2.waitKey(0)
    cv2.destroyWindow('matches')
    
    
def find_matching_coordinates(keypoints_1, keypoints2, matches):
    coordinates_1 = []
    coordinates_2 = []
    for match in matches:
        image_1_idx = match.queryIdx
        image_2_idx = match.trainIdx
        (x1,y1) = keypoints_1[image_1_idx].pt
        (x2,y2) = keypoints_2[image_2_idx].pt
        coordinates_1.append((x1,y1))
        coordinates_2.append((x2,y2))
    return coordinates_1, coordinates_2


def calculate_mean_distance(coordinates_1, coordinates_2):
    all_distances = 0
    merged_coordinates = list(zip(coordinates_1, coordinates_2))
    for coordinate in merged_coordinates:
        x_difference = coordinate[0][0] - coordinate[1][0]
        y_difference = coordinate[0][1] - coordinate[1][1]
        distance = math.hypot(x_difference, y_difference)
        all_distances = all_distances + distance
    return all_distances / len(merged_coordinates)

import statistics

def calculate_median_distance(coordinates_1, coordinates_2):
    distances = []
    merged_coordinates = list(zip(coordinates_1, coordinates_2))
    for coordinate in merged_coordinates:
        x_difference = coordinate[0][0] - coordinate[1][0]
        y_difference = coordinate[0][1] - coordinate[1][1]
        distance = math.hypot(x_difference, y_difference)
        distances.append(distance)
    return statistics.median(distances)


def calculate_gsd(altitude):
    """
    Calculate the Ground Sampling Distance (GSD) in cm/pixel.
    
    Arguments:
    altitude -- The distance of the camera from the ground (i.e. the flight height).
    
    Returns:
    gsd -- the Ground Sampling Distance in cm/pixel.
    """
    
    # Fixed parameters
    """
        image_width:    the image width in pixels.
        image_height:   the image height in pixels.
        focal_Lenght:   the focal lenght of the camera in mm
        sensor_width:   the sensor width of the camera in mm
        Find those data at https://www.raspberrypi.com/documentation/accessories/camera.html
        and at https://projects.raspberrypi.org/en/projects/code-for-your-astro-pi-mission-space-lab-experiment/4
    """
    focal_length = 5 # millimeters
    sensor_width = 6.287 # millimeters
    image_width = 4056 # pixels
    image_height = 3040 # pixels
    
    # Calculate the physical width and height of one pixel in millimeters
    pixel_width = sensor_width / image_width
    pixel_height = sensor_width / image_height
    # Calculate the Ground Sampling Distance
    gsd = (100 * pixel_width * altitude) / focal_length
    return gsd

def calculate_speed_in_kmps(feature_distance, GSD, time_difference):
    # distance = feature_distance * GSD / 100000
    average_feature_distance_on_earth = feature_distance * GSD / 100000
    distance_ratio = average_feature_distance_on_earth / distance
    actual_distance = distance * distance_ratio
    speed = actual_distance / time_difference
    return speed


image_1 = 'milkyway_001.jpg'
image_2 = 'milkyway_002.jpg'


time_difference = get_time_difference(image_1, image_2) #get time difference between images
image_1_cv, image_2_cv = convert_to_cv(image_1, image_2) #create opencfv images objects
altitude = 416600  # meters
GSD = calculate_gsd(altitude)
keypoints_1, keypoints_2, descriptors_1, descriptors_2 = calculate_features(image_1_cv, image_2_cv, 1000) #get keypoints and descriptors
matches = calculate_matches(descriptors_1, descriptors_2) #match descriptors
filtered_matches_ransac = apply_ransac_filter(keypoints_1, keypoints_2, matches)
# Apply the ratio filter
filtered_matches_ratio = apply_ratio_filter(matches)
# display_matches(image_1_cv, keypoints_1, image_2_cv, keypoints_2, filtered_matches_ratio) #display matches
# coordinates_1, coordinates_2 = find_matching_coordinates(keypoints_1, keypoints_2, filtered_matches_ratio)
display_matches(image_1_cv, keypoints_1, image_2_cv, keypoints_2, filtered_matches_ransac) #display matches
coordinates_1, coordinates_2 = find_matching_coordinates(keypoints_1, keypoints_2, filtered_matches_ransac)
# average_feature_distance = calculate_mean_distance(coordinates_1, coordinates_2)
average_feature_distance = calculate_median_distance(coordinates_1, coordinates_2)
speed = calculate_speed_in_kmps(average_feature_distance, GSD, time_difference)
print(speed)
