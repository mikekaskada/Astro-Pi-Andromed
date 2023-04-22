"""
This code performs the following steps:

    Imports the required libraries: cv2, imageio, and numpy.
    Initializes an empty list result_images to store the final result images.
    Loops over a range of 711 values to process each image.
        Loads two images: "combined_{:03d}.png" and "milkyway_{:03d}.jpg", where {:03d}
        is a placeholder for the loop index.
        Crops 95 pixels from the left and 90 pixels from the right of the second image.
        Resizes the second image to 60% of its original size.
        Creates a new image that is the same size as the first image.
        Pastes the second image onto the new image at the top-left corner.
        Adds the new image to the list of result images.
    Converts the list of result images to a NumPy array.
    Saves the result images as an animated GIF and as a video.
"""

import cv2
import imageio.v2 as imageio
import numpy as np

# Create a list to store the result images
result_images = []

for i in range(711):
    # Load the first image
    image1 = cv2.imread("combined_{:03d}.png".format(i + 1))

    # Load the second image
    image2 = cv2.imread("milkyway_{:03d}.jpg".format(i + 1))

    # Crop 95 pixels from the left and 90 pixels from the right of the second image
    image2 = image2[:, 95:-90, :]

    # Resize the second image to 60% of its original size
    height, width = image2.shape[:2]
    new_height, new_width = int(height * 0.6), int(width * 0.6)
    image2 = cv2.resize(image2, (new_width, new_height), cv2.INTER_CUBIC)

    # Create a new image that is the same size as image1
    result = image1.copy()

    # Paste image2 onto the result image at the top-left corner
    result[0:image2.shape[0], 0:image2.shape[1]] = image2

    # Add the result image to the list of result images
    result_images.append(result)

# Convert the list of result images to a NumPy array


# Save the result images as an animated GIF
imageio.mimsave("result.gif", result_images, fps=5)

# Save the result images as a video
imageio.mimsave("result.mp4", result_images, fps=5)
