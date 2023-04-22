import cv2

# Load the first image
image1 = cv2.imread("combined_001.png")

# Load the second image
image2 = cv2.imread("milkyway_001.jpg")

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

# Display the result image
cv2.imshow("Result", result)
cv2.waitKey(0)
cv2.destroyAllWindows()
