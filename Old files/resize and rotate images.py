import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageOps

def rotate_image(image, angle):
    image_center = tuple(np.array(image.size) / 2)
    rotated_image = image.rotate(angle, resample=Image.Resampling.BICUBIC, center=image_center)
    return rotated_image

data = pd.read_csv("C:\\Users\\MK\\Desktop\\Calculate ISS speed\\milkyway\\ISS data with speed.csv")
for i in range(len(data)):
    image_path = "C:\\Users\\MK\\Desktop\\Calculate ISS speed\\milkyway\\" + data['image file'][i]
    angle = -data['rotation'][i]  # Change the sign of the angle for clockwise rotation
    image = Image.open(image_path)
    image = rotate_image(image, angle)
    
    # Resize image
    new_size = tuple(int(dim/6) for dim in image.size)
    image = image.resize(new_size, resample=Image.Resampling.BICUBIC)
    
    output_path = "C:\\Users\\MK\\Desktop\\Calculate ISS speed\\milkyway\\combined\\" + data['image file'][i]
    image.save(output_path, exif=image.info['exif'], quality=95)  # Save the image with the original metadata (EXIF)
    print(i)