import os
from colormath.color_diff import delta_e_cie2000
from colormath.color_objects import LabColor, sRGBColor
from colormath import color_conversions
from PIL import Image
import numpy as np


def calculate_delta_e(color1, color2):
    return delta_e_cie2000(color1, color2)


def rgb_to_lab(r, g, b):
    srgb_color = sRGBColor(r / 255.0, g / 255.0, b / 255.0)
    lab_color = color_conversions.convert_color(srgb_color, LabColor)
    return lab_color


def extract_lab_values(image_path):
    image = Image.open(image_path)
    image = image.convert('RGB')
    image_data = np.array(image)
    image_data = image_data.reshape(-1, 3)
    lab_values = []
    for rgb in image_data:
        lab = rgb_to_lab(rgb[0], rgb[1], rgb[2])
        lab_values.append(lab)
    return lab_values


folder1_path = 'E:\\wangjing\\Fre_Aware_LLIE\\data\\Image_restoration\\LL_dataset\\LOLv1\\val\\high'
folder2_path = 'E:\\wangjing\\Fre_Aware_LLIE\\results\\test\\experiment\\SGF\\SGF'

images_folder1 = os.listdir(folder1_path)
images_folder2 = os.listdir(folder2_path)

# Sort the lists to make sure the images are processed in the same order
images_folder1.sort()
images_folder2.sort()

all_delta_e_values = []

for image_name1, image_name2 in zip(images_folder1, images_folder2):
    image_path1 = os.path.join(folder1_path, image_name1)
    image_path2 = os.path.join(folder2_path, image_name2)

    lab_values1 = extract_lab_values(image_path1)
    lab_values2 = extract_lab_values(image_path2)

    delta_e_values = []
    for lab1, lab2 in zip(lab_values1, lab_values2):
        delta_e = calculate_delta_e(lab1, lab2)
        delta_e_values.append(delta_e)

    average_delta_e = np.mean(delta_e_values)
    all_delta_e_values.append(average_delta_e)
    print(f"DeltaE between {image_name1} and {image_name2}: {average_delta_e}")

# Calculate and print the average DeltaE value for all images
average_delta_e_all = np.mean(all_delta_e_values)
print(f"Average DeltaE for all images: {average_delta_e_all}")
