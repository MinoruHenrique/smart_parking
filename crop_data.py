import cv2
from utils import load_spaces, crop_from_space
import os

SPACES_NAME = "spaces/v4.txt"
DATA_FOLDER = r"D:\Smart parking\Data\CTIC_2023_v4_data\CTIC_2023_v4_filtered"
DEST_FOLDER = r"D:\Smart parking\Data\CTIC_cropped_no_label"
images = os.listdir(DATA_FOLDER)

spaces = load_spaces(SPACES_NAME)

# print(images)
print(spaces)
for image_dir in images:
    img = cv2.imread(os.path.join(DATA_FOLDER, image_dir))
    for space in spaces[1:9]:
        n_files = len(os.listdir(DEST_FOLDER))
        img_cropped = crop_from_space(img, space)
        file_name = os.path.join(DEST_FOLDER, f"{n_files}.jpg")
        cv2.imwrite(file_name, img_cropped)