import os
import cv2
import sys
from PIL import Image
import numpy as np 

#WARNING: CORRUPTED IMAGE WILL PRODUCE AN ERROR "PREMATURE END OF JPEG FILE"

#Make a new directory if doesn't exists
if not os.path.isdir(os.path.join("224_dataset_ori_rgb")):
    os.mkdir(os.path.join("224_dataset_ori_rgb"))

#Loop through the list of files
labels = []
file_name = []
names = []
for file_label in os.listdir("dont_touch"):
    if file_label != ".DS_Store":
        for img in os.listdir(os.path.join("dont_touch", file_label)):
            file_name.append(os.path.join("dont_touch", file_label, img))
            labels.append(file_label)
            names.append(img)

assert len(labels) == len(file_name) == len(names)

for index, image in enumerate(file_name):
    try:
        with open(image, 'rb') as image_curr:
            curr_image = cv2.imread(image)
            preprocessed_image  = cv2.resize(curr_image, (224,224), interpolation=cv2.INTER_CUBIC)
            if not os.path.isdir(os.path.join("224_dataset_ori_rgb", labels[index])):
                os.mkdir(os.path.join("224_dataset_ori_rgb", labels[index]))
            cv2.imwrite(os.path.join("224_dataset_ori_rgb", labels[index], names[index] ), preprocessed_image)
            print(f"{str(curr_image)} is acceptable")
            cv2.waitKey(0)
    except Exception as e:
        print(str(img) + " can't be saved because it is corrupted")


