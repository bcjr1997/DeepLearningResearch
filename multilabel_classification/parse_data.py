import pandas as pd
import os
import csv
import numpy as np
import cv2

def get_data():
    data_csv = pd.read_csv(os.path.join("../planet_amazon/", "train_v2.csv"))
    image_path = os.path.join("../planet_amazon/", "train-jpg-small")
    image_data = []
    label_data = []
    unique_labels = set()

    for index, img in enumerate(data_csv["image_name"]):
        if(index < 10000):
            image = cv2.imread(os.path.join(image_path, img + ".jpg"))/255
            image_data.append(image)
        else:
            break

    for index, tags in enumerate(data_csv["tags"]):
        if(index < 10000):
            label_data.append(tags.split())
            for i in tags.split():
                unique_labels.add(i)
        else:
            break

    for index, labels in enumerate(label_data):
        label_data[index] = convert_labels_to_array(labels, sorted(unique_labels))

    image_data = np.array(image_data)
    print("Current : {}".format(image_data.shape))
    label_data = np.array(label_data)
    permutation = np.random.permutation(image_data.shape[0])
    image_data = image_data[permutation]
    label_data = label_data[permutation]
    
    #Split Training and Validation data
    train_image_data = image_data[:8000]
    train_label_data = image_data[:8000]
    train_filename = data_csv["image_name"][:8000]
    valid_image_data = image_data[8000: 10000]
    valid_label_data = image_data[8000: 10000]
    valid_filename = data_csv["image_name"][8000: 10000]
    return train_image_data, train_label_data, train_filename, valid_image_data, valid_label_data, valid_filename, sorted(unique_labels)

def convert_labels_to_array(labels, unique_labels):
    new_arr = np.zeros(len(unique_labels), dtype=np.float32)
    for index, label in enumerate(unique_labels):
        for tag in labels:
            if tag == label:
                new_arr[index] = 1.0
    return new_arr
