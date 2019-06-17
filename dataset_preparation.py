import os
import random
import sys
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
from math import sqrt
import json	
from sklearn.preprocessing import OneHotEncoder
import skimage.io as io
from PIL import Image

def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def load_image(image_path):
  with tf.gfile.FastGFile(image_path, 'rb') as reader:
    curr_img = reader.read()
  return curr_img

def convert_to_tfrecords(images, labels, file_name):
  writer = tf.python_io.TFRecordWriter(file_name)

  for index in range(len(images)):

    if not index % 1000:
      print("Data Converted for {}: {}/{}".format(file_name, index, len(images)))

    img = load_image(images[index])
    label = labels[index]
    height = 224
    width = 224
    
    dataset_features = {
      'image_raw': _bytes_feature(img),
      'filename': _bytes_feature(os.path.basename(images[index]).encode('utf-8')),
      'label': _int64_feature(int(label)),
      'height': _int64_feature(height),
      'width': _int64_feature(width),
      'channels': _int64_feature(3) 
    }

    example = tf.train.Example(features=tf.train.Features(feature=dataset_features))

    writer.write(example.SerializeToString())
  
  writer.close()

def split_data(data, proportion):
    """
    Split a numpy array into two parts of `proportion` and `1 - proportion`
    
    Args:
        - data: numpy array, to be split along the first axis
        - proportion: a float less than 1
    """
    size = data.shape[0]
    split_idx = int(proportion * size)
    return data[:split_idx], data[split_idx:]

def convert_labels_to_numeric(labels):
  unique_label = list(set(labels))
  label_dict = {}

  for index, i in enumerate(unique_label):
    label_dict[i] = index

  for index, i in enumerate(labels):
    if i in label_dict:
      labels[index] = label_dict[i]

  with open('labels_code.txt', 'w') as f:
    f.write(json.dumps(label_dict))

  return labels

def main(cli_args):
    #Get the data from the dataset
    DATASET_PATH = os.path.join("../Datasets/224_dataset_ori_rgb/")
    images = np.array(glob.glob(os.path.join(DATASET_PATH, '*', '*.JPG')))
    labels = np.array([os.path.basename(os.path.dirname(path)) for path in images])
    labels = convert_labels_to_numeric(labels)
    
    permutation = np.random.permutation(images.shape[0])
    images = images[permutation]
    labels = labels[permutation]

    train_imgs, test_imgs = split_data(images, 0.8)
    test_imgs, valid_imgs = split_data(test_imgs, 0.5)

    train_label, test_label = split_data(labels, 0.8)
    test_label, valid_label = split_data(test_label, 0.5)

    convert_to_tfrecords(train_imgs, train_label, "train_dataset.tfrecord")
    convert_to_tfrecords(test_imgs, test_label, "test_dataset.tfrecord")
    convert_to_tfrecords(valid_imgs, valid_label, "valid_dataset.tfrecord")

if __name__ == "__main__":
    main(sys.argv[1:])