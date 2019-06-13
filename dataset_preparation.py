import os
import random
import sys
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
from math import sqrt	
from sklearn.preprocessing import OneHotEncoder

def split_data(data, proportion):
    """
    Split a numpy array into two parts of `proportion` and `1 - proportion`
    
    Args:
        - data: numpy array, to be split along the first axis
        - proportion: a float less than 1
    """
    size = len(data)
    split_idx = int(proportion * size)
    return data[:split_idx], data[split_idx:]

def load_images(image):
    img = open(image, 'rb').read()
    return img

#Methods from Tensorflow's example
def _float64_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def make_train_TFRecords(train_imgs, train_labels):
    file_name = "train_data.tfrecords"
    options = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.GZIP)
    writer = tf.python_io.TFRecordWriter(file_name, options=options)
    
    for i in range(len(train_imgs)):
        
        if not i % 1000:
            print("Data (Training) : {} / {}".format(i , len(train_imgs)))

        #Load the image
        curr_img = load_images(train_imgs[i])
        curr_label = train_labels[i]
        #Create a feature 
        feature = {'train/label': _bytes_feature(tf.compat.as_bytes(curr_label.tostring())),
                   'train/image': _bytes_feature(tf.compat.as_bytes(curr_img))}

        protocol_buff = tf.train.Example(features=tf.train.Features(feature= feature))

        writer.write(protocol_buff.SerializeToString())

    writer.close()

def make_test_TFRecords(test_imgs, test_labels):
    file_name = "test_data.tfrecords"
    options = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.GZIP)
    writer = tf.python_io.TFRecordWriter(file_name, options=options)
    
    for i in range(len(test_imgs)):
        
        if not i % 1000:
            print("Data (Testing) : {} / {}".format(i , len(test_imgs)))

        #Load the image
        curr_img = load_images(test_imgs[i])
        curr_label = test_labels[i]
        #Create a feature 
        feature = {'test/label': _bytes_feature(tf.compat.as_bytes(curr_label.tostring())),
                   'test/image': _bytes_feature(tf.compat.as_bytes(cv2.imencode('.jpg', curr_img).tostring()))}

        protocol_buff = tf.train.Example(features=tf.train.Features(feature= feature))

        writer.write(protocol_buff.SerializeToString())

    writer.close()

def make_valid_TFRecords(valid_imgs, valid_labels):
    file_name = "valid_data.tfrecords"
    options = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.GZIP)
    writer = tf.python_io.TFRecordWriter(file_name, options=options)
    
    for i in range(len(valid_imgs)):
        
        if not i % 1000:
            print("Data (Validation) : {} / {}".format(i , len(valid_imgs)))

        #Load the image
        curr_img = load_images(valid_imgs[i])
        curr_label = valid_labels[i]

        #Create a feature 
        feature = {'valid/label': _bytes_feature(tf.compat.as_bytes(curr_label.tostring())),
                   'valid/image': _bytes_feature(tf.compat.as_bytes(cv2.imencode('.jpg', curr_img).tostring()))}

        protocol_buff = tf.train.Example(features=tf.train.Features(feature= feature))

        writer.write(protocol_buff.SerializeToString())

    writer.close()


def main(cli_args):
    #Constants
    #Get the data from the dataset
    DATASET_PATH = os.path.join("../Datasets/224_dataset_ori_rgb/")
    images = glob.glob(os.path.join(DATASET_PATH, '*', '*.JPG'))
    labels = [os.path.basename(os.path.dirname(path)) for path in images]
    encoder = OneHotEncoder(handle_unknown='ignore')
    labels = encoder.fit_transform(list(zip(images, labels))).toarray()

    #Shuffle the data
    data = list(zip(images, labels))
    random.shuffle(data)
    images, labels = zip(*data)
    
    #Split the data into ratios
    train_imgs, test_imgs = split_data(images, 0.8)
    test_imgs, valid_imgs = split_data(images, 0.5)

    train_labels, test_labels = split_data(labels, 0.8)
    test_labels, valid_labels = split_data(labels, 0.5)
    #Create a TFRecords File
    make_train_TFRecords(train_imgs, train_labels)
    make_test_TFRecords(test_imgs, test_labels)
    make_valid_TFRecords(valid_imgs, valid_labels)

if __name__ == "__main__":
    main(sys.argv[1:])