import argparse
import os
import parser
import sys
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import csv
import cv2
import pandas as pd
from model import initiate_vgg_model
import util
import matplotlib.image as mpimg

def read_and_decode_tfrecords(tfrecord_file):
    features = {
        'image_raw': tf.FixedLenFeature([], tf.string),
        'filename': tf.FixedLenFeature([], tf.string),
        'label': tf.FixedLenFeature([], tf.int64),
        'height': tf.FixedLenFeature([], tf.int64),
        'width': tf.FixedLenFeature([], tf.int64),
        'channels': tf.FixedLenFeature([], tf.int64),
    }

    curr_example = tf.parse_single_example(tfrecord_file, features)
    image_shape = tf.stack([curr_example['height'], curr_example['width'], curr_example['channels']])
    image = tf.image.decode_image(curr_example['image_raw'])
    label = curr_example['label']
    filename = curr_example['filename']
    return [image, image_shape, label, filename]

#Argsparse
def main(cli_args):
    #Constants
    DATASET_PATH = os.path.join("../Datasets/serengeti_dataset_tfrecords/")
    LEARNING_RATE = 0.0001
    EPOCHS = 100
    STOPPING_COUNT = 50 
    BATCH_SIZE = 32
    NUM_CLASSES = 47 
    Z_SCORE = 1.96

    dataset = tf.data.TFRecordDataset(['train_dataset.tfrecord'])
    dataset = dataset.map(read_and_decode_tfrecords)
    iterator = dataset.make_one_shot_iterator()
    el = iterator.get_next()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        try:
            while True:
                data = sess.run(el)
                if not np.array_equal(data[0].shape, data[1]):
                    print("{} and {} are different".format(data[0].shape, data[1]))
        except:
            pass


    
if __name__ == "__main__":
    main(sys.argv[1:])
