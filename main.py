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

#Argsparse
def main(cli_args):

    #Constants
    DATASET_PATH = os.path.join("./greyscale_dataset/")
    LEARNING_RATE = 0.0001
    EPOCHS = 100
    STOPPING_COUNT = 50 
    BATCH_SIZE = 32
    NUM_CLASSES = 47 
    Z_SCORE = 1.96

    #Load data
    images, labels, unique_labels = util.load_data(DATASET_PATH)
    #Check to see if images and labels share the same length
    assert images.shape[0] == labels.shape[0]

    #Placeholders
    x = tf.placeholder(tf.float32, [None, 224, 224, 3], name='input_placeholder')
    y = tf.placeholder(tf.float32, [None, len(unique_labels)], name='labels')

    #Dataset
    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    iterator = dataset.make_initializable_iterator()
    next_element = iterator.get_next()

    #REMINDER: tf.global_variables_initializer() does not initialise tf.data iterator
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(iterator.initializer, feed_dict={x: images, y:labels})
        for i in range(10):
            value = sess.run(next_element)
            print(value)

      
if __name__ == "__main__":
    main(sys.argv[1:])
