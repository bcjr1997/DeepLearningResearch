import argparse
import os
import parser
import sys
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import csv
import cv2
import glob
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
    LEARNING_RATE = 0.01
    EPOCHS = 55
    BATCH_SIZE = 128
    NUM_CLASSES = 47 
    Z_SCORE = 1.96
    WEIGHT_DECAY = 0.0005

    print("Current Setup:-")
    print("Learning Rate: {}, Epochs: {}, Batch Size: {}, Confidence Interval Z-Score {}, Number of classes: {}, Weight Decay: {}".format(LEARNING_RATE, EPOCHS, BATCH_SIZE, Z_SCORE, NUM_CLASSES, WEIGHT_DECAY))

    #Placeholders
    x = tf.placeholder(tf.float32, [None, 224, 224, 3], name='input_placeholder')
    y = tf.placeholder(tf.float32, [None, NUM_CLASSES], name='labels')

    #Train Dataset
    dataset_len = np.array(glob.glob(os.path.join("../Datasets/224_dataset_ori_rgb/", '*', '*.JPG')))
    train_dataset_len = dataset_len.shape[0] * 0.8
    train_dataset = tf.data.TFRecordDataset([os.path.join(DATASET_PATH, 'train_dataset.tfrecord')])
    train_dataset = train_dataset.map(read_and_decode_tfrecords)
    #train_dataset = train_dataset.shuffle(buffer_size=10000, reshuffle_each_iteration=True)
    train_dataset = train_dataset.batch(BATCH_SIZE)
    train_dataset = train_dataset.repeat(EPOCHS)
    train_iterator = train_dataset.make_one_shot_iterator()
    train_el = train_iterator.get_next()

    #Valid Dataset
    valid_dataset = tf.data.TFRecordDataset([os.path.join(DATASET_PATH, 'valid_dataset.tfrecord')])
    valid_dataset = valid_dataset.map(read_and_decode_tfrecords)
    #valid_dataset = valid_dataset.shuffle(buffer_size=10000, reshuffle_each_iteration=True)
    valid_dataset = valid_dataset.batch(BATCH_SIZE)
    valid_dataset = valid_dataset.repeat(EPOCHS)
    valid_iterator = valid_dataset.make_one_shot_iterator()
    valid_el = valid_iterator.get_next()

    #Test Dataset
    test_dataset = tf.data.TFRecordDataset([os.path.join(DATASET_PATH, 'test_dataset.tfrecord')])
    test_dataset = test_dataset.map(read_and_decode_tfrecords)
    #test_dataset = test_dataset.shuffle(buffer_size=10000, reshuffle_each_iteration=True)
    test_dataset = test_dataset.batch(BATCH_SIZE)
    test_dataset = test_dataset.repeat(EPOCHS)
    test_iterator = test_dataset.make_one_shot_iterator()
    test_el = test_iterator.get_next()

    #Model
    _, outputLayer = initiate_vgg_model(x, NUM_CLASSES)

    optimizer = tf.contrib.opt.AdamWOptimizer(weight_decay=WEIGHT_DECAY, learning_rate=LEARNING_RATE, name="AdamWeightDecay")
    cross_entropy = util.cross_entropy_op(y, outputLayer)
    global_step_tensor = util.global_step_tensor('global_step_tensor')
    train_op = util.train_op(cross_entropy, global_step_tensor, optimizer)
    conf_matrix = util.confusion_matrix_op(y, outputLayer, NUM_CLASSES)
    saver = tf.train.Saver()

    if not os.path.exists(os.path.join("./dl_research/")):
        os.mkdir(os.path.join("./dl_research/"))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(EPOCHS):

            if epoch == 18:
                LEARNING_RATE = 0.005
                print("Learning Rate changed to {} at epoch {}".format(LEARNING_RATE, epoch))
            elif epoch == 29:
                LEARNING_RATE = 0.001
                WEIGHT_DECAY = 0.0
                print("Learning Rate changed to {} at epoch {}".format(LEARNING_RATE, epoch))
                print("Weight Decay changed to {} at epoch {}".format(WEIGHT_DECAY, epoch))
            elif epoch == 42:
                LEARNING_RATE = 0.0005
                print("Learning Rate changed to {} at epoch {}".format(LEARNING_RATE, epoch))
            elif epoch == 51:
                LEARNING_RATE = 0.0001
                print("Learning Rate changed to {} at epoch {}".format(LEARNING_RATE, epoch))

            print("Current Epoch: {}".format(epoch))

            train_highest_acc = 0
            for i in range(int(train_dataset_len/BATCH_SIZE)):
                print("Current Training Iteration : {}/{}".format(i, int(train_dataset_len/BATCH_SIZE)))
                train_image_data = sess.run(train_el)
                train_label = util.one_hot_encoding(train_image_data[2],NUM_CLASSES)
                train_acc, train_conf_mtx = util.training(BATCH_SIZE, x, y, train_image_data[0]/255, train_label, sess, train_op, conf_matrix, NUM_CLASSES)
                value1, value2 = util.confidence_interval(train_acc, Z_SCORE, train_image_data[0].shape[0])
                print("Training Confidence Interval: [{} , {}]".format(value2, value1))
                
                if train_highest_acc <= train_acc:
                    train_highest_acc = train_acc
                    print("Highest Training Accuracy Reached: {}".format(train_highest_acc))
            
            valid_highest_acc = 0
            for i in range(int((dataset_len.shape[0] * 0.1)/BATCH_SIZE)):
                print("Current Validation Iteration : {}/{}".format(i, int((dataset_len.shape[0] * 0.1)/BATCH_SIZE)))
                valid_image_data = sess.run(valid_el)
                valid_label = util.one_hot_encoding(valid_image_data[2], NUM_CLASSES)
                valid_acc, valid_conf_mtx, valid_avg_valid_ce = util.validation(BATCH_SIZE, x, y, valid_image_data[0]/255, valid_label, sess, cross_entropy, conf_matrix, NUM_CLASSES)
                value1, value2 = util.confidence_interval(valid_acc, Z_SCORE, valid_image_data[0].shape[0])
                print("Validation Confidence Interval: [{} , {}]".format(value2, value1))

                if valid_highest_acc <= valid_acc:
                    valid_highest_acc = valid_acc
                    print("Highest Training Accuracy Reached: {}".format(valid_highest_acc))

            test_highest_acc = 0
            for i in range(int((dataset_len.shape[0] * 0.1)/BATCH_SIZE)):
                print("Current Testing Iteration : {}/{}".format(i, int((dataset_len.shape[0] * 0.1)/BATCH_SIZE)))
                test_image_data = sess.run(test_el)
                test_label = util.one_hot_encoding(test_image_data[2], NUM_CLASSES)
                test_acc, test_ce, test_conf_mtx = util.test(BATCH_SIZE, x, y, test_image_data[0]/255, test_label, sess, cross_entropy, conf_matrix, NUM_CLASSES)
                value1, value2 = util.confidence_interval(test_acc, Z_SCORE, test_image_data[0].shape[0])
                print("Testing Confidence Interval: [{} , {}]".format(value2, value1))

                if test_highest_acc <= test_acc:
                    test_highest_acc = test_acc
                    print("Highest Training Accuracy Reached: {}".format(test_highest_acc))

            #For every epoch, we will save the model
            saver.save(sess, os.path.join("./dl_research/", "dl_research"))
    
if __name__ == "__main__":
    main(sys.argv[1:])
