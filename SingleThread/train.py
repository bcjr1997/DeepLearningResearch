import argparse
import os
import parser
import sys
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import csv
import glob
import pandas as pd
from model import initiate_vgg_model
import util
from math import floor

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
def main():
    #Constants
    DATASET_PATH = os.path.join("/content/drive/'My Drive'/dataset")
    LEARNING_RATE_1 = 0.01
    EPOCHS = 55
    BATCH_SIZE = 128
    NUM_CLASSES = 47 
    Z_SCORE = 1.96
    WEIGHT_DECAY_1 = 0.0005

    print("Current Setup:-")
    print("Starting Learning Rate: {}, Epochs: {}, Batch Size: {}, Confidence Interval Z-Score {}, Number of classes: {}, Starting Weight Decay: {}".format(LEARNING_RATE_1, EPOCHS, BATCH_SIZE, Z_SCORE, NUM_CLASSES, WEIGHT_DECAY_1))

    #Placeholders
    x = tf.placeholder(tf.float32, [None, 224, 224, 3], name='input_placeholder')
    y = tf.placeholder(tf.float32, [None, NUM_CLASSES], name='labels')
    learning_rate = tf.placeholder(tf.float32, shape=[], name='learning_rate')
    weight_decay = tf.placeholder(tf.float32, shape=[], name="weight_decay")

    #Train Dataset
    dataset_len = 515190
    NUM_SHARDS = 25
    train_dataset_len = dataset_len * 0.8
    train_dataset = tf.data.TFRecordDataset([os.path.join('/content/drive/My Drive/dataset', 'train_dataset.tfrecord')])
    #train_dataset = train_dataset.interleave(lambda x: tf.data.TFRecordDataset(x), cycle_length = 1)
    #print(train_dataset)
    train_dataset = train_dataset.map(read_and_decode_tfrecords)
    #train_dataset = train_dataset.shuffle(buffer_size=10000, reshuffle_each_iteration=True)
    #train_dataset = train_dataset.shuffle(buffer_size = floor(train_dataset_len/NUM_SHARDS) + 1)
    train_dataset = train_dataset.batch(BATCH_SIZE)
    train_iterator = train_dataset.make_initializable_iterator()
    train_el = train_iterator.get_next()

    #Model
    _, outputLayer = initiate_vgg_model(x, NUM_CLASSES)

    optimizer = tf.contrib.opt.AdamWOptimizer(weight_decay=weight_decay, learning_rate=learning_rate, name="AdamWeightDecay")
    cross_entropy = util.cross_entropy_op(y, outputLayer)
    global_step_tensor = util.global_step_tensor('global_step_tensor')
    train_op = util.train_op(cross_entropy, global_step_tensor, optimizer)
    conf_matrix = util.confusion_matrix_op(y, outputLayer, NUM_CLASSES)
    saver = tf.train.Saver()

    if not os.path.exists(os.path.join("./dl_research_train/")):
        os.mkdir(os.path.join("./dl_research_train/"))

    train_df = pd.DataFrame({'epoch': [], 'iteration':[], 'train_accuracy': [], 'train_conf_matrix':[],
                            'positive_confidence_interval': [], 'negative_confidence_interval': [], 'learning_rate': [],
                            'weight_decay': []})

    with tf.Session() as sess:
        with np.printoptions(threshold=np.inf):
            merged_summary = tf.summary.merge_all()
            writer = tf.summary.FileWriter("./tensorboard_logs/")
            writer.add_graph(sess.graph)
            train_highest_acc = 0 
            sess.run(tf.global_variables_initializer())
            for epoch in range(EPOCHS):

                if epoch == 18:
                    LEARNING_RATE_1 = 0.005
                    print("Learning Rate changed to {} at epoch {}".format(LEARNING_RATE_1, epoch))
                elif epoch == 29:
                    LEARNING_RATE_1 = 0.001
                    WEIGHT_DECAY_1 = 0.0
                    print("Learning Rate changed to {} at epoch {}".format(LEARNING_RATE_1, epoch))
                    print("Weight Decay changed to {} at epoch {}".format(WEIGHT_DECAY_1, epoch))
                elif epoch == 42:
                    LEARNING_RATE_1 = 0.0005
                    print("Learning Rate changed to {} at epoch {}".format(LEARNING_RATE_1, epoch))
                elif epoch == 51:
                    LEARNING_RATE_1 = 0.0001
                    print("Learning Rate changed to {} at epoch {}".format(LEARNING_RATE_1, epoch))

                print("Current Epoch: {}".format(epoch))

                sess.run(train_iterator.initializer)
                for i in range(int(train_dataset_len/BATCH_SIZE)):
                    print("Current Training Iteration : {}/{}".format(i, int(train_dataset_len/BATCH_SIZE)))
                    train_image_data = sess.run(train_el)
                    train_label = util.one_hot_encoding(train_image_data[2],NUM_CLASSES)
                    train_acc, train_conf_mtx, summary = util.training(BATCH_SIZE, x, y, learning_rate, weight_decay, train_image_data[0]/255, train_label, sess, train_op, conf_matrix, NUM_CLASSES, LEARNING_RATE_1, WEIGHT_DECAY_1, merged_summary)
                    train_value1, train_value2 = util.confidence_interval(train_acc, Z_SCORE, train_image_data[0].shape[0])
                    print("Training Confidence Interval: [{} , {}]".format(train_value2, train_value1))
                    output_data = {'epoch': epoch, 'iteration':i, 'train_accuracy': train_acc, 'train_conf_matrix':train_conf_mtx,
                                   'positive_confidence_interval': train_value1, 'negative_confidence_interval': train_value2, 'learning_rate': LEARNING_RATE_1,
                                   'weight_decay': WEIGHT_DECAY_1}
                    train_df = train_df.append(output_data, ignore_index=True)
                    
                    if train_highest_acc <= train_acc:
                        train_highest_acc = train_acc
                        print("Highest Training Accuracy Reached: {}".format(train_highest_acc))
                        #For every epoch, we will save the model
                        saver.save(sess, os.path.join("./dl_research_train/", "model.ckpt"))
                        writer.add_summary(summary, epoch)
                        print("Latest Model is saving and Tensorboard Logs are updated")
            
            train_df.to_csv(r"./train_results.csv", header=True, index=False, encoding='utf-8')      

if __name__ == "__main__":
    main()   
