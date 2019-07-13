# -*- coding: utf-8 -*-
"""distributed_training

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1Ke4_gVmlV3m7r1e6Xaj_gQXJRjlmh8oV
"""

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
#from model import initiate_vgg_model
#import util
from math import floor

from google.colab import drive
drive.mount('/content/drive')

#Model, x = placeholder
def start_model(x, num_classes):
    #VGG Architecture
    with tf.name_scope('vgg_arch_model') as scope:
        conv_1 = tf.layers.conv2d(x, 64, 3, padding='same', activation=tf.nn.relu,
                                    kernel_regularizer = tf.contrib.layers.l2_regularizer(scale=0.01),
                                    bias_regularizer=tf.contrib.layers.l2_regularizer(scale=0.01),
                                    name='conv_1')

        max_pool_1 = tf.layers.max_pooling2d(conv_1, 2, 2, padding='same')

        conv_2 = tf.layers.conv2d(max_pool_1, 128, 3, padding='same', activation=tf.nn.relu,
                                    kernel_regularizer = tf.contrib.layers.l2_regularizer(scale=0.01),
                                    bias_regularizer=tf.contrib.layers.l2_regularizer(scale=0.01),
                                    name='conv_2')

        max_pool_2 = tf.layers.max_pooling2d(conv_2, 2, 2, padding='same') 

        conv_3 = tf.layers.conv2d(max_pool_2, 256, 3, padding='same', activation=tf.nn.relu,
                                    kernel_regularizer = tf.contrib.layers.l2_regularizer(scale=0.01),
                                    bias_regularizer=tf.contrib.layers.l2_regularizer(scale=0.01),
                                    name='conv_3')

        conv_4 = tf.layers.conv2d(conv_3, 256, 3, padding='same', activation=tf.nn.relu,
                                    kernel_regularizer = tf.contrib.layers.l2_regularizer(scale=0.01),
                                    bias_regularizer=tf.contrib.layers.l2_regularizer(scale=0.01),
                                    name='conv_4')

        max_pool_3 = tf.layers.max_pooling2d(conv_4, 2, 2, padding='same')

        conv_5 = tf.layers.conv2d(max_pool_3, 512, 3, padding='same', activation=tf.nn.relu,
                                    kernel_regularizer = tf.contrib.layers.l2_regularizer(scale=0.01),
                                    bias_regularizer=tf.contrib.layers.l2_regularizer(scale=0.01),
                                    name='conv_5')

        conv_6 = tf.layers.conv2d(conv_5, 512, 3, padding='same', activation=tf.nn.relu,
                                    kernel_regularizer = tf.contrib.layers.l2_regularizer(scale=0.01),
                                    bias_regularizer=tf.contrib.layers.l2_regularizer(scale=0.01),
                                    name='conv_6')

        max_pool_4 = tf.layers.max_pooling2d(conv_6, 2, 2, padding='same')

        conv_7 = tf.layers.conv2d(max_pool_4, 512, 3, padding='same', activation=tf.nn.relu,
                                    kernel_regularizer = tf.contrib.layers.l2_regularizer(scale=0.01),
                                    bias_regularizer=tf.contrib.layers.l2_regularizer(scale=0.01),
                                    name='conv_7')

        conv_8 = tf.layers.conv2d(conv_7, 512, 3, padding='same', activation=tf.nn.relu,
                                    kernel_regularizer = tf.contrib.layers.l2_regularizer(scale=0.01),
                                    bias_regularizer=tf.contrib.layers.l2_regularizer(scale=0.01),
                                    name='conv_8')

        max_pool_5 = tf.layers.max_pooling2d(conv_8, 2, 2, padding='same')

        flat_layer = tf.contrib.layers.flatten(max_pool_5)
        #inclusion of batch normalization vs dropout based on vgg.py model?
        fc_layer_1 = tf.layers.dense(flat_layer,4096,activation=tf.nn.relu, name='fc_layer_1')

        #Batch Normalization
        batch_norm_layer_1 = tf.layers.batch_normalization(fc_layer_1, training=True)
        #Dropout
        dropout_layer_1 = tf.layers.dropout(batch_norm_layer_1, 0.5)

        fc_layer_2 = tf.layers.dense(dropout_layer_1,4096,activation=tf.nn.relu, name='fc_layer_2')

        #Batch Normalization
        batch_norm_layer_2 = tf.layers.batch_normalization(fc_layer_2, training=True)
        #Dropout
        dropout_layer_2 = tf.layers.dropout(batch_norm_layer_2, 0.5)

        #num_output is not yet defined
        #no activation function on the output?
        output = tf.layers.dense(dropout_layer_2, num_classes, name='output')

    tf.identity(output, name='output')
    return scope, output

from tensorflow.python.client import device_lib

def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type]
  
get_available_gpus()

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
  
def get_training_dataset(DATASET_PATH, BATCH_SIZE, EPOCHS):
  dataset_len = 515190
  train_dataset_len = dataset_len * 0.8
  train_dataset = tf.data.TFRecordDataset([os.path.join(DATASET_PATH, 'test_dataset.tfrecord')])
  train_dataset = train_dataset.map(read_and_decode_tfrecords)
  #train_dataset = train_dataset.shuffle(buffer_size=10000, reshuffle_each_iteration=True)
  train_dataset = train_dataset.batch(BATCH_SIZE)
  train_dataset = train_dataset.repeat(EPOCHS)
  train_iterator = train_dataset.make_initializable_iterator()
  train_el = train_iterator.get_next()
  return train_iterator, train_el

def cross_entropy_op(y_placeholder, output):
	cross_ent = tf.nn.softmax_cross_entropy_with_logits(labels=y_placeholder, logits=output, name="cross_entropy")
	tf.summary.histogram("cross_entropy", cross_ent)
	return cross_ent

def train_op(cross_entropy_op, global_step_tensor, optimizer):
	training_operation = optimizer.minimize(cross_entropy_op, global_step=global_step_tensor, name="training_op")
	return training_operation

def global_step_tensor_op(name):
	global_step_tensor = tf.get_variable(
	name, 
	trainable=False, 
	shape=[], 
	initializer=tf.zeros_initializer)
	return global_step_tensor

def one_hot_encoding(labels, num_classes):
	return np.eye(num_classes)[labels.astype(int)]

PS_OPS = [
    'Variable', 'VariableV2', 'AutoReloadVariable', 'MutableHashTable',
    'MutableHashTableOfTensors', 'MutableDenseHashTable'
]
    
# see https://github.com/tensorflow/tensorflow/issues/9517
def assign_to_device(device, ps_device):
    """Returns a function to place variables on the ps_device.

    Args:
        device: Device for everything but variables
        ps_device: Device to put the variables on. Example values are /GPU:0 and /CPU:0.

    If ps_device is not set then the variables will be placed on the default device.
    The best device for shared varibles depends on the platform as well as the
    model. Start with CPU:0 and then test GPU:0 to see if there is an
    improvement.
    """
    def _assign(op):
        node_def = op if isinstance(op, tf.NodeDef) else op.node_def
        if node_def.op in PS_OPS:
            return ps_device
        else:
            return device
    return _assign
  
def run_training_step():
  #Constants
  DATASET_PATH = os.path.join(".")
  LEARNING_RATE = 0.01
  EPOCHS = 55
  BATCH_SIZE = 128
  NUM_CLASSES = 47 
  Z_SCORE = 1.96
  WEIGHT_DECAY = 0.0005
  
  with tf.Session() as sess , tf.device("/device:CPU:0"):
      #Get data from TFrecords
      train_iterator, train_el = get_training_dataset(DATASET_PATH, BATCH_SIZE, EPOCHS)
    
      #Placeholders
      x = tf.placeholder(tf.float32, [None, 224, 224, 3], name='input_placeholder')
      y = tf.placeholder(tf.float32, [None, NUM_CLASSES], name='labels')
      learning_rate = tf.placeholder(tf.float32, shape=[], name='learning_rate')
      weight_decay = tf.placeholder(tf.float32, shape=[], name="weight_decay")
    
      #Model
      _, outputLayer = start_model(x, NUM_CLASSES)

      optimizer = tf.contrib.opt.AdamWOptimizer(weight_decay=weight_decay, learning_rate=learning_rate, name="AdamWeightDecay")
      global_step_tensor = global_step_tensor_op('global_step_tensor')
      cross_entropy = cross_entropy_op(y, outputLayer)
      update_opt = train_op(cross_entropy, global_step_tensor, optimizer)
      sess.run([tf.global_variables_initializer(), train_iterator.initializer])
      
      for i in range(5):
        train_image_data = sess.run(train_el)
        train_label = one_hot_encoding(train_image_data[2], NUM_CLASSES)
        with tf.device(assign_to_device("/device:XLA_GPU:0", "/device:XLA_CPU:0")):
          _, loss_val = sess.run([update_opt, tf.reduce_mean(cross_entropy)], feed_dict={x: train_image_data[0]/255, y: train_label, learning_rate: LEARNING_RATE, weight_decay: WEIGHT_DECAY })
          print(f"Step : {i} , Loss : {loss_val}")

#Main function
tf.reset_default_graph()
run_training_step()