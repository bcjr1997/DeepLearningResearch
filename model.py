import os
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

def initiate_vgg_model(x, num_classes):
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

