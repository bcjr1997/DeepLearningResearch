import os
import tensorflow as tf
import util
import matplotlib.pyplot as plt
import numpy as np

def initiate_vgg_model(features, labels, num_classes, weight_decay, learning_rate, handle="training"):
    #VGG Architecture
    with tf.name_scope('vgg_arch_model') as scope:
        #Declare feature input for the model
        x_placeholder = tf.reshape(features, [-1, 256, 256, 3])
        conv_1 = tf.layers.conv2d(x_placeholder, 64, 3, padding='same', activation=tf.nn.relu,
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
        logits = tf.layers.dense(dropout_layer_2, num_classes, name='output')

    tf.identity(logits, name='output')

    #Declare Loss and Optimizer functions
    label_placeholder = labels
    optimizer = tf.contrib.opt.MomentumWOptimizer(weight_decay=weight_decay, learning_rate=learning_rate, momentum=0.9,  name="MomentumWeightDecay")
    global_step_tensor = util.global_step_tensor('global_step_tensor')

    with tf.name_scope("cost"):
        cross_entropy = util.cross_entropy_op(label_placeholder, logits, "cross_entropy")
        train_op = util.train_op(cross_entropy, global_step_tensor, optimizer)
        tf.summary.scalar(str(handle+"_loss"), cross_entropy)

    with tf.name_scope("confusion_matrix"):
        conf_matrix_op = util.confusion_matrix_op(label_placeholder, logits, num_classes)
        conf_mtx = tf.reshape(tf.cast(conf_matrix_op, tf.float32), [1, num_classes, num_classes, 1])
        tf.summary.image('confusion_image', conf_mtx)

    with tf.name_scope("accuracy"):
        accuracy = tf.equal(tf.argmax(logits, 1), tf.argmax(label_placeholder,1))
        accuracy = tf.reduce_mean(tf.cast(accuracy, tf.float32))
        tf.summary.scalar(str(handle+"_accuracy"), accuracy)
    return scope, train_op, cross_entropy, conf_matrix_op, accuracy, tf.round(logits), tf.round(label_placeholder)

