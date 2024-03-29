import argparse
import os
import parser
import sys
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import glob
import util
from basic_model import initiate_vgg_model
from math import floor
from parse_data import get_data
import itertools


import os, os.path as osp, sys
import argparse

def get_parser():
    parser = argparse.ArgumentParser(
        description = 'Deep learning multiclass classification'
    )
    parser.add_argument('-dataset', '--datasetPath',
        help = 'Path to dataset'
    )

    parser.add_argument('-rate', '--learningRate',
        help = 'Set the learning rate'
    )

    parser.add_argument('-epochs', '--epochs',
        help = 'Set the epochs'
    )

    parser.add_argument('-batch-size', '--batchSize',
        help = 'Set the batch size'
    )

    parser.add_argument('-classes', '--numClasses',
        help = 'Set the number of classes'
    )

    parser.add_argument('-zscore', '--zScore',
        help = 'Set the Z score'
    )

    parser.add_argument('-decay', '--weightDecay',
        help = 'Set the weight decay'
    )

    main(parser)


#Argsparse
def main(args_parser):
    #Dataset
    parser = args_parser
    args   = parser.parse_args()
    
    train_image_data, train_label_data, train_filename, valid_image_data, valid_label_data, valid_filename, unique_classes = get_data()
    #tf.reset_default_graph()
    DATASET_PATH = args.datasetPath
    LEARNING_RATE_1 = args.learningRate
    EPOCHS = args.epochs
    BATCH_SIZE = args.batchSize
    NUM_CLASSES = len(unique_classes) 
    Z_SCORE = args.zScore
    WEIGHT_DECAY_1 = args.weightDecay

    print("Current Setup:-")
    print("Starting Learning Rate: {}, Epochs: {}, Batch Size: {}, Confidence Interval Z-Score {}, Number of classes: {}, Starting Weight Decay: {}".format(LEARNING_RATE_1, EPOCHS, BATCH_SIZE, Z_SCORE, NUM_CLASSES, WEIGHT_DECAY_1))

    #Placeholders
    learning_rate = tf.placeholder(tf.float32, shape=[], name='learning_rate')
    weight_decay = tf.placeholder(tf.float32, shape=[], name="weight_decay")

    #Dataset
    training_dataset = tf.data.Dataset.from_generator(lambda: itertools.zip_longest(train_image_data, train_label_data, train_filename),
                                                            output_types=(tf.float32, tf.float32),
                                                            output_shapes=(tf.TensorShape([None, None, 3]), 
                                                                        tf.TensorShape([None]),
                                                                        tf.TensorShape([None])))
                                                                        
    training_dataset = training_dataset.repeat(EPOCHS).batch(BATCH_SIZE).prefetch(1)
    train_iterator = training_dataset.make_initializable_iterator()
    train_features, train_labels, train_filename = train_iterator.get_next()

    valid_dataset = tf.data.Dataset.from_generator(lambda: itertools.zip_longest(valid_image_data, valid_label_data, valid_filename),
                                                            output_types=(tf.float32, tf.float32),
                                                            output_shapes=(tf.TensorShape([None, None, 3]), 
                                                                        tf.TensorShape([None]),
                                                                        tf.TensorShape([None])))
                                                                        
    valid_dataset = valid_dataset.repeat(EPOCHS).batch(BATCH_SIZE).prefetch(1)
    valid_iterator = valid_dataset.make_initializable_iterator()
    valid_features, valid_labels, valid_filename = valid_iterator.get_next()

    #Model
    _, train_op, train_cross_entropy, train_conf_matrix_op, train_accuracy = initiate_vgg_model(train_features, train_labels, train_filename, NUM_CLASSES, weight_decay, learning_rate, handle="training", reuse_model=None)
    _, _, valid_cross_entropy, valid_conf_matrix_op, valid_accuracy = initiate_vgg_model(valid_features, valid_labels, valid_filename, NUM_CLASSES, weight_decay, learning_rate, handle="validation", reuse_model=True)
    
    saver = tf.train.Saver()

    if not os.path.exists(os.path.join( "./short_dl_research_train/")):
        os.mkdir(os.path.join("./short_dl_research_train/"))
    
    with tf.Session() as sess:
        with np.printoptions(threshold=np.inf):
            train_writer = tf.summary.FileWriter("./short_tensorboard_training_logs/")
            valid_writer = tf.summary.FileWriter("./short_tensorboard_validation_logs/")
            train_writer.add_graph(sess.graph)
            valid_writer.add_graph(sess.graph)
            train_highest_acc = 0
            valid_highest_acc = 0
            sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])

            for epoch in range(EPOCHS):
                print("Current Epoch: {}/{}".format(epoch, EPOCHS))
                i = 0
                try:
                    sess.run(train_iterator.initializer)
                    while True:
                        print("Current Training Iteration : {}/{}".format(i, floor(int(157252)/BATCH_SIZE)))
                        train_acc, _, _, train_ce = util.training(BATCH_SIZE, NUM_CLASSES,learning_rate, weight_decay, sess, train_op, train_conf_matrix_op, LEARNING_RATE_1, WEIGHT_DECAY_1, train_cross_entropy, train_accuracy)
                        train_value1, train_value2 = util.confidence_interval(train_acc, Z_SCORE, BATCH_SIZE)
                        print("Training Accuracy : {}".format(train_acc))
                        print("Training Loss (Cross Entropy) : {}".format(train_ce))
                        print("Training Confidence Interval: [{} , {}]".format(train_value2, train_value1))
                        if train_highest_acc <= train_acc:
                            train_highest_acc = train_acc
                            print("Highest Training Accuracy Reached: {}".format(train_highest_acc))
                        #For every epoch, we will save the model
                            saver.save(sess, os.path.join("./short_dl_research_train/", "model.ckpt"))
                            print("Latest Model is saving and Tensorboard Logs are updated")  
                        train_writer.add_summary(tf.summary.merge_all().eval(), epoch * (floor(int(157252)/BATCH_SIZE)) + i)
                        i = i + 1
                except tf.errors.OutOfRangeError:
                    print("End of the training dataset, proceed to validation")
                    pass

                j = 0
                try:
                    sess.run(valid_iterator.initializer)
                    while True:
                        print("Current Validation Iteration : {}/{}".format(j, floor(int(19657)/BATCH_SIZE)))
                        valid_acc, _, valid_ce = util.validation(BATCH_SIZE, NUM_CLASSES,learning_rate, weight_decay, sess, valid_conf_matrix_op, LEARNING_RATE_1, WEIGHT_DECAY_1, valid_cross_entropy, valid_accuracy)
                        valid_value1, valid_value2 = util.confidence_interval(valid_acc, Z_SCORE, BATCH_SIZE)
                        print("Validation Accuracy : {}".format(valid_acc))
                        print("validation Loss (Cross Entropy) : {}".format(valid_ce))
                        print("Validation Confidence Interval: [{} , {}]".format(valid_value2, valid_value1))
                        if valid_highest_acc <= valid_acc:
                            valid_highest_acc = valid_acc
                            print("Highest Validation Accuracy Reached: {}".format(valid_highest_acc))
                        valid_writer.add_summary(tf.summary.merge_all().eval(), epoch * (floor(int(19657)/BATCH_SIZE)) + j)
                        j = j + 1
                except tf.errors.OutOfRangeError:
                    print("End of validation dataset, go to the next epoch")
                    pass
                        
if __name__ == "__main__":
    get_parser()   
