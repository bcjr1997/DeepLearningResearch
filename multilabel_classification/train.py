import argparse
import os
import parser
import sys
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import glob
import util
from model import initiate_vgg_model
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
    return parser


#Argsparse
def main():
    #Dataset
    image_data, label_data, unique_classes, image_filename = get_data()
    
    parser = get_parser()
    args   = parser.parse_args()
    
    
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
    dataset_len = len(image_data)
    training_dataset = tf.data.Dataset.from_generator(lambda: itertools.zip_longest(image_data, label_data),
                                                            output_types=(tf.float32, tf.float32),
                                                            output_shapes=(tf.TensorShape([None, None, 3]), 
                                                                        tf.TensorShape([None])))
                                                                        
    training_dataset = training_dataset.repeat(EPOCHS).batch(BATCH_SIZE).prefetch(1)
    training_iterator = training_dataset.make_one_shot_iterator()
    train_features, train_labels = training_iterator.get_next()
    #Model
    _, train_op, train_cross_entropy, train_conf_matrix_op, train_accuracy, train_predictions, true_predictions = initiate_vgg_model(train_features, train_labels, NUM_CLASSES, weight_decay, learning_rate, handle="training")
    #tf.summary.scalar("training_confusion_matrix", tf.reshape(tf.cast(conf_matrix_op, tf.float32),[1, NUM_CLASSES, NUM_CLASSES, 1]))

    saver = tf.train.Saver()

    if not os.path.exists(os.path.join( "./short_dl_research_train/")):
        os.mkdir(os.path.join("./short_dl_research_train/"))
    
    with tf.Session() as sess:
        with np.printoptions(threshold=np.inf):
            writer = tf.summary.FileWriter("./short_tensorboard_logs/")
            writer.add_graph(sess.graph)
            merged_summary = tf.summary.merge_all()
            train_highest_acc = 0
            sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])

            for epoch in range(EPOCHS):

                print("Current Epoch: {}".format(epoch))
                for i in range(1000):
                    print("Current Training Iteration : {}/{}".format(i, 1000))
                    train_acc, _, _, train_ce, train_summary = util.training(BATCH_SIZE, NUM_CLASSES,learning_rate, weight_decay, sess, train_op, train_conf_matrix_op, LEARNING_RATE_1, WEIGHT_DECAY_1, train_cross_entropy, merged_summary, train_accuracy)
                    train_value1, train_value2 = util.confidence_interval(train_acc, Z_SCORE, 32)
                    predictions, truth = sess.run([train_predictions, true_predictions])
                    print("Training Accuracy : {}".format(train_acc))
                    print("Training Prediction : {}".format(predictions))
                    print("True values : {}".format(truth))
                    print("Training Loss (Cross Entropy) : {}".format(train_ce))
                    print("Training Confidence Interval: [{} , {}]".format(train_value2, train_value1))
                    if train_highest_acc <= train_acc:
                        train_highest_acc = train_acc
                        print("Highest Training Accuracy Reached: {}".format(train_highest_acc))
                    #For every epoch, we will save the model
                        saver.save(sess, os.path.join("./short_dl_research_train/", "model.ckpt"))
                        print("Latest Model is saving and Tensorboard Logs are updated")  
                    writer.add_summary(train_summary, epoch * int((dataset_len * 0.8)/BATCH_SIZE) + i)

if __name__ == "__main__":
    main()   
