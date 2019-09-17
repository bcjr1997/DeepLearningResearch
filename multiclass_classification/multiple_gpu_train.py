import argparse
import os
import parser
import sys
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import glob
import util
from parallel_model import initiate_vgg_model
from math import floor

#Argsparse
def main():
    #Constants
    DATASET_PATH = os.path.join(".")
    LEARNING_RATE_1 = 0.0001
    EPOCHS = 2
    BATCH_SIZE = 32
    NUM_CLASSES = 48 
    Z_SCORE = 1.96
    WEIGHT_DECAY_1 = 0.0005

    print("Current Setup:-")
    print("Starting Learning Rate: {}, Epochs: {}, Batch Size: {}, Confidence Interval Z-Score {}, Number of classes: {}, Starting Weight Decay: {}".format(LEARNING_RATE_1, EPOCHS, BATCH_SIZE, Z_SCORE, NUM_CLASSES, WEIGHT_DECAY_1))

    #Get the number of GPUs
    NUM_GPUS = util.get_available_gpus()

    print("Number of GPUs available : {}".format(NUM_GPUS))
    with tf.device('/cpu:0'):
        tower_grads = []
        reuse_vars = False
        dataset_len = 1207350

        #Placeholders
        learning_rate = tf.placeholder(tf.float32, shape=[], name='learning_rate')
        weight_decay = tf.placeholder(tf.float32, shape=[], name="weight_decay")

        for i in range(NUM_GPUS):
            with tf.device(util.assign_to_device('/gpu:{}'.format(i), ps_device='/cpu:0')):

                #Need to split data between GPUs
                train_features, train_labels, train_filenames = util.train_input_fn(DATASET_PATH, BATCH_SIZE, EPOCHS)
                print("At GPU {}, Train Features : {}".format(i, train_features))

                #Model
                _, train_op, tower_grads, train_cross_entropy, train_conf_matrix_op, train_accuracy, reuse_vars = initiate_vgg_model(train_features, train_labels, train_filenames, NUM_CLASSES, weight_decay, learning_rate, reuse=reuse_vars, tower_grads=tower_grads, gpu_num=i, handle="training")
                #tf.summary.scalar("training_confusion_matrix", tf.reshape(tf.cast(conf_matrix_op, tf.float32),[1, NUM_CLASSES, NUM_CLASSES, 1]))

        tower_grads = util.average_gradients(tower_grads)
        train_op = train_op.apply_gradients(tower_grads)

        saver = tf.train.Saver()

        if not os.path.exists(os.path.join("./multi_dl_research_train/")):
            os.mkdir(os.path.join("./multi_dl_research_train/"))
                
        with tf.Session() as sess:
            with np.printoptions(threshold=np.inf):
                writer = tf.summary.FileWriter("./multi_tensorboard_logs/")
                writer.add_graph(sess.graph)
                merged_summary = tf.summary.merge_all()
                train_highest_acc = 0
                sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])

                for epoch in range(EPOCHS):
                    if epoch == 18:
                        LEARNING_RATE_1 = 0.00005
                        print("Learning Rate changed to {} at epoch {}".format(LEARNING_RATE_1, epoch))
                    elif epoch == 29:
                        LEARNING_RATE_1 = 0.00001
                        WEIGHT_DECAY_1 = 0.0
                        print("Learning Rate changed to {} at epoch {}".format(LEARNING_RATE_1, epoch))
                        print("Weight Decay changed to {} at epoch {}".format(WEIGHT_DECAY_1, epoch))
                    elif epoch == 42:
                        LEARNING_RATE_1 = 0.000005
                        print("Learning Rate changed to {} at epoch {}".format(LEARNING_RATE_1, epoch))
                    elif epoch == 51:
                        LEARNING_RATE_1 = 0.000001
                        print("Learning Rate changed to {} at epoch {}".format(LEARNING_RATE_1, epoch))

                    print("Current Epoch: {}".format(epoch))
                    for i in range(2):
                        print("Current Training Iteration : {}/{}".format(i, 10))
                        train_acc, _, _, train_ce, train_summary = util.training(BATCH_SIZE, NUM_CLASSES,learning_rate, weight_decay, sess, train_op, train_conf_matrix_op, LEARNING_RATE_1, WEIGHT_DECAY_1, train_cross_entropy, merged_summary, train_accuracy)
                        train_value1, train_value2 = util.confidence_interval(train_acc, Z_SCORE, 32)
                        print("Training Accuracy : {}".format(train_acc))
                        print("Training Loss (Cross Entropy) : {}".format(train_ce))
                        print("Training Confidence Interval: [{} , {}]".format(train_value2, train_value1))
                        if train_highest_acc <= train_acc:
                            train_highest_acc = train_acc
                            print("Highest Training Accuracy Reached: {}".format(train_highest_acc))
                        #For every epoch, we will save the model
                            saver.save(sess, os.path.join("./multi_dl_research_train/", "model.ckpt"))
                            print("Latest Model is saving and Tensorboard Logs are updated")  
                        writer.add_summary(train_summary, epoch * int((dataset_len * 0.8)/BATCH_SIZE) + i)

if __name__ == "__main__":
    main()   
