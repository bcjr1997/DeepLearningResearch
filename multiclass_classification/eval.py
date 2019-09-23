"""File that test the model"""
import tensorflow as tf
import util
import os
import numpy as np
from model import initiate_vgg_model
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

def main(args = [ ]):
    parser = get_parser()
    args   = parser.parse_args()
    
    with tf.device('/cpu:0'):
        #tf.reset_default_graph()
        DATASET_PATH = args.datasetPath
        LEARNING_RATE_1 = args.learningRate
        EPOCHS = args.epochs
        BATCH_SIZE = args.batchSize
        NUM_CLASSES = args.numClasses
        Z_SCORE = args.zScore
        WEIGHT_DECAY_1 = args.weightDecay

        #Placeholders
        learning_rate = tf.placeholder(tf.float32, shape=[], name='learning_rate')
        weight_decay = tf.placeholder(tf.float32, shape=[], name="weight_decay")

        #Dataset
        test_features, test_labels, test_filenames = util.test_input_fn(DATASET_PATH, BATCH_SIZE, EPOCHS)

        #Model
        _, _, test_cross_entropy, test_conf_matrix_op, test_accuracy = initiate_vgg_model(test_features, test_labels, test_filenames, NUM_CLASSES, weight_decay, learning_rate, handle="testing")
        saver = tf.train.Saver()

        with tf.Session() as sess:
            with np.printoptions(threshold=np.inf):
                if not os.path.isdir("./hcc_output/"):
                    raise Exception("Model file not found. Use Train.py to train a model")
                else:
                    saver.restore(sess, "./hcc_output/model.ckpt")
                    print("Model restored from Saver files")

                writer = tf.summary.FileWriter("./short_tensorboard_logs/")
                writer.add_graph(sess.graph)
                merged_summary = tf.summary.merge_all()
                sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
                
                for i in range(100):
                    print("Current Testing Iteration : {}/{}".format(i, 100))
                    summary, _, test_ce, test_acc = util.test(BATCH_SIZE, learning_rate, weight_decay, sess, test_cross_entropy, test_conf_matrix_op, NUM_CLASSES, LEARNING_RATE_1, WEIGHT_DECAY_1, merged_summary, test_accuracy)
                    test_value1, test_value2 = util.confidence_interval(test_acc, Z_SCORE, 32)
                    print("Testing Accuracy : {}".format(test_acc))
                    print("Testing Loss (Cross Entropy) : {}".format(test_ce))
                    print("Testing Confidence Interval: [{} , {}]".format(test_value2, test_value1))
                    writer.add_summary(summary, i)

if __name__ == "__main__":
    main()