import argparse
import os
import parser
import sys
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import csv
import pandas as pd
from model import initiate_basic_model, initiate_vgg_model
import util
#Argsparse
def main(cli_args):
    parser = argparse.ArgumentParser(description="CSCE 496 Project")
    parser.add_argument('--input_dir', type=str, default='./small_dataset_rgb/', help = 'Dataset Path Input')
    parser.add_argument('--model_dir',type=str,default='./project/',help='directory where model graph and weights are saved')
    parser.add_argument('--epoch' , type=int, default=1, help = "Epoch : number of iterations for the model")
    parser.add_argument('--batch_size', type=int, default=32, help = "Batch Size")
    parser.add_argument('--model', type=int, help=" '1' for basic model, '2' for best model")
    parser.add_argument('--stopCount', type=int, default = 100, help="Number of times for dropping accuracy before early stopping")
    args_input = parser.parse_args(cli_args)

    if args_input.input_dir:
        input_dir = args_input.input_dir
    else:
        raise ValueError("Provide a valid input data path")

    if args_input.model_dir:
        model_dir = args_input.model_dir
    else:
        raise ValueError("Provide a valid model data path")

    if args_input.epoch:
        epochs = args_input.epoch
    else:
        raise ValueError("Epoch value cannot be null and has to be an integer")

    if args_input.batch_size:
        batch_size = args_input.batch_size
    else:
        raise ValueError("Batch Size value cannot be null and has to be an integer")
    
    if args_input.model:
        model = args_input.model
    else:
        raise ValueError("Model selection must not be empty") 

    if args_input.stopCount:
        stop_counter = args_input.stopCount
    else:
        raise ValueError("StopCount have to be an int") 

    #Make output model dir
    if os.path.exists(model_dir) == False:
        os.mkdir(model_dir)

    #Load Data
    print("Loading Data")
    train_images, train_labels, test_images, test_labels, val_images, val_labels, unique_labels = util.load_data(input_dir)
    x = tf.placeholder(tf.float32, [None, 224, 224, 3], name='input_placeholder')
    y = tf.placeholder(tf.float32, [None, len(unique_labels)], name='labels')

    #Specify Model
    if(str(model) == '1'):
        _, outputLayer = initiate_vgg_model(x, len(unique_labels))

    #Run Training with early stopping and save output
    counter = stop_counter
    prev_winner = 0
    curr_winner = 0
    learn_rate=0.0001
    optimizer = tf.train.AdamOptimizer(learning_rate=learn_rate)
    cross_entropy = util.cross_entropy_op(y , outputLayer)
    global_step_tensor = util.global_step_tensor('global_step_tensor')
    train_op = util.train_op(cross_entropy, global_step_tensor, optimizer)
    conf_matrix = util.confusion_matrix_op(y, outputLayer, len(unique_labels))
    saver = tf.train.Saver()
    df = pd.DataFrame({'epoch': [], 'train_accuracy': [], 'train_conf_matrix':[],
                      'valid_accuracy':[], "valid_cross_entropy": [], 'valid_conf_matrix':[],
                      'test_accuracy':[], 'test_cross_entropy':[], 'test_conf_matrix': [],
                      'positive_confidence_interval': [], 'negative_confidence_interval': [], 'learning_rate': [],
                      'labels_org': []})
    with tf.Session() as session:
        with np.printoptions(threshold=np.inf):
            session.run(tf.global_variables_initializer())
            counter = stop_counter
            print("Start Session")
            for epoch in range (epochs):
                if counter > 0:
                    print("Epoch : " + str(epoch))
                    train_acc, train_conf_matrix = util.training(batch_size, x , y, train_images,
                                                                train_labels, session,
                                                                train_op,conf_matrix, len(unique_labels))
                    valid_acc, valid_conf_matrix, valid_ce = util.validation(batch_size, x , y, val_images,
                                                val_labels, session,
                                                cross_entropy,
                                                conf_matrix,len(unique_labels))
                    if epoch == 0:
                        prev_winner = valid_acc
                        print("Saving.......")
                        saver.save(session, os.path.join("./project/", "project"))
                    else:
                        curr_winner = valid_acc
                        if (curr_winner > prev_winner) and (counter > 0):
                            prev_winner = curr_winner

                            print("Saving.......")
                            saver.save(session, os.path.join("./project/", "project"))
                        else:
                            counter -= 1

                    test_acc, test_ce, test_conf_mtx = util.test(batch_size, x , y, test_images,
                                            test_labels, session,
                                            cross_entropy, conf_matrix, len(unique_labels))
                        #Calculate the confidence interval
                    value1 , value2 = util.confidence_interval(test_acc, 1.96, test_images.shape[0])
                    print("Confidence Interval : " + str(value1) + " , " + str(value2))
                    output_data = {'epoch': epoch, 'train_accuracy': train_acc, 'train_conf_matrix':train_conf_matrix,
                                    'valid_accuracy':valid_acc, "valid_cross_entropy": valid_ce, 'valid_conf_matrix':valid_conf_matrix,
                                    'test_accuracy':test_acc , 'test_cross_entropy':test_ce, 'test_conf_matrix': test_conf_mtx,
                                    'positive_confidence_interval': value1, 'negative_confidence_interval': value2,
                                    'learning_rate': learn_rate, 'labels_org': unique_labels}
                    print(output_data)
                    df = df.append(output_data, ignore_index=True)
                else:
                    break
            df.to_csv(r"./results_data.csv", header=True, index=False, encoding='utf-8')        
if __name__ == "__main__":
    main(sys.argv[1:])
