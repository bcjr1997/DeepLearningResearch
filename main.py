import argparse
import os
import parser
import sys
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import csv
import cv2
import pandas as pd
from model import initiate_vgg_model
import util



#Argsparse
def main(cli_args):

    #Constants
    DATASET_PATH = os.path.join("../Datasets/224_dataset_ori_rgb/")
    LEARNING_RATE = 0.0001
    EPOCHS = 100
    STOPPING_COUNT = 50 
    BATCH_SIZE = 32
    NUM_CLASSES = 47 
    Z_SCORE = 1.96

    #Load data
    util.load_data_with_TFRecords(DATASET_PATH)
    #Check to see if images and labels share the same length
    

      
if __name__ == "__main__":
    main(sys.argv[1:])
