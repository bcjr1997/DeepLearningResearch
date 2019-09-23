import os
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
from math import sqrt
from tensorflow.python.client import device_lib

PS_OPS = ['Variable', 'VariableV2', 'AutoReloadVariable']

def assign_to_device(device, ps_device='/cpu:0'):
    def _assign(op):
        node_def = op if isinstance(op, tf.NodeDef) else op.node_def
        if node_def.op in PS_OPS:
            return "/" + ps_device
        else:
            return device

    return _assign

def train_input_fn(DATASET_PATH, BATCH_SIZE, EPOCHS):
    train_dataset = tf.data.TFRecordDataset([os.path.join(DATASET_PATH, 'small_train_dataset.tfrecord')])
    train_dataset = train_dataset.repeat(EPOCHS)
    train_dataset = train_dataset.map(read_and_decode_tfrecords, num_parallel_calls=1)
    train_dataset = train_dataset.batch(BATCH_SIZE)
    train_dataset = train_dataset.repeat(EPOCHS)
    train_dataset = train_dataset.prefetch(1)
    train_iterator = train_dataset.make_initializable_iterator()
    train_next_element = train_iterator.get_next()
    return train_next_element, train_iterator

def valid_input_fn(DATASET_PATH, BATCH_SIZE, EPOCHS):
    valid_dataset = tf.data.TFRecordDataset([os.path.join(DATASET_PATH, 'small_valid_dataset.tfrecord')])
    valid_dataset = valid_dataset.repeat(EPOCHS)
    valid_dataset = valid_dataset.map(read_and_decode_tfrecords, num_parallel_calls=1)
    valid_dataset = valid_dataset.batch(BATCH_SIZE)
    valid_dataset = valid_dataset.repeat(EPOCHS)
    valid_dataset = valid_dataset.prefetch(1)
    valid_iterator = valid_dataset.make_initializable_iterator()
    valid_next_element = valid_iterator.get_next()
    return valid_next_element, valid_iterator

def test_input_fn(DATASET_PATH, BATCH_SIZE, EPOCHS):
    test_dataset = tf.data.TFRecordDataset([os.path.join(DATASET_PATH, 'small_test_dataset.tfrecord')])
    test_dataset = test_dataset.map(read_and_decode_tfrecords, num_parallel_calls=1)
    test_dataset = test_dataset.batch(BATCH_SIZE)
    test_dataset = test_dataset.repeat(EPOCHS)
    test_dataset = test_dataset.prefetch(1)
    test_iterator = test_dataset.make_initializable_iterator()
    test_next_element = test_iterator.get_next()
    return test_next_element

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
    return image, label, filename

def split_data(data, proportion):
    """
    Split a numpy array into two parts of `proportion` and `1 - proportion`
    
    Args:
        - data: numpy array, to be split along the first axis
        - proportion: a float less than 1
    """
    size = data.shape[0]
    split_idx = int(proportion * size)
    return data[:split_idx], data[split_idx:]

def confidence_interval(accuracy, constant, n):
	error = 1 - accuracy
	calculation = (error * (1 - error)) / n
	value1 = error + (constant * sqrt(calculation))
	value2 = error - (constant * sqrt(calculation))
	return value1, value2

def one_hot_encoding(labels, num_classes):
	return np.eye(num_classes)[labels.astype(int)]

def confusion_matrix_op(y, output, num_classes):
	conf_mtx = tf.confusion_matrix(
    			tf.argmax(y, axis=1), 
    			tf.argmax(output, axis=1), 
    			num_classes=num_classes)	
	return conf_mtx

def cross_entropy_op(y_placeholder, output, func_name):
	cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_placeholder, logits=output, name=func_name)
	#cross_ent = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_placeholder, logits=output, name=func_name)
	return tf.reduce_mean(cross_ent)

def train_op(cross_entropy_op, global_step_tensor, optimizer):
	training_operation = optimizer.minimize(cross_entropy_op, global_step=global_step_tensor, name="training_op")
	return training_operation

#Declaring global step tensor
def global_step_tensor(name):
	global_step_tensor = tf.get_variable(
	name, 
	trainable=False, 
	shape=[], 
	initializer=tf.zeros_initializer)
	return global_step_tensor

def training(batch_size, NUM_CLASSES, learning_rate, weight_decay, session, train_op, confusion_matrix_op, LEARNING_RATE, WEIGHT_DECAY, cross_entropy_op, accuracy_op):
	with np.printoptions(threshold=np.inf):
		train_cost , confusion_mtx, train_ce, train_acc = session.run([train_op, confusion_matrix_op, tf.reduce_mean(cross_entropy_op), accuracy_op],
															  feed_dict = {learning_rate: LEARNING_RATE, weight_decay: WEIGHT_DECAY})
		#This prints the values across each class
	return train_acc, confusion_mtx, train_cost, train_ce

def validation(batch_size, NUM_CLASSES, learning_rate, weight_decay, session, confusion_matrix_op, LEARNING_RATE, WEIGHT_DECAY, cross_entropy_op, accuracy_op):
	with np.printoptions(threshold=np.inf):
		valid_ce, conf_matrix, valid_acc = session.run([cross_entropy_op, confusion_matrix_op, accuracy_op],
													  feed_dict = {learning_rate: LEARNING_RATE, weight_decay: WEIGHT_DECAY})
		#This prints the values across each class
	return valid_acc, conf_matrix, valid_ce

def test(batch_size, learning_rate, weight_decay, session, cross_entropy_op, confusion_matrix_op, num_classes, LEARNING_RATE, WEIGHT_DECAY, merged_summary, accuracy_op):
	with np.printoptions(threshold=np.inf):
		confusion_mtx, test_ce, test_acc = session.run([merged_summary, confusion_matrix_op, tf.reduce_mean(cross_entropy_op), accuracy_op],
															  feed_dict = {learning_rate: LEARNING_RATE, weight_decay: WEIGHT_DECAY})
		#This prints the values across each class
	return confusion_mtx, test_ce, test_acc 

def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if "GPU" in x.device_type]

def average_gradients(tower_grads):
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0)

            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)

        # Average over the 'tower' dimension.
        grad = tf.concat(grads, 0)
        grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads
