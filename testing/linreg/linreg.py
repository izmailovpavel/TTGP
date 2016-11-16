import tensorflow as tf
import numpy as np

from input import NUM_FEATURES

def inference(x):
    """
    Builds the part of the computational graph, required for inference
    """
    with tf.variable_scope('Linreg'):
        W = tf.get_variable('weights', [NUM_FEATURES, 1], initializer=tf.constant_initializer(0.0), dtype=x.dtype)
        b = tf.get_variable('bias', [1], initializer=tf.constant_initializer(0.0), dtype=x.dtype)
        y = tf.matmul(x, W) + b
    return y

def loss(y, y_true):
    """
    Builds the part of the graph, required to calculate loss
    """
    loss = tf.reduce_mean(tf.squared_difference(tf.reshape(y, [-1]), tf.reshape(y_true, [-1])),
        name='MSE')
    return loss

def train(loss, lr):
    return tf.train.GradientDescentOptimizer(lr).minimize(loss)
