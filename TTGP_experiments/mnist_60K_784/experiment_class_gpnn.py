import tensorflow as tf
import os
import numpy as np
from tensorflow.contrib.layers import batch_norm

from TTGP.covariance import SE_multidim
from TTGP.projectors import FeatureTransformer, LinearProjector
from TTGP.gpc_runner import GPCRunner

class NN(FeatureTransformer):
    
    def __init__(self, H1=32, H2=64, H3=100, d=2):

        with tf.name_scope('layer_1'):
            self.W_conv1 = self.weight_var('W1', [5, 5, 1, H1])
            self.b1 = self.bias_var('b1', [H1])        
        with tf.name_scope('layer_2'):
            self.W_conv2 = self.weight_var('W2', [5, 5, H1, H2])
            self.b2 = self.bias_var('b2', [H2])
        with tf.name_scope('layer_3'):
            self.W3 = self.weight_var('W3', [7 * 7 * H2, H3])
            self.b3 = self.bias_var('b3', [H3])
        with tf.name_scope('layer_4'):
            self.W4 = self.weight_var('W4', [H3, d])
        
        self.H1 = H1
        self.H2 = H2
        self.H3 = H3
        self.d = d
        self.reuse = False
        
    @staticmethod
    def weight_var(name, shape, trainable=True):
#        init = tf.orthogonal_initializer()(shape=shape, dtype=tf.float64)
        init = tf.truncated_normal(shape, stddev=0.1, dtype=tf.float32)
        return tf.get_variable(name, initializer=init, 
                                dtype=tf.float32, trainable=trainable)
    @staticmethod
    def bias_var(name, shape, trainable=True):
        init = tf.constant(0.1, shape=shape, dtype=tf.float32) 
        return tf.get_variable(name, initializer=init, 
                                dtype=tf.float32, trainable=trainable)

    @staticmethod
    def conv2d(x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    @staticmethod
    def max_pool_2x2(x):
          return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                                          strides=[1, 2, 2, 1], padding='SAME')

    def transform(self, x, test=False):
        x_image = tf.cast(tf.reshape(x, [-1, 28, 28, 1]), tf.float32)

        h_conv1 = tf.nn.relu(self.conv2d(x_image, self.W_conv1) + self.b1)
        h_pool1 = self.max_pool_2x2(h_conv1)

        h_conv2 = tf.nn.relu(self.conv2d(h_pool1, self.W_conv2) + self.b2)
        h_pool2 = self.max_pool_2x2(h_conv2)

        h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*self.H2])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, self.W3) + self.b3)
        
        projected = tf.matmul(h_fc1, self.W4) 
        projected = tf.cast(projected, tf.float64)

        # Rescaling
        projected = tf.cast(projected, tf.float32)
        projected = batch_norm(projected, decay=0.99, center=False, scale=False,
                                is_training=(not test), reuse=self.reuse, scope="batch_norm")
        projected = tf.cast(projected, tf.float64)
        projected /= 3
        self.reuse = True

        projected = tf.minimum(projected, 1)
        projected = tf.maximum(projected, -1)
        return projected

    def initialize(self, sess):
        sess.run(tf.variables_initializer(self.get_params()))

    def get_params(self):
        return [self.W_conv1, self.b1, self.W_conv2, self.b2, 
                self.W3, self.b3, self.W4]

    def out_dim(self):
        return self.d

    def save_weights(self, sess):
        W1, b1, W2, b2, W3, b3, W4 = sess.run(self.get_params())
        np.save('W1.npy', W1)
        np.save('b1.npy', b1)
        np.save('W2.npy', W2)
        np.save('b2.npy', b2)
        np.save('W3.npy', W3)
        np.save('b3.npy', b3)
        np.save('W4.npy', W4)


with tf.Graph().as_default():
    data_dir = "data_class/"
    n_inputs = 10
    mu_ranks = 10
    projector = NN(H1=32, H2=64, H3=1024, d=4)
    C = 10

    cov = SE_multidim(C, 0.7, 0.2, 0.1, projector)

    lr = 1e-2
    decay = (7, 0.2)
    n_epoch = 20
    batch_size = 200
    data_type = 'numpy'
    log_dir = 'log'
    save_dir = None#'models/gpnn_100_100_2.ckpt'
    model_dir = save_dir
    load_model = False#True
    num_threads = 3
    
    runner=GPCRunner(data_dir, n_inputs, mu_ranks, cov,
                lr=lr, decay=decay, n_epoch=n_epoch, batch_size=batch_size,
                data_type=data_type, log_dir=log_dir, save_dir=save_dir,
                model_dir=model_dir, load_model=load_model, num_threads=num_threads)
    runner.run_experiment()
