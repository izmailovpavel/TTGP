import tensorflow as tf
import os
import numpy as np
from tensorflow.contrib.layers import batch_norm

from gptt_embed.covariance import SE_multidim
from gptt_embed.projectors import FeatureTransformer, LinearProjector
from gptt_embed.gpc_runner import GPCRunner


HEIGHT, WIDTH = 24, 24

class NN(FeatureTransformer):
    
    def __init__(self, H1=32, H2=64, H3=100, H4=100, d=2):

        with tf.name_scope('layer_1'):
            self.W_conv1 = self.weight_var('W1', [5, 5, 3, H1])
            self.b1 = self.bias_var('b1', [H1])        
        with tf.name_scope('layer_2'):
            self.W_conv2 = self.weight_var('W2', [5, 5, H1, H2])
            self.b2 = self.bias_var('b2', [H2])
        with tf.name_scope('layer_3'):
            self.W3 = self.weight_var('W3', [36 * H2, H3])
            self.b3 = self.bias_var('b3', [H3])
        with tf.name_scope('layer_4'):
            self.W4 = self.weight_var('W4', [H3, H4])
            self.b4 = self.bias_var('b4', [H4])
        with tf.name_scope('layer_5'):
            self.W5 = self.weight_var('W5', [H4, d])
        
        self.H1 = H1
        self.H2 = H2
        self.H3 = H3
        self.H4 = H4
        self.d = d
        self.batch_size = batch_size
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
          return tf.nn.max_pool(x, ksize=[1, 3, 3, 1],
                                          strides=[1, 2, 2, 1], padding='SAME')

    def transform(self, x, test=False):
        batch_size = x.get_shape()[0].value
        x_image = tf.cast(tf.reshape(x, [-1, HEIGHT, WIDTH, 3]), tf.float32)

        h_conv1 = tf.nn.relu(self.conv2d(x_image, self.W_conv1) + self.b1)
        h_pool1 = self.max_pool_2x2(h_conv1)

        h_conv2 = tf.nn.relu(self.conv2d(h_pool1, self.W_conv2) + self.b2)
        h_pool2 = self.max_pool_2x2(h_conv2)

        h_pool2_flat = tf.reshape(h_pool2, [batch_size, -1])
        print('h_pol2_flat', h_pool2_flat.get_shape())
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, self.W3) + self.b3)
        print('h_fc1', h_fc1.get_shape())
        h_fc2 = tf.nn.relu(tf.matmul(h_fc1, self.W4) + self.b4)
        print('h_fc2', h_fc2.get_shape())
        
        projected = tf.matmul(h_fc2, self.W5) 
        print('projected', projected.get_shape())
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
        bn_vars = tf.get_collection(tf.GraphKeys.VARIABLES, scope='batch_norm')
        return [self.W_conv1, self.b1, self.W_conv2, self.b2, 
                self.W3, self.b3, self.W4, self.b4, self.W5] + bn_vars

    def out_dim(self):
        return self.d

def tr_preprocess_op(img):

    img = tf.reshape(img, [32, 32, 3])
    distorted_image = tf.random_crop(img, [HEIGHT, WIDTH, 3])
    distorted_image = tf.image.random_flip_left_right(distorted_image)
    distorted_image = tf.image.random_brightness(distorted_image, max_delta=63)
    distorted_image = tf.image.random_contrast(distorted_image, lower=0.2, upper=1.8)
    res = tf.image.per_image_standardization(distorted_image)
    res = tf.reshape(res, [-1])
    return res

def te_preprocess_op(img):

    img = tf.reshape(img, [32, 32, 3])
    resized_image = tf.image.resize_image_with_crop_or_pad(img, HEIGHT, WIDTH)
    res = tf.image.per_image_standardization(resized_image)
    res = tf.reshape(res, [-1])
    return res


with tf.Graph().as_default():
    data_dir = "data_class/"
    n_inputs = 10
    mu_ranks = 10
    C = 10
    lr = 5e-3
    decay = (30, 0.2)
    n_epoch = 300
    batch_size = 200
    data_type = 'numpy'
    log_dir = 'log'
    save_dir = None#'models/gpnn_100_100_2.ckpt'
    model_dir = None#save_dir
    load_model = False#True

    projector = NN(H1=64, H2=128, H3=512, H4=128, d=8)
    cov = SE_multidim(C, 0.7, 0.2, 0.1, projector)
    
    runner=GPCRunner(data_dir, n_inputs, mu_ranks, cov,
                lr=lr, decay=decay, n_epoch=n_epoch, batch_size=batch_size,
                preprocess_op=tr_preprocess_op, te_preprocess_op=te_preprocess_op,
                data_type=data_type, log_dir=log_dir, save_dir=save_dir,
                model_dir=model_dir, load_model=load_model)
    runner.run_experiment()
