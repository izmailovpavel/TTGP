import tensorflow as tf
import os
import numpy as np
from tensorflow.contrib.layers import batch_norm

from gptt_embed.covariance import SE_multidim
from gptt_embed.projectors import FeatureTransformer, LinearProjector
from gptt_embed.gpc_runner import GPCRunner


HEIGHT, WIDTH = 24, 24

class NN(FeatureTransformer):
    
    def __init__(self, Hc1=64, Hc2=64, Hc3=128, Hc4=128, 
            Hc5=128, Hc6=128, Hd1=512, Hd2=512, d=2):

        # Convolutional
        with tf.name_scope('conv1'):
            self.W_conv1 = self.weight_var('W_conv1', [3, 3, 3, Hc1])
            self.b_conv1 = self.bias_var('b_conv1', [Hc1])        
        with tf.name_scope('conv2'):
            self.W_conv2 = self.weight_var('W_conv2', [3, 3, Hc1, Hc2])
            self.b_conv2 = self.bias_var('b_conv2', [Hc2])
        with tf.name_scope('conv3'):
            self.W_conv3 = self.weight_var('W_conv3', [3, 3, Hc2, Hc3])
            self.b_conv3 = self.bias_var('b_conv3', [Hc3])        
        with tf.name_scope('conv4'):
            self.W_conv4 = self.weight_var('W_conv4', [3, 3, Hc3, Hc4])
            self.b_conv4 = self.bias_var('b_conv4', [Hc4])
        with tf.name_scope('conv5'):
            self.W_conv5 = self.weight_var('W_conv5', [3, 3, Hc4, Hc5])
            self.b_conv5 = self.bias_var('b_conv5', [Hc5])
        with tf.name_scope('conv6'):
            self.W_conv6 = self.weight_var('W_conv6', [3, 3, Hc5, Hc6])
            self.b_conv6 = self.bias_var('b_conv6', [Hc6])

        # Fully-connected
        with tf.name_scope('fc1'):
            self.W_fc1 = self.weight_var('W_fc1', [36 * Hc6, Hd1])
            self.b_fc1 = self.bias_var('b_fc1', [Hd1])
        with tf.name_scope('fc2'):
            self.W_fc2 = self.weight_var('W_fc2', [Hd1, Hd2])
            self.b_fc2 = self.bias_var('b_fc2', [Hd2])
        with tf.name_scope('fc3'):
            self.W_fc3 = self.weight_var('W_fc3', [Hd2, d])
        
        self.d = d
        self.batch_size = batch_size
        self.reuse = False
        
    @staticmethod
    def weight_var(name, shape, trainable=True):
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
    def max_pool_3x3(x):
          return tf.nn.max_pool(x, ksize=[1, 3, 3, 1],
                                          strides=[1, 2, 2, 1], padding='SAME')

    def transform(self, x, test=False):
        batch_size = x.get_shape()[0].value
        x_image = tf.cast(tf.reshape(x, [-1, HEIGHT, WIDTH, 3]), tf.float32)

        # Convolutional layer 1
        h_preact1 = self.conv2d(x_image, self.W_conv1) + self.b_conv1
        norm1 = batch_norm(h_preact1, decay=0.99, is_training=(not test), 
                           reuse=self.reuse, scope="norm_conv1")
        h_conv1 = tf.nn.relu(norm1)

        # Convolutional layer 2
        h_preact2 = self.conv2d(h_conv1, self.W_conv2) + self.b_conv2
        norm2 = batch_norm(h_preact2, decay=0.99, is_training=(not test), 
                           reuse=self.reuse, scope="norm_conv2")
        h_conv2 = tf.nn.relu(norm2)

        # Pooling
        h_pool1 = self.max_pool_3x3(h_conv2)


        # Convolutional layer 3
        h_preact3 = self.conv2d(h_pool1, self.W_conv3) + self.b_conv3
        norm3 = batch_norm(h_preact3, decay=0.99, is_training=(not test), 
                           reuse=self.reuse, scope="norm_conv3")
        h_conv3 = tf.nn.relu(norm3)

        # Convolutional layer 4
        h_preact4 = self.conv2d(h_conv3, self.W_conv4) + self.b_conv4
        norm4 = batch_norm(h_preact4, decay=0.99, is_training=(not test), 
                           reuse=self.reuse, scope="norm_conv4")
        h_conv4 = tf.nn.relu(norm4)

        # Pooling
        h_pool2 = self.max_pool_3x3(h_conv4)

        # Convolutional layer 5
        h_preact5 = self.conv2d(h_pool2, self.W_conv5) + self.b_conv5
        norm5 = batch_norm(h_preact5, decay=0.99, is_training=(not test), 
                           reuse=self.reuse, scope="norm_conv5")
        h_conv5 = tf.nn.relu(norm5)

        # Convolutional layer 6
        h_preact6 = self.conv2d(h_conv5, self.W_conv6) + self.b_conv6
        norm6 = batch_norm(h_preact6, decay=0.99, is_training=(not test), 
                           reuse=self.reuse, scope="norm_conv6")
        h_conv6 = tf.nn.relu(norm6)

        # Dense layer 1
        h_flat = tf.reshape(h_conv6, [batch_size, -1])
        print('h_pol2_flat', h_flat.get_shape())
        h_preact_fc1 = tf.matmul(h_flat, self.W_fc1) + self.b_fc1
        norm_fc1 = batch_norm(h_preact_fc1, decay=0.99, is_training=(not test), 
                           reuse=self.reuse, scope="norm_fc1")
        h_fc1 = tf.nn.relu(norm_fc1)

        # Dense layer 2
        h_preact_fc2 = tf.matmul(h_fc1, self.W_fc2) + self.b_fc2
        norm_fc2 = batch_norm(h_preact_fc2, decay=0.99, is_training=(not test), 
                           reuse=self.reuse, scope="norm_fc2")
        h_fc2 = tf.nn.relu(norm_fc2)

        # Dense layer 3
        projected = tf.matmul(h_fc2, self.W_fc3) 
        projected = tf.cast(projected, tf.float64)

        # Rescaling
        projected = tf.cast(projected, tf.float32)
        projected = batch_norm(projected, decay=0.99, center=False, scale=False,
                                is_training=(not test), reuse=self.reuse, 
                                scope="norm_fc3")
        projected = tf.cast(projected, tf.float64)
        projected /= 3
        self.reuse = True

        projected = tf.minimum(projected, 1)
        projected = tf.maximum(projected, -1)
        return projected

    def initialize(self, sess):
        sess.run(tf.variables_initializer(self.get_params()))
        
    def get_params(self):
        bn_vars = []
        for scope in ["norm_conv1", "norm_conv2", "norm_conv3", 
                      "norm_conv4", "norm_conv5", "norm_conv6",
                      "norm_fc1", "norm_fc2", "norm_fc3"]:
            bn_vars += tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)
        return bn_vars + [self.W_conv1, self.b_conv1, self.W_conv2, self.b_conv2,
                self.W_conv3, self.b_conv3, self.W_conv4, self.b_conv4,
                self.W_conv5, self.b_conv5, self.W_conv6, self.b_conv6,
                self.W_fc1, self.b_fc1, self.W_fc2, self.b_fc2, self.W_fc3] 

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
    lr = 1e-3
    decay = (30, 0.2)
    n_epoch = 300
    batch_size = 200
    data_type = 'numpy'
    log_dir = 'log'
    save_dir = None#'models/gpnn_100_100_2.ckpt'
    model_dir = None#save_dir
    load_model = False#True
    num_threads=3

    projector = NN(Hc1=128, Hc2=128, Hc3=256, Hc4=256, Hc5=256, Hc6=256,
            Hd1=1536, Hd2=512, d=9)
    cov = SE_multidim(C, 0.7, 0.2, 0.1, projector)
    
    runner=GPCRunner(data_dir, n_inputs, mu_ranks, cov,
                lr=lr, decay=decay, n_epoch=n_epoch, batch_size=batch_size,
                preprocess_op=tr_preprocess_op, te_preprocess_op=te_preprocess_op,
                data_type=data_type, log_dir=log_dir, save_dir=save_dir,
                model_dir=model_dir, load_model=load_model, 
                num_threads=num_threads)
    runner.run_experiment()
