import tensorflow as tf
import numpy as np
import os

from gptt_embed.gp import GP
from gptt_embed.covariance import SE
from gptt_embed.projectors import FeatureTransformer, LinearProjector
from gptt_embed.gp_runner import GPRunner

data_basedir1 = "/Users/IzmailovPavel/Documents/Education/Programming/DataSets/"
data_basedir2 = "/Users/IzmailovPavel/Documents/Education/Projects/GPtf/experiments/"

class NN(FeatureTransformer):
    
    def __init__(self, H1=10, H2=10, d=2, D=2):

        with tf.name_scope('layer_1'):
            self.W1 = self.weight_var('W1', [D, H1])
            self.b1 = self.bias_var('b1', [H1])
        
        with tf.name_scope('layer_1'):
            self.W2 = self.weight_var('W2', [H1, H2])
            self.b2 = self.bias_var('b2', [H2])

        with tf.name_scope('layer_1'):
            self.W3 = self.weight_var('W3', [H2, d])
            self.b3 = self.bias_var('b3', [d])

        self.d = d
        
    @staticmethod
    def weight_var(name, shape, trainable=True):
        init = tf.orthogonal_initializer()(shape=shape, dtype=tf.float64)
        return tf.get_variable(name, initializer=init, 
                                dtype=tf.float64, trainable=trainable)
    @staticmethod
    def bias_var(name, shape, trainable=True):
        init = tf.constant(0., shape=shape, dtype=tf.float64) 
        return tf.get_variable(name, initializer=init, 
                                dtype=tf.float64, trainable=trainable)

    def transform(self, x):
#        l1 = 2 * (tf.sigmoid(tf.matmul(x, self.W1) + self.b1) - 0.5)
        l1 = tf.sigmoid(tf.matmul(x, self.W1) + self.b1)
#        l2 = 2 * (tf.sigmoid(tf.matmul(l1, self.W2) + self.b2) - 0.5)
        l2 = tf.matmul(l1, self.W2) + self.b2
        #l3 = 2 * (tf.sigmoid(tf.matmul(l2, self.W3) + self.b3) - 0.5)
        projected = l2

        # Rescaling
        mean, scale = tf.nn.moments(projected, axes=[0])
        scale += 1e-8
        projected = (projected - mean[None, :]) / scale[None, :]
        projected /= 3

        projected = tf.minimum(projected, 1)
        projected = tf.maximum(projected, -1)
        return projected

    def initialize(self, sess):
        sess.run(tf.variables_initializer(self.get_params()))

    def get_params(self):
        return [self.W1, self.b1, self.W2, self.b2, self.W3, self.b3]

    def out_dim(self):
        return self.d

with tf.Graph().as_default():
    data_dir = ""
    n_inputs = 20
    mu_ranks = 15
    projector = NN(H1=5, H2=2)#LinearProjector(D=2, d=2)
    cov = SE(0.7, 0.2, 0.1, projector)
    lr = 1e-2
    decay = None#(50, 0.1)
    n_epoch = 50
    batch_size = 200
    data_type = 'numpy'
    log_dir = 'log'
    save_dir = 'models/proj_nn.ckpt'
    model_dir = save_dir
    load_model = True
    
    runner=GPRunner(data_dir, n_inputs, mu_ranks, cov,
                lr=lr, decay=decay, n_epoch=n_epoch, batch_size=batch_size,
                data_type=data_type, log_dir=log_dir, save_dir=save_dir,
                model_dir=model_dir, load_model=load_model)
    runner.run_experiment()
