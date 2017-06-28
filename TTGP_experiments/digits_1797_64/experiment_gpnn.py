import tensorflow as tf
import os
import numpy as np

from gptt_embed.covariance import SE
from gptt_embed.projectors import FeatureTransformer, LinearProjector
from gptt_embed.gp_runner import GPRunner

data_basedir1 = "/Users/IzmailovPavel/Documents/Education/Programming/DataSets/"
data_basedir2 = "/Users/IzmailovPavel/Documents/Education/Projects/GPtf/experiments/"

class NN(FeatureTransformer):
    
    def __init__(self, H1=100, H2=100, d=2, D=64, p=0.3):

        with tf.name_scope('layer_1'):
            self.W1 = self.weight_var('W1', [D, H1])
            self.b1 = self.bias_var('b1', [H1])        
        with tf.name_scope('layer_2'):
            self.W2 = self.weight_var('W2', [H1, H2])
            self.b2 = self.bias_var('b2', [H2])
        with tf.name_scope('layer_3'):
            self.W3 = self.weight_var('W3', [H2, d])

        self.p = p
        self.d = d
        
    @staticmethod
    def weight_var(name, shape, W=None, trainable=True):
        if not W is None:
            init = tf.constant(W, dtype=tf.float64)
        else:
            init = tf.orthogonal_initializer()(shape=shape, dtype=tf.float64)
        return tf.get_variable(name, initializer=init, 
                                dtype=tf.float64, trainable=trainable)

    @staticmethod
    def bias_var(name, shape, b=None, trainable=True):
        if not b is None:
            init = tf.constant(b, dtype=tf.float64)
        else:
            init = tf.constant(0., shape=shape, dtype=tf.float64) 

        return tf.get_variable(name, initializer=init, 
                                dtype=tf.float64, trainable=trainable)

    def transform(self, x):
        l1 = tf.sigmoid(tf.matmul(x, self.W1) + self.b1)
#        l1_d = tf.nn.dropout(l1, self.p)
        l2 = tf.sigmoid(tf.matmul(l1, self.W2) + self.b2)
#        l2_d = tf.nn.dropout(l2, self.p)
        l3 = tf.matmul(l2, self.W3)
        projected = l3

        # Rescaling
        mean, variance = tf.nn.moments(projected, axes=[0])
        scale = tf.rsqrt(variance + 1e-8)
        projected = (projected - mean[None, :]) * scale[None, :]
        projected /= 3

        projected = tf.minimum(projected, 1)
        projected = tf.maximum(projected, -1)
        return projected

    def initialize(self, sess):
        sess.run(tf.variables_initializer(self.get_params()))

    def get_params(self):
        return [self.W1, self.b1, self.W2, self.b2, self.W3]

    def out_dim(self):
        return self.d

    def save_weights(self, sess):
        W1, b1, W2, b2, W3 = sess.run(self.get_params())
        np.save('W1.npy', W1)
        np.save('b1.npy', b1)
        np.save('W2.npy', W2)
        np.save('b2.npy', b2)
        np.save('W3.npy', W3)

with tf.Graph().as_default():
    data_dir = "data_reg/"
    n_inputs = 10
    mu_ranks = 10
    projector = NN(H1=20, H2=20, d=4)
    cov = SE(0.7, 0.2, 0.1, projector)
    lr = 1e-2
    decay = (50, 0.2)
    n_epoch = 100
    batch_size = 200
    data_type = 'numpy'
    log_dir = 'log'
    save_dir = 'models/gpnn_100_100_4.ckpt'
    model_dir = save_dir
    load_model = False#True
    
    runner=GPRunner(data_dir, n_inputs, mu_ranks, cov,
                lr=lr, decay=decay, n_epoch=n_epoch, batch_size=batch_size,
                data_type=data_type, log_dir=log_dir, save_dir=save_dir,
                model_dir=model_dir, load_model=load_model)
    runner.run_experiment()
