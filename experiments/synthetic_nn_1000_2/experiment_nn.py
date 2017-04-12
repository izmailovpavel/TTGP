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
#            W1 = np.load('P1.npy')
#            b1 = np.load('b1.npy')
            self.W1 = self.weight_var('W1', [D, H1])#, W1, trainable=False)
            self.b1 = self.bias_var('b1', [H1])#, b1), trainable=False)
        
        with tf.name_scope('layer_1'):
#            W2 = np.load('P2.npy')
#            b2 = np.load('b2.npy')
            self.W2 = self.weight_var('W2', [H1, H2])#, W2, trainable=False)
            self.b2 = self.bias_var('b2', [H2])#, b2, trainable=False)

        with tf.name_scope('layer_1'):
#            W3 = np.load('P3.npy')
            self.W3 = self.weight_var('W3', [H2, d])#, W3, trainable=False)
            #self.b3 = self.bias_var('b3', [d])

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
#        l1 = 2 * (tf.sigmoid(tf.matmul(x, self.W1) + self.b1) - 0.5)
        l1 = tf.sigmoid(tf.matmul(x, self.W1) + self.b1)
        l2 = tf.sigmoid(tf.matmul(l1, self.W2) + self.b2)
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
    data_dir = ""
    n_inputs = 20
    mu_ranks = 15
    projector = NN(H1=10, H2=10)#LinearProjector(D=2, d=2)
    cov = SE(0.7, 0.2, 0.1, projector)
    lr = 1e-1
    decay = None#(50, 0.1)
    n_epoch = 50
    batch_size = 200
    data_type = 'numpy'
    log_dir = 'log'
    save_dir = 'models/proj_nn.ckpt'
    model_dir = save_dir
    load_model = False#True
    
    runner=GPRunner(data_dir, n_inputs, mu_ranks, cov,
                lr=lr, decay=decay, n_epoch=n_epoch, batch_size=batch_size,
                data_type=data_type, log_dir=log_dir, save_dir=save_dir,
                model_dir=model_dir, load_model=load_model)
    runner.run_experiment()
