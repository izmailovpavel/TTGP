import tensorflow as tf
import os
import numpy as np
from tensorflow.contrib.layers import batch_norm

from TTGP.covariance import SE_multidim
from TTGP.projectors import FeatureTransformer, LinearProjector, Identity
from TTGP.gpc_runner import GPCRunner

class NN(FeatureTransformer):
    
    def __init__(self, H1=1000, H2=1000, H3=500, H4=50, d=4, D=8, p=0.5):

        with tf.name_scope('layer_1'):
            self.W1 = self.weight_var('W1', [D, H1])
            self.b1 = self.bias_var('b1', [H1])        
        with tf.name_scope('layer_2'):
            self.W2 = self.weight_var('W2', [H1, H2])
            self.b2 = self.bias_var('b2', [H2])
        with tf.name_scope('layer_3'):
            self.W3 = self.weight_var('W3', [H2, H3])
            self.b3 = self.bias_var('b3', [H3])
        with tf.name_scope('layer_4'):
            self.W4 = self.weight_var('W4', [H3, H4])
            self.b4 = self.bias_var('b4', [H4])
        with tf.name_scope('layer_5'):
            self.W5 = self.weight_var('W5', [H4, d])

        self.p = p
        self.d = d
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

    def transform(self, x, test=False):

        # layer 1
        x_input = tf.cast(x, tf.float32)
        h_preact1 = tf.matmul(x_input, self.W1) + self.b1
        h_1 = tf.nn.relu(h_preact1)

        # layer 2
        h_preact2 = tf.matmul(h_1, self.W2) + self.b2
        h_2 = tf.nn.relu(h_preact2)

        # layer 3
        h_preact3 = tf.matmul(h_2, self.W3) + self.b3
        h_3 = tf.nn.relu(h_preact3)

        # layer 4
        h_preact4 = tf.matmul(h_3, self.W4) + self.b4
        h_4 = tf.nn.relu(h_preact4)

        # layer 5
        h_preact5 = tf.matmul(h_4, self.W5) 
        projected = h_preact5

        projected = tf.cast(projected, tf.float32)
        projected = batch_norm(projected, decay=0.999, center=False, scale=False,
                                is_training=(not test), reuse=self.reuse, scope="norm")
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
        for scope in ["norm"]:
            bn_vars += tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)
        return bn_vars + [self.W1, self.b1, self.W2, self.b2,
                self.W3, self.b3, self.W4, self.b4, self.W5] 

    def save_weights(self, sess):
        np.save('models/W1.npy', sess.run(self.W1))
        np.save('models/b1.npy', sess.run(self.b1))
        np.save('models/W2.npy', sess.run(self.W2))
        np.save('models/b2.npy', sess.run(self.b2))
        np.save('models/W3.npy', sess.run(self.W3))
        np.save('models/b3.npy', sess.run(self.b3))
        np.save('models/W4.npy', sess.run(self.W4))
        np.save('models/b4.npy', sess.run(self.b4))
        np.save('models/W5.npy', sess.run(self.W5))

    def out_dim(self):
        return self.d

with tf.Graph().as_default():
    data_dir = "data/"
    n_inputs = 10
    mu_ranks = 10
    projector = NN(H1=1000, H2=1000, H3=500, H4=50, d=2)
    C = 2

    cov = SE_multidim(C, 0.7, 0.2, 0.1, projector)

    lr = 5e-3
    decay = (3, 0.2)
    n_epoch = 10
    batch_size = 5000
    data_type = 'numpy'
    log_dir = 'log'
    save_dir = None#'models/gpnn.ckpt'
    model_dir = None#save_dir
    load_model = False#True
    
    runner=GPCRunner(data_dir, n_inputs, mu_ranks, cov,
                lr=lr, decay=decay, n_epoch=n_epoch, batch_size=batch_size,
                data_type=data_type, log_dir=log_dir, save_dir=save_dir,
                model_dir=model_dir, load_model=load_model, print_freq=500,
                num_threads=3)
    runner.run_experiment()
