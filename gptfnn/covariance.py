import tensorflow as tf
import numpy as np

import t3f
import t3f.kronecker as kron
from t3f import ops, TensorTrain, TensorTrainBatch
from tensorflow.contrib.opt import ScipyOptimizerInterface

class SE:

    def __init__(self, sigma_f, l, sigma_n, projector, trainable=True):
        """Squared Exponential kernel.
        Args:
            sigma_f: process variance
            l: process length scale
            sigma_n: noise variance
            projector: `FeatureTransformer` object
            trainable: Bool, parameters are trainable iff True
        """
        self.sigma_f = tf.get_variable('Process_variance', [1], 
                                initializer=tf.constant_initializer(sigma_f), 
                                dtype=tf.float64, trainable=trainable)
        self.l = tf.get_variable('Process_lengthscale', [1], 
                                initializer=tf.constant_initializer(l), 
                                dtype=tf.float64, trainable=trainable)
        self.sigma_n = tf.get_variable('Noise_variance', [1], 
                                initializer=tf.constant_initializer(sigma_n), 
                                dtype=tf.float64, trainable=trainable)
        self.projector = projector

    def project(self, x, name=None):
        """Transforms the features of x with projector.
        Args:
            x: batch of data to be transformed through the projector.
        """
        with tf.name_scope(name, 'SE_project', [x]):
            return self.projector.transform(x)

    def kron_cov(self, kron_dists, eig_correction=1e-2, name=None):
        """Computes the covariance matrix, given a kronecker product 
        representation of distances.

        Args:
            kron_dists: kronecker product representation of pairwise
                distances.
            eig_correction: eigenvalue correction for numerical stability.
            name: name for the op.
        """
        with tf.name_scope(name, 'SE_kron_cov', [kron_dists]):
            res_cores = []
            for core_idx in range(kron_dists.ndims()):
                core = kron_dists.tt_cores[core_idx]
                cov_core = (self.sigma_f**(2./ kron_dists.ndims())* 
                            tf.exp(-core/(2. * (self.l**2.))))
                cov_core += tf.reshape(eig_correction *
                            tf.eye(core.get_shape()[1].value, dtype=tf.float64),
                            core.get_shape())
                res_cores.append(cov_core)
            res_shape = kron_dists.get_raw_shape()
            res_ranks = kron_dists.get_tt_ranks()
            return TensorTrain(res_cores, res_shape, res_ranks)

    def __call__(self, x1, x2, name=None):
        return self.cov(x1, x2, name)

#    def get_params(self):
#        return [self.sigma_f, self.l, self.sigma_n, self.P]

    def initialize(self, sess):
        """Runs the initializers for kernel parameters.

        Args:
            sess: `tf.Session` object
        """
        self.projector.initialize(sess)
        sess.run(tf.variables_initializer([self.sigma_f, self.l, self.sigma_n]))
