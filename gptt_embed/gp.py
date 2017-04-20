import tensorflow as tf
import numpy as np

import t3f
import t3f.kronecker as kron
from t3f import ops, TensorTrain, TensorTrainBatch

from gptt_embed.misc import _kron_tril, _kron_logdet

class GP:

    def __init__(self, cov, inputs, x_init, y_init, mu_ranks): 
        '''Gaussian Process model for regression.
        
        Args:
            cov: covariance function.
            inputs: inducing inputs — InputsGrid object.
            x_init, y_init: tensors of data for initialization of GP parameters.
            mu_ranks: TT-ranks of mu — expectations of the process at
                inducing inputs.
        '''
        self.cov = cov
        self.inputs = inputs
        self.inputs_dists = inputs.kron_dists()
        with tf.variable_scope('gp_var_params'):
            self.sigma_l = self._get_sigma_l()
            self.mu = self._get_mu(mu_ranks, x_init, y_init)
        self.N = 0 # Size of the training set

    def initialize(self, sess):
        """Initializes the variational and covariance parameters.

        Args:
            sess: a `Session` instance
        """
        self.cov.initialize(sess)
        sess.run(tf.variables_initializer(self.sigma_l.tt_cores))
        sess.run(tf.variables_initializer(self.mu.tt_cores))

    def get_params(self):
        """Returns a list of all the parameters of the model.
        """
        
        gp_var_params = list(self.mu.tt_cores + self.sigma_l.tt_cores)
        cov_params = self.cov.get_params()
        return cov_params + gp_var_params


    def _get_mu(self, ranks, x, y):
        """Initializes latent inputs expectations mu.

        Either loads pretrained values of tt-cores of mu, or initializes it
        according to optimal formulas from the given data.

        Args:
            ranks: tt-ranks of mu
            x: features of a batch of objects
            y: targets of a batch of objects
        """
        w = self.inputs.interpolate_on_batch(self.cov.project(x))
        Sigma = ops.tt_tt_matmul(self.sigma_l, ops.transpose(self.sigma_l))
        temp = ops.tt_tt_matmul(w, y)        
        anc = ops.tt_tt_matmul(Sigma, temp) 
        res = TensorTrain([core[0, :, :, :, :] for core in anc.tt_cores], 
                tt_ranks=[1]*(anc.ndims()+1))
        res = res
        for i in range(1, anc.get_shape()[0]):
            elem = TensorTrain([core[i, :, :, :, :] for core in anc.tt_cores],
                    tt_ranks=[1]*(anc.ndims()+1))
            res = ops.add(res, elem)
        mu_ranks = [1] + [ranks] * (res.ndims() - 1) + [1]
        return t3f.get_variable('tt_mu', initializer=TensorTrain(res.tt_cores, 
                                    res.get_raw_shape(), mu_ranks))

    def _get_sigma_l(self):
        """Initializes latent inputs covariance Sigma_l.
        """
        shapes = self.inputs.npoints
        cov = self.cov
        inputs_dists = self.inputs_dists
        K_mm = cov.kron_cov(inputs_dists)    
        return t3f.get_variable('sigma_l', 
                                initializer=kron.cholesky(K_mm))

    def predict(self, x_test, name=None):
        '''Predicts the value of the process at new points.

        Args:
            w_test: interpolation vectors at test points.
            name: name of the op.
        '''
        inputs_dists = self.inputs_dists
        expectation = self.mu
        with tf.name_scope(name, 'GP_Predict', [x_test]):
            w_test = self.inputs.interpolate_on_batch(self.cov.project(x_test))
            K_mm = self.cov.kron_cov(inputs_dists)
            K_mm_noeig = self.cov.kron_cov(inputs_dists, eig_correction=0.)
            K_xm = ops.tt_tt_matmul(K_mm_noeig, w_test)
            K_mm_inv = kron.inv(K_mm)
            y = ops.tt_tt_flat_inner(K_xm, 
                                       t3f.tt_tt_matmul(K_mm_inv, expectation))
            return y

    def elbo(self, w, y, name=None):
        '''Evidence lower bound.
        
        Args:
            w: interpolation vector for the current batch.
            y: target values for the current batch.
        '''
        
        with tf.name_scope(name, 'ELBO', [w, y]):

            l = tf.cast(tf.shape(y)[0], tf.float64) # batch size
            N = tf.cast(self.N, dtype=tf.float64) 

            y = tf.reshape(y, [-1, 1])
           
            mu = self.mu
            sigma_l = _kron_tril(self.sigma_l)
            ops.transpose(sigma_l)
            sigma = ops.tt_tt_matmul(sigma_l, ops.transpose(sigma_l))
            sigma_logdet = _kron_logdet(sigma_l)

            cov = self.cov
            inputs_dists = self.inputs_dists
            K_mm = cov.kron_cov(inputs_dists)
            
            K_mm_inv = kron.inv(K_mm)
            K_mm_logdet = kron.slog_determinant(K_mm)[1]
            K_mm_noeig = cov.kron_cov(inputs_dists, eig_correction=0.)

            tilde_K_ii = l * (self.cov.sigma_n**2 + self.cov.sigma_f**2)
            tilde_K_ii -= tf.reduce_sum(ops.tt_tt_flat_inner(w, 
                                                 ops.tt_tt_matmul(K_mm, w)))
            
            elbo = 0
            elbo += - tf.log(tf.abs(cov.sigma_n)) * l 
            elbo -= (tf.reduce_sum(
                tf.square(y[:,0] - ops.tt_tt_flat_inner(w, mu)))
                    / (2 * cov.sigma_n**2))
            elbo += - tilde_K_ii/(2 * cov.sigma_n**2)
            elbo += - (ops.tt_tt_flat_inner(w, ops.tt_tt_matmul(sigma, w))
                    / (2 * cov.sigma_n**2))
            elbo /= l
            elbo += - K_mm_logdet / (2 * N)
            elbo += sigma_logdet / (2 * N)
            elbo += - ops.tt_tt_flat_inner(sigma, K_mm_inv) / (2 * N)
            elbo += - ops.tt_tt_flat_inner(mu, 
                                   ops.tt_tt_matmul(K_mm_inv, mu)) / (2 * N)
            return -elbo[0]
    
    def fit(self, x, y, N, lr, global_step, name=None):
        """Fit the GP to the data.

        Args:
            w: interpolation vector for the current batch
            y: target values for the current batch
            N: number of training points
            lr: learning rate for the optimization method
            global_step: global step tensor
            name: name for the op.
        """
        self.N = N
        with tf.name_scope(name, 'fit', [x, y]):
            w = self.inputs.interpolate_on_batch(self.cov.project(x))
            fun = self.elbo(w, y)
            #return fun, tf.train.AdamOptimizer(learning_rate=lr).minimize(fun,
            #    global_step=global_step) 
            optimizer = tf.train.AdamOptimizer(learning_rate=lr)
            #optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr)
            return fun, optimizer.minimize(fun, global_step=global_step)

