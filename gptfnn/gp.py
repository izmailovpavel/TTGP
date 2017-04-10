import tensorflow as tf
import numpy as np

import t3f
import t3f.kronecker as kron
from t3f import ops, TensorTrain, TensorTrainBatch
from tensorflow.contrib.opt import ScipyOptimizerInterface

from misc import _kron_tril, _kron_logdet

class GP:

    def __init__(self, cov, inputs, x_init, y_init, mu_ranks=5, 
            load_mu_sigma=False):
        '''Gaussian Process model.
        
        Args:
            cov: covariance function.
            inputs: inducing inputs — InputsGrid object.
            mu_ranks: TT-ranks of mu — expectations of the process at
                inducing inputs.
            load_mu_sigma: wether or not to load pretrained values for mu and 
                sigma.
        '''
        self.cov = cov
        self.inputs = inputs
        self.inputs_dists = inputs.kron_dists()
        self.sigma_l = self._get_sigma_l(load=load_mu_sigma) 
        self.mu = self._get_mu(mu_ranks, x_init, y_init, load=load_mu_sigma)
        self.N = 0 # Size of the training set

    def _get_mu(self, ranks, x, y, load=False):
        """Initializes latent inputs expectations mu.

        Either loads pretrained values of tt-cores of mu, or initializes it
        according to optimal formulas from the given data.

        Args:
            ranks: tt-ranks of mu
            x: features of a batch of objects
            y: targets of a batch of objects
            load: bool, loads pretrained values if True
        """
        if load:
            mu_r = ranks
            shapes = self.inputs.npoints
            mu_cores = []
            for i in range(len(shapes)):
                mu_cores.append(np.load('temp/mu_core'+str(i)+'.npy'))
            mu_shape = (tuple(shapes), tuple([1] * len(shapes)))
            mu_ranks = [1] + [mu_r] * (len(shapes) - 1) + [1]
            tt_mu_init = TensorTrain(mu_cores, mu_shape, mu_ranks)
            tt_mu = t3f.cast(t3f.get_variable('tt_mu', initializer=tt_mu_init), 
                              tf.float64)
            return tt_mu
        
        else:
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

    def _get_sigma_l(self, load=False):
        """Initializes latent inputs covariance Sigma_l.

        Either loads pretrained values of kronecker-cores of mu, or initializes 
        it.

        Args:
            load: bool, loads pretrained values if True
        """
        shapes = self.inputs.npoints
        if load:
            sigma_cores = []
            for i in range(len(shapes)):
                sigma_cores.append(np.load('temp/sigma_l_core'+str(i)+'.npy'))
            sigma_shape = (tuple(shapes), tuple(shapes))
            sigma_ranks = [1] * (len(shapes) + 1)
            tt_sigma_l_init = t3f.cast(TensorTrain(sigma_cores, sigma_shape, 
                                        sigma_ranks), tf.float64)
            tt_sigma_l = t3f.get_variable('tt_sigma_l', 
                                    initializer=tt_sigma_l_init)   
            return tt_sigma_l
        else:
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
            sigma_l = self._kron_tril(self.sigma_l)
            ops.transpose(sigma_l)
            sigma = ops.tt_tt_matmul(sigma_l, ops.transpose(sigma_l))
            sigma_logdet = self._kron_logdet(sigma_l)

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
    
    def fit(self, x, y, N, lr=0.5, name=None):
        """Fit the GP to the data.

        Args:
            w: interpolation vector for the current batch.
            y: target values for the current batch.
            N: number of training points.
            lr: learning rate for the optimization method.
            name: name for the op.
        """
        self.N = N
        with tf.name_scope(name, 'fit', [x, y]):
            w = self.inputs.interpolate_on_batch(self.cov.project(x))
            fun = self.elbo(w, y)
            print('Adadelta, lr=', lr)
            return fun, tf.train.AdamOptimizer(learning_rate=lr).minimize(fun)
    
    def get_mu_sigma_cores(self):
        return self.mu.tt_cores, self.sigma_l.tt_cores


