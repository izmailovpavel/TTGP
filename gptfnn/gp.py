import tensorflow as tf
import numpy as np

import t3f
import t3f.kronecker as kron
from t3f import ops, TensorTrain, TensorTrainBatch
from tensorflow.contrib.opt import ScipyOptimizerInterface

class SE:

    def __init__(self, sigma_f, l, sigma_n, P, trainable):
        """Squared Exponentia kernel.
        """
        d, D = P.shape
        self.P = tf.get_variable('Projection_matrix', [d, D],
                                initializer=tf.constant_initializer(P), 
                                dtype=tf.float64, trainable=trainable)
        self.sigma_f = tf.get_variable('Process_variance', [1], 
                                initializer=tf.constant_initializer(sigma_f), 
                                dtype=tf.float64, trainable=trainable)
        self.l = tf.get_variable('Process_lengthscale', [1], 
                                initializer=tf.constant_initializer(l), 
                                dtype=tf.float64, trainable=trainable)
        self.sigma_n = tf.get_variable('Noise_variance', [1], 
                                initializer=tf.constant_initializer(sigma_n), 
                                dtype=tf.float64, trainable=trainable)

    def project(self, x):
        projected = tf.matmul(x, tf.transpose(self.P))
        projected = tf.minimum(projected, 1)
        projected = tf.maximum(projected, -1)
        return projected

    def kron_cov(self, kron_dists, eig_correction=1e-2, name=None):
        """Computes the covariance matrix, given a kronecker product 
        representation of distances.

        Args:
            kron_dists: kronecker product representation of pairwise i
                distances.
            eig_correction: eigenvalue correction for numerical stability.
            name: name for the op.
        """
        with tf.name_scope(name, 'SEcov', [kron_dists]):
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

    def get_params(self):
        return [self.sigma_f, self.l, self.sigma_n, self.P]


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
        #self.m = inputs.size
        self.sigma_l = self._get_sigma_l(load=load_mu_sigma) 
        self.mu = self._get_mu(mu_ranks, x_init, y_init, load=load_mu_sigma)
        self.N = 0 # Size of the training set

    def _get_mu(self, ranks, x, y, load=False):
        """
        Computes optimal mu.
        """
        if load:
            mu_r = ranks
            shapes = self.inputs.npoints
            mu_cores = []
            for i in range(len(shapes)):
                mu_cores.append(np.load('mu_core'+str(i)+'.npy'))
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

    def _get_sigma_l(self, load=False, name=None):
        shapes = self.inputs.npoints
        if load:
            sigma_cores = []
            for i in range(len(shapes)):
                sigma_cores.append(np.load('sigma_l_core'+str(i)+'.npy'))
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
        with tf.name_scope(name, 'Predict', [x_test]):
            w_test = self.inputs.interpolate_on_batch(self.cov.project(x_test))
            K_mm = self.cov.kron_cov(inputs_dists)
            K_mm_noeig = self.cov.kron_cov(inputs_dists, eig_correction=0.)
            K_xm = ops.tt_tt_matmul(K_mm_noeig, w_test)
            K_mm_inv = kron.inv(K_mm)
            y = ops.tt_tt_flat_inner(K_xm, 
                                       t3f.tt_tt_matmul(K_mm_inv, expectation))
            return y
        
    @staticmethod
    def _kron_tril(kron_mat, name=None):
        '''Computes the lower triangular part of a kronecker-factorized matrix.

        Note, that it computes it as a product of the lower triangular parts
        of the elements of the product, which is not exactly the lower 
        triangular part.
        '''
        with tf.name_scope(name, 'Kron_tril', [kron_mat]):
            mat_l_cores = []
            for core_idx in range(kron_mat.ndims()):
                core = kron_mat.tt_cores[core_idx][0, :, :, 0]
                mat_l_cores.append(tf.matrix_band_part(core,-1, 0)
                                                    [None, :, :, None])
            mat_l_shape = kron_mat.get_raw_shape()
            mat_l_ranks = kron_mat.get_tt_ranks()
            mat_l = TensorTrain(mat_l_cores, mat_l_shape, mat_l_ranks)
            return mat_l
    
    @staticmethod
    def _kron_logdet(kron_mat, name=None):
        '''Computes the logdet of a kronecker-factorized matrix.
        '''
        with tf.name_scope(name, 'Kron_logdet', [kron_mat]):
            i_shapes = kron_mat.get_raw_shape()[0]
            pows = tf.cast(tf.reduce_prod(i_shapes), kron_mat.dtype)
            logdet = 0.
            for core_idx in range(kron_mat.ndims()):
                core = kron_mat.tt_cores[core_idx][0, :, :, 0]
                core_pow = pows / i_shapes[core_idx].value
                logdet += (core_pow * 
                    tf.reduce_sum(tf.log(tf.abs(tf.diag_part(core)))))
            logdet *= 2
            return logdet

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
            #sigma_logdet = 2 * kron.slog_determinant(sigma_l)[1]
            sigma_logdet = self._kron_logdet(sigma_l)

            cov = self.cov
            inputs_dists = self.inputs_dists
            K_mm = cov.kron_cov(inputs_dists)
            
            K_mm_inv = kron.inv(K_mm)
            K_mm_logdet = kron.slog_determinant(K_mm)[1]
            K_mm_noeig = cov.kron_cov(inputs_dists, eig_correction=0.)

            #Lambda_i = ops.tt_tt_matmul(w, ops.transpose(w))
            tilde_K_ii = l * (self.cov.sigma_n**2 + self.cov.sigma_f**2)
            tilde_K_ii -= tf.reduce_sum(ops.tt_tt_flat_inner(w, 
                                                 ops.tt_tt_matmul(K_mm, w)))
            
            elbo = 0
            elbo += - tf.log(tf.abs(cov.sigma_n)) * l 
            elbo -= (tf.reduce_sum(
                tf.square(y[:,0] - ops.tt_tt_flat_inner(w, mu)))
                    / (2 * cov.sigma_n**2))
            elbo += - tilde_K_ii/(2 * cov.sigma_n**2)
            #elbo += - (tf.reduce_sum(ops.tt_tt_flat_inner(sigma, Lambda_i))
            #        / (2 * cov.sigma_n**2))
            elbo += - (ops.tt_tt_flat_inner(w, ops.tt_tt_matmul(sigma, w))
                    / (2 * cov.sigma_n**2))
            elbo /= l
            elbo += - K_mm_logdet / (2 * N)
            elbo += sigma_logdet / (2 * N)
            elbo += - ops.tt_tt_flat_inner(sigma, K_mm_inv) / (2 * N)
            elbo += - ops.tt_tt_flat_inner(mu, 
                                   ops.tt_tt_matmul(K_mm_inv, mu)) / (2 * N)
            return -elbo[0]
    
    def fit_stoch(self, x, y, N, lr=0.5, name=None):
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


def r2(y_pred, y_true, name=None):
    """r2 score.
    """
    with tf.name_scope(name, 'r2_score', [y_pred, y_true]):
        mse_score = mse(y_pred, y_true)
        return 1. - mse_score / mse(tf.ones_like(y_true) * 
                    tf.reduce_mean(y_true), y_true)


def mse(y_pred, y_true, name=None):
    """MSE score.
    """
    with tf.name_scope(name, 'mse', [y_pred, y_true]):
        mse = tf.reduce_mean(tf.squared_difference(tf.reshape(y_pred, [-1]), 
                             tf.reshape(y_true, [-1])), name='MSE')
        return mse

