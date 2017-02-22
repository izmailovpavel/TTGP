import tensorflow as tf
import numpy as np

import t3f
import t3f.kronecker as kron
from t3f import ops, TensorTrain
from tt_batch import *
from input import NUM_FEATURES, make_tensor

class SE:

    def __init__(self, sigma_f, l, sigma_n):
        """Squared Exponentia kernel.
        """
        self.sigma_f = tf.get_variable('Process_variance', [1], 
                                initializer=tf.constant_initializer(sigma_f), 
                                dtype=tf.float64)
        self.l = tf.get_variable('Process_lengthscale', [1], 
                                initializer=tf.constant_initializer(l), 
                                dtype=tf.float64)
        self.sigma_n = tf.get_variable('Noise_variance', [1], 
                                initializer=tf.constant_initializer(sigma_n), 
                                dtype=tf.float64)
        
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
        return [self.sigma_f, self.l, self.sigma_n]


class GP:

    def __init__(self, cov, inputs, mu_ranks=5):
        '''Gaussian Process model.
        
        Args:
            cov: covariance function.
            inputs: inducing inputs — InputsGrid object.
            mu_ranks: TT-ranks of mu — expectations of the process at
                inducing inputs.
        '''
        self.cov = cov
        self.inputs = inputs
        self.inputs_dists = inputs.kron_dists()
        self.m = inputs.size
        self.mu = self._get_mu(mu_ranks)
        self.sigma_l = self._get_sigma_l() 
        self.N = 0 # Size of the training set

    def _get_mu(self, ranks):
        '''Returns a variable representing mu.

        TT-cores of mu are initialized with gaussian random vectors.
        Args:
            ranks: TT-ranks of mu.
        '''
        mu_r = ranks
        shapes = self.inputs.npoints
        mu_cores = [np.random.randn(mu_r, shape_i, 1, mu_r) for shape_i in 
                    shapes[1:-1]]
        mu_cores = [np.random.randn(1, shapes[0], 1, mu_r)] + mu_cores
        mu_cores = mu_cores + [np.random.randn(mu_r, shapes[-1], 1, 1)]
        mu_shape = (tuple(shapes), tuple([1] * len(shapes)))
        mu_ranks = [1] + [mu_r] * (len(shapes) - 1) + [1]
        tt_mu_init = TensorTrain(mu_cores, mu_shape, mu_ranks)
        tt_mu = t3f.cast(t3f.get_variable('tt_mu', initializer=tt_mu_init), 
                          tf.float64)
        return tt_mu

    def _get_sigma_l(self):
        '''Reterns a variable, represrnting sigma_l.

        TT-cores of sigma_l are initialized with identity matrices.
        '''
        shapes = self.inputs.npoints
        sigma_cores = [np.eye(shape_i)[None, :, :, None] for shape_i in shapes]
        sigma_shape = (tuple(shapes), tuple(shapes))
        sigma_ranks = [1] * (len(shapes) + 1)
        tt_sigma_l_init = TensorTrain(sigma_cores, sigma_shape, sigma_ranks)
        tt_sigma_l = t3f.cast(t3f.get_variable('tt_sigma_l', 
                                    initializer=tt_sigma_l_init), tf.float64)
        return tt_sigma_l

    def predict(self, w_test, name=None):
        '''Predicts the value of the process at new points.

        Args:
            w_test: interpolation vectors at test points.
            name: name of the op.
        '''
        inputs_dists = self.inputs_dists
        expectation = self.mu
        with tf.name_scope(name, 'Predict', [w_test]):
            K_mm = self.cov.kron_cov(inputs_dists)
            K_mm_noeig = self.cov.kron_cov(inputs_dists, eig_correction=0.)
            K_xm = batch_tt_tt_matmul(K_mm_noeig, w_test)
            K_mm_inv = kron.inv(K_mm)
            y = batch_tt_tt_flat_inner(K_xm, 
                                       t3f.tt_tt_matmul(K_mm_inv, expectation))
            return y
        
    @staticmethod
    def _kron_tril(kron_mat, name=None):
        '''Computes the lower triangular part of a kronecker-factorized matrix.

        Note, that it computes it as a product of the lower triangular parts
        of the elements of the product, which is not exactly the lower 
        triangular part.
        '''
        with tf.name_scope(name, 'Kron_band_part', [kron_mat]):
            mat_l_cores = []
            for core_idx in range(kron_mat.ndims()):
                core = kron_mat.tt_cores[core_idx][0, :, :, 0]
                mat_l_cores.append(tf.matrix_band_part(core,-1, 0)
                                                    [None, :, :, None])
            mat_l_shape = kron_mat.get_raw_shape()
            mat_l_ranks = kron_mat.get_tt_ranks()
            mat_l = TensorTrain(mat_l_cores, mat_l_shape, mat_l_ranks)
            return mat_l

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
            sigma = ops.tt_tt_matmul(sigma_l, ops.transpose(sigma_l))
            sigma_logdet = 2 * kron.slog_determinant(sigma_l)[1]

            cov = self.cov
            inputs_dists = self.inputs_dists
            K_mm = cov.kron_cov(inputs_dists)
            
            K_mm_inv = kron.inv(K_mm)
            K_mm_logdet = kron.slog_determinant(K_mm)[1]
            K_mm_noeig = cov.kron_cov(inputs_dists, eig_correction=0.)

            Lambda_i = batch_tt_tt_matmul(w, batch_transpose(w))

            tilde_K_ii = l * (self.cov.sigma_n**2 + self.cov.sigma_f**2)
            tilde_K_ii -= tf.reduce_sum(batch_quadratic_form(K_mm, w, w))
            
            elbo = 0
            elbo += - tf.log(tf.abs(cov.sigma_n)) * l 
            elbo -= (tf.reduce_sum(
                        tf.square(y - batch_tt_tt_flat_inner(w, mu)[:, :, 0]))
                    / (2 * cov.sigma_n**2))
            elbo += - tilde_K_ii/(2 * cov.sigma_n**2)
            elbo += - (tf.reduce_sum(batch_tt_tt_flat_inner(sigma, Lambda_i))
                    / (2 * cov.sigma_n**2))
            elbo /= l
            elbo += - K_mm_logdet / (2 * N)
            elbo += sigma_logdet / (2 * N)
            elbo += - ops.tt_tt_flat_inner(sigma, K_mm_inv) / (2 * N)
            elbo += - ops.quadratic_form(K_mm_inv, mu, mu) / (2 * N)    
            return -elbo[0, 0]

    def fit(self, w, y, N, lr=0.5, name=None):
        """Fit the GP to the data.

        Args:
            w: interpolation vector for the current batch.
            y: target values for the current batch.
            N: number of training points.
            lr: learning rate for the optimization method.
            name: name for the op.
        """
        self.N = N
        with tf.name_scope(name, 'fit', [w, y]):
            fun = self.elbo(w, y)
            print('Adadelta, lr=', lr)
            return fun, tf.train.AdamOptimizer(learning_rate=lr).minimize(fun)


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

