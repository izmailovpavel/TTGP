import tensorflow as tf
import numpy as np

import t3f
import t3f.kronecker as kron
from t3f import ops, TensorTrain
from tt_batch import *
from input import NUM_FEATURES, make_tensor

class SE:

    def __init__(self, sigma_f, l, sigma_n):
        """
        Parameters should be float
        """

        self.sigma_f = tf.get_variable('Process_variance', [1], initializer=tf.constant_initializer(sigma_f), dtype=tf.float64)
        self.l = tf.get_variable('Process_lengthscale', [1], initializer=tf.constant_initializer(l), dtype=tf.float64)
        self.sigma_n = tf.get_variable('Noise_variance', [1], 
                                       initializer=tf.constant_initializer(sigma_n), 
                                       dtype=tf.float64)#, trainable=False)

    def kron_cov(self, kron_dists, with_noise=True, name=None):
        """
        Computes the covariance matrix, given a kronecker product representation
        of distances.
        """
        # TODO: this is wrong.

        with tf.name_scope(name, 'SEcov', [kron_dists]):
            res_cores = []
            print(kron_dists.tt_cores[0].get_shape())
            for core_idx in range(kron_dists.ndims()):
                core = kron_dists.tt_cores[core_idx]
                cov_core = (self.sigma_f**(2./ kron_dists.ndims())* 
                            tf.exp(-core/(2. * (self.l**2.))))
                if with_noise:
                    cov_core += (tf.cast(tf.equal(core, 0.0), tf.float64) *
                                 self.sigma_n**2.)
                res_cores.append(cov_core)
            res_shape = kron_dists.get_raw_shape()
            res_ranks = kron_dists.get_tt_ranks()
            return TensorTrain(res_cores, res_shape, res_ranks)

    def cov(self, x1, x2, name=None):
        """
        Returns a covariance matrix for the two given arrays
        """
        with tf.name_scope(name, 'SEcov', [x1, x2]):
            dists = squared_dists(x1, x2)
            return self.sigma_f**2. * tf.exp(- dists / (2. * (self.l ** 2.))) + tf.cast(tf.equal(dists, 0.0), tf.float64) * self.sigma_n**2.

    def __call__(self, x1, x2, name=None):
        return self.cov(x1, x2, name)

    @staticmethod
    def bounds():
        return [(1e-2, 1e3)]*3

    
    def get_params(self):
        return [self.sigma_f, self.l, self.sigma_n]

class GP:

    def __init__(self, cov, inputs):
        self.cov = cov
        self.inputs = inputs
        self.W_tr = None
        self.inputs_dists = inputs.kron_dists()
        self.inputs_full = make_tensor(inputs.full(), 'inputs')
        self.m = inputs.size
        
        tt_mu_init = t3f.random_matrix((self.inputs.npoints, None), tt_rank=2)
        tt_mu = t3f.cast(t3f.get_variable('tt_mu', initializer=tt_mu_init), 
                          tf.float64)

        self.mu = tt_mu
       # self.mu = tf.get_variable('mu', [self.m, 1], 
       #                           initializer=tf.constant_initializer(0.), 
       #                           dtype=tf.float64, trainable=True)
        
        tt_sigma_l_init = t3f.random_matrix((self.inputs.npoints, 
                                                self.inputs.npoints), tt_rank=1)

        tt_sigma_l = t3f.cast(t3f.get_variable('tt_sigma_l', 
                                    initializer=tt_sigma_l_init), tf.float64)

        self.sigma_l = tt_sigma_l 
       # self.sigma_l = tf.get_variable('Sigma_L', [self.m, self.m], 
       #                                 initializer=tf.constant_initializer(np.eye(self.m)), 
       #                                 dtype=tf.float64, trainable=True)
        self.N = 0 # Size of the training set

    def predict(self, w_test, name=None):
        inputs_dists = self.inputs_dists
        expectation = self.mu
        with tf.name_scope(name, 'Predict', [w_test]):
            K_mm = self.cov.kron_cov(inputs_dists)
            K_xm = batch_tt_tt_matmul(K_mm, w_test)
            K_mm_inv = kron.inv(K_mm)
            print(K_xm.get_raw_shape())
            print(t3f.tt_tt_matmul(K_mm_inv, expectation).get_raw_shape())
            print('((')
            #y = batch_tt_tt_matmul(K_xm, t3f.tt_tt_matmul(K_mm_inv, expectation))
            y = batch_tt_tt_flat_inner(K_xm, t3f.tt_tt_matmul(K_mm_inv, expectation))
            print('y', y.get_shape())
            return y

    def elbo(self, W, y, name=None):
        with tf.name_scope(name, 'ELBO', [W, y]):
            W_full = W #batch_full(W)

            # Computing band part
            sigma_l_cores = []
            for core_idx in range(self.sigma_l.ndims()):
                core = self.sigma_l.tt_cores[core_idx] 
                sigma_l_cores.append(tf.matrix_band_part(core,-1, 0))
            sigma_l_shape = self.sigma_l.get_raw_shape()
            sigma_l_ranks = self.sigma_l.get_tt_ranks()
            sigma_l = TensorTrain(sigma_l_cores, sigma_l_shape, sigma_l_ranks)

            l = tf.cast(tf.shape(y)[0], tf.float64) # batch size
            y = tf.reshape(y, [-1, 1])
            cov = self.cov
            #inputs = self.inputs.full()
            inputs_dists = self.inputs_dists
            #N = tf.cast(self.N
            N = tf.cast(self.N, dtype=tf.float64)
            
            sigma = tf.matmul(sigma_l, tf.transpose(sigma_l))
            mu = self.mu
            #K_mm = cov(inputs, inputs)
            K_mm = cov.kron_cov(inputs_dists)
            K_mm_no_noise = cov.kron_cov(inputs_dists, with_noise=False)
            
            K_mm_cho = kron.cholesky(K_mm)
            K_mm_inv = kron.inv(K_mm)
            K_mm_logdet = kron.slog_determinant(K_mm)[1]
            #K_mm_logdet = 2 * tf.reduce_sum(tf.log(tf.diag_part(K_mm_cho))) 
            K_mm_inv__mu = ops.tt_tt_matmul(K_mm_inv, mu)
            k_i = batch_tt_tt_matmul(K_mm_no_noise, W)
            print(k_i.get_shape(), 'k_i')
            zeros = tf.zeros((1,1), dtype=tf.float64) 
            tilde_K_ii = l * cov(zeros, zeros) - tf.reduce_sum(tf.einsum('ij,ji->i', tf.transpose(k_i), tf.matmul(K_mm_inv, k_i)))
            Lambda_i = tf.matmul(K_mm_inv, tf.matmul(k_i, tf.matmul(tf.transpose(k_i), K_mm_inv))) / cov.sigma_n**2
            
            elbo = 0
            
            elbo += - tf.log(tf.abs(cov.sigma_n)) * l 
            elbo -= tf.reduce_sum(tf.square(y - tf.matmul(tf.transpose(k_i), 
                                            K_mm_inv__mu)))/(2 * cov.sigma_n**2)
            elbo += - tilde_K_ii/(2 * cov.sigma_n**2)
            elbo += - tf.reduce_sum(tf.einsum('ij,ji->i', sigma, Lambda_i)) / 2
            elbo /= l
            elbo += - K_mm_logdet / (2 * N)
            elbo += tf.reduce_sum(tf.log(tf.abs(tf.diag_part(sigma_l)))) / N
            elbo += -tf.reduce_sum(tf.einsum('ij,ji->i', sigma, K_mm_inv)) / (2 * N)
            elbo += - tf.matmul(tf.transpose(mu), tf.matmul(K_mm_inv, mu)) / (2 * N)    
            return -elbo
    
    def check_interpolation(self, W, X):
        inputs = self.inputs.full()
        K_mm = self.cov.kron_cov(self.inputs_dists)
        K_mm_2 = self.cov(inputs, inputs)
        K_mm_nonoise = self.cov.kron_cov(self.inputs_dists, with_noise=False)
        
        k_i = self.cov(inputs, X)
        k_i_2 = batch_tt_tt_matmul(K_mm_nonoise, W)
        return tf.reduce_sum(tf.square(tf.transpose(k_i) - batch_full(k_i_2)[:, :, 0]))
        #k_i = cov(inputs, X)
        #k_i = tf.matmul(K_mm, tf.transpose(W))

    def fit(self, W, y, N, lr=0.5, name=None):
        self.N = N
        self.W_tr = W
        with tf.name_scope(name, 'fit', [W, y]):
            fun = self.elbo(W, y)
            print('Adadelta, lr=', lr)
#            return fun, tf.train.GradientDescentOptimizer(learning_rate=lr).minimize(fun)
#            return fun, tf.train.AdadeltaOptimizer(learning_rate=lr, rho=0.9).minimize(fun)
            return fun, tf.train.AdamOptimizer(learning_rate=lr).minimize(fun)

    def initialize_mu_sigma(self, X, y, name=None):
        # TODO: think of a clever initialization
        with tf.name_scope(name, 'MuSigma', [X, y]):
            y = tf.reshape(y, [-1, 1])
            cov = self.cov
            inputs = self.inputs.full()
            K_nm = cov(X, inputs)
            K_mn = tf.transpose(K_nm)
            K_mnK_nm = tf.matmul(K_mn, K_nm)
            K_mm = cov(inputs, inputs)
            K_mm_inv = tf.matrix_inverse(K_mm)
            
            Sigma = tf.matrix_inverse(K_mm + K_mnK_nm / cov.sigma_n**2)
            mu = tf.matmul(K_mm, tf.matmul(Sigma, tf.matmul(K_mn, y))) / cov.sigma_n**2
            A = tf.matmul(K_mm, tf.matmul(Sigma, K_mm))
            sigma_l = tf.cholesky(A)
            mu_assign = tf.assign(self.mu, mu)
            Sigma_assign = tf.assign(self.sigma_l, A)
            return tf.group(mu_assign, Sigma_assign)

def squared_dists(x1, x2, name=None):
    """
    Returns a matrix of pairwise distances
    """
    with tf.name_scope(name, 'PairwiseDist', [x1, x2]):
        x1_norms = tf.reshape(tf.reduce_sum(tf.square(x1), 1), [-1, 1])
        x2_norms = tf.reshape(tf.reduce_sum(tf.square(x2), 1), [1, -1])
        dists = x1_norms + x2_norms
        dists = dists - 2 * tf.matmul(x1, tf.transpose(x2))
        return dists

def r2(y_pred, y_true, name=None):
    with tf.name_scope(name, 'r2_score', [y_pred, y_true]):
        mse_score = mse(y_pred, y_true)
        return 1. - mse_score / mse(tf.ones_like(y_true) * tf.reduce_mean(y_true), y_true)

def mse(y_pred, y_true, name=None):
    """
    Builds the part of the graph, required to calculate loss
    """
    with tf.name_scope(name, 'mse', [y_pred, y_true]):
        mse = tf.reduce_mean(tf.squared_difference(tf.reshape(y_pred, [-1]), tf.reshape(y_true, [-1])),
            name='MSE')
        return mse

