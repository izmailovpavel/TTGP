import tensorflow as tf
import numpy as np

from tensorflow.contrib.opt.python.training.external_optimizer import ScipyOptimizerInterface
from input import NUM_FEATURES


class SE:

    def __init__(self, sigma_f, l, sigma_n):
        """
        Parameters should be float
        """

        self.sigma_f = tf.get_variable('Process_variance', [1], initializer=tf.constant_initializer(sigma_f), dtype=tf.float64)
        self.l = tf.get_variable('Process_lengthscale', [1], initializer=tf.constant_initializer(l), dtype=tf.float64)
        self.sigma_n = tf.get_variable('Noise_variance', [1], initializer=tf.constant_initializer(sigma_n), dtype=tf.float64)


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
        self.m = inputs.get_shape()[0].value
        self.mu = tf.get_variable('mu', [self.m, 1], initializer=tf.constant_initializer(0.), dtype=tf.float64, trainable=False)
        self.Sigma = tf.get_variable('Sigma', [self.m, self.m], initializer=tf.constant_initializer(0.), dtype=tf.float64, trainable=False)

    def predict(self, x_test, name=None):
        #self.mu_sigma(X, y)  
        inputs = self.inputs
        expectation, covariance = self.mu, self.Sigma
        with tf.name_scope(name, 'Predict', [x_test]):
            K_xm = self.cov(x_test, inputs)
            K_mm = self.cov(inputs, inputs)
            K_mm_inv = tf.matrix_inverse(K_mm)
            y = tf.matmul(K_xm, tf.matmul(K_mm_inv, expectation))
            return y

    def elbo(self, X, y, name=None):
        with tf.name_scope(name, 'ELBO', [X, y]):
            y = tf.reshape(y, [-1, 1])
            cov = self.cov
            inputs = self.inputs
            K_nm = cov(X, inputs)
            K_mn = tf.transpose(K_nm)
            K_mnK_nm = tf.matmul(K_mn, K_nm)
            K_mm = cov(inputs, inputs)
            K_mm_cho = tf.cholesky(K_mm)
            K_mm_inv = tf.matrix_inverse(K_mm)
            K_mm_logdet = 2 * tf.reduce_sum(tf.log(tf.diag_part(K_mm_cho))) 
            A = K_mm + K_mnK_nm / (cov.sigma_n**2)
            #print((K_mnK_nm / (cov.sigma_n**2)).get_shape())
            A_cho = tf.cholesky(A)
            A_inv = tf.matrix_inverse(A)
            A_logdet = 2 * tf.reduce_sum(tf.log(tf.diag_part(A_cho)))
            K_mn_y = tf.matmul(K_mn, y)
            
            y_B_inv_y =  tf.matmul(tf.transpose(y), y) / cov.sigma_n**2 - tf.matmul(tf.transpose(K_mn_y), tf.matmul(A_inv, K_mn_y))/cov.sigma_n**4
            B_logdet = A_logdet + 2 *  y.get_shape()[0].value * tf.log(tf.abs(cov.sigma_n)) - K_mm_logdet
            zeros = tf.zeros((1,1), dtype=tf.float64) 
            elbo = - B_logdet - y_B_inv_y - ((cov.cov(zeros, zeros)) * y.get_shape()[0].value - tf.trace(tf.matmul(K_mm_inv, K_mnK_nm)))/cov.sigma_n**2
            
            return -elbo

#    def fit(self, X, y, lr=0.5, name=None):
#        with tf.name_scope(name, 'fit', [X, y]):
#            fun = self.elbo(X, y)
#            return tf.train.GradientDescentOptimizer(lr).minimize(fun)

    def fit(self, X, y, maxiter, name=None):
        maxiter = maxiter or 10
        print('maxiter:', maxiter)
        with tf.name_scope(name, 'fit', [X, y]):
            fun = self.elbo(X, y)
            self.optimizer = ScipyOptimizerInterface(fun, method='L-BFGS-B', 
                    options={'maxiter': maxiter, 'disp': True})#, inequalities=inequalities)#, equalities=equalities)
    
    def run_fit(self, sess, feed_dict):
        def lcb():
            print(1)
        def cb(w):
            i = 0
            while True:
                i += 1
                print('Iteration', i, ':', w)
                yield None
        print('Minimizing...')
        print(sess.run(self.optimizer._vars))
        return self.optimizer.minimize(session=sess, feed_dict=feed_dict, step_callback=cb, loss_callback=lcb)


    def mu_sigma(self, X, y, name=None):
        with tf.name_scope(name, 'MuSigma', [X, y]):
            y = tf.reshape(y, [-1, 1])
            cov = self.cov
            inputs = self.inputs
            K_nm = cov(X, inputs)
            K_mn = tf.transpose(K_nm)
            K_mnK_nm = tf.matmul(K_mn, K_nm)
            K_mm = cov(inputs, inputs)
            K_mm_inv = tf.matrix_inverse(K_mm)
            
            Sigma = tf.matrix_inverse(K_mm + K_mnK_nm / cov.sigma_n**2)
            mu = tf.matmul(K_mm, tf.matmul(Sigma, tf.matmul(K_mn, y))) / cov.sigma_n**2
            A = tf.matmul(K_mm, tf.matmul(Sigma, K_mm))
            mu_assign = tf.assign(self.mu, mu)
            Sigma_assign = tf.assign(self.Sigma, A)
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

