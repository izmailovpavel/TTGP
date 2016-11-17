import tensorflow as tf
import numpy as np

from scipy import optimize as opt 
from copy import deepcopy
from tensorflow.contrib.opt.python.training.external_optimizer import ScipyOptimizerInterface
from input import NUM_FEATURES


class SE:

#    def __init__(self, sigma_f, l, sigma_n):
#        """
#        Parameters should be float
#        """
#
#        self.sigma_f = tf.get_variable('Process_variance', [1], initializer=tf.constant_initializer(sigma_f), dtype=tf.float64)
#        self.l = tf.get_variable('Process_lengthscale', [1], initializer=tf.constant_initializer(l), dtype=tf.float64)
#        self.sigma_n = tf.get_variable('Noise_variance', [1], initializer=tf.constant_initializer(sigma_n), dtype=tf.float64)

    @staticmethod
    def cov(x1, x2, w, name=None):
        """
        Returns a covariance matrix for the two given arrays
        """
        with tf.name_scope(name, 'SEcov', [x1, x2]):
            dists = squared_dists(x1, x2)
            sigma_f, l, sigma_n = tf.unpack(w)
            return sigma_f**2. * tf.exp(- dists / (2. * (l ** 2.))) + tf.cast(tf.equal(dists, 0.0), tf.float64) * sigma_n**2.

    def __call__(self, x1, x2, w, name=None):
        return self.cov(x1, x2, w, name)

    @staticmethod
    def bounds():
        return [(1e-2, 1e3)]*3

    
class GP:

    def __init__(self, cov, w, inputs):
        self.cov = cov
        self.inputs = tf.get_variable('inputs', inputs.shape, initializer=tf.constant_initializer(inputs), dtype=tf.float64, trainable=False)
        print(self.inputs.get_shape())
        self.w = tf.get_variable('w', w.shape, initializer=tf.constant_initializer(w), dtype=tf.float64, trainable=False)
        self.m = inputs.shape[0]
        self.mu = tf.get_variable('mu', [self.m, 1], initializer=tf.constant_initializer(0.), dtype=tf.float64, trainable=False)
        self.Sigma = tf.get_variable('Sigma', [self.m, self.m], initializer=tf.constant_initializer(0.), dtype=tf.float64, trainable=False)
        self._make_placeholders()        

    def _predict_build_graph(self, x_test, name=None):
        inputs = self.inputs
        expectation, covariance = self.mu, self.Sigma
        with tf.name_scope(name, 'Predict', [x_test]):
            K_xm = self.cov(x_test, inputs, self.w)
            K_mm = self.cov(inputs, inputs, self.w)
            K_mm_inv = tf.matrix_inverse(K_mm)
            self.pred_op = tf.matmul(K_xm, tf.matmul(K_mm_inv, expectation))

    def elbo(self, X, y, w, name=None):
        with tf.name_scope(name, 'ELBO', [X, y]):
            sigma_n = w[-1]
            y = tf.reshape(y, [-1, 1])
            cov = self.cov
            inputs = self.inputs
            K_nm = cov(X, inputs, w)
            K_mn = tf.transpose(K_nm)
            K_mnK_nm = tf.matmul(K_mn, K_nm)
            K_mm = cov(inputs, inputs, w)
            K_mm_cho = tf.cholesky(K_mm)
            K_mm_inv = tf.matrix_inverse(K_mm)
            K_mm_logdet = 2 * tf.reduce_sum(tf.log(tf.diag_part(K_mm_cho))) 
            A = K_mm + K_mnK_nm / (sigma_n**2)
            A_cho = tf.cholesky(A)
            A_inv = tf.matrix_inverse(A)
            A_logdet = 2 * tf.reduce_sum(tf.log(tf.diag_part(A_cho)))
            K_mn_y = tf.matmul(K_mn, y)
            n = tf.cast(tf.shape(y)[0], tf.float64)

            y_B_inv_y =  tf.matmul(tf.transpose(y), y) / sigma_n**2 - tf.matmul(tf.transpose(K_mn_y), tf.matmul(A_inv, K_mn_y))/sigma_n**4
            B_logdet = A_logdet + 2 * n * tf.log(tf.abs(sigma_n)) - K_mm_logdet
            zeros = tf.zeros((1,1), dtype=tf.float64) 
            elbo = - B_logdet - y_B_inv_y - ((cov.cov(zeros, zeros, w)) * n - tf.trace(tf.matmul(K_mm_inv, K_mnK_nm)))/sigma_n**2
            return -elbo

#    def fit(self, X, y, maxiter):
#        with tf.Graph().as_default():
#            w_ph = tf.placeholder(tf.float64, self.w.get_shape(), 'w')
#            x_ph = tf.placeholder(tf.float64, X.shape, 'x_tr')
#            y_ph = tf.placeholder(tf.float64, y.shape, 'y_tr'
#            elbo = self.elbo(x_ph, y_ph, w_ph)
#            grad = tf.gradients(elbo, w_ph)
#            with tf.Session() as sess:
#                def _fun(w):
#                    elbo_val, grad_val = sess.run([elbo, grad], {x_ph: X, y_ph:y, w_ph:w})
#                    return elbo_val, grad_val.reshape(-1)
#            res = opt.minimize(_fun, self.w, method='L-BFGS-B', options={'maxiter': maxiter, 'disp':1}) 
#        self.w = res['x']
#        print(self.w)

    def _make_placeholders(self):
        self.x_tr_ph = tf.placeholder(tf.float64, shape=[None, None], name='x_tr_ph')
        self.y_tr_ph = tf.placeholder(tf.float64, shape=[None], name='y_tr_ph')
        self.x_te_ph = tf.placeholder(tf.float64, shape=[None, None], name='x_te_ph')
        self.y_te_ph = tf.placeholder(tf.float64, shape=[None,], name='y_te_ph')
        self.w_ph = tf.placeholder(tf.float64, self.w.get_shape(), 'w_ph') 
        self._fit_build_graph(self.x_tr_ph, self.y_tr_ph) 
        self._predict_build_graph(self.x_te_ph)

    def _fit_build_graph(self, X, y, name=None):
        with tf.name_scope(name, 'fit', [X, y]):
            self.elbo_op = self.elbo(X, y, self.w_ph)
            self.grad_op = tf.gradients(self.elbo_op, self.w_ph)
            self.mu_sigma_upd = self.mu_sigma(X, y)

    def fit(self, X, y, maxiter, sess):
        def _fun(w):
            feed_dict = {self.x_tr_ph: X, self.y_tr_ph: y, self.w_ph: w}
            fun, grad = sess.run([self.elbo_op, self.grad_op], feed_dict)
            return fun.reshape(-1)[0], grad[0]
        #np.save('A', _fun(self.w.eval())[0])
        if maxiter:
            res = opt.minimize(fun=_fun, x0=self.w.eval(), jac=True, bounds=self.cov.bounds(), method='L-BFGS-B', options={'maxiter': maxiter, 'disp':1}) 
            sess.run(tf.assign(self.w, res['x']))
        print(self.w.eval())
        sess.run(self.mu_sigma_upd, {self.x_tr_ph: X, self.y_tr_ph: y})

    def predict(self, X, sess):
        feed_dict = {self.x_te_ph: X}
        preds = sess.run(self.pred_op, feed_dict)
        return preds

    def mu_sigma(self, X, y, name=None):
        with tf.name_scope(name, 'MuSigma', [X, y]):
            y = tf.reshape(y, [-1, 1])
            cov = self.cov
            sigma_n = self.w[-1]
            inputs = self.inputs
            K_nm = cov(X, inputs, self.w)
            K_mn = tf.transpose(K_nm)
            K_mnK_nm = tf.matmul(K_mn, K_nm)
            K_mm = cov(inputs, inputs, self.w)
            K_mm_inv = tf.matrix_inverse(K_mm)
            
            Sigma = tf.matrix_inverse(K_mm + K_mnK_nm / sigma_n**2)
            mu = tf.matmul(K_mm, tf.matmul(Sigma, tf.matmul(K_mn, y))) / sigma_n**2
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

