import tensorflow as tf
import numpy as np

import t3f
import t3f.kronecker as kron
from t3f import ops, TensorTrain, TensorTrainBatch

from gptt_embed.misc import _kron_tril, _kron_logdet
from gptt_embed.gp_base import _TTGPbase

class TTGPR:

    def __init__(self, cov, inputs, x_init, y_init, mu_ranks): 
        '''Gaussian Process model for regression.
        
        Args:
            cov: covariance function.
            inputs: inducing inputs — InputsGrid object.
            x_init, y_init: tensors of data for initialization of GP parameters.
            mu_ranks: TT-ranks of mu — expectations of the process at
                inducing inputs.
        '''
        self.gp = _TTGPbase(cov, inputs, x_init, y_init, mu_ranks)
        self.N = 0 # Size of the training set

    def initialize(self, sess):
        """Initializes the variational and covariance parameters.

        Args:
            sess: a `Session` instance
        """
        self.gp.initialize(sess)

    def get_params(self):
        """Returns a list of all the parameters of the model.
        """
        return self.gp.get_params()        

    def predict(self, x, with_variance=False, name=None):
        '''Predicts the value of the process x points.

        Args:
            x: data features.
            with_variance: wether or not to return prediction variance
            name: name of the op.
        '''
        return self.gp.predict_process_value(x, with_variance=with_variance)

    def elbo(self, w, y, name=None):
        '''Evidence lower bound.
        
        Args:
            w: interpolation vector for the current batch.
            y: target values for the current batch.
        '''
        
        with tf.name_scope(name, 'ELBO', [w, y]):

            l = tf.cast(tf.shape(y)[0], tf.float64) # batch size
            N = tf.cast(self.N, dtype=tf.float64) 

            y = tf.reshape(y, [-1])
           
            mu = self.gp.mu
            sigma_l = _kron_tril(self.gp.sigma_l)
            sigma = ops.tt_tt_matmul(sigma_l, ops.transpose(sigma_l))
            
            sigma_n = self.gp.cov.noise_variance()
            
            K_mm = self.gp.K_mm()

            tilde_K_ii = l * self.gp.cov.cov_0()
            tilde_K_ii -= tf.reduce_sum(ops.tt_tt_flat_inner(w, 
                                                 ops.tt_tt_matmul(K_mm, w)))

            

            # Likelihood
            elbo = 0
            elbo -= tf.reduce_sum(tf.square(y - ops.tt_tt_flat_inner(w, mu)))
            elbo -= tilde_K_ii 
            elbo -= ops.tt_tt_flat_inner(w, ops.tt_tt_matmul(sigma, w))
            elbo /= 2 * sigma_n**2 * l
            elbo += self.gp.complexity_penalty() / N
            elbo -=  tf.log(tf.abs(sigma_n))  
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
            w = self.gp.inputs.interpolate_on_batch(self.gp.cov.project(x))
            fun = self.elbo(w, y)
            optimizer = tf.train.AdamOptimizer(learning_rate=lr)
            return fun, optimizer.minimize(fun, global_step=global_step)

