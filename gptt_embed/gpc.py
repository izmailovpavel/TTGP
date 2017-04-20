import tensorflow as tf
import numpy as np

import t3f
import t3f.kronecker as kron
from t3f import ops, TensorTrain, TensorTrainBatch

from gptt_embed.misc import _kron_tril, _kron_logdet
from gp_base import _TTGPbase

class GPC:

    def __init__(self, covs, inputs, x_init, y_init, mu_ranks): 
        '''Gaussian Process model for multiclass classification.
        
        Args:
            covs: a list of covariances; one covariance for each class; covs
            should share the same FeatureTransformer.
            inputs: inducing inputs — InputsGrid object.
            x_init, y_init: tensors of data for initialization of GP parameters.
            mu_ranks: TT-ranks of mu — expectations of the process at
                inducing inputs.
        '''
#        self.covs = covs
        self.inputs = inputs
        self.inputs_dists = inputs.kron_dists()
        self.n_class = len(covs)
        self.gps = []
        for c, cov in enumerate(covs):
            print('GPC/__init__/c', c)
            y_init_c = self._prepare_labels(y_init, c)
            self.gps.append(_TTGPbase(cov, inputs, x_init, y_init_c))
        self.N = 0 # Size of the training set

    @staticmethod
    def _prepare_labels(y, c):
        """Prepares labels for GP initialization. 

        Returns a vector y_c, s.t. 
            y_c == 1 if y == c; y_c == -1 if y ≠ c;
        Args:
            y: target values
            c: class label
        """
        return 2 * tf.cast(tf.equal(y, c), tf.float64) - 1

    def initialize(self, sess):
        """Initializes the variational and covariance parameters.

        Args:
            sess: a `Session` instance
        """
        for gp in self.gps:
            gp.initialize(sess)

    def get_params(self):
        """Returns a list of all the parameters of the model.
        """
        params = []
        for gp in zip(self.gps):
            params += gp.get_params()
        return params

    def _process_predicions(self, x, with_variance=False):
        """Returns prediction means (and variances) for GPs for all classes.
        Args:
            x: data features
            with_variance: if True, returns process variance at x
        """
        means, variances = []
        for gp in self.gps:
            if with_variance:
                mean, var = append(gp.predict_process_value(x, with_variance))
                means.append(mean)
                variances.append(var)
            else:
                mean = append(gp.predict_process_value(x, with_variance))
                means.append(mean)
        means = tf.concat(means, axis=1)
        if not with_variance:
            return means
        variances = tf.concat(variances, axis=1)
        return means, variances


    def predict(self, x):
        '''Predicts the labels at points x.

        Note, this function predicts the label that has the highest expectation.
        This is not equivalent to the process with highest posterior (?).
        Args:
            x: data features.
        '''
        preds = self._process_predictions(x)
        return tf.argmax(preds, axis=1)  

    def elbo(self, w, y, name=None):
        '''Evidence lower bound.
        
        Args:
            w: interpolation vector for the current batch.
            y: target values for the current batch.
        '''
        
        with tf.name_scope(name, 'ELBO', [w, y]):

            means, variances = self._process_predictions(x, with_variance=True)

            l = tf.cast(tf.shape(y)[0], tf.float64) # batch size
            N = tf.cast(self.N, dtype=tf.float64) 

            y = tf.reshape(y, [-1, 1])
            indices = tf.concat([tf.range(l)[:, None], y], axis=1)
            # means for true classes
            means_c = tf.gather_nd(means, indices)
            print('GPC/elbo/means_c', means_c.get_shape())
           
            # Likelihood
            elbo = 0
            elbo += tf.reduce_sum(means_c)
            log_sum_exp_bound = tf.log(tf.reduce_sum(tf.exp(means + variances/2),
                                                                        axis=1))
            print('GPC/elbo/log_sum_exp_bound', log_sum_exp_bound.get_shape())
            elbo -= tf.reduce_sum(log_sum_exp_bound)
            elbo /= l
            elbo += self.gp.complexity_penalty() / N
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

