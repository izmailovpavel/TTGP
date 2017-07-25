import tensorflow as tf
import numpy as np

import t3f
import t3f.kronecker as kron
from t3f import ops, TensorTrain, TensorTrainBatch, batch_ops

from TTGP.misc import _kron_tril, _kron_logdet, pairwise_quadratic_form

class TTGPC:

    def __init__(self, cov, inputs, x_init, y_init, mu_ranks): 
        '''Gaussian Process model for multiclass classification.
        
        Args:
            cov: a multidimensional covariance.
            should share the same FeatureTransformer.
            inputs: inducing inputs - InputsGrid object.
            x_init, y_init: tensors of data for initialization of GP parameters.
            mu_ranks: TT-ranks of mu - expectations of the process at
                inducing inputs.
        '''
        self.inputs = inputs
        self.inputs_dists = inputs.kron_dists()
        self.n_class = cov.ndim
        self.cov = cov
        self.sigma_ls = self._get_sigma_ls()
        self.mus = self._get_mus(mu_ranks, x_init, y_init)
        self.N = 0 # Size of the training set

    def _get_mus(self, ranks, x_init, y_init):
        w = self.inputs.interpolate_on_batch(self.cov.project(x_init))
        Sigma = ops.tt_tt_matmul(self.sigma_ls[0], ops.transpose(self.sigma_ls[0]))
        temp = ops.tt_tt_matmul(w, y_init)        
        anc = ops.tt_tt_matmul(Sigma, temp) 
        res = TensorTrain([core[0, :, :, :, :] for core in anc.tt_cores], 
                tt_ranks=[1]*(anc.ndims()+1))
        res = res
        for i in range(1, anc.get_shape()[0]):
            elem = TensorTrain([core[i, :, :, :, :] for core in anc.tt_cores],
                    tt_ranks=[1]*(anc.ndims()+1))
            res = ops.add(res, elem)
        mu_ranks = [1] + [ranks] * (res.ndims() - 1) + [1]
        mu_cores = []
        for core in res.tt_cores:
            mu_cores.append(tf.tile(core[None, ...], [self.n_class, 1, 1, 1, 1]))
        return t3f.get_variable('tt_mus', 
            initializer=TensorTrainBatch(mu_cores, res.get_raw_shape(), mu_ranks))

    def _get_sigma_ls(self):
        cov = self.cov
        inputs_dists = self.inputs_dists
        K_mm = cov.kron_cov(inputs_dists)    
        return t3f.get_variable('sigma_ls', initializer=kron.cholesky(K_mm))

    def initialize(self, sess):
        """Initializes the variational and covariance parameters.

        Args:
            sess: a `Session` instance
        """
        self.cov.initialize(sess)
        sess.run(tf.variables_initializer(self.sigma_ls.tt_cores))
        sess.run(tf.variables_initializer(self.mus.tt_cores))

    def get_params(self):
        """Returns a list of all the parameters of the model.
        """
        gp_var_params = list(self.mus.tt_cores + self.sigma_ls.tt_cores)
        cov_params = self.cov.get_params()
        return cov_params + gp_var_params

    def _K_mms(self, eig_correction=1e-2):
        """Returns covariance matrix computed at inducing inputs. 
        """
        return self.cov.kron_cov(self.inputs_dists, eig_correction)

    def _predict_process_values(self, x, with_variance=False, test=False):
        w = self.inputs.interpolate_on_batch(self.cov.project(x, test=test))

        mean = batch_ops.pairwise_flat_inner(w, self.mus)
        if not with_variance:
            return mean
        K_mms = self._K_mms()

        sigma_ls = _kron_tril(self.sigma_ls)
        variances = []
        sigmas = ops.tt_tt_matmul(sigma_ls, ops.transpose(sigma_ls))
        variances = pairwise_quadratic_form(sigmas, w, w)
        variances -= pairwise_quadratic_form(K_mms, w, w)
        variances += self.cov.cov_0()[None, :]
        return mean, variances

    def predict(self, x, test=False):
        '''Predicts the labels at points x.

        Note, this function predicts the label that has the highest expectation.
        This is not equivalent to the process with highest posterior (?).
        Args:
            x: data features.
        '''
        preds = self._predict_process_values(x, test=test)
        return tf.argmax(preds, axis=1)  

    def complexity_penalty(self):
        """Returns the complexity penalty term for ELBO. 
        """
        mus = self.mus
        sigma_ls = _kron_tril(self.sigma_ls)
        sigmas = ops.tt_tt_matmul(sigma_ls, ops.transpose(sigma_ls))
        sigmas_logdet = _kron_logdet(sigma_ls)

        K_mms = self._K_mms()
        K_mms_inv = kron.inv(K_mms)
        K_mms_logdet = kron.slog_determinant(K_mms)[1]

        penalty = 0
        penalty += - K_mms_logdet
        penalty += sigmas_logdet
        penalty += - ops.tt_tt_flat_inner(sigmas, K_mms_inv)
        penalty += - ops.tt_tt_flat_inner(mus, 
                               ops.tt_tt_matmul(K_mms_inv, mus))
        return penalty / 2
        

    def elbo(self, x, y, name=None):
        '''Evidence lower bound.
        
        Args:
            w: interpolation vector for the current batch.
            y: target values for the current batch.
        '''
        
        with tf.name_scope(name, 'ELBO', [x, y]):

            means, variances = self._predict_process_values(x, with_variance=True)

            l = tf.cast(tf.shape(y)[0], tf.float64) # batch size
            N = tf.cast(self.N, dtype=tf.float64) 

            y = tf.reshape(y, [-1, 1])
            indices = tf.concat([tf.range(tf.cast(l, tf.int64))[:, None], y], axis=1)

            # means for true classes
            means_c = tf.gather_nd(means, indices)
           
            # Likelihood
            elbo = 0
            elbo += tf.reduce_sum(means_c)
            log_sum_exp_bound = tf.log(tf.reduce_sum(tf.exp(means + variances/2),
                                                                        axis=1))
            elbo -= tf.reduce_sum(log_sum_exp_bound)
            elbo /= l
            elbo += tf.reduce_sum(self.complexity_penalty()) / N
            return -elbo
    
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
            fun = self.elbo(x, y)
            optimizer = tf.train.AdamOptimizer(learning_rate=lr)
            return fun, optimizer.minimize(fun, global_step=global_step)
