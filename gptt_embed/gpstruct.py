"""
Module implementing the TT-GPstruct, a model for structured prediction based on
GP framework.
"""

import tensorflow as tf
import numpy as np

import t3f
import t3f.kronecker as kron
from t3f import ops, TensorTrain, TensorTrainBatch, batch_ops
from tf.contrib import crf

from gptt_embed.misc import _kron_tril, _kron_logdet, pairwise_quadratic_form

class TTGPstruct:

  def __init__(self, cov, inputs, x_init, y_init, mu_ranks):
    """Creates a `TTGPstruct` object for structured GP prediction.

    Args:
      cov:
      inputs: `InputsGrid` object.
      x_init: features for initialization.
      y_init: target values for initialization.
      mu_ranks: TT-ranks of means of the proceses at inducing inputs mu.
    """
    self.inputs = inputs
    self.inputs_dists = inputs.kron_dists()
    self.n_labels = cov.ndim
    self.cov = cov
    self.sigma_ls = self._get_sigma_ls()
    self.mus = self._get_mus(mu_ranks, x_init, y_init)
    self.N = 0 # Size of the training set

  def _get_mus(self, ranks, x_init, y_init):
    """Initialization of mus.
    """
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
    """Initialization of sigmas.
    """
    cov = self.cov
    inputs_dists = self.inputs_dists
    K_mm = cov.kron_cov(inputs_dists)    
    return t3f.get_variable('sigma_ls', initializer=kron.cholesky(K_mm))

  def initialize(self, sess):
    """Initializes variational and covariance parameters.

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

  def _sample_f(self, mus, sigmas):
    """Samples a value from all the processes.
    """
    #eps = tf.random_normal([self.nlabels])
    pass


  def _binary_complexity_penalty(self):
    """Computes the complexity penalty for binary potentials.

    This function computes KL-divergence between prior and variational 
    distribution over binary potentials.
    """
    pass

  def _unary_complexity_inducing_inputs(self):
    """Computes the complexity penalty for unary potentials.

    This function computes KL-divergence between prior and variational 
    distribution over the values of GPs at inducing inputs.

    Returns:
      A scalar `tf.Tensor` containing the complexity penalty for the processes.
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
    return tf.reduce_sum(penalty) / 2
      

  def elbo(self, x, y, seq_lens, name=None):
    '''Evidence lower bound.
    
    A doubly stochastic procedure based on reparametrization trick is used to 
    approximate the gradients of the ELBO with respect to the variational
    parameters.
    Args:
      x: `tf.Tensor` of shape `batch_size` x `max_seq_len` x D; 
      sequences of features for the current batch.
      y: `tf.Tensor` of shape `batch_size` x `max_seq_len`; target values for 
      the current batch.
      seq_lens: `tf.Tensor` of shape `bach_size`; lenghts of input sequences.

    Returns:
      A scalar `tf.Tensor` containing a stochastic approximation of the evidence
      lower bound.
    '''
    means, variances = self._predict_process_values(x, with_variance=True)

    l = tf.cast(tf.shape(y)[0], tf.float64) # batch size
    N = tf.cast(self.N, dtype=tf.float64) 

    y = tf.reshape(y, [-1, 1])
    indices = tf.concat([tf.range(tf.cast(l, tf.int64))[:, None], y], axis=1)

    # means for true classes
    means_c = tf.gather_nd(means, indices)
    print('GPC/elbo/means_c', means_c.get_shape())
    
    # Likelihood
    elbo = 0
    elbo += tf.reduce_sum(means_c)

    # Log-partition function expectation estimate
    sample_unary, sample_binary = self.sample_potentials(x)
    log_Z = crf.crf_log_norm(sample_unary, seq_lens, sample_binary)

    log_sum_exp_bound = tf.log(tf.reduce_sum(tf.exp(means + variances/2),
                                                                axis=1))
    print('GPC/elbo/log_sum_exp_bound', log_sum_exp_bound.get_shape())
    elbo -= tf.reduce_sum(log_sum_exp_bound)
    elbo /= l
    
    print('GPC/elbo/complexity_penalty', self._unary_complexity_penalty().get_shape())
    print('GPC/elbo/complexity_penalty', self._complexity_penalty_inducing_inputs().get_shape())
    elbo += self._unary_complexity_penalty() / N #is this right?
    elbo += self._complexity_penalty_inducing_inputs() / N
    return -elbo
  
  def fit(self, x, y, seq_lens, N, lr, global_step):
    """Fit the model.

    Args:
      x: `tf.Tensor` of shape `batch_size` x `max_seq_len` x D; 
      sequences of features for the current batch.
      y: `tf.Tensor` of shape `batch_size` x `max_seq_len`; target values for 
      the current batch.
      seq_lens: `tf.Tensor` of shape `bach_size`; lenghts of input sequences.
      N: number of training instances.
      lr: learning rate for the optimization method.
      global_step: global step tensor.
    """
    self.N = N
    fun = self.elbo(x, y)
    optimizer = tf.train.AdamOptimizer(learning_rate=lr)
    return fun, optimizer.minimize(fun, global_step=global_step)

  def predict(self, x, seq_lens, n_samples, test=False):
    '''Predicts the labels for the given sequences.

    Approximately finds the configuration of the graphical model which has
    the lowest expected energy based on sampling.
    Args:
      x: `tf.Tensor` of shape `batch_size` x `max_seq_len` x D; 
      sequences of features for the current batch.
      y: `tf.Tensor` of shape `batch_size` x `max_seq_len`; target values for 
      the current batch.
      seq_lens: `tf.Tensor` of shape `bach_size`; lenghts of input sequences.
      n_samples: number of samples used to estimate the optimal labels.
    '''
    pass
