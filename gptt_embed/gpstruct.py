"""
Module implementing the TT-GPstruct, a model for structured prediction based on
GP framework.
"""

import tensorflow as tf
import numpy as np

import t3f
import t3f.kronecker as kron
from t3f import ops
from t3f import TensorTrain
from t3f import TensorTrainBatch
from t3f import batch_ops
from tensorflow.contrib.crf import crf_sequence_score
from tensorflow.contrib.crf import crf_log_norm

from gptt_embed.misc import _kron_tril
from gptt_embed.misc import _kron_logdet
from gptt_embed.misc import pairwise_quadratic_form
from gptt_embed.misc import _kron_sequence_pairwise_quadratic_form

class TTGPstruct:

  def __init__(self, cov, inputs, mu_ranks):
    """Creates a `TTGPstruct` object for structured GP prediction.

    Args:
      cov: covarianse object.
      inputs: `InputsGrid` object.
      mu_ranks: TT-ranks of means of the proceses at inducing inputs mu.
    """
    self.inputs = inputs
    self.inputs_dists = inputs.kron_dists()
    self.n_labels = cov.ndim
    self.cov = cov
    self.d = self.cov.projector.out_dim()
    self.N = 0 # Size of the training set
    self.bin_mu, self.bin_sigma_l = self._get_bin_vars()
    self.sigma_ls = self._get_sigma_ls()
    self.mus = self._get_mus(mu_ranks)

  def _get_bin_vars(self):
    """Initializes binary potential variational parameters.
    """
    init_mu = tf.zeros([self.n_labels**2], dtype=tf.float64)
    # TODO: Should we use diagonal variational covariance?
    init_sigma_l = tf.eye(self.n_labels**2, dtype=tf.float64)
    bin_mu = tf.get_variable('bin_mu', initializer=init_mu, dtype=tf.float64)
    bin_sigma_l = tf.get_variable('bin_sigma_l', initializer=init_sigma_l, 
        dtype=tf.float64)
    return bin_mu, bin_sigma_l

  def _get_mus(self, mu_ranks):
    """Initialize expectations of var distribution over unary potentials.
       
    Args:
      mu_ranks: TT-ranks of mus.
    """

    # TODO: is this a good initialization?
    x_init = tf.random_normal([mu_ranks, self.d], dtype=tf.float64)
    y_init = tf.random_normal([mu_ranks], dtype=tf.float64)

    w = self.inputs.interpolate_on_batch(self.cov.project(x_init))
    y_init_cores = [tf.reshape(y_init, (-1, 1, 1, 1, 1))]
    for core_idx in range(1, w.ndims()):
      y_init_cores += [tf.ones((mu_ranks, 1, 1, 1, 1), dtype=tf.float64)]
      y_init = t3f.TensorTrainBatch(y_init_cores)

    Sigma = ops.tt_tt_matmul(self.sigma_ls[0], ops.transpose(self.sigma_ls[0]))
    res_batch = t3f.tt_tt_matmul(Sigma, t3f.tt_tt_matmul(w, y_init))
    res = res_batch[0]
    for i in range(1, mu_ranks):
      res = res + res_batch[i]

    mu_ranks = [1] + [mu_ranks] * (res.ndims() - 1) + [1]
    mu_cores = []
    for core in res.tt_cores:
        mu_cores.append(tf.tile(core[None, ...], [self.n_labels, 1, 1, 1, 1]))
    return t3f.get_variable('tt_mus', 
        initializer=TensorTrainBatch(mu_cores, res.get_raw_shape(), mu_ranks))

  def _get_sigma_ls(self):
    """Initialize covariance matrix of var distribution over unary potentials.
    """
    cov = self.cov
    inputs_dists = self.inputs_dists
    K_mm = cov.kron_cov(inputs_dists)    
    return t3f.get_variable('sigma_ls', initializer=kron.cholesky(K_mm))

  def initialize(self, sess):
    """Initialize variational and covariance parameters.

    Args:
        sess: a `tf.Session` instance.
    """
    self.cov.initialize(sess)
    sess.run(tf.variables_initializer(self.sigma_ls.tt_cores))
    sess.run(tf.variables_initializer(self.mus.tt_cores))
    sess.run(tf.variables_initializer([self.bin_mu, self.bin_sigma_l]))

  def get_params(self):
    """Returns a list of all the parameters of the model.
    """
    bin_var_params = list(self.bin_mu_l + self.bin_sigma_l)
    un_var_params = list(self.mus.tt_cores + self.sigma_ls.tt_cores)
    var_params = bin_var_params + un_var_params
    cov_params = self.cov.get_params()
    return cov_params + gp_var_params

  def _K_mms(self, eig_correction=1e-2):
    """Returns covariance matrices computed at inducing inputs for all labels. 

    Args:
      eig_correction: eigenvalue correction for numerical stability.
    """
    return self.cov.kron_cov(self.inputs_dists, eig_correction)

  def _binary_complexity_penalty(self):
    """Computes the complexity penalty for binary potentials.

    This function computes negative KL-divergence between variational 
    distribution and prior over binary potentials.
    Returns:
      A scalar `tf.Tensor` containing the complexity penalty for the variational
      distribution over binary potentials.
    """
    # TODO: test this
    # TODO: should we use other kernels for binary potentials?
    K_bin = tf.eye(self.n_labels * self.n_labels)
    K_bin_log_det = tf.zeros([1])
    K_bin_inv = tf.eye(self.n_labels * self.n_labels)

    S_bin_l = self.bin_sigma_l
    S_bin = tf.matmul(S_bin_l, tf.transpose(S_bin_l))
    S_bin_logdet = _kron_logdet(S_bin_l)
    mu_bin = self.bin_mu
    
    KL = K_bin_logdet - S_bin_logdet
    KL += tf.einsum('ij,ji->i', K_bin_inv, S_bin)
    KL += tf.matmul(mu_bin, tf.transpose(tf.matmul(K_bin_inv, mu_bin)))
    KL = KL / 2
    return -KL

  def _unary_complexity_penalty(self):
    """Computes the complexity penalty for unary potentials.

    This function computes KL-divergence between prior and variational 
    distribution over the values of GPs at inducing inputs.

    Returns:
      A scalar `tf.Tensor` containing the complexity penalty for GPs 
      determining unary potentials.
    """
    # TODO: test this
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
      
  @staticmethod
  def _compute_pairwise_dists(x):
    """Computes pairwise distances in feature space for a batch of sequences.

    Args:
      x: `tf.Tensor` of shape `batch_size` x `max_seq_len` x d; 
      sequences of features for the current batch.

    Returns:
      A `tf.Tensor` of shape `batch_size` x `max_seq_len` x `max_seq_len`; for
      each sequence in the batch it contains a matrix of pairwise distances
      between it's elements in the feature space.
    """
    # TODO: test this
    x_norms = tf.reduce_sum(x**2, axis=2)[:, :, None]
    x_norms = x_norms + tf.transpose(x_norms, [0, 2, 1])
    batch_size, max_len, d = x.get_shape()
    print('_compute_pairwise_dists/x_norms', x_norms.get_shape(), '=', 
      batch_size, max_len, max_len)
    scalar_products = tf.einsum('bid,bjd->bij', x, x)
    print('_compute_pairwise_dists/scalar_products', 
        scalar_products.get_shape(), '=', batch_size, max_len, max_len)
    dists = x_norms - scalar_products
    return dists

  def _latent_vars_distribution(self, x, seq_lens):
    """Computes the parameters of the variational distribution over potentials.

    Args:
      x: `tf.Tensor` of shape `batch_size` x `max_seq_len` x d; 
      sequences of features for the current batch.
      seq_lens: `tf.Tensor` of shape `bach_size`; lenghts of input sequences.

    Returns:
      A tuple containing 4 `tf.Tensors`.
      `m_un`: a `tf.Tensor` of shape  `n_labels` x `batch_size` x `max_seq_len`;
        the expectations of the unary potentials.
      `S_un`: a `tf.Tensor` of shape 
        `n_labels` x `batch_size` x `max_seq_len` x `max_seq_len`; the
        covariance matrix of unary potentials.
      `m_bin`: a `tf.Tensor` of shape `max_seq_len`^2; the expectations
        of binary potentials.
      `S_bin`: a `tf.Tensor` of shape `max_seq_len`^2 x `max_seq_len`^2; the
        covariance matrix of binary potentials.
    """
    # TODO: check
    d, max_len, batch_size  = x.get_shape()
    mask = tf.sequence_mask(seq_lens, maxlen=max_len)
    indices = tf.where(sequence_mask)
    
    x_flat = tf.gather_nd(x, indices)
    print('_latent_vars_distribution/x_flat', x_flat.get_shape(), '=',
        '?', 'x', d)

    w = self.inputs.interpolate_on_batch(self.cov.project(x_flat))
    m_un_flat = batch_ops.pairwise_flat_inner(w, self.mu_s)
    print('_latent_vars_distribution/m_un_flat', m_un_flat.get_shape(), '=',
        '?', 'x', n_labels)
    m_un = tf.scatter_nd(indices, m_un_flat)

    dists = self._compute_pairwise_dists(x)
    sigma_f = self.cov.noise_variance()
    K_nn = self.cov.cov_for_squared_dists(dists) 
    K_nn += sigma_f**2 * tf.eye(max_len, batch_shape=[batch_size])
    print('_latent_vars_distribution/K_nn', K_nn.get_shape(), '=',
        n_labels, 'x', batch_size, 'x', max_len, 'x', max_len)
    
    sigmas = ops.tt_tt_matmul(self.sigma_l, self.sigma_l)
    K_mms = self._K_mms()

    S_un = K_nn
    S_un += _kron_sequence_pairwise_quadratic_form(sigmas, w, seq_lens, max_len)
    S_un -= _kron_sequence_pairwise_quadratic_form(K_mms, w, seq_lens, max_len)

    m_bin = self.bin_mu
    S_bin = tf.matmul(self.bin_sigma_l, tf.transpose(self.bin_sigma_l))
    return m_un, S_un, m_bin, m_un

  def _sample_f(self, m_un, S_un, m_bin, S_bin):
    """Samples a value from all the processes.

    Args:
      A tuple containing 4 `tf.Tensors`.
      m_un: a `tf.Tensor` of shape  `n_labels` x `batch_size` x `max_seq_len`;
        the expectations of the unary potentials.
      S_un: a `tf.Tensor` of shape 
        `n_labels` x `batch_size` x `max_seq_len` x `max_seq_len`; the
        covariance matrix of unary potentials.
      m_bin: a `tf.Tensor` of shape `max_seq_len`^2; the expectations
        of binary potentials.
      S_bin: a `tf.Tensor` of shape `max_seq_len`^2 x `max_seq_len`^2; the
        covariance matrix of binary potentials.
    """
    # TODO: check
    
    eps_un = tf.random_normal(m_un.get_shape())
    eps_bin = tf.random_normal(m_bin.get_shape())

    S_un_l = tf.cholesky(S_un)
    S_bin_l = tf.cholesky(S_bin)

    f_un = m_un + tf.einsum('lsij,lsj->lsi', S_un_l, eps_un)
    f_bin = m_bin + tf.matmul(S_bin_l, eps_bin)

    n_labels, batch_size, max_seq_len = m_un.get_shape()
    print('_sample_f/f_un', f_un.get_shape(), '=',
        n_labels, batch_size, max_seq_len)
    print('_sample_f/f_bin', f_bin.get_shape(), '=',
        max_seq_len**2)
    return f_un, f_bin

  def elbo(self, x, y, seq_lens):
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
    # TODO: check

    m_un, S_un, m_bin, S_bin = self._latent_vars_distribution(x, seq_lens)
    f_un, f_bin = self._sample(m_un, S_un, m_bin, S_bin)

#    l = tf.cast(tf.shape(y)[0], tf.float64) # batch size
    batch_size = tf.shape(y)[0]
#    N = tf.cast(self.N, dtype=tf.float64) 
    N = self.N 

    unnormalized_log_likelihood = crf_sequence_score(
        tf.transpose(m_un, [1, 2, 0]), y, seq_lens, 
        tf.transpose(m_bin, [self.n_labels, self.n_labels]))
    log_partition_estimate = crf_log_norm(
        tf.transpose(f_un, [1, 2, 0]), seq_lens, 
        tf.transpose(f_bin, [self.n_labels, self.n_labels]))
    
    # Likelihood
    elbo = 0
    elbo += unnormalized_log_likelihood
    elbo -= log_partition_estimate
    elbo /= batch_size
    
    print('GPC/elbo/complexity_penalty', self._unary_complexity_penalty().get_shape())
    print('GPC/elbo/complexity_penalty', self._binary_complexity_penalty().get_shape())
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
    # TODO: check 
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
    # TODO: implement 
    # How do we do this?
    pass
