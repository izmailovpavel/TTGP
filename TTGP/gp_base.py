import tensorflow as tf
import numpy as np

import t3f
import t3f.kronecker as kron
from t3f import ops, TensorTrain, TensorTrainBatch

from TTGP.misc import _kron_tril, _kron_logdet

class _TTGPbase:
  """A class representing Gaussian Process.

  This class is meant to be used in GP models for regression and
  classification. It stores covariance and variational parameters and provides
  methods for computing the complexity penalty term for ELBO of different GP
  models. The inducing inputs should be situated on a multidimensional grid, 
  and the variational parameters (see [Titsias 2009, Hensman 2015]) are 
  represented in the Tensor Train format ([Oseledets 2011]).
  """

  def __init__(self, cov, inputs, x_init, y_init, mu_ranks):
    '''
    Args:
      cov: covariance function.
      inputs: inducing inputs - InputsGrid object.
      x_init, y_init: tensors of data for initialization of GP parameters.
      mu_ranks: TT-ranks of mu - expectations of the process at
          inducing inputs.
    '''
    self.cov = cov
    self.inputs = inputs
    self.inputs_dists = inputs.kron_dists()
    self.sigma_l = self._get_sigma_l()
    self.mu = self._get_mu(mu_ranks, x_init, y_init)

  def initialize(self, sess):
    """Initializes the variational and covariance parameters.

    Args:
      sess: a `tf.Session` instance.
    """
    # TODO: how to avoid reinitializing projector?
    self.cov.initialize(sess)
    sess.run(tf.variables_initializer(self.sigma_l.tt_cores))
    sess.run(tf.variables_initializer(self.mu.tt_cores))

  def get_params(self):
    """Returns a list of all the parameters of the model.
    """
    # TODO: how to avoid including projector multiple times here?
    gp_var_params = list(self.mu.tt_cores + self.sigma_l.tt_cores)
    cov_params = self.cov.get_params()
    return cov_params + gp_var_params


  def _get_mu(self, ranks, x, y):
    """Initializes latent inputs expectations mu.

    Either loads pretrained values of tt-cores of mu, or initializes it
    according to optimal formulas from the given data.

    Args:
      ranks: tt-ranks of mu
      x: features of a batch of objects
      y: targets of a batch of objects
    """
    # TODO: test if this is needed.
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

  def _get_sigma_l(self):
    """Initializes latent inputs covariance Sigma_l.
    """
    shapes = self.inputs.npoints
    cov = self.cov
    inputs_dists = self.inputs_dists
    K_mm = cov.kron_cov(inputs_dists)    
    return t3f.get_variable('sigma_l', initializer=kron.cholesky(K_mm))

  def complexity_penalty(self):
    """Returns the complexity penalty term for ELBO of different GP models. 
    """
    mu = self.mu
    sigma_l = _kron_tril(self.sigma_l)
    sigma = ops.tt_tt_matmul(sigma_l, ops.transpose(sigma_l))
    sigma_logdet = _kron_logdet(sigma_l)

    K_mm = self.K_mm()
    K_mm_inv = kron.inv(K_mm)
    K_mm_logdet = kron.slog_determinant(K_mm)[1]

    elbo = 0
    elbo += - K_mm_logdet
    elbo += sigma_logdet
    elbo += - ops.tt_tt_flat_inner(sigma, K_mm_inv)
    elbo += - ops.tt_tt_flat_inner(mu, 
                           ops.tt_tt_matmul(K_mm_inv, mu))
    return elbo / 2

  def K_mm(self, eig_correction=1e-2):
    """Returns covariance matrix computed at inducing inputs. 
    """
    return self.cov.kron_cov(self.inputs_dists, eig_correction)

  def predict_process_value(self, x, with_variance=False):
    """Predicts the value of the process at point x.

    Args:
      x: data features
      with_variance: if True, returns process variance at x
    """
    mu = self.mu
    w = self.inputs.interpolate_on_batch(self.cov.project(x))

    mean = ops.tt_tt_flat_inner(w, mu)
    if not with_variance:
      return mean
    K_mm = self.K_mm()
    variance = self.cov.cov_0() 
    sigma_l_w = ops.tt_tt_matmul(ops.transpose(self.sigma_l), w)
    variance += ops.tt_tt_flat_inner(sigma_l_w, sigma_l_w)
    variance -= ops.tt_tt_flat_inner(w, ops.tt_tt_matmul(K_mm, w))
    return mean, variance
