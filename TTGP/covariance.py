import tensorflow as tf
import numpy as np

import t3f
import t3f.kronecker as kron
from t3f import ops, TensorTrain, TensorTrainBatch

# TODO: write an abstract base class for covariances

class SE:

  def __init__(self, sigma_f, l, sigma_n, projector, trainable=True, 
              name_append=''):
      """Squared Exponential kernel.
      Args:
        sigma_f: process variance
        l: process length scale
        sigma_n: noise variance
        projector: `FeatureTransformer` object
        trainable: Bool, parameters are trainable iff True
      """
      with tf.name_scope('SE'+name_append):
        self.sigma_f = tf.get_variable('Process_variance'+name_append, [1], 
                            initializer=tf.constant_initializer(sigma_f), 
                            dtype=tf.float64, trainable=trainable)
        self.l = tf.get_variable('Process_lengthscale'+name_append, [1], 
                            initializer=tf.constant_initializer(l), 
                            dtype=tf.float64, trainable=trainable)
        self.sigma_n = tf.get_variable('Noise_variance'+name_append, [1], 
                            initializer=tf.constant_initializer(sigma_n), 
                            dtype=tf.float64, trainable=trainable)
      self.projector = projector

  def project(self, x, name=None):
      """Transforms the features of x with projector.
      Args:
        x: batch of data to be transformed through the projector.
      """
      with tf.name_scope(name, 'SE_project', [x]):
        return self.projector.transform(x)

  def kron_cov(self, kron_dists, eig_correction=1e-2, name=None):
      """Computes the covariance matrix, given a kronecker product 
      representation of distances.

      Args:
        kron_dists: kronecker product representation of pairwise
            distances.
        eig_correction: eigenvalue correction for numerical stability.
        name: name for the op.
      """
      with tf.name_scope(name, 'SE_kron_cov', [kron_dists]):
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

#  def __call__(self, x1, x2, name=None):
#      return self.cov(x1, x2, name)

  def get_params(self):
    cov_params = [self.sigma_f, self.l, self.sigma_n]
    projector_params = self.projector.get_params()
    return cov_params + projector_params

  def initialize(self, sess):
    """Runs the initializers for kernel parameters.

    Args:
      sess: `tf.Session` object
    """
    self.projector.initialize(sess)
    sess.run(tf.variables_initializer([self.sigma_f, self.l, self.sigma_n]))

  def feature_dim(self):
    """Returns the dimensionality of feature space used.
    """
    return self.projector.out_dim()

  def noise_variance(self):
    """Returns the noise variance of the process.
    """
    return self.sigma_n

  def cov_0(self):
    """Returns covariance between a point and itself.
    """
    #return self.sigma_n**2 + self.sigma_f**2
    return self.sigma_f**2


class SE_multidim:
  """Multidimensional SE kernel.

  This class is meant to be used in multiclass gp classification.
  """
  # TODO: merge this class with SE

  def __init__(self, n_dims, sigma_f, l, sigma_n, projector, trainable=True):
      """Squared Exponential kernel.
      Args:
        n_dims: number of dimensions
        sigma_f: process variance
        l: process length scale
        sigma_n: noise variance
        projector: `FeatureTransformer` object
        trainable: Bool, parameters are trainable iff True
      """
      with tf.name_scope('SE_multidim'):
        sigma_f_init = tf.cast(tf.fill([n_dims], sigma_f), dtype=tf.float64)
        l_init = tf.cast(tf.fill([n_dims], l), dtype=tf.float64) 
        sigma_n_init = tf.cast(tf.fill([n_dims], sigma_n), dtype=tf.float64)
        self.sigma_f = tf.get_variable('Process_variance',  
                            initializer=sigma_f_init, 
                            dtype=tf.float64, trainable=trainable)
        self.l = tf.get_variable('Process_lengthscale', 
                            initializer=l_init, 
                            dtype=tf.float64, trainable=trainable)
        self.sigma_n = tf.get_variable('Noise_variance', 
                            initializer=sigma_n_init, 
                            dtype=tf.float64, trainable=trainable)
      self.projector = projector
      self.ndim = n_dims

  def project(self, x, test=False, name=None):
      """Transforms the features of x with projector.
      Args:
        x: batch of data to be transformed through the projector.
      """
      with tf.name_scope(name, 'SE_project', [x]):
        return self.projector.transform(x, test=test)

  def kron_cov(self, kron_dists, eig_correction=1e-2, name=None):
    """Computes the covariance matrix, given a kronecker product 
    representation of distances.

    Args:
      kron_dists: kronecker product representation of pairwise
          distances.
      eig_correction: eigenvalue correction for numerical stability.
      name: name for the op.
    """
    with tf.name_scope(name, 'SE_kron_cov', [kron_dists]):
      res_cores = []
      sigma_f = self.sigma_f[:, None, None, None, None]
      l = self.l[:, None, None, None, None]
#      sigma_n = self.sigma_n[:, None, None, None, None]
      for core_idx in range(kron_dists.ndims()):
        core = kron_dists.tt_cores[core_idx][None, :]
        cov_core = (sigma_f**(2./ kron_dists.ndims()) * 
                    tf.exp(-core/(2. * l**2.)))
        cov_core += tf.tile(eig_correction *
                    tf.eye(core.get_shape()[2].value, dtype=tf.float64)
                    [None, None, :, :, None],
                    [self.ndim, 1, 1, 1, 1])
        res_cores.append(cov_core)
      res_shape = kron_dists.get_raw_shape()
      res_ranks = kron_dists.get_tt_ranks()
      return TensorTrainBatch(res_cores, res_shape, res_ranks)

  def cov_for_squared_dists(self, sq_dists, eig_correction=1e-2): 
    """Computes the covariance matrix given distances between objects.

    Note that this function doesn't add noise variance. 
    Args:
      sq_dists: `tf.Tensor` of shape ... x N x M; two innermost dimensions
        contain matrices of pairwisesquared distances.
      eig_correction: eigenvalue correction for numerical stability.

    Returns:
      A `tf.Tensor` of shape `ndim` x... x N x M ; two innermost dimenstions 
      contain covariance matrices.
    """
    # TODO: check this
    n_extra_dims = len(sq_dists.get_shape())
    sigma_f = self.sigma_f
    l = self.l
    for i in range(n_extra_dims):
      sigma_f = sigma_f[:, None]
      l = l[:, None]
    sq_dists = sq_dists[None, :]
    cov = sigma_f ** 2 * tf.exp(-sq_dists / (2 * l**2))
    print('cov_for_squared_dists/cov', cov.get_shape(), '=', 
        [self.ndim] + sq_dists.get_shape().as_list()[1:])
    return cov

  def get_params(self):
    cov_params = [self.sigma_f, self.l, self.sigma_n]
    projector_params = self.projector.get_params()
    return cov_params + projector_params

  def initialize(self, sess):
    """Runs the initializers for kernel parameters.

    Args:
      sess: `tf.Session` object
    """
    self.projector.initialize(sess)
    sess.run(tf.variables_initializer([self.sigma_f, self.l, self.sigma_n]))

  def feature_dim(self):
    """Returns the dimensionality of feature space used.
    """
    return self.projector.out_dim()

  def noise_variance(self):
    """Returns the noise variance of the process.
    """
    return self.sigma_n

  def cov_0(self):
    """Returns covariance between a point and itself.
    """
    # TODO: check this!
#    return self.sigma_n**2 + self.sigma_f**2
    return self.sigma_f**2


class BinaryKernel:
  def __init__(self, n_labels, alpha, trainable=True):
    """Kernel for binary potentials in GPstruct.

    The covariance matrices for this kernel are of the form `alpha**2 I`, 
    where `I` is the identity matrix.
    Args:
      alpha: Scale parameter.
      trainable: Bool, parameters are trainable iff True.
    """
    self.alpha = tf.Variable(initial_value=alpha, dtype=tf.float64)
    self.n_labels = n_labels

  def cov(self):
    """Returns the covariance matrix.
    """
    return self.alpha**2 * tf.eye(self.n_labels**2, dtype=tf.float64)

  def cov_logdet(self):
    """Returns the covariance logdet.
    """
    # TODO: check
    return 2 * self.n_labels**2 * tf.log(tf.abs(self.alpha))
