import numpy as np
import tensorflow as tf
import t3f 
import t3f.kronecker as kr
from t3f import TensorTrain, TensorTrainBatch
import time

class InputsGrid:
  """A class representing inducing inputs placed on a grid.
  """

  def __init__(self, ndims, left=0., right=1., npoints=10):
    """
    Args:
      ndims: Number of dimensions of the grid.
      left: list or a number; the left bound of the interval for each 
        dimension.
      right: list or a number; the right bound of the interval for each 
        dimension.
      npoints: list or an interval; the number of points in the grid for 
        each dimension.
    """
    self.ndims = ndims
    self.inputs_tf = []
    self.steps_tf = []
    self.npoints = []
    if not hasattr(left, '__iter__'):
      left = [left] * ndims
    if not hasattr(right, '__iter__'):
      right = [right] * ndims
    if not hasattr(npoints, '__iter__'):
      npoints = [npoints] * ndims
    for i in range(ndims):
      h = (right[i] - left[i]) / (npoints[i]-1)
      self.inputs_tf.append(tf.cast(tf.linspace(left[i]-h, right[i]+h, 
                                                npoints[i]+2),
                                      dtype=tf.float64))
      self.steps_tf.append(h)
      self.npoints.append(npoints[i] + 2)
    self.steps_tf = tf.constant(self.steps_tf, dtype=tf.float64)
    self.left = tf.constant(left, dtype=tf.float64)
    self.right = tf.constant(right, dtype=tf.float64)

  def _add_padding(self):
    """Adds one point on each side of the grid for each dimension"""
    for i in range(self.ndims):
      inputs_i = self.inputs[i]
      n_points = self.npoints[i]
      left, right = self.left[i], self.right[i]
      h = (right - left) / (len(inputs_i)-1)
      self.inputs_tf.append(tf.cast(tf.linspace(left - h, right+h, n_points+2),
                                      dtype=tf.float64))

  def kron_dists(self):
    """Computes pairwise squared distances as kronecker-product matrix.
    
    This function returns a tt-tensor with tt-ranks 1 (a Kronecker product),
    with tt-cores, equal to squared pairwise distances between the grid 
    points in each dimension. This matrix can then be used to compute the 
    covariance matrix.
    """
    dist_dims = []
    for dim in range(self.ndims):
      dists = ((self.inputs_tf[dim]**2)[:, None] + 
              (self.inputs_tf[dim]**2)[None, :] -
              2*self.inputs_tf[dim][:, None]* self.inputs_tf[dim][None, :])
      dist_dims.append(dists[None, :, :, None])
    res_ranks = [1] * (self.ndims + 1)
    res_shape_i = tuple(self.npoints)
    res_shape = (res_shape_i, res_shape_i)
    return TensorTrain(dist_dims, res_shape, res_ranks)

  def interpolate_on_batch(self, x):
    """
    Computes the interpolation matrix for K_im matrix.

    Args:
      x: batch of points
    Returns:
      A `TensorTrainBatch` object, representing the matrix W_i: 
      K_im = W_i K_mm.
    """

    #n_test = x.get_shape()[0].value
    n_test = tf.cast(tf.shape(x), tf.int64)[0]
    n_inputs_dims = self.npoints
    w_cores = []
    hs_tensor = self.steps_tf
    closest_left = tf.cast(tf.floordiv(x-self.left[None, :], hs_tensor), tf.int64)
    
    y_indices = tf.reshape(
                    tf.tile(tf.range(n_test, dtype=tf.int64)[:, None], [1, 4]),
                    [-1])

    for dim in range(self.ndims):
      x_dim = x[:, dim][:, None]
      inputs_dim = self.inputs_tf[dim]
      core_shape = tf.concat([tf.cast([n_test], tf.int32), 
            [n_inputs_dims[dim]]], axis=0)
      core = tf.zeros(core_shape, dtype=tf.float64)
      x_indices = closest_left[:, dim][:, None] + tf.range(4, dtype=tf.int64)[None, :]
      x_indices = tf.maximum(x_indices, 0)
      x_indices = tf.minimum(x_indices, n_inputs_dims[dim]-1)

      s = tf.abs(x_dim - tf.gather(inputs_dim, x_indices)) / hs_tensor[dim]
      s_q = tf.pow(s, 3) / 2
      s_s = 2.5 * tf.pow(s, 2)
      values_03 = (- s_q +  s_s - 4 * s + 2)
      values_0 = values_03[:, 0:1]
      values_3 = values_03[:, 3:4]
      values_12 = (3 * s_q[:, 1:3] - s_s[:, 1:3] + 1)
      values = tf.concat([values_0, values_12, values_3], axis=1)
      indices = tf.concat([y_indices[:, None], 
                           tf.reshape(x_indices, [-1])[:, None]], axis=1)

      core = core + tf.scatter_nd(indices, tf.reshape(values, [-1]), 
                                      [n_test, n_inputs_dims[dim]])
      core = tf.expand_dims(core, 1)
      core = tf.expand_dims(core, -1)
      core = tf.expand_dims(core, -1)
      w_cores.append(core)
    W = TensorTrainBatch(w_cores)
    return W
