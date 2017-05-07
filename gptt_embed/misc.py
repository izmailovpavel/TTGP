import tensorflow as tf
import numpy as np

import t3f
import t3f.kronecker as kron
from t3f import ops, TensorTrain, TensorTrainBatch

def r2(y_pred, y_true, name=None):
    """r2 score.
    """
    with tf.name_scope(name, 'r2_score', [y_pred, y_true]):
        mse_score = mse(y_pred, y_true)
        return 1. - mse_score / mse(tf.ones_like(y_true) * 
                    tf.reduce_mean(y_true), y_true)


def mse(y_pred, y_true, name=None):
    """MSE score.
    """
    with tf.name_scope(name, 'mse', [y_pred, y_true]):
        mse = tf.reduce_mean(tf.squared_difference(tf.reshape(y_pred, [-1]), 
                             tf.reshape(y_true, [-1])), name='MSE')
        return mse

def accuracy(y_pred, y_true, name=None):
    """Acuracy score.
    """
    correct_prediction = tf.equal(y_pred, y_true)
    return tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

def num_correct(y_pred, y_true, name=None):
    """Number of correct predictions.
    """
    correct_prediction = tf.equal(y_pred, y_true)
    return tf.reduce_sum(tf.cast(correct_prediction, tf.float32))

        
def _kron_tril(kron_mat, name=None):
    '''Computes the lower triangular part of a kronecker-factorized matrix.

    Note, that it computes it as a product of the lower triangular parts
    of the elements of the product, which is not exactly the lower 
    triangular part.
    '''
    is_batch = isinstance(kron_mat, TensorTrainBatch)
    with tf.name_scope(name, 'Kron_tril', [kron_mat]):
        mat_l_cores = []
        for core_idx in range(kron_mat.ndims()):
            if is_batch:
                core = kron_mat.tt_cores[core_idx][:, 0, :, :, 0]
                core_tril = tf.matrix_band_part(core,-1, 0)[:, None, :, :, None]
            else:
                core = kron_mat.tt_cores[core_idx][0, :, :, 0]
                core_tril = tf.matrix_band_part(core,-1, 0)[None, :, :, None]
            mat_l_cores.append(core_tril)
        mat_l_shape = kron_mat.get_raw_shape()
        mat_l_ranks = kron_mat.get_tt_ranks()
        if is_batch:
            mat_l = TensorTrainBatch(mat_l_cores, mat_l_shape, mat_l_ranks)
        else:
            mat_l = TensorTrain(mat_l_cores, mat_l_shape, mat_l_ranks)
        return mat_l


def _kron_logdet(kron_mat, name=None):
    '''Computes the logdet of a kronecker-factorized matrix.
    '''
    with tf.name_scope(name, 'Kron_logdet', [kron_mat]):
        is_batch = isinstance(kron_mat, TensorTrainBatch)
        i_shapes = kron_mat.get_raw_shape()[0]
        pows = tf.cast(tf.reduce_prod(i_shapes), kron_mat.dtype)
        logdet = 0.
        for core_idx in range(kron_mat.ndims()):
            core_pow = pows / i_shapes[core_idx].value
            if is_batch:
                core = kron_mat.tt_cores[core_idx][:, 0, :, :, 0]
                logdet += (core_pow * tf.reduce_sum(tf.log(tf.abs(
                                    tf.matrix_diag_part(core))), axis=1))
            else:
                core = kron_mat.tt_cores[core_idx][0, :, :, 0]
                logdet += (core_pow * tf.reduce_sum(tf.log(tf.abs(
                                    tf.matrix_diag_part(core)))))
        logdet *= 2
        return logdet

def pairwise_quadratic_form(A, b, c):
  """???

      res[i, j] = t3f.flat_inner(tt[i], t3f.matmul(matrix[j], tt[i]))
    or more shortly
      res[i, j] = tt[i]^T * matrices[j] * tt[i]
    but is more efficient.

  Args:
    A: TensorTrainBatch of TT-matrices.
    b: TensorTrainBatch.
    c: TensorTrainBatch.

  Returns:
    tf.tensor with the matrix of pairwise scalar products (flat inners).
  """
  ndims = A.ndims()
  curr_core_A = A.tt_cores[0][:, 0, :, :, :]
  curr_core_b = b.tt_cores[0][:, 0, :, :, :]
  curr_core_c = c.tt_cores[0][:, 0, :, :, :]
  res = tf.einsum('qikd,pkjf,pijb->pqbdf', curr_core_A, curr_core_c,
                  curr_core_b)
  for core_idx in range(1, ndims):
    curr_core_A = A.tt_cores[core_idx]
    curr_core_b = b.tt_cores[core_idx]
    curr_core_c = c.tt_cores[core_idx]
    res = tf.einsum('qcikd,pekjf,paijb,pqace->pqbdf', curr_core_A, curr_core_c,
                    curr_core_b, res)

  return tf.squeeze(res)
