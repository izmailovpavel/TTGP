import tensorflow as tf
import numpy as np

import t3f
import t3f.kronecker as kron
from t3f import ops, TensorTrain, TensorTrainBatch

def r2(y_pred, y_true):
    """r2 score.
    """
    mse_score = mse(y_pred, y_true)
    return 1. - mse_score / mse(tf.ones_like(y_true) * 
                tf.reduce_mean(y_true), y_true)


def mse(y_pred, y_true):
    """MSE score.
    """
    mse = tf.reduce_mean(tf.squared_difference(tf.reshape(y_pred, [-1]), 
                         tf.reshape(y_true, [-1])), name='MSE')
    return mse

def accuracy(y_pred, y_true):
    """Acuracy score.
    """
    correct_prediction = tf.equal(y_pred, y_true)
    return tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

def accuracy_struct(y_pred, y_true):
    """Accuracy score for structured prediction.
    """
    sum_right = 0
    sum_len = 0
    for y, pred in list(zip(y_true, y_pred)):
        sum_right += np.sum(y[:len(pred)] == np.array(pred))
        sum_len += len(pred)
    return sum_right / sum_len

def num_correct(y_pred, y_true):
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

def _kron_sequence_pairwise_quadratic_form(mat, vec, seq_lens, max_seq_len):
  """Computes the pairwise quadratic form `vec.T` `mat` `vec` for sequence data.

  This function is implemented for computing the variational covariance matrix 
  for binary potentials in GPstruct model.

  Args:
    mat: a kronecker (all TT-ranks equal to 1) `t3f.TensorTrainBatch` of shape 
      `n_labels` x `M` x `M`.
    vec: a kronecker (all TT-ranks equal to 1) `t3f.TensorTrainBatch` of shape 
      `sum_lens` x `M`, where `sum_lens` is the sum of `seq_lens`.
    seq_lens: `tf.Tensor` of shape `bach_size`; lenghts of input sequences.
    max_seq_len: a number, maximum length of a sequence in the batch.

  Return:
    a `tf.Tensor` of shape 
    `n_labels` x `batch_size` x `max_seq_len` x `max_seq_len`.
  """
  sequence_mask = tf.sequence_mask(seq_lens, maxlen=max_seq_len)
  indices = tf.cast(tf.where(sequence_mask), tf.int32)
  n_labels = mat.batch_size
  dtype = mat.dtype
  batch_size = tf.shape(seq_lens)[0]
  result = tf.ones([n_labels, batch_size, max_seq_len, max_seq_len], 
      dtype=dtype)

  if mat.ndims() != vec.ndims():
    raise ValueError('`mat` and `vec` should have the same ndims')

  for core_idx in range(mat.ndims()):
    cur_core_mat = mat.tt_cores[core_idx][:, 0, :, :, 0]
    cur_core_vec = vec.tt_cores[core_idx][:, 0, :, 0, 0]
    dim = tf.shape(cur_core_vec)[-1]
    cur_shape = tf.concat([[batch_size], [max_seq_len], [dim]], axis=0)
    cur_shape = tf.cast(cur_shape, dtype=tf.int32)
    cur_core_vec = tf.scatter_nd(indices, cur_core_vec, cur_shape)
    core_res = tf.einsum('sia,lab,sjb->lsij', 
        cur_core_vec, cur_core_mat, cur_core_vec) 
    result *= core_res
    
  return result
