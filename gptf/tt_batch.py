import numpy as np
import tensorflow as tf
import t3f 
import t3f.kronecker as kr
from t3f import TensorTrain

# This is a trmporary module implementing a batch of TT-matrices and
# some operations, required by this project. It should be replaced when
# the required features are implemented in t3f

class BatchTTMatrices():

    def __init__(self, cores):
        
        batch_cores = cores
        self.tt_cores = batch_cores
    
    def ndims(self):
        return len(self.tt_cores)

    def get_nelems(self):
        return self.tt_cores[0].get_shape()[0]

    def get_tt_ranks(self):
        return tf.TensorShape([core.get_shape()[1] for core in self.tt_cores]
                + [1])

    def get_raw_shape(self):
        return (tf.TensorShape([core.get_shape()[2] for core in self.tt_cores]),
               tf.TensorShape([core.get_shape()[3] for core in self.tt_cores]))

def batch_full(tt):
    """Computes the full matrix of the given batch.
    """
    num_dims = tt.ndims()
    ranks = tt.get_tt_ranks()    
    res = tt.tt_cores[0]
    n_elems = tt.get_nelems().value
    for i in range(1, num_dims):
        res = tf.reshape(res, (n_elems, -1, ranks[i].value))
        curr_core = tf.reshape(tt.tt_cores[i], (n_elems, ranks[i].value, -1))
        res = tf.matmul(res, curr_core)
    raw_shape = tt.get_raw_shape()
    intermediate_shape = [n_elems]
    for i in range(num_dims):
        intermediate_shape.append(raw_shape[0][i])
        intermediate_shape.append(raw_shape[1][i])
    res = tf.reshape(res, tf.TensorShape(intermediate_shape))
    transpose = []
    for i in range(0, 2 * num_dims, 2):
        transpose.append(i)
    for i in range(1, 2 * num_dims, 2):
        transpose.append(i)
    transpose = [0] + [elem + 1 for elem in transpose]
    res = tf.transpose(res, transpose)
    tt_shape = (n_elems, np.prod(tt.get_raw_shape()[0]).value, 
                np.prod(tt.get_raw_shape()[1]).value)
    return tf.reshape(res, tt_shape)


def batch_subsample(tt_batch, batch_size, targets=None):
    """
    Generates a subsample of the batch.

    Creates the queue with tf.train.slice_input_producer and then batches the 
    output.
    Args:
        tt_batch: BatchTTMatrices object.
        batch_size: size of the subsample.
        targets: a tensor of target values. Use this argument to generate
            batches of both features and targets.
    """
    if targets is not None:
        tensors = tt_batch.tt_cores + [targets]
        sample = tf.train.slice_input_producer(tensors)
        batch = tf.train.batch(sample, batch_size)
        return BatchTTMatrices(batch[:-1]), batch[-1]
    else:
        sample = tf.train.slice_input_producer(tt_batch.tt_cores)
        batch = tf.train.batch(sample, batch_size)
        return BatchTTMatrices(batch)


def batch_tt_tt_flat_inner(tt_a, tt_b):
    """
    Inner product of the given tt-matrices.
    
    Args:
        tt_a: a batch of tt-matrices or a tt-matrix.
        tt_b: a batch of tt-matrices or a tt-matrix.
    """
    a_is_batch = isinstance(tt_a, BatchTTMatrices)
    b_is_batch = isinstance(tt_b, BatchTTMatrices)
    a_core = tt_a.tt_cores[0]
    b_core = tt_b.tt_cores[0]
    ndims = tt_a.ndims()

    if a_is_batch and b_is_batch:
        res = tf.einsum('naijb,ncijd->nbd', a_core, b_core)
    elif a_is_batch:
        res = tf.einsum('naijb,cijd->nbd', a_core, b_core)
    elif b_is_batch:
        res = tf.einsum('aijb,ncijd->nbd', a_core, b_core)
    else:
        res = tf.einsum('aijb,cijd->bd', a_core, b_core)

    for core_idx in range(1, ndims):
        a_core = tt_a.tt_cores[core_idx]
        b_core = tt_b.tt_cores[core_idx] 
        if a_is_batch and b_is_batch:
            res = tf.einsum('nac,naijb,ncijd->nbd', res, a_core, b_core)
        elif a_is_batch:
            res = tf.einsum('nac,naijb,cijd->nbd', res, a_core, b_core)
        elif b_is_batch:
            res = tf.einsum('nac,aijb,ncijd->nbd', res, a_core, b_core)
        else:
            res = tf.einsum('ac,aijb,cijd->bd', res, a_core, b_core)

    return res


def batch_transpose(tt_matrix):
    """Transpose a batch of TT-matrices.

    Args:
      tt_matrix: batch of tt-matrices.
    """

    transposed_tt_cores = []
    for core_idx in range(tt_matrix.ndims()):
        curr_core = tt_matrix.tt_cores[core_idx]
        transposed_tt_cores.append(tf.transpose(curr_core, (0, 1, 3, 2, 4)))

    tt_matrix_shape = tt_matrix.get_raw_shape()
    return BatchTTMatrices(transposed_tt_cores)


def batch_tt_tt_matmul(tt_matrix_a, tt_matrix_b):
    """Multiplies two TT-matrices and returns the TT-matrix of the result.
    
    Args:
        tt_a: a batch of tt-matrices or a tt-matrix.
        tt_b: a batch of tt-matrices or a tt-matrix.
    """

    a_is_batch = isinstance(tt_matrix_a, BatchTTMatrices)
    b_is_batch = isinstance(tt_matrix_b, BatchTTMatrices)
    res_batch = a_is_batch or b_is_batch
    if a_is_batch:
        n_elems = tt_matrix_a.get_nelems()
    elif b_is_batch:
        n_elems = tt_matrix_b.get_nelems()
    if a_is_batch and b_is_batch:
        if tt_matrix_a.get_nelems() != tt_matrix_b.get_nelems():
            raise ValueError('Batches have different number of elements')

    ndims = tt_matrix_a.ndims()
    if tt_matrix_b.ndims() != ndims:
        raise ValueError('Arguments should have the same number of dimensions, '
                         'got %d and %d instead.' % (ndims, tt_matrix_b.ndims()))
    result_cores = []
    a_shape = tt_matrix_a.get_raw_shape()
    a_ranks = tt_matrix_a.get_tt_ranks()
    b_shape = tt_matrix_b.get_raw_shape()
    b_ranks = tt_matrix_b.get_tt_ranks()
    
    for core_idx in range(ndims):
        a_core = tt_matrix_a.tt_cores[core_idx]
        b_core = tt_matrix_b.tt_cores[core_idx]
        if a_is_batch and b_is_batch:
            curr_res_core = tf.einsum('naijb,ncjkd->nacikbd', a_core, b_core)
        elif a_is_batch:
            curr_res_core = tf.einsum('naijb,cjkd->nacikbd', a_core, b_core)
        elif b_is_batch:
            curr_res_core = tf.einsum('aijb,ncjkd->nacikbd', a_core, b_core)
        else:
            curr_res_core = tf.einsum('aijb,cjkd->acikbd', a_core, b_core)

        res_left_rank = a_ranks[core_idx] * b_ranks[core_idx]
        res_right_rank = a_ranks[core_idx + 1] * b_ranks[core_idx + 1]
        left_mode = a_shape[0][core_idx]
        right_mode = b_shape[1][core_idx]
        if res_batch:
            core_shape = (n_elems, res_left_rank, left_mode, right_mode, 
                    res_right_rank)
        else:
            core_shape = (res_left_rank, left_mode, right_mode, res_right_rank)
        
        core_shape = tf.TensorShape(core_shape)
        curr_res_core = tf.reshape(curr_res_core, core_shape)
        result_cores.append(curr_res_core)
    
    if res_batch:
        return BatchTTMatrices(result_cores)
    return TensorTrain(result_cores)


def batch_quadratic_form(A, b, c):
    """Computes the quadratic form b^t A c where A is a TT-matrix.
    """
    b_is_batch = isinstance(b, BatchTTMatrices)
    c_is_batch = isinstance(c, BatchTTMatrices)
    if c_is_batch:
        return batch_tt_tt_flat_inner(A, batch_tt_tt_matmul(b, batch_transpose(c)))
    else:
        return batch_tt_tt_flat_inner(A, batch_tt_tt_matmul(b, t3f.ops.transpose(c)))

