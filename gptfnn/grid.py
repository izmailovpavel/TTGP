import numpy as np
import tensorflow as tf
import t3f 
import t3f.kronecker as kr
from t3f import TensorTrain, TensorTrainBatch

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
        #self.inputs = []
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
            #self.inputs.append(np.linspace(left[i], right[i], npoints[i]))
        #self._add_padding()
        #self.left = [dim[0] for dim in self.inputs]
        #self.right = [dim[-1] for dim in self.inputs]
        #self.npoints = [len(dim) for dim in self.inputs]
        #self.size = np.prod([dim.size for dim in self.inputs])
        #self.steps = [dim[1] - dim[0] for dim in self.inputs]
        self.steps_tf = tf.constant(self.steps_tf, dtype=tf.float64)

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
        n_test = x.get_shape()[0].value
        n_inputs_dims = self.npoints
        w_cores = []
        hs_tensor = self.steps_tf
        closest_left = tf.cast(tf.floordiv(x, hs_tensor), tf.int64)

        for dim in range(self.ndims):
            x_dim = x[:, dim]
            inputs_dim = self.inputs_tf[dim]
            core = tf.zeros((n_test, n_inputs_dims[dim]), dtype=tf.float64)
            for j in range(4):
                y_indices = tf.range(n_test, dtype=tf.int64)
                x_indices = closest_left[:, dim] + j
                idx = tf.logical_and(x_indices >= 0, 
                        x_indices < n_inputs_dims[dim])
                y_indices = tf.boolean_mask(y_indices, idx)
                x_indices = tf.boolean_mask(x_indices, idx)
                s = tf.abs(tf.boolean_mask(x_dim, idx) - 
                           tf.gather(inputs_dim, x_indices)) / hs_tensor[dim]
                s_is_1 = tf.logical_and(s <= 2, s > 1)
                s_is_0 = s <= 1
                indices_1 = tf.concat([tf.boolean_mask(y_indices, s_is_1)[:, None], 
                                       tf.boolean_mask(x_indices, s_is_1)[:, None]], 
                                       axis=1)
                indices_0 = tf.concat([tf.boolean_mask(y_indices, s_is_0)[:, None], 
                                       tf.boolean_mask(x_indices, s_is_0)[:, None]], 
                                       axis=1)
                indices = tf.concat([indices_1, indices_0], axis=0)
                s_1 = tf.boolean_mask(s, s_is_1)
                s_0 = tf.boolean_mask(s, s_is_0)
                values_1 = (- s_1**3 / 2 +  5 * s_1**2 /2 - 4 * s_1 + 2)
                values_0 = (3 * s_0**3 / 2 - 5 * s_0**2 / 2 + 1)
                values = tf.concat([values_1, values_0], axis=0)
                core_j = tf.sparse_to_dense(sparse_indices=indices, 
                            sparse_values=values, validate_indices=False,
                            output_shape=(n_test, n_inputs_dims[dim]))
                core = core + core_j
            w_cores.append(core)
        W = TensorTrainBatch([tf.reshape(core, [core.get_shape()[0].value, 1, 
                                           core.get_shape()[1].value, 1, 1])
                                           for core in w_cores])
        return W
