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
        self.inputs = []
        self.inputs_tf = []
        if not hasattr(left, '__iter__'):
            left = [left] * ndims
        if not hasattr(right, '__iter__'):
            right = [right] * ndims
        if not hasattr(npoints, '__iter__'):
            npoints = [npoints] * ndims
        for i in range(ndims):
            self.inputs.append(np.linspace(left[i], right[i], npoints[i]))
        self._add_padding()
        self.left = [dim[0] for dim in self.inputs]
        self.right = [dim[-1] for dim in self.inputs]
        self.npoints = [len(dim) for dim in self.inputs]
        self.size = np.prod([dim.size for dim in self.inputs])
        self.steps = [dim[1] - dim[0] for dim in self.inputs]
        self.steps_tf = tf.constant([dim[1, 0] - dim[0, 0] for dim in 
                                    self.inputs], dtype=tf.float64)

    def _add_padding(self):
        """Adds one point on each side of the grid for each dimension"""
        for i in range(self.ndims):
            inputs_i = self.inputs[i]
            n_points = inputs_i.size
            left, right = inputs_i[0], inputs_i[-1]
            h = (right - left) / (len(inputs_i)-1)
            self.inputs[i] = np.array([left - h] + inputs_i.tolist() 
                                      + [right+h])[:, None]
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
            dists = ((self.inputs[dim]**2)[:, None] + 
                    (self.inputs[dim]**2)[None, :] -
                    2*self.inputs[dim][:, None]* self.inputs[dim][None, :])
            dist_dims.append(dists[None, :, :])#, None])
        res_ranks = [1] * (self.ndims + 1)
        res_shape_i = tuple(self.npoints)
        res_shape = (res_shape_i, res_shape_i)
        return TensorTrain(dist_dims, res_shape, res_ranks)

    def interpolate_kernel(self, points):
        """Computes the interpolation matrix for computing kernel.
        
        TODO: Has to be rewritten more efficiently.

        Args:
            points: an NxD numpy array, the training data

        Returns:
            A list of Nxn_points tensorflow tensors of length ndims, 
            where n_points is the number of points in each dimension in
            the grid.
        """
        n_test, n_inputs = points.shape[0], self.size
        n_inputs_dims = self.npoints
        w = []
        for dim in range(self.ndims):
            w.append(np.zeros((n_test, n_inputs_dims[dim])))

            hs = self.steps

        for dim in range(self.ndims):
            h = hs[dim]
            point_dim = points[:, dim][:, None]
            inputs_dim = self.inputs[dim].T
            temp = point_dim - inputs_dim
            temp[temp <= 0] = np.inf
            closest_left = np.argmin(temp, axis=1)
            for j in range(-1, 3):
                y_indices = np.arange(n_test)
                x_indices = closest_left + j
                idx = np.logical_and(x_indices >= 0, 
                        x_indices < n_inputs_dims[dim])
                y_indices = y_indices[idx]
                x_indices = x_indices[idx]
                s = np.abs(point_dim[idx, 0] - inputs_dim[0, x_indices]) / h
                s_1 = np.logical_and(s <= 2, s > 1)
                s_0 = s <= 1
                w[dim][y_indices[s_1], x_indices[s_1]] = (- s[s_1]**3 / 2 +  
                                            5 * s[s_1]**2 /2 - 4 * s[s_1] + 2)
                w[dim][y_indices[s_0], x_indices[s_0]] = (3 * s[s_0]**3 / 2 -  
                                            5 * s[s_0]**2 / 2 + 1)

        W = []
        for dim in range(self.ndims):
            W.append(tf.convert_to_tensor(w[dim])) 
        return W
    
    def interpolate_on_batch(self, x):
        """
        Computes the interpolation matrix for K_im matrix.

        Args:
            x: batch of points
        Returns:
            A `TensorTrainBatch` object, representing the matrix W_i: 
            K_im = W_i K_mm.
        """
        n_test, n_inputs = x.get_shape()[0].value, self.size 
        n_inputs_dims = self.npoints
        w_cores = []
        hs_tensor = self.steps_tf
        closest_left = tf.cast(tf.floordiv(x, hs_tensor), tf.int64)

        for dim in range(self.ndims):
            #print('interpolate_to_batch, closest left', closest_left.get_shape())
            x_dim = x[:, dim]
            inputs_dim = self.inputs_tf[dim]
            core = tf.zeros((n_test, n_inputs_dims[dim]), dtype=tf.float64)
            for j in range(4):
                y_indices = tf.range(n_test, dtype=tf.int64)
                x_indices = closest_left[:, dim] + j
                idx = tf.logical_and(x_indices >= 0, 
                        x_indices < n_inputs_dims[dim])
                #print('interpolate_on_batch, idx.dtype', idx.dtype)
                #print('interpolate_on_batch, idx.shape', idx.get_shape())
                #print('interpolate_on_batch, x_indices.shape', x_indices.get_shape())
                #print('interpolate_on_batch, y_indices.shape', y_indices.get_shape())
                y_indices = tf.boolean_mask(y_indices, idx)
                x_indices = tf.boolean_mask(x_indices, idx)
                #print('interpolate_on_batch, x_dim.shape', x_dim.get_shape())
                #print('interpolate_on_batch, inputs_dim.shape', inputs_dim.get_shape())
                #print('interpolate_on_batch, inputs_dim[x_indices].shape', 
                #        tf.gather(inputs_dim, x_indices).get_shape())
                s = tf.abs(tf.boolean_mask(x_dim, idx) - 
                           tf.gather(inputs_dim, x_indices)) / hs_tensor[dim]
                #print('interpolate_on_batch, s.shape', s.get_shape())
                s_is_1 = tf.logical_and(s <= 2, s > 1)
                s_is_0 = s <= 1
                #print('interpolate_on_batch, x_indices[s_1].shape', 
                #                        tf.boolean_mask(x_indices, s_is_1).shape)
                #print('interpolate_on_batch, y_indices[s_1].shape', 
                #                        tf.boolean_mask(y_indices, s_is_1).shape)
                indices_1 = tf.concat([tf.boolean_mask(y_indices, s_is_1)[:, None], 
                                       tf.boolean_mask(x_indices, s_is_1)[:, None]], 
                                       axis=1)
                indices_0 = tf.concat([tf.boolean_mask(y_indices, s_is_0)[:, None], 
                                       tf.boolean_mask(x_indices, s_is_0)[:, None]], 
                                       axis=1)
                #print('interpolate_on_batch, indices_1.shape', indices_1.get_shape())
                #print('interpolate_on_batch, indices_0.shape', indices_0.get_shape())
                indices = tf.concat([indices_1, indices_0], axis=0)
                #print('interpolate_on_batch, indices.shape', indices.get_shape())
                s_1 = tf.boolean_mask(s, s_is_1)
                s_0 = tf.boolean_mask(s, s_is_0)
                values_1 = (- s_1**3 / 2 +  5 * s_1**2 /2 - 4 * s_1 + 2)
                values_0 = (3 * s_0**3 / 2 - 5 * s_0**2 / 2 + 1)
                values = tf.concat([values_1, values_0], axis=0)
                #print('interpolate_on_batch, values.shape', values.shape)
                core_j = tf.sparse_to_dense(sparse_indices=indices, 
                            sparse_values=values, validate_indices=False,
                            output_shape=(n_test, n_inputs_dims[dim]))
                core = core + core_j
            w_cores.append(core)
        W = TensorTrainBatch([tf.reshape(core, [core.get_shape()[0].value, 1, 
                                           core.get_shape()[1].value, 1, 1])
                                           for core in w_cores])
        return W
                
                


