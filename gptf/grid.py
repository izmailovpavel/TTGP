import numpy as np
import tensorflow as tf
import t3f 
import t3f.kronecker as kr
from t3f.tensor_train import TensorTrain

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

    def _add_padding(self):
        """Adds one point on each side of the grid for each dimension"""
        for i in range(self.ndims):
            inputs_i = self.inputs[i]
            left, right = inputs_i[0], inputs_i[-1]
            h = (right - left) / (len(inputs_i)-1)
            self.inputs[i] = np.array([left - h] + inputs_i.tolist() 
                                      + [right+h])[:, None]

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
            dist_dims.append(dists[None, :, :, None])
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

            W = np.zeros((n_test, n_inputs))
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

    def full(self):
        """Returns the full np.array of point coordinates.
        
        Note: the array might be extremely large.
        """
        return np.hstack([coord.reshape(-1)[:, None] for coord 
                          in np.meshgrid(*self.inputs, indexing='ij')])
